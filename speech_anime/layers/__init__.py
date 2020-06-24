import torch
import saber
from saber.nn.layers import easy_create
from .misc import MultiplicativeNoise, PcaUnprojection
from .pblstm import PyramidBiLSTM
from .freq_lstm import FreqLstm
from .attentions import create_self_atten, _Attention
from .rnn import _create_gru, _create_lstm
from .lstm2d import LSTM2d


__first_forward__ = dict()
__support_layers__ = {
    "lstm2d": LSTM2d,
    "freq-lstm": FreqLstm,
    "lstm": _create_lstm,
    "gru": _create_gru,
    "attn": create_self_atten,
    "mul-noise": MultiplicativeNoise,
    "pBLSTM": PyramidBiLSTM,
    **easy_create.LayerParser.__layer_types__
}


def create(layer_info, verbose=False):
    assert isinstance(layer_info, (tuple, list))
    assert len(layer_info) > 0
    assert layer_info[0] in __support_layers__,\
        "'{}' is not support!".format(layer_info[0])
    layer_info = list(layer_info)
    _creation = __support_layers__[layer_info[0]]
    parser = easy_create.LayerParser(
        layer_info, _creation,
        can_ignore_keys=["residual", "condition", "weight_norm"]
    )
    return parser.create(verbose), parser


def build_layers(tag, layer_info_list, hparams):
    # shared parameters
    verbose = hparams.model.verbose
    weight_norm = hparams.model.weight_norm
    # build layers
    layers = torch.nn.ModuleList()
    parsers = []
    for layer_info in layer_info_list:
        layer_info = list(layer_info)
        layer_info.append("weight_norm={}".format(weight_norm))
        module, parser = create(layer_info, verbose=False)
        layers.append(module)
        parsers.append(parser)
    if verbose:
        saber.nn.layers.LayerParser.print_table(f"build '{tag}'", *parsers)
    return layers, parsers


def forward_layer(
    tag, inputs, module, parser,
    condition=None, align_dict=None, latent_dict=None,
    **kwargs
):
    # maybe combine inputs with condition
    if condition is not None and parser.get("cat_condition"):
        cat_dim = parser.get("cat_condition")
        if cat_dim < 0: cat_dim += inputs.dim()
        assert cat_dim > 0
        assert condition.dim() == 2

        shape = [-1]
        for i in range(1, inputs.dim()):
            if cat_dim == i:
                shape.append(-1)
            else:
                condition = condition.unsqueeze(i)
                shape.append(inputs.shape[i])
        condition = condition.expand(*shape)
        inputs = torch.cat((inputs, condition), dim=cat_dim)
    # forward
    if isinstance(module, (torch.nn.LSTM, torch.nn.GRU, PyramidBiLSTM)):
        # return output and hidden state
        ret, _ = module(inputs, **kwargs)
    elif isinstance(module, _Attention):
        # return output and alignment
        x = inputs
        _ahead = parser.query_radius - 1
        _after = parser.query_radius
        mid = x.size(1)//2 + parser.get("query_offset", 0)
        stt = mid - _ahead
        end = mid + _after
        query = x[:, stt: end, :]
        ret, align = module(query=query, key=x)
        if isinstance(align_dict, dict):
            align_dict[tag] = align
    else:
        ret = module(inputs, **kwargs)
    assert torch.is_tensor(ret)
    return ret


def forward(
    tag, inputs, layers, parsers, training=True,
    condition=None, align_dict=None, latent_dict=None,
    **kwargs
):

    def _liststr(vals):
        return "(" + ",".join(str(x).rjust(4) for x in vals) + ")"

    if tag not in __first_forward__:
        __first_forward__[tag] = training

    if __first_forward__[tag]:
        saber.log.info(f"Module '{tag}'")
        saber.log.info(f"- inputs   | {' ':13} | {_liststr(inputs.shape)}")

    history_inputs = []
    x = inputs
    for i, (module, parser) in enumerate(zip(layers, parsers)):
        # cache input
        history_inputs.append(x)
        out = forward_layer(
            tag="{}{}".format(tag, str(i).zfill(2)),
            inputs=x,
            module=module,
            parser=parser,
            align_dict=align_dict,
            condition=condition,
        )
        if __first_forward__[tag]:
            if condition is not None and parser.get("condition", False):
                concat_shape = list(x.shape)
                concat_shape[1] += condition.shape[1]
                saber.log.info(f"- concat   | {' ':13} | {_liststr(concat_shape)}")
            saber.log.info(f"- layer {i:02d} | {parser.name:13} | {_liststr(out.shape)}")
        # skip connection
        skip = parser.get("skip_connect")
        if isinstance(skip, int):
            residual = history_inputs[skip]
            out = out + residual
        # next layer
        x = out

    __first_forward__[tag] = False
    return x
