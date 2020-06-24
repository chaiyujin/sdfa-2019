import re
import torch
import torch.nn.functional as F
import numpy as np
from saber.utils import log


def freeze(m):
    for param in m.parameters():
        param.requires_grad = False
    return m


def unfreeze(m):
    for param in m.parameters():
        param.requires_grad = True
    return m


def count_parameters(graph):
    parameters = filter(lambda p: p.requires_grad, graph.parameters())
    params = np.sum([np.prod(p.size()) for p in parameters])
    params_str = [str(params // np.power(1000, s) % 1000)
                  for s in range(8) if params // np.power(1000, s) > 0]
    params_str = ",".join(reversed(params_str))
    log.info("Model {} has totally {} trainable parameters".format(
             graph.__class__.__name__, params_str))
    return params


""" reduce tensor """


def sum(tensor, dim=None, keepdim=False):
    if dim is None:
        # sum up all dim
        return torch.sum(tensor)
    # sum up given dim
    if isinstance(dim, int):
        dim = [dim]
    dim = sorted(dim)
    for d in dim:
        tensor = tensor.sum(dim=d, keepdim=True)
    if not keepdim:
        for i, d in enumerate(dim):
            tensor.squeeze_(d-i)
    return tensor


def mean(tensor, dim=None, keepdim=False):
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    # mean given dim
    if isinstance(dim, int):
        dim = [dim]
    dim = sorted(dim)
    for d in dim:
        tensor = tensor.mean(dim=d, keepdim=True)
    if not keepdim:
        for i, d in enumerate(dim):
            tensor.squeeze_(d-i)
    return tensor


def elements(tensor, dim=None):
    """ return number of elements of given dims """
    ret = 1
    if dim is None:
        dim = range(tensor.dim())
    if isinstance(dim, int):
        dim = [dim]
    for d in dim:
        ret *= tensor.size(d)
    return ret


""" weight norm """


def weight_norm(module):
    def fn(module):
        if hasattr(module, "weight"):
            module = torch.nn.utils.weight_norm(module)
        return module
    module.apply(fn)
    return module


def remove_weight_norm(module):
    def fn(module):
        if hasattr(module, "weight_v") and hasattr(module, "weight_g"):
            module = torch.nn.utils.remove_weight_norm(module)
        return module
    module.apply(fn)
    return module


""" initialization """


def _analyze_activation(activation):
    if activation is None:
        activation = "linear"
    if activation.find("leaky_relu@") == 0:
        # with negative_slope
        if re.match(r"leaky_relu@a:[\d\.]+", activation) is None:
            raise ValueError("Invalid '{}'. You may want "
                             "`leaky_relu@a:<neg_slope>`".format(activation))
        return "leaky_relu", float(activation[13:])
    elif activation.find("lrelu@") == 0:
        # with negative_slope
        if re.match(r"lrelu@a:[\d\.]+", activation) is None:
            raise ValueError("Invalid '{}'. You may want "
                             "`lrelu@a:<neg_slope>`".format(activation))
        return "leaky_relu", float(activation[8:])
    elif activation.find("glu@dim:") == 0:
        return "glu", int(activation[8:])
    else:
        return activation, 0.0


def glorot_init(m):
    torch.nn.init.xavier_normal_(m.weight)
    if hasattr(m, "bias") and m.bias is not None:
        m.bias.data.zero_()


def kaiming_init(m, nonlinearity, mode="fan_in"):
    nonlinearity, a = _analyze_activation(nonlinearity)
    torch.nn.init.kaiming_normal_(tensor=m.weight, a=a, mode=mode,
                                  nonlinearity=nonlinearity)
    if hasattr(m, "bias") and m.bias is not None:
        m.bias.data.zero_()


def zero_init(m):
    torch.nn.init.zeros_(m.weight)
    if hasattr(m, "bias") and m.bias is not None:
        torch.nn.init.zeros_(m.bias)


def orthogonal_init(m):
    assert isinstance(m, (torch.nn.GRU, torch.nn.LSTM,
                          torch.nn.GRUCell, torch.nn.LSTMCell))
    for name, param in m.named_parameters():
        if name.find("weight_ih") >= 0:
            torch.nn.init.xavier_uniform_(param)
        elif name.find("weight_hh") >= 0:
            torch.nn.init.orthogonal_(param)
        elif name.find("bias") >= 0:
            torch.nn.init.zeros_(param)
        else:
            raise NameError("unknown param {}".format(name))
    return m


""" activation """


def parse_activation(name):
    if name == "relu":
        return torch.nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return torch.nn.Sigmoid()
    elif name == "softmax":
        return torch.nn.Softmax()
    elif name == "softmax2d":
        return torch.nn.Softmax2d()
    elif name == "tanh":
        return torch.nn.Tanh()
    elif name == "softplus":
        return torch.nn.Softplus()
    elif name is None or name == "linear":
        return Identity()
    elif name.find("glu") == 0:
        assert name.find("@dim:") > 0
        _, dim = _analyze_activation(name)
        return GLU(dim)
    elif name.find("leaky_relu") == 0 or name.find("lrelu") == 0:
        if name == "leaky_relu" or name == "lrelu":
            a = 0.01
        else:
            _, a = _analyze_activation(name)
        return torch.nn.LeakyReLU(negative_slope=a, inplace=True)
    else:
        raise ValueError("Do not support activation of name {}".format(name))


""" padding for convolution """


def check_padding(padding):
    assert isinstance(padding, (int, str, tuple, list)),\
        "'padding' is not int, str, tuple or list"
    # return (torch_padding, padding_mode)
    if isinstance(padding, str):
        assert padding in ["same", "valid", "causal"]
        return 0, padding
    else:
        return padding, None


def get_pad_tuple(size, kernel_size, stride, dilation, padding):
    padlr = (size // stride - 1) * stride + dilation * (kernel_size-1) + 1 - size
    if padding == "same":
        right = padlr // 2
        left = padlr - right
        return (left, right)
    elif padding == "causal":
        return (padlr, 0)
    elif padding == "valid":
        return (0, 0)
    else:
        raise ValueError("unknown padding mode: {}".format(padding))


def conv_pad(inputs, kernel_size, stride, dilation=1, padding="same"):
    """
    `inputs` should be in shape: (B, C, T) or (B, C, W, H).
    `kernel_size` and `stride` should be <class 'int'>, if inputs is (B, C, T).
    `padding` should be in ['same', 'valid', 'causal']
    """

    assert inputs.dim() == 3 or inputs.dim() == 4, "inputs.dim() == {}, shoule be 3 or 4".format(inputs.dim())
    assert padding in ["same", "valid", "causal"]
    if padding == "valid":
        return inputs
    # same or causal
    if inputs.dim() == 3:  # B, C, T
        if not isinstance(kernel_size, int) and len(kernel_size) == 1: kernel_size = kernel_size[0]
        if not isinstance(stride, int) and len(stride) == 1: stride = stride[0]
        if not isinstance(dilation, int) and len(dilation) == 1: dilation = dilation[0]
        assert isinstance(kernel_size, int), "'kernel_size' should be <class 'int'>, not {}".format(type(stride))
        assert isinstance(stride, int), "'stride' should be <class 'int'>, not {}".format(type(stride))
        assert isinstance(dilation, int), "'dilation' should be <class 'int'>, not {}".format(type(stride))
        _pad_tup = get_pad_tuple(inputs.size(-1), kernel_size, stride, dilation, padding)
        ret = F.pad(inputs, _pad_tup)
        return ret
    else:
        def _to_tuple(x):
            return tuple(x) if isinstance(x, (list, tuple)) else (x, x)
        ksz = _to_tuple(kernel_size)
        hop = _to_tuple(stride)
        dil = _to_tuple(dilation)
        # last dim, second last dim
        _pad_tup = (get_pad_tuple(inputs.size(-1), ksz[-1], hop[-1], dil[-1], padding) +
                    get_pad_tuple(inputs.size(-2), ksz[-2], hop[-2], dil[-2], padding))
        return F.pad(inputs, _pad_tup)


def conv_unpad(inputs, want_size, kernel_size, stride, dilation=1, padding="same"):
    if not isinstance(want_size, (list, tuple)):
        want_size = [want_size]
    assert inputs.dim() == 3 or inputs.dim() == 4, "inputs.dim() == {}, shoule be 3 or 4".format(inputs.dim())
    assert padding in ["same", "valid", "causal"]
    assert len(want_size) == inputs.dim() - 2
    # for valid padding, check want_size
    if padding == "valid":
        inputs_size = list(inputs.shape[2:])
        assert inputs_size == want_size, f"inputs: {inputs_size}, want_size: {want_size}"
        return inputs
    # same or causal
    if inputs.dim() == 3:  # B, C, T
        assert isinstance(kernel_size, int), "'kernel_size' should be <class 'int'>, not {}".format(type(stride))
        assert isinstance(stride, int), "'stride' should be <class 'int'>, not {}".format(type(stride))
        assert isinstance(dilation, int), "'dilation' should be <class 'int'>, not {}".format(type(stride))
        _pad_tup = get_pad_tuple(want_size[-1], kernel_size, stride, dilation, padding)
        si, ei = _pad_tup[0], inputs.shape[-1] - _pad_tup[1]
        return inputs[:, :, si: ei]
    else:
        def _to_tuple(x):
            return tuple(x) if isinstance(x, (list, tuple)) else (x, x)
        ksz = _to_tuple(kernel_size)
        hop = _to_tuple(stride)
        dil = _to_tuple(dilation)
        # last dim, second last dim
        _pad_tup = (get_pad_tuple(want_size[-1], ksz[-1], hop[-1], dil[-1], padding) +
                    get_pad_tuple(want_size[-2], ksz[-2], hop[-2], dil[-2], padding))
        si0, ei0 = _pad_tup[0], inputs.shape[-1] - _pad_tup[1]  # last dim
        si1, ei1 = _pad_tup[2], inputs.shape[-2] - _pad_tup[3]  # second last dim
        return inputs[:, :, si1: ei1, si0: ei0]


def size_after_conv(lengths, kernel_size, stride, dilation=1, padding="same"):
    if lengths is None:
        return None
    _pad_tup = get_pad_tuple(lengths, kernel_size, stride, dilation, padding)
    lengths = lengths + _pad_tup[0] + _pad_tup[1] - dilation * (kernel_size - 1) - 1
    return lengths // stride + 1


""" mask """


def generate_invalid_mask(lengths, max_length=None, dtype=torch.bool):
    if lengths is None:
        return None
    if max_length is None:
        max_length = torch.max(lengths)
    mask = lengths.new_zeros((lengths.size(0), max_length), dtype=dtype)
    for i in range(mask.size(0)):
        assert lengths[i] <= max_length
        if lengths[i] < max_length:
            mask[i, lengths[i]:] = 1
    return mask


def generate_mask(lengths, max_length=None, dtype=torch.bool):
    if lengths is None:
        return None
    if max_length is None:
        max_length = torch.max(lengths)
    mask = lengths.new_zeros((lengths.size(0), max_length), dtype=dtype)
    for i in range(mask.size(0)):
        assert lengths[i] <= max_length
        mask[i, :min(lengths[i], max_length)] = 1
    return mask


def generate_position(lengths, max_length=None):
    if max_length is None:
        max_length = torch.max(lengths)
    position = torch.LongTensor([
        np.concatenate((
            np.arange(1, lengths[i] + 1, dtype=np.long),
            np.zeros(max_length - lengths[i], dtype=np.long)
        ))
        for i in range(lengths.size(0))
    ]).to(lengths.device)
    return position


""" timestep """


def right_shift_one_timestep(tensor, dim=-1, pad_value=0):
    if len(tensor.size()) == 3:
        # abandon the last, and append zero at first
        if dim == -1 or dim == 2:
            last = tensor[:, :,  -1].unsqueeze(2)
            left = tensor[:, :, :-1]
        elif dim == -2 or dim == 1:
            # BTC
            last = tensor[:,  -1, :].unsqueeze(1)
            left = tensor[:, :-1, :]
        else:
            raise ValueError("Should not shift batch dim")
    elif len(tensor.size()) == 2:
        if dim == -1 or dim == 1:
            last = tensor[:,  -1].unsqueeze(1)
            left = tensor[:, :-1]
        else:
            raise ValueError("Should not shift batch dim")
    # replace the first with 0's
    first = torch.zeros_like(last).fill_(pad_value)
    return torch.cat((first, left), dim=dim)


def get_timestep(tensor, time_dim, index):
    return tensor.index_select(
        dim=time_dim if time_dim >= 0 else tensor.dim() + time_dim,
        index=torch.LongTensor([
            index if index >= 0 else tensor.size(time_dim) + index
        ]).to(tensor.device))


def last_timestep(tensor, time_dim):
    return get_timestep(tensor, time_dim, -1)


""" misc """


def one_hot(tensor, n, fill_with=1):
    vec = tensor.new_zeros(tensor.size() + (n,))
    vec.scatter_(tensor.dim(), tensor.unsqueeze(-1), fill_with)
    return vec


def sort_index(length, descending=True):
    _, sort_idx = torch.sort(length, descending=descending)
    revs_idx = sort_idx.new_zeros(*sort_idx.size())
    for i in range(len(revs_idx)):
        revs_idx[sort_idx[i]].fill_(i)
    return sort_idx, revs_idx


def is_parallel(model):
    return isinstance(model, (
        torch.nn.parallel.DataParallel,
        torch.nn.parallel.DistributedDataParallel,
        torch.nn.parallel.DistributedDataParallelCPU
    ))


class scale_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        grad_out = grad * ctx.scale
        return grad_out, None


# ------------- #
# simple layers #
# ------------- #


class GradScaler(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self._scale = float(scale)

    def forward(self, inputs):
        return scale_grad.apply(inputs, self._scale)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GLU(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    def forward(self, x):
        return torch.nn.functional.glu(x, dim=self._dim)
