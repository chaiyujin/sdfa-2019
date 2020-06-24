import torch
import torch.nn.functional as F
from .. import functions as fn


class ILayerExtended(object):

    @staticmethod
    def default_args():
        return dict(
            # prev module
            prev_activation=None,
            prev_batch_norm=None,
            prev_bn_first=False,
            prev_dropout=None,
            prev_drop_always=False,
            # after module
            activation=None,
            batch_norm=None,
            bn_first=False,
            dropout=None,
            drop_always=False,
            # weight
            init_method="kaiming",
            init_nonlinearity=None,
            weight_norm=False,
        )

    def _ext_init(self, out_features, **kwargs):
        defaults = self.default_args()

        # extend original module
        assert hasattr(self, "weight")

        # previous module
        prev_activation  = kwargs.get("prev_activation",  defaults["prev_activation"])
        prev_batch_norm  = kwargs.get("prev_batch_norm",  defaults["prev_batch_norm"])
        prev_bn_first    = kwargs.get("prev_bn_first",    defaults["prev_bn_first"])
        prev_dropout     = kwargs.get("prev_dropout",     defaults["prev_dropout"])
        prev_drop_always = kwargs.get("prev_drop_always", defaults["prev_drop_always"])
        self._ext_prev_act = fn.parse_activation(prev_activation)
        self._ext_prev_drop = prev_dropout if prev_dropout is not None else 0.0
        self._ext_prev_drop_always = prev_drop_always
        self._ext_prev_bn = fn.Identity()
        self._ext_prev_bn_first = prev_bn_first
        if prev_batch_norm is not None:
            batch_norm_type = kwargs.get("batch_norm_type")
            assert batch_norm_type in ["1d", "2d"], "'batch_norm_type' is not given as '1d' or '2d'"
            self._ext_prev_bn = getattr(torch.nn, "BatchNorm" + batch_norm_type)(out_features, **prev_batch_norm)

        # normally after module
        activation  = kwargs.get("activation",  defaults["activation"])
        batch_norm  = kwargs.get("batch_norm",  defaults["batch_norm"])
        bn_first    = kwargs.get("bn_first",    defaults["bn_first"])
        dropout     = kwargs.get("dropout",     defaults["dropout"])
        drop_always = kwargs.get("drop_always", defaults["drop_always"])
        self._ext_post_act = fn.parse_activation(activation)
        self._ext_post_drop = dropout if dropout is not None else 0.0
        self._ext_post_drop_always = drop_always
        self._ext_post_bn = fn.Identity()
        self._ext_post_bn_first = bn_first
        if batch_norm is not None:
            batch_norm_type = kwargs.get("batch_norm_type")
            assert batch_norm_type in ["1d", "2d"], "'batch_norm_type' is not given as '1d' or '2d'"
            self._ext_post_bn = getattr(torch.nn, "BatchNorm" + batch_norm_type)(out_features, **batch_norm)

        # about weight
        init_method = kwargs.get("init_method",       defaults["init_method"])
        init_nonlin = kwargs.get("init_nonlinearity", defaults["init_nonlinearity"])
        weight_norm = kwargs.get("weight_norm",       defaults["weight_norm"])
        assert init_method in ["kaiming", "glorot", "default"],\
            f"init_method shoule be 'kaiming', 'glorot', 'default', not {init_method}"
        if init_method == "kaiming":
            fn.kaiming_init(self, init_nonlin)
        elif init_method == "glorot":
            fn.glorot_init(self)
        if weight_norm:
            torch.nn.utils.weight_norm(self)

    def _ext_prev_module(self, x):
        # act, bn
        if self._ext_prev_bn_first:
            x = self._ext_prev_bn(x)
            x = self._ext_prev_act(x)
        else:
            x = self._ext_prev_act(x)
            x = self._ext_prev_bn(x)
        # dropout
        if self._ext_prev_drop > 0.0:
            training = self.training or self._ext_prev_drop_always
            x = F.dropout(input=x, p=self._ext_prev_drop, training=training)
        return x

    def _ext_post_module(self, x):
        # act, bn
        if self._ext_post_bn_first:
            x = self._ext_post_bn(x)
            x = self._ext_post_act(x)
        else:
            x = self._ext_post_act(x)
            x = self._ext_post_bn(x)
        # dropout
        if self._ext_post_drop > 0.0:
            training = self.training or self._ext_post_drop_always
            x = F.dropout(input=x, p=self._ext_post_drop, training=training)
        return x
