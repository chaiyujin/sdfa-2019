import math
import torch
import saber
import numpy as np


def create_self_atten(
    name, memory_size, num_units, query_radius,
    smooth=False, scale_score_at_eval=1.0,
    num_k=None, softmax=False, scale_x=6.0,
    num_heads=None,
    **kwargs
):
    qry_size = memory_size
    key_size = memory_size
    if name == "bah":
        return BahdanauAttention(
            num_units, qry_size, key_size,
            query_radius=query_radius,
            smooth=smooth,
            scale_score_at_eval=scale_score_at_eval
        )
    elif name == "gmm":
        assert num_k is not None
        return GmmAttention(
            num_units, qry_size, key_size, num_k,
            query_radius=query_radius,
            softmax=softmax, scale_x=scale_x
        )
    elif name == "prod":
        return ProdAttention(num_units, qry_size, key_size, query_radius=query_radius)
    elif name == "multi-head":
        assert num_heads is not None
        return MultiHeadAttention(num_units, qry_size, key_size, key_size, num_heads, query_radius=query_radius)
    else:
        raise NotImplementedError()


class _Attention(torch.nn.Module):
    def __init__(self, num_units, query_size, key_size, value_size=None, same_kv=False, query_radius=1):
        super().__init__()
        self.qry_size = query_size
        self.qry_length = query_radius * 2 - 1
        self.key_size = key_size
        self.val_size = value_size or key_size
        self.num_units = num_units
        self._same_kv = same_kv
        # project query
        self._conv_query = saber.nn.layers.Conv1d(
            query_size, query_size,
            kernel_size=self.qry_length,
            stride=self.qry_length,
            bias=False
        )

    def forward(self, query, key, value=None):
        if value is None:  # default value from key
            value = key
        if self._same_kv:  # check same
            assert key.data_ptr() == value.data_ptr()
        assert query.shape[1] == self.qry_length and query.shape[2] == self.qry_size,\
            "query should be in shape (N, {}, {}), but {}".format(self.qry_length, self.qry_size, query.shape)
        assert key.size(2)   == self.key_size, "key should be in shape (N, T, {})".format(self.key_size)
        assert value.size(2) == self.val_size, "value should be in shape (N, T, {})".format(self.val_size)
        assert key.size(1) == value.size(1),\
            "key, value has different length: {} != {}".format(
                key.size(1), value.size(1))
        # project query
        query = self._conv_query(query.permute(0, 2, 1).contiguous())\
                    .permute(0, 2, 1).contiguous()
        align = self.get_alignment(query, key)
        assert align.size(1) == query.size(1)
        assert align.size(2) == key.size(1)
        context = torch.bmm(align, value)
        return context, align

    def get_alignment(self, query, key):
        """
        Input:
            query: (N, T_q, C_q)
            key:   (N, T_k, C_k)
        Return:
            alignment: (N, T_q, T_k)
        """
        raise NotImplementedError()


def _smoothing_normalization(e, dim=-1):
    return torch.sigmoid(e) / torch.sum(torch.sigmoid(e), dim=dim, keepdim=True)


class BahdanauAttention(_Attention):
    def __init__(self, num_units, query_size, key_size, query_radius=1, smooth=False, scale_score_at_eval=1.0):
        super().__init__(num_units, query_size, key_size, query_radius=query_radius, value_size=None, same_kv=False)
        self.score_scaling = scale_score_at_eval
        # projections
        self.proj_key = saber.nn.layers.FullyConnected(self.key_size, self.num_units, bias=False, init_method="glorot")
        self.proj_qry = saber.nn.layers.FullyConnected(self.qry_size, self.num_units, bias=False, init_method="glorot")
        self.v = saber.nn.layers.FullyConnected(self.num_units, 1, bias=False, init_method="glorot")
        self.b = torch.nn.Parameter(torch.zeros((1, 1, self.num_units)))
        # normalize function
        if not smooth:
            self.normalize = torch.nn.functional.softmax
        else:
            self.normalize = _smoothing_normalization

    def _get_score(self, query, key):
        assert query.size(1) == 1
        B, N, _ = key.size()
        b = self.b
        s = self.v(torch.tanh(query+key+b)).view(B, 1, N)
        return s

    def get_alignment(self, query, key):
        # project
        qry = self.proj_qry(query)
        key = self.proj_key(key)
        # get score
        score = self._get_score(query=qry, key=key)
        if not self.training:
            score = score * self.score_scaling
        # normalize score
        align = self.normalize(score, dim=-1)
        return align


class GmmAttention(_Attention):
    def __init__(self, num_units, query_size, key_size, num_k, query_radius=1, softmax=False, scale_x=6.0):
        super().__init__(num_units, query_size, key_size, query_radius=query_radius, value_size=key_size, same_kv=True)
        self.num_k = num_k
        self.softmax = softmax
        self.scale_x = float(scale_x)
        # projections
        self.proj_qry = torch.nn.Sequential(
            saber.nn.layers.FullyConnected(self.qry_size,  self.num_units, bias=False, activation="leaky_relu@a:0.01"),
            saber.nn.layers.FullyConnected(self.qry_size,  self.num_units, bias=False, activation="leaky_relu@a:0.01"),
            saber.nn.layers.FullyConnected(self.num_units, self.num_k*3,   bias=False, activation="linear")
        )

    def _get_parameters(self, query):
        x = self.proj_qry(query.squeeze(1))
        alpha_hat, beta_hat, kappa_hat = x.chunk(3, dim=1)
        # parameters
        if self.softmax:
            alpha = torch.nn.functional.softmax(alpha_hat, dim=1)
        else:
            # grave paper, more smooth
            alpha = torch.exp(alpha_hat) / float(self.num_k)
        beta  = torch.exp(beta_hat)
        kappa = kappa_hat
        # (N, num_k, 1)
        return alpha.unsqueeze(2), beta.unsqueeze(2), kappa.unsqueeze(2)

    def get_alignment(self, query, key):
        # prepare query
        assert query.size(1) == 1
        assert query.dim() == 3
        # parameters
        alpha, beta, kappa = self._get_parameters(query)
        # position
        length = key.size(1)
        x = torch.arange(start=0, end=length, dtype=torch.float32)\
                 .unsqueeze(0).unsqueeze(0)\
                 .expand(query.size(0), self.num_k, -1)\
                 .to(query.device) / float(length) - 0.5  # -0.5 ~ 0.5
        x = x * self.scale_x  # x: (N, num_k, len)
        kappa = kappa.expand(-1, -1, length)  # kappa: (N, num_k, len)
        # align: (N, 1, len)
        align = torch.sum(
            alpha * torch.exp(-beta * (x - kappa) ** 2),
            dim=1, keepdim=True
        )
        return align


class ProdAttention(_Attention):
    def __init__(self, num_units, query_size, key_size, query_radius=1, num_layers=1):
        super().__init__(num_units, query_size, key_size, query_radius=query_radius, value_size=None, same_kv=False)
        self._layers_qry = torch.nn.ModuleList()
        self._layers_key = torch.nn.ModuleList()
        self._scaling = 1.0 / math.sqrt(num_units)
        for i in range(num_layers):
            qin_size = num_units if i > 0 else self.qry_size
            kin_size = num_units if i > 0 else self.key_size
            activation = "lrelu@a:0.2" if (i < num_layers - 1) else "linear"
            self._layers_qry.append(saber.nn.layers.FullyConnected(
                qin_size, num_units, bias=False,
                activation=activation, init_method="glorot"
            ))
            self._layers_key.append(saber.nn.layers.FullyConnected(
                kin_size, num_units, bias=False,
                activation=activation, init_method="glorot"
            ))

    def proj_key(self, key):
        for m in self._layers_key:
            key = m(key)
        return key

    def proj_qry(self, qry):
        for m in self._layers_qry:
            qry = m(qry)
        return qry

    def get_alignment(self, query, key):
        # project query and key
        qry = self.proj_qry(query)
        key = self.proj_key(key)
        # product
        score = torch.bmm(qry, key.transpose(2, 1)) * self._scaling
        align = torch.nn.functional.softmax(score, dim=-1)
        return align


class MultiHeadAttention(_Attention):
    def __init__(self, num_units, query_size, key_size, value_size, num_heads):
        raise NotImplementedError()
        super().__init__(num_units, query_size, key_size, value_size)
        self.num_heads = num_heads
        self._multi_head = torch.nn.MultiheadAttention(
            embed_dim = num_units,
            num_heads = num_heads,
            kdim      = key_size,
            vdim      = value_size
        )
        self.proj_qry = saber.nn.layers.Linear(self.qry_size, self.num_units, bias=False)

    def forward(self, query, key, value=None):
        query = self.proj_qry(query)
        # send into torch module
        qry = query.transpose(1, 0)
        key = key.transpose(1, 0)
        val = key if value is None else value.transpose(1, 0)
        out, align = self._multi_head(qry, key, val)
        out = out.transpose_(1, 0)
        return out, align
