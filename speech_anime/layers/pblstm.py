import torch


class PyramidBiLSTM(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers    = 1,
        batch_first   = True,
        bidirectional = True,
        dropout       = 0,
        bias          = True,
        down_sample   = 2,
        down_inputs   = False,
        down_style    = "concat",
    ):
        super().__init__()
        assert batch_first, "PyramidBiLSTM only support 'batch_first'"
        assert bidirectional, "PyramidBiLSTM only support 'bidirectional'"
        assert down_style in ["concat", "drop"]
        assert type(down_sample) is int, "PyramidBiLSTM, 'down_sample' should be {}".format(type(int))
        self.down_style = down_style
        self.down_sample = down_sample
        self.down_inputs = down_inputs
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layers = torch.nn.ModuleList()
        inp, out = input_size, hidden_size
        if down_style == "concat" and down_inputs:
            inp *= self.down_sample
        for i_layer in range(num_layers):
            self.layers.append(torch.nn.LSTM(
                input_size    = inp,
                hidden_size   = out,
                batch_first   = True,
                bidirectional = True,
                bias          = bias,
                num_layers    = 1,
                dropout       = 0
            ))
            if down_style == "drop":
                inp = out * 2
            elif down_style == "concat":
                inp = out * 2 * down_sample

    def forward(self, x):
        for i_layer, layer in enumerate(self.layers):
            x = self._maybe_down_sample(i_layer, x)
            x, _ = layer(x)
            x = self._maybe_dropout(i_layer, x)
        return x, None

    def _maybe_down_sample(self, i, x):
        if (self.down_sample <= 1) or (i == 0 and not self.down_inputs):
            return x
        # x: B, T, C
        if self.down_style == "drop":
            x = x[:, ::self.down_sample, :]
        elif self.down_style == "concat":
            bsz, time, ch = x.size()
            if time % self.down_sample != 0:
                x = x[: :-(time % self.down_sample), :]
            x = x.contiguous().view(bsz, time//self.down_sample, ch*self.down_sample)
        else:
            raise NotImplementedError()
        return x

    def _maybe_dropout(self, i, x):
        if i + 1 < len(self.layers):
            x = self.dropout(x)
        return x
