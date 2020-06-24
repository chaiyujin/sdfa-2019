import torch
import saber


class FreqLstm(torch.nn.Module):
    # input should be 4 dim
    def __init__(self, input_size, freq_length, hidden_size, output_size, bias=True, mode="full", **kwargs):
        super().__init__()
        assert mode in ["full", "last"]
        self._mode = mode
        self._freq_len = freq_length
        self._lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )
        if self._mode == "full":
            self._proj_size = self._freq_len * 2 * hidden_size
        else:
            self._proj_size = 2 * hidden_size
        self._out_size = output_size
        self._proj = saber.nn.layers.FullyConnected(
            self._proj_size,
            output_size,
            bias=bias
        )

    def forward(self, x):
        dim_4 = False
        if x.dim() == 4:
            assert x.dim() == 4, "inputs should be (B,C,F,T)"
            bsz, ch, fq, t = x.size()
            x = x.permute(0, 3, 2, 1).contiguous()\
                .view(bsz * t, fq, ch)
            dim_4 = True
        elif x.dim() == 3:
            bsz, ch, fq = x.size()
            x = x.permute(0, 2, 1).contiguous()
            t = 1
        else:
            raise ValueError(f"[freq-lstm] Given inputs {x.shape} is not supported!")

        # check frequency bins
        assert fq == self._freq_len, "inputs should have '{}' freq bins, but '{}'".format(self._freq_len, fq)

        # forward
        if self._mode == 'full':
            x, _ = self._lstm(x)  # BT, F, C
            x = x.contiguous().view(bsz * t, self._proj_size)  # BT, FC
            x = self._proj(x).view(bsz, t, self._out_size)  # B, T, C
            x = x.permute(0, 2, 1).contiguous()  # B, C, T
        else:
            _, (x, _) = self._lstm(x)
            x = x.permute(1, 0, 2).contiguous()  # BT, dC
            x = x.view(bsz * t, self._proj_size)
            x = self._proj(x).view(bsz, t, self._out_size)
            x = x.permute(0, 2, 1).contiguous()  # B, C, T

        # return
        if dim_4:
            return x.unsqueeze(2)  # B, C, 1, T
        else:
            return x
