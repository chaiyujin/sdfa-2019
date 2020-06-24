import torch


class LSTM2d(torch.nn.Module):
    def __init__(self, hidden_channels, nb_layers, **kwargs):
        super().__init__()
        self._chs = hidden_channels
        self._layers = nb_layers
        self._rnn_freq_list = torch.nn.ModuleList()
        self._rnn_time_list = torch.nn.ModuleList()
        for _ in range(self._layers):
            self._rnn_freq_list.append(torch.nn.LSTM(
                hidden_channels,
                hidden_channels//2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ))
            self._rnn_time_list.append(torch.nn.LSTM(
                hidden_channels,
                hidden_channels//2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ))

    def forward(self, inputs):
        assert inputs.dim() == 4  # N, C, F, T
        assert inputs.size(1) == self._chs
        N,C,F,T = inputs.size()
        x = inputs.permute(0, 2, 3, 1)  # N, F, T, C

        for i in range(self._layers):
            _rnn_freq = self._rnn_freq_list[i]
            _rnn_time = self._rnn_time_list[i]
            residual = x
            # along freq
            x = x.transpose(2, 1).contiguous()\
                .view(N*T,F,C)
            x, _ = _rnn_freq(x)
            x = x.view(N, T, F, C)

            # along time
            x = x.transpose(2, 1).contiguous()\
                .view(N*F,T,C)
            x, _ = _rnn_time(x)
            x = x.view(N, F, T, C) + residual

        x = x.permute(0, 3, 1, 2).contiguous()
        return x
