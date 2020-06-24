import torch


def _create_rnn(name, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional):
    return getattr(torch.nn, name)(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional
    )


def _create_gru(input_size, hidden_size, num_layers, bias=False, batch_first=True, dropout=0, bidirectional=False, **kwargs):
    return _create_rnn("GRU", input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)


def _create_lstm(input_size, hidden_size, num_layers, bias=False, batch_first=True, dropout=0, bidirectional=False, **kwargs):
    return _create_rnn("LSTM", input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
