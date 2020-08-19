import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class BiLSTM(nn.Module):

    def __init__(self, dim_in, dim_hid, nlayer, drop_rate=0., out_pad=0.):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(
            dim_in, dim_hid, nlayer, batch_first=True,
            dropout=drop_rate, bidirectional=True
        )
        self.nlayer = nlayer
        self.dim_hid = dim_hid
        self.out_pad = out_pad
        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.nlayer * 2, batch_size, self.dim_hid).to(self.device),
            torch.zeros(self.nlayer * 2, batch_size, self.dim_hid).to(self.device)
        )   # ( (nlayer * 2, n_sent, dimh) * 2 )

    def forward(self, x, slen):
        """
        Args:
            x (n_sent, max_slen, dim): Padded and sorted FloatTensor
            slen (n_sent): LongTensor whose each element corresponds to a sentence length
        
        Returns:
            output (batch, max_slen, dimh * 2): FloatTensor padded with `out_pad`
            hid_n, cell_n (nlayer * 2, batch, dimh): FloatTensor of hidden or cell states
        """
        # Pack
        total_length = x.size(1)
        x = pack(x, slen, batch_first=True)             # (sum(slen), dim)

        init_hc = self.init_hidden(len(slen))           # ( (nlayer * 2, batch, dimh) * 2 )
        out, (hid_n, cell_n) = self.bilstm(x, init_hc)  # (sum(slen), dimh * 2), ( (nlayer * 2, batch, dimh) * 2 )

        # Unpack
        out, _ = unpack(
            out, batch_first=True,
            padding_value=self.out_pad, total_length=total_length
        )   # (batch, max_slen, dimh * 2)

        return out, (hid_n, cell_n)


class MLP(nn.Module):

    def __init__(self, dim_list, drop_rate=0., activation=""):
        """
        Args:
            dim_list: list [dim_in, dim_hid * n, dim_out]
            drop_rate: float [0, 1]
            activation: str {"relu", "tanh", "sigmoid", ""}
        """
        assert activation in {"relu", "tanh", "sigmoid", ""}, "invalid activation function"
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()
        for k in range(len(dim_list) - 2):
            self.hidden.append(nn.Linear(dim_list[k], dim_list[k + 1]))
            self.add_activation(self.hidden, activation)
            if drop_rate > 0.:
                self.hidden.append(nn.Dropout(drop_rate))
        self.hidden.append(nn.Linear(dim_list[-2], dim_list[-1]))
    
    def add_activation(self, module_list, activation):
        if activation == "relu":
            module_list.append(nn.ReLU())
        elif activation == "tanh":
            module_list.append(nn.Tanh())
        elif activation == "sigmoid":
            module_list.append(nn.Sigmoid())
        else:
            pass
    
    def forward(self, x):
        """
        Args:
            x (*, dim_in): FloatTensor
        
        Returns:
            x (*, dim_out): FloatTensor
        """
        for layer in self.hidden:
            x = layer(x)

        return x
