import torch
import torch.nn as nn
import torch.nn.functional as F

from nns.modules import BiLSTM


class Model(nn.Module):

    def __init__(self, args, embs):
        super(Model, self).__init__()
        # Embedder
        self.embedder = nn.Embedding(embs.shape[0], embs.shape[1], padding_idx=0)
        embs = torch.from_numpy(embs)[1:]   # replace the given padding embedding with a zero vector
        pad = torch.zeros(1, embs.size(1))
        self.embedder.weight = nn.Parameter(torch.cat([pad, embs], dim=0))

        # Encoders
        self.seq_encoder = BiLSTM(
            args.dim_tok, args.dim_hid, args.nlayer_enc, args.drop_seq, args.out_pad
        )
        self.seq_dropout = nn.Dropout(p=args.drop_seq)
        self.att_key_encoder = nn.Linear(args.dim_hid * 2, args.dim_hid * 2)
        self.att_query = nn.Linear(args.dim_hid * 2, 1, bias=False)
        self.pred_encoder = nn.Linear(args.dim_hid * 2, 2)

        self.device = args.device
        self.fix_emb = args.fix_emb

    def forward(self, batch):
        """ Predict whether a given sentence is subjective or objective.
        
        Args:
            text_tensor: LongTensor (batch, max_slen)
            lengths: LongTensor (batch)
            labels: LongTensor (batch)

        Returns:
            preds: FloatTensor (batch, 2)
            labels: LongTensor (batch)
        """
        # Send input tensors to CPU/GPU
        text_tensor, lengths, labels = batch
        text_tensor = text_tensor.to(self.device)
        lengths = lengths.to(self.device)
        labels = labels.to(self.device)

        # Embed tokens
        if self.fix_emb:
            with torch.no_grad():                       # optional: do not update pre-trained embeddings
                tok_embs = self.embedder(text_tensor)   # (batch, max_slen, dim_tok)
        else:
            tok_embs = self.embedder(text_tensor)       # (batch, max_slen, dim_tok)

        # Encode sequences of token embeddings
        seq_embs, _ = self.seq_encoder(tok_embs, lengths)  # (batch, max_slen, dim_hid * 2)

        # Make padding mask for attention.
        # Can be written short:
        #   mask_tensor = torch.empty_like(text_tensor).fill_(-1e+12)
        #   for ...
        #   mask_tensor = mask_tensor.unsqueeze(-1).to(self.device)
        mask_tensor = torch.empty_like(text_tensor)     # (batch, max_slen)
        mask_tensor = mask_tensor.fill_(-1e+12)         # fill with -inf
        for i, slen in enumerate(lengths):
            mask_tensor[i][:slen] = torch.zeros(slen)   # fill the sentences with zeros
        mask_tensor = mask_tensor.unsqueeze(-1)         # (batch, max_slen, 1)
        mask_tensor = mask_tensor.to(self.device)       # send to cpu/gpu

        # Apply attention
        att_keys = self.att_key_encoder(seq_embs)       # (batch, max_slen, dim_hid * 2)
        att_weights = self.att_query(att_keys)          # (batch, max_slen, 1)
        att_weights = att_weights + mask_tensor         # fill the paddings with -inf to make them 0 in the normalized attention weights
        att_weights = torch.softmax(
            att_weights, dim=-1
        )                                               # take softmax over the last dimension
        att_weights = att_weights.transpose(1, 2)       # (batch, 1, max_slen)
        att_outs = torch.bmm(
            att_weights, seq_embs
        )                                               # (batch, 1, dim_hid * 2)
        att_outs = att_outs.squeeze(1)                  # (batch, dim_hid * 2)

        # Transformation for prediction
        preds = self.pred_encoder(att_outs)             # (batch, 2)

        return preds, labels
