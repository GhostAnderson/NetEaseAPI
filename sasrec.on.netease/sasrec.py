import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, n_items, hidden_dim=64, max_seq_len=50, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)  # +1 for padding
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.attn_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4, dropout=dropout),
            num_layers=n_layers
        )

        self.output_layer = nn.Linear(hidden_dim, n_items)

    def forward(self, input_seq):
        seq_len = input_seq.size(1)
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0).expand_as(input_seq)

        item_embs = self.item_emb(input_seq)
        pos_embs = self.pos_emb(positions)
        x = item_embs + pos_embs
        x = self.dropout(x).transpose(0, 1)  # [seq_len, batch, dim]

        attn_output = self.attn_layers(x).transpose(0, 1)  # [batch, seq_len, dim]
        final_hidden = attn_output[:, -1, :]  # 取最后一个位置

        logits = self.output_layer(final_hidden)  # [batch, n_items]
        return logits
