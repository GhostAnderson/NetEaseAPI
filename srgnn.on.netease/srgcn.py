import torch
import torch.nn as nn
import torch.nn.functional as F

class SRGNN(nn.Module):
    def __init__(self, n_items, hidden_size=100):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(n_items, hidden_size)
        self.gnn = GNNCell(hidden_size)
        self.linear_one = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_two = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_three = nn.Linear(hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(hidden_size * 2, hidden_size, bias=True)

    def forward(self, alias_inputs, unique_items, session):
        # embedding lookup
        item_embs = self.embedding(torch.LongTensor(unique_items))  # [n_nodes, hidden]
        A = self.build_adj(len(unique_items), session)
        hidden = self.gnn(A, item_embs)
        seq_hidden = hidden[alias_inputs]  # map alias inputs

        # attention
        last = seq_hidden[-1]
        q1 = self.linear_one(last).view(1, -1)
        q2 = self.linear_two(seq_hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # [T, 1]
        s = torch.sum(alpha * seq_hidden, 0)  # [hidden]
        session_rep = self.linear_transform(torch.cat([s, last], 0))
        scores = torch.matmul(self.embedding.weight, session_rep)
        return scores

    def build_adj(self, num_nodes, session):
        A = torch.zeros(num_nodes, num_nodes)
        for i in range(len(session) - 1):
            src = session[i]
            tgt = session[i + 1]
            if src != tgt:
                try:
                    u = session.index(src)
                    v = session.index(tgt)
                    A[u][v] = 1
                except ValueError:
                    continue
        # row normalize
        A = A / (A.sum(1, keepdim=True) + 1e-8)
        return A

class GNNCell(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.in_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, A, hidden):
        input_in = torch.matmul(A, self.in_linear(hidden))
        input_out = torch.matmul(A.t(), self.out_linear(hidden))
        inputs = input_in + input_out
        hidden = self.gru(inputs, hidden)
        return hidden
