import torch
from model import SASRec
import numpy as np
import json

import collections
# 92871,93765

class SASRecRecommender:
    def __init__(self, model_path, item2id, max_len=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.item2id = item2id
        self.id2item = {v: k for k, v in item2id.items()}
        ARGS = collections.namedtuple('ARGS', ['device', 'hidden_units', 'maxlen', 'dropout_rate', 'num_blocks', 'num_heads'])
        args = ARGS(self.device, 50, 100, 0.2, 2, 1)
        self.model = SASRec(10000, len(item2id), args).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.max_len = max_len

    def recommend(self, session_items, top_k=10):
        ids = [self.item2id[str(i)] for i in session_items if str(i) in self.item2id]
        if not ids:
            return ["无有效item，无法推荐"]
        padded = [0] * (self.max_len - len(ids)) + ids[-self.max_len:]
        input_tensor = np.array([padded])
        item_indices = np.array([list(range(1, len(self.id2item)+1))])
        with torch.no_grad():
            results = self.model.predict(None, input_tensor, item_indices)
            topk = torch.topk(results, k=top_k).indices.squeeze(0).tolist()
            return [self.id2item[i] for i in topk if i in self.id2item]
