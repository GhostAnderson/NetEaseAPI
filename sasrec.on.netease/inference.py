import torch
from sasrec import SASRec

class SASRecRecommender:
    def __init__(self, model_path, item2id, max_len=50):
        self.model = SASRec(n_items=len(item2id), max_seq_len=max_len)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.item2id = item2id
        self.id2item = {v: k for k, v in item2id.items()}
        self.max_len = max_len

    def recommend(self, session, top_k=20):
        ids = [self.item2id[i] for i in session if i in self.item2id]
        padded = [0] * (self.max_len - len(ids)) + ids[-self.max_len:]
        input_seq = torch.LongTensor(padded).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(input_seq)
            top_items = torch.topk(logits, k=top_k).indices.squeeze(0).tolist()
        return [self.id2item[i] for i in top_items]
