import torch
from torch.utils.data import Dataset
import numpy as np

class SessionDataset(Dataset):
    def __init__(self, filepath, item2id=None):
        self.sessions = []
        self.item_set = set()
        with open(filepath, 'r') as f:
            for line in f:
                sid, items = line.strip().split()
                item_seq = items.split(',')
                self.sessions.append(item_seq)
                self.item_set.update(item_seq)
        
        self.item2id = item2id or {item: idx for idx, item in enumerate(sorted(self.item_set))}
        self.id2item = {v: k for k, v in self.item2id.items()}
        self.data = [list(map(self.item2id.get, session)) for session in self.sessions]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        session = self.data[idx]
        alias_inputs = []
        unique_items = list(dict.fromkeys(session))  # 保序去重
        for i in session:
            alias_inputs.append(unique_items.index(i))
        return {
            'session': session,
            'alias_inputs': alias_inputs,
            'unique_items': unique_items,
        }

