import torch
from torch.utils.data import Dataset
import random

class SessionDataset(Dataset):
    def __init__(self, filepath, max_len=50):
        self.max_len = max_len
        self.sessions = []
        self.item_set = set()

        # 加载并解析 session 数据
        with open(filepath) as f:
            for line in f:
                _, item_str = line.strip().split()
                items = item_str.split(',')
                self.sessions.append(items)
                self.item_set.update(items)

        self.item2id = {item: idx + 1 for idx, item in enumerate(sorted(self.item_set))}  # 0 for padding
        self.id2item = {v: k for k, v in self.item2id.items()}

        self.samples = self._build_samples()

    def _build_samples(self):
        data = []
        for session in self.sessions:
            ids = [self.item2id[i] for i in session if i in self.item2id]
            for i in range(1, len(ids)):
                input_seq = ids[:i][-self.max_len:]
                padded = [0] * (self.max_len - len(input_seq)) + input_seq
                target = ids[i]
                data.append((padded, target))
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]
        return torch.LongTensor(input_seq), torch.LongTensor([target])

    def get_item_mappings(self):
        return self.item2id, self.id2item

    def train_test_split(self, ratio=0.8):
        random.shuffle(self.samples)
        split = int(len(self.samples) * ratio)
        train_data = torch.utils.data.Subset(self, range(split))
        test_data = torch.utils.data.Subset(self, range(split, len(self.samples)))
        return train_data, test_data
