import torch
import torch.nn as nn
import torch.optim as optim
from sasrec import SASRec
from data_loader import SessionDataset
import random
from tqdm import tqdm

# Load data
from data_loader import SessionDataset
from torch.utils.data import DataLoader

dataset = SessionDataset("netease.filtered.txt", max_len=50)
train_data, test_data = dataset.train_test_split()

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

device = torch.device('mps')

model = SASRec(n_items=len(dataset.item2id), hidden_dim=64, max_seq_len=50).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for input_seq, target in tqdm(train_loader):
        input_seq, target = input_seq.to(device), target.to(device)
        logits = model(input_seq)
        loss = criterion(logits, target.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu().item()

    # Evaluation
    model.eval()
    hits = 0
    total = 0
    with torch.no_grad():
        for seq, target in test_data:
            input_seq = torch.LongTensor(seq).unsqueeze(0).to(torch.device('mps'))
            logits = model(input_seq).cpu()
            top_k = torch.topk(logits, k=20).indices.squeeze(0).tolist()
            hits += int(target in top_k)
            total += 1

    recall = hits / total if total > 0 else 0
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Recall@20: {recall:.4f}")
    torch.save(model.state_dict(), "sasrec.pth")
