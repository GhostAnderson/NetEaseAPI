import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import os
from data_loader import SessionDataset
from srgcn import SRGNN

def split_data(data, split_ratio=0.8):
    random.shuffle(data)
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]

def recall_at_k(pred_items, target_item, k=20):
    return int(target_item in pred_items[:k])

# Load dataset
full_dataset = SessionDataset('dataset.txt')
item2id = full_dataset.item2id

# Split dataset
train_data, test_data = split_data(full_dataset.data)

# Model setup
model = SRGNN(n_items=len(item2id), hidden_size=100)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Load existing model if exists
MODEL_PATH = 'model.pth'
BEST_MODEL_PATH = 'best_model.pth'
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print("✅ 模型已加载")

EPOCHS = 10
best_recall = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    # --- Training ---
    for session in tqdm(train_data):
        if len(session) < 2:
            continue
        alias_inputs = []
        unique_items = list(dict.fromkeys(session))
        for i in session:
            alias_inputs.append(unique_items.index(i))
        input_items = session[:-1]
        target_item = session[-1]

        optimizer.zero_grad()
        scores = model(alias_inputs[:-1], unique_items, input_items)
        target = torch.LongTensor([target_item])
        loss = criterion(scores.view(1, -1), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # --- Evaluation ---
    model.eval()
    hits = 0
    total = 0
    with torch.no_grad():
        for session in test_data:
            if len(session) < 2:
                continue
            alias_inputs = []
            unique_items = list(dict.fromkeys(session))
            input_items = session[:-1]
            target_item = session[-1]
            try:
                alias_inputs = [unique_items.index(i) for i in input_items]
                scores = model(alias_inputs, unique_items, input_items)
                topk = torch.topk(scores, k=20).indices.tolist()
                hits += recall_at_k(topk, target_item, k=20)
                total += 1
            except:
                continue

    recall = hits / total if total > 0 else 0
    print(f"📦 Epoch {epoch+1}, Loss: {total_loss:.4f}, Recall@20: {recall:.4f}")

    # Save best model
    if recall > best_recall:
        best_recall = recall
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"🚀 最佳模型更新，Recall@20 = {best_recall:.4f}")

    # Save latest model
    torch.save(model.state_dict(), MODEL_PATH)
