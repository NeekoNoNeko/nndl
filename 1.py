import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import glob
import os
import unicodedata
import string
import time
import random
import numpy as np
import matplotlib.pyplot as plt

# ======================
# 设备
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================
# 字符处理
# ======================
allowed_characters = string.ascii_letters + " .,;'" + "_"
n_letters = len(allowed_characters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )

def letterToIndex(letter):
    return allowed_characters.find(letter) if letter in allowed_characters else allowed_characters.find("_")

def lineToTensor(line):
    return torch.tensor([letterToIndex(c) for c in line], dtype=torch.long)

# ======================
# Dataset
# ======================
class NamesDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
        labels_set = set()

        files = glob.glob(os.path.join(data_dir, "*.txt"))
        for filename in files:
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)

            lines = open(filename, encoding="utf-8").read().strip().split("\n")
            for name in lines:
                name = unicodeToAscii(name)
                if len(name) == 0:
                    continue
                self.data.append(lineToTensor(name))
                self.labels.append(label)

        self.labels_uniq = list(labels_set)
        self.label_map = {l: i for i, l in enumerate(self.labels_uniq)}

        self.label_tensors = [
            torch.tensor(self.label_map[l], dtype=torch.long)
            for l in self.labels
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.label_tensors[idx], self.data[idx]

# ======================
# collate_fn（关键）
# ======================
def collate_fn(batch):
    labels, texts = zip(*batch)

    labels = torch.stack(labels)
    lengths = torch.tensor([len(t) for t in texts])

    texts_padded = pad_sequence(texts, batch_first=True)

    return labels, texts_padded, lengths

# ======================
# 模型（GRU + Embedding）
# ======================
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        x = self.embedding(x)

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, hidden = self.rnn(packed)
        out = self.fc(hidden.squeeze(0))

        return F.log_softmax(out, dim=1)

# ======================
# 训练函数
# ======================
def train_model(model, train_loader, epochs=20, lr=0.003):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    use_amp = torch.cuda.is_available()

    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for labels, texts, lengths in train_loader:
            labels = labels.to(device)
            texts = texts.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(texts, lengths)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(texts, lengths)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    return losses

# ======================
# 评估
# ======================
def evaluate(model, loader, classes):
    confusion = torch.zeros(len(classes), len(classes))

    model.eval()
    with torch.no_grad():
        for labels, texts, lengths in loader:
            texts = texts.to(device)
            outputs = model(texts, lengths)
            preds = torch.argmax(outputs, dim=1).cpu()

            for i in range(len(labels)):
                confusion[labels[i]][preds[i]] += 1

    for i in range(len(classes)):
        if confusion[i].sum() > 0:
            confusion[i] /= confusion[i].sum()

    plt.imshow(confusion.numpy())
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.show()

# ======================
# 主流程
# ======================
if __name__ == "__main__":

    dataset = NamesDataset("data/names")
    print(f"Total samples: {len(dataset)}")

    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = CharRNN(
        vocab_size=n_letters,
        hidden_size=128,
        output_size=len(dataset.labels_uniq)
    ).to(device)

    print(model)

    start = time.time()
    losses = train_model(model, train_loader, epochs=20)
    end = time.time()

    print(f"Training time: {end - start:.2f}s")

    plt.plot(losses)
    plt.title("Training Loss")
    plt.show()

    evaluate(model, test_loader, dataset.labels_uniq)