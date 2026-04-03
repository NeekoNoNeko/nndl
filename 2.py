import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import random
import time
import matplotlib.pyplot as plt

# ======================
# 设备
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# 字符表
# ======================
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS

# ======================
# 数据处理
# ======================
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    with open(filename, encoding='utf-8') as f:
        return [unicodeToAscii(line.strip()) for line in f]

category_lines = {}
all_categories = []

for filename in glob.glob('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    category_lines[category] = readLines(filename)

n_categories = len(all_categories)

# ======================
# Dataset（批量化关键）
# ======================
class NameDataset(Dataset):
    def __init__(self):
        self.data = []

        for category in all_categories:
            for line in category_lines[category]:
                if len(line) == 0:
                    continue
                self.data.append((category, line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        category, line = self.data[idx]

        category_idx = all_categories.index(category)

        input_tensor = torch.tensor(
            [all_letters.find(c) for c in line],
            dtype=torch.long
        )

        target_tensor = torch.tensor(
            [all_letters.find(c) for c in line[1:]] + [n_letters - 1],
            dtype=torch.long
        )

        return category_idx, input_tensor, target_tensor

# ======================
# collate_fn（padding）
# ======================
def collate_fn(batch):
    categories, inputs, targets = zip(*batch)

    categories = torch.tensor(categories)

    lengths = torch.tensor([len(x) for x in inputs])

    inputs = pad_sequence(inputs, batch_first=True)
    targets = pad_sequence(targets, batch_first=True, padding_value=-100)

    return categories, inputs, targets, lengths

# ======================
# 模型（Embedding + GRU）
# ======================
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.category_emb = nn.Embedding(n_categories, hidden_size)

        self.rnn = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, categories, inputs, lengths):
        # inputs: [B, T]
        emb = self.embedding(inputs)

        cat_emb = self.category_emb(categories)
        cat_emb = cat_emb.unsqueeze(1).repeat(1, emb.size(1), 1)

        x = torch.cat([emb, cat_emb], dim=2)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.rnn(packed)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output = self.fc(output)

        return output

# ======================
# 训练
# ======================
def train_model(model, loader, epochs=10, lr=0.003):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    use_amp = torch.cuda.is_available()

    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for categories, inputs, targets, lengths in loader:
            categories = categories.to(device)
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(categories, inputs, lengths)
                    loss = criterion(outputs.view(-1, n_letters), targets.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(categories, inputs, lengths)
                loss = criterion(outputs.view(-1, n_letters), targets.view(-1))
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    return losses

# ======================
# 采样（生成名字）
# ======================
def sample(model, category, start_letter='A', max_length=20):
    with torch.no_grad():
        category_idx = torch.tensor([all_categories.index(category)]).to(device)

        input = torch.tensor([[all_letters.find(start_letter)]], device=device)

        hidden = None
        output_name = start_letter

        for _ in range(max_length):
            emb = model.embedding(input)
            cat_emb = model.category_emb(category_idx).unsqueeze(1)

            x = torch.cat([emb, cat_emb], dim=2)

            out, hidden = model.rnn(x, hidden)
            out = model.fc(out[:, -1, :])

            topi = torch.argmax(out, dim=1).item()

            if topi == n_letters - 1:
                break

            letter = all_letters[topi]
            output_name += letter

            input = torch.tensor([[topi]], device=device)

        return output_name

def samples(model, category, letters="ABC"):
    for l in letters:
        print(sample(model, category, l))

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    dataset = NameDataset()

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = CharRNN(n_letters, 128, n_letters).to(device)

    start = time.time()
    losses = train_model(model, loader, epochs=15)
    print("Training time:", time.time() - start)

    plt.plot(losses)
    plt.title("Loss")
    plt.show()

    print("\nRussian:")
    samples(model, "Russian", "RUS")

    print("\nGerman:")
    samples(model, "German", "GER")

    print("\nChinese:")
    samples(model, "Chinese", "CHI")