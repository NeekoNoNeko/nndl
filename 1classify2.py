import torch
import string
import unicodedata
from io import open
import glob
import os
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------- 设备配置 ---------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 注意：不再设置 torch.set_default_device(device)
print(f"Using device = {device}")

# ---------------------------- 字符处理 ---------------------------------
allowed_characters = string.ascii_letters + " .,;'" + "_"
n_letters = len(allowed_characters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )

def letterToIndex(letter):
    if letter not in allowed_characters:
        return allowed_characters.find("_")
    else:
        return allowed_characters.find(letter)

def lineToTensor(line):
    # 始终在 CPU 上创建 tensor
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# ---------------------------- Dataset ---------------------------------
class NamesDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_time = time.localtime
        labels_set = set()
        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        text_files = glob.glob(os.path.join(data_dir, '*.txt'))
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            for name in lines:
                ascii_name = unicodeToAscii(name)
                if ascii_name:
                    self.data.append(ascii_name)
                    self.data_tensors.append(lineToTensor(ascii_name))  # CPU tensor
                    self.labels.append(label)

        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.labels_tensors[idx],
                self.data_tensors[idx],
                self.labels[idx],
                self.data[idx])

# ---------------------------- 模型定义 ---------------------------------
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=False)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[-1])
        output = self.softmax(output)
        return output

# ---------------------------- 批处理辅助函数 ---------------------------
def collate_batch(batch):
    # batch 中的 tensor 都在 CPU 上
    text_tensors = [item[1].squeeze(1) for item in batch]
    lengths = [t.size(0) for t in text_tensors]
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    text_tensors_sorted = [text_tensors[i] for i in sorted_indices]
    lengths_sorted = [lengths[i] for i in sorted_indices]
    labels = torch.cat([item[0] for item in batch])
    labels_sorted = labels[sorted_indices]
    padded = rnn_utils.pad_sequence(text_tensors_sorted, batch_first=False)
    packed = rnn_utils.pack_padded_sequence(padded, lengths_sorted, enforce_sorted=True)
    return packed, labels_sorted, lengths_sorted

# ---------------------------- 训练函数（批处理版本） -------------------
def train_batched(rnn, training_dataset, n_epoch=10, batch_size=64,
                  report_every=5, learning_rate=0.15, criterion=nn.NLLLoss()):
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_batch, drop_last=False)
    all_losses = []
    dev = next(rnn.parameters()).device

    print(f"Training on {len(training_dataset)} samples, batch_size={batch_size}")

    for epoch in range(1, n_epoch + 1):
        epoch_loss = 0.0
        n_batches = 0
        for packed_input, labels, _ in dataloader:
            # 将数据移到模型所在设备
            packed_input = packed_input.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()
            output = rnn(packed_input)
            loss = criterion(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        all_losses.append(avg_loss)

        if epoch % report_every == 0:
            print(f"{epoch} ({epoch/n_epoch:.0%}): average batch loss = {avg_loss:.6f}")

    return all_losses

# ---------------------------- 评估函数（需适配设备） ---------------------
def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i

def evaluate(rnn, testing_data, classes):
    confusion = torch.zeros(len(classes), len(classes))
    rnn.eval()
    dev = next(rnn.parameters()).device
    with torch.no_grad():
        for i in range(len(testing_data)):
            label_tensor, text_tensor, label, _ = testing_data[i]
            # 将 text_tensor 移到设备（原始在 CPU）
            text_tensor = text_tensor.to(dev)
            label_tensor = label_tensor.to(dev)
            output = rnn(text_tensor)
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    # 归一化
    for i in range(len(classes)):
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy())
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

# ---------------------------- 主程序 ----------------------------------
if __name__ == "__main__":
    alldata = NamesDataset("data/names")
    print(f"Loaded {len(alldata)} names, {len(alldata.labels_uniq)} languages")

    # 使用 CPU generator（默认）
    generator = torch.Generator().manual_seed(2024)
    train_set, test_set = torch.utils.data.random_split(
        alldata, [0.85, 0.15],
        generator=generator
    )
    print(f"Train: {len(train_set)}, Test: {len(test_set)}")

    n_hidden = 128
    rnn = CharRNN(n_letters, n_hidden, len(alldata.labels_uniq))
    rnn.to(device)          # 显式移到 GPU（如果可用）
    print(rnn)

    start = time.time()
    all_losses = train_batched(rnn, train_set, n_epoch=27, batch_size=64,
                               learning_rate=0.15, report_every=5)
    end = time.time()
    print(f"Training took {end-start:.2f} seconds")

    plt.figure()
    plt.plot(all_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    evaluate(rnn, test_set, classes=alldata.labels_uniq)