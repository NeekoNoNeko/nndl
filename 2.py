import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    init_method = f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    if world_size > 1:
        setup(rank, world_size)
        model = ToyModel()
        ddp_model = DDP(model)
    else:
        print("World size is 1, using non-distributed model.")
        model = ToyModel()
        ddp_model = model

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    inputs = torch.randn(20, 10)
    outputs = ddp_model(inputs)
    labels = torch.randn(20, 5)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    if world_size > 1:
        cleanup()
    print(f"Finished basic DDP example on rank {rank}.")


def run_demo(demo_fn, world_size):
    if world_size > 1:
        mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)
    else:
        demo_fn(0, world_size)


if __name__ == "__main__":
    n_devices = torch.accelerator.device_count()
    if n_devices == 0:
        print("No accelerator found. Running CPU DDP demo with 2 processes.")
        world_size = 2
    else:
        print(f"Found {n_devices} accelerator(s). Running CPU DDP demo with {n_devices} process(es).")
        world_size = n_devices if n_devices > 1 else 1

    run_demo(demo_basic, world_size)