import os
import torch
import numpy as np
import traceback as tb
os.environ["PYTHONUTF8"] = "1"

## Basic Usage

torch._logging.set_logs(graph_code=True)

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


opt_foo1 = torch.compile(foo, backend="aot_eager")
print(opt_foo1(torch.randn(3, 3), torch.randn(3, 3)))


@torch.compile(backend="aot_eager")
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


print(opt_foo2(torch.randn(3, 3), torch.randn(3, 3)))


def inner(x):
    return torch.sin(x)


@torch.compile
def outer(x, y):
    a = inner(x)
    b = torch.cos(y)
    return a + b


print(outer(torch.randn(3, 3), torch.randn(3, 3)))


t = torch.randn(10, 100)


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))


mod1 = MyModule()
mod1.compile()
print(mod1(torch.randn(3, 3)))

mod2 = MyModule()
mod2 = torch.compile(mod2)
print(mod2(torch.randn(3, 3)))



## Demonstrating Speedups

def foo3(x):
    y = x + 1
    z = torch.nn.functional.relu(y)
    u = z * 2
    return u


opt_foo3 = torch.compile(foo3)


# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


inp = torch.randn(4096, 4096).cuda()
print("compile:", timed(lambda: opt_foo3(inp))[1])
print("eager:", timed(lambda: foo3(inp))[1])

# turn off logging for now to prevent spam
torch._logging.set_logs(graph_code=False)

eager_times = []
for i in range(10):
    _, eager_time = timed(lambda: foo3(inp))
    eager_times.append(eager_time)
    print(f"eager time {i}: {eager_time}")
print("~" * 10)

compile_times = []
for i in range(10):
    _, compile_time = timed(lambda: opt_foo3(inp))
    compile_times.append(compile_time)
    print(f"compile time {i}: {compile_time}")
print("~" * 10)



eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
assert speedup > 1
print(
    f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x"
)
print("~" * 10)



## Benefits over TorchScript

def f1(x, y):
    if x.sum() < 0:
        return -y
    return y


# Test that `fn1` and `fn2` return the same result, given the same arguments `args`.
def test_fns(fn1, fn2, args):
    out1 = fn1(*args)
    out2 = fn2(*args)
    return torch.allclose(out1, out2)


inp1 = torch.randn(5, 5)
inp2 = torch.randn(5, 5)

traced_f1 = torch.jit.trace(f1, (inp1, inp2))
print("traced 1, 1:", test_fns(f1, traced_f1, (inp1, inp2)))
print("traced 1, 2:", test_fns(f1, traced_f1, (-inp1, inp2)))

compile_f1 = torch.compile(f1)
print("compile 1, 1:", test_fns(f1, compile_f1, (inp1, inp2)))
print("compile 1, 2:", test_fns(f1, compile_f1, (-inp1, inp2)))
print("~" * 10)



torch._logging.set_logs(graph_code=True)


def f2(x, y):
    return x + y


inp1 = torch.randn(5, 5)
inp2 = 3

script_f2 = torch.jit.script(f2)
try:
    script_f2(inp1, inp2)
except:
    tb.print_exc()

compile_f2 = torch.compile(f2)
print("compile 2:", test_fns(f2, compile_f2, (inp1, inp2)))
print("~" * 10)

## Graph Breaks

def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


opt_bar = torch.compile(bar)
inp1 = torch.ones(10)
inp2 = torch.ones(10)
opt_bar(inp1, inp2)
opt_bar(inp1, -inp2)

# Reset to clear the torch.compile cache
torch._dynamo.reset()
opt_bar(inp1, inp2)
opt_bar(inp1, -inp2)

# Reset to clear the torch.compile cache
torch._dynamo.reset()

opt_bar_fullgraph = torch.compile(bar, fullgraph=True)
try:
    opt_bar_fullgraph(torch.randn(10), torch.randn(10))
except:
    tb.print_exc()

from functorch.experimental.control_flow import cond


@torch.compile(fullgraph=True)
def bar_fixed(a, b):
    x = a / (torch.abs(a) + 1)

    def true_branch(y):
        return y * -1

    def false_branch(y):
        # NOTE: torch.cond doesn't allow aliased outputs
        return y.clone()

    b = cond(b.sum() < 0, true_branch, false_branch, (b,))
    return x * b


bar_fixed(inp1, inp2)
bar_fixed(inp1, -inp2)