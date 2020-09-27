# %%
import torch
import numpy as np
import time
from torch import nn

# %%
x = torch.rand(5, 3)
x = torch.zeros(5, 3, dtype=torch.long)
x = torch.tensor([5.5, 234, -123])
x = x.new_ones(5, 3, dtype=torch.double)
x = torch.randn_like(x, dtype=torch.double)
print(x)
print(x.size())
x.t_()
print(x.size())

# %%
print(x.new_ones(3, 5))
print(torch.ones(5, 3))

# %%
x = torch.ones(5, 3)
y = torch.rand(5, 5)
print(x * y)

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

# %% [markdown]
# ## Compare running time between CPU and GPU

# %%
t = time.time()
x = torch.randn(10000, 10000)
y = torch.randn(10000, 10000)
for i in range(1000):
    z = x * y + 100
print(z[0, 0])
print("time in CPU: " + str(time.time() - t))

# %%
t = time.time()
x = torch.randn(10000, 10000, device=device)
y = torch.randn(10000, 10000, device=device)
for i in range(1000):
    z = x * y + 100
print(z[0, 0])
print("time in GPU: " + str(time.time() - t))

# %% [markdown]
# ## Samples of autograd

# %%
x = torch.ones(2, 2, requires_grad=True)
print(x)

# %%
y = x + 2
print(y)


# %%
print(y.grad_fn)

# %%
z = y * y * 3
out = z.mean()
print(z, out)


# %%
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# %%
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
print(z, out)

# %%
x = torch.rand(2, 2, requires_grad=True)
y = torch.tensor([[1, 2], [3, 4]], dtype=torch.double)
z = x * y + 2

# %%
out = z.mean()

# %%
z.backward(torch.ones(2, 2, dtype=torch.float))

# %%
print(x.grad)

# %%
out.backward()
print(x.grad)

# %%
x = torch.ones(4, 4, device=device, requires_grad=True)
print(x)

# %%
y = x * 3
print(y)

# %%
v = torch.ones(4, 4, device=device)
y.backward(v * 2)
print(x)

# %%
print(x.grad)

# %%
x = torch.ones(2, 2, requires_grad=True)
print(x)

# %%
y = x + 2
print(y)

# %%
z = y * y * 3
out = z.mean()

print(z, out)

# %%
out.backward()

# %%
x.grad

# %%
y.grad

# %%
z.grad

# %%
x = torch.ones(4, 4, device=device, requires_grad=True)
y = 3 * x
y.requires_grad_(True)
z = y * y
out = z.mean()


# %%
out.backward()

# %%
y.grad

# %%
x.grad

# %%
out

# %%
x

# %%
x = torch.tensor([3, 2, 1], requires_grad=True, dtype=torch.float)
a = np.ones((1, 3))


# %%
out = x * x * 3 * torch.from_numpy(a)
out = out.mean()
out.backward()
x.grad

# %%
x[2]

# %%
x = torch.tensor([1,2,3,4])
# %%
y = x.view(2, 2)
# %%
print(x)
print(y)

# %%
sm = nn.Softmax()

# %%
a = torch.randn(2, 10)
print(a)
# %%
print(sm(a))
# %%
