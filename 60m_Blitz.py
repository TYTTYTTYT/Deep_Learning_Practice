# %%
import torch
import numpy as np
import time


# %%
device = torch.device("cuda")

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
out.backward()
print(x.grad)

# %%
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
