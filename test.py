# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()                   # Display the plot

x = torch.ones(100, 100)
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!


# %% [markdown]

# # This is the biggist
# ### and a smaller one

# %%
msg = "Hello again"
print(msg)

# %%
origin = open('origin.data', 'rb')


# %%
origin.seek(0)
a = origin.read(4)
a

# %%
magic = int.from_bytes(a, byteorder='big', signed=False)
magic

# %%
firstImage = np.zeros((28, 28), dtype=int)

# %%
origin.seek(16 + 28 * 28 * 59980)
for i in range(28):
    for j in range(28):
        firstImage[i, j] = int.from_bytes(origin.read(1), byteorder='big', signed=False)


# %%
plt.imshow(firstImage, cmap='gray', vmin=0, vmax=255)

# %%
f = origin.read(100)


# %%
f

# %%
origin.close()

# %%
