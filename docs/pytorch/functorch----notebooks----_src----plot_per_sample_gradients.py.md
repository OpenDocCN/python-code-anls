# `.\pytorch\functorch\notebooks\_src\plot_per_sample_gradients.py`

```
"""
==========================
Per-sample-gradients
==========================

What is it?
--------------------------------------------------------------------
Per-sample-gradient computation is computing the gradient for each and every
sample in a batch of data. It is a useful quantity in differential privacy
and optimization research.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# Here's a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 1 input channel, 32 output channels, 3x3 kernel size, stride 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 32 input channels, 64 output channels, 3x3 kernel size, stride 1
        self.fc1 = nn.Linear(9216, 128)  # Fully connected layer from 9216 to 128 neurons
        self.fc2 = nn.Linear(128, 10)  # Fully connected layer from 128 to 10 neurons

    def forward(self, x):
        x = self.conv1(x)  # First convolutional layer
        x = F.relu(x)  # ReLU activation
        x = self.conv2(x)  # Second convolutional layer
        x = F.relu(x)  # ReLU activation
        x = F.max_pool2d(x, 2)  # Max pooling with 2x2 kernel and stride 2
        x = torch.flatten(x, 1)  # Flatten the tensor to prepare for fully connected layers
        x = self.fc1(x)  # First fully connected layer
        x = F.relu(x)  # ReLU activation
        x = self.fc2(x)  # Second fully connected layer
        output = F.log_softmax(x, dim=1)  # Log softmax for output probabilities
        output = x  # Direct output before log softmax (for per-sample gradients)
        return output


def loss_fn(predictions, targets):
    return F.nll_loss(predictions, targets)  # Negative log likelihood loss


# Let's generate a batch of dummy data. Pretend that we're working with an
# MNIST dataset where the images are 28 by 28 and we have a minibatch of size 64.
device = "cuda"
num_models = 10
batch_size = 64
data = torch.randn(batch_size, 1, 28, 28, device=device)  # Random tensor of shape [64, 1, 28, 28]
targets = torch.randint(10, (64,), device=device)  # Random integer targets for 64 examples

# In regular model training, one would forward the batch of examples and then
# call .backward() to compute gradients:

model = SimpleCNN().to(device=device)  # Instantiate SimpleCNN model on CUDA device
predictions = model(data)  # Forward pass to get predictions
loss = loss_fn(predictions, targets)  # Calculate loss
loss.backward()  # Backward pass to compute gradients


# Conceptually, per-sample-gradient computation is equivalent to: for each sample
# of the data, perform a forward and a backward pass to get a gradient.
def compute_grad(sample, target):
    sample = sample.unsqueeze(0)  # Add batch dimension to the sample tensor
    target = target.unsqueeze(0)  # Add batch dimension to the target tensor
    prediction = model(sample)  # Forward pass with the model
    loss = loss_fn(prediction, target)  # Calculate loss
    return torch.autograd.grad(loss, list(model.parameters()))  # Compute gradients


def compute_sample_grads(data, targets):
    sample_grads = [compute_grad(data[i], targets[i]) for i in range(batch_size)]  # Compute gradients for each sample
    sample_grads = zip(*sample_grads)  # Transpose list of gradients
    sample_grads = [torch.stack(shards) for shards in sample_grads]  # Stack gradients into tensors
    return sample_grads  # Return list of per-sample gradients


per_sample_grads = compute_sample_grads(data, targets)  # Compute per-sample gradients

# sample_grads[0] is the per-sample-grad for model.conv1.weight
# model.conv1.weight.shape is [32, 1, 3, 3]; notice how there is one gradient
# per sample in the batch for a total of 64.
print(per_sample_grads[0].shape)


######################################################################
# Per-sample-grads using functorch
# --------------------------------------------------------------------
# We can compute per-sample-gradients efficiently by using function transforms.
# First, let's create a stateless functional version of ``model`` by using
# ``functorch.make_functional_with_buffers``.
# 导入必要的函数和模块：从functorch中导入grad, make_functional_with_buffers, vmap
from functorch import grad, make_functional_with_buffers, vmap

# 使用make_functional_with_buffers函数，将model转换为函数式表示，并返回转换后的函数fmodel、参数params和缓冲buffers
fmodel, params, buffers = make_functional_with_buffers(model)


# 接下来，定义一个函数来计算模型给定单个输入时的损失，而不是批量输入。
# 这个函数接受参数、输入样本和目标样本，因为我们将对它们进行变换。
# 由于模型最初是为处理批量而设计的，我们将使用torch.unsqueeze来添加一个批量维度。
def compute_loss(params, buffers, sample, target):
    batch = sample.unsqueeze(0)  # 将输入样本添加一个批量维度
    targets = target.unsqueeze(0)  # 将目标样本添加一个批量维度
    predictions = fmodel(params, buffers, batch)  # 使用函数式模型预测输出
    loss = loss_fn(predictions, targets)  # 计算预测输出与目标的损失
    return loss


# 使用grad函数创建一个新的函数，用于计算相对于compute_loss的第一个参数（即params）的梯度。
ft_compute_grad = grad(compute_loss)

# ft_compute_grad用于计算单个（样本，目标）对的梯度。
# 我们可以使用vmap来让它计算整个样本和目标的批量的梯度。
# 注意，in_dims=(None, None, 0, 0)表示我们希望在数据和目标的第0维上映射ft_compute_grad，并对每个样本使用相同的params和buffers。
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

# 最后，使用转换后的函数来计算每个样本的梯度：
ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)
for per_sample_grad, ft_per_sample_grad in zip(per_sample_grads, ft_per_sample_grads):
    # 断言每个样本的梯度是否在数值上接近
    assert torch.allclose(per_sample_grad, ft_per_sample_grad, atol=1e-6, rtol=1e-6)

# 一个快速的注意事项：关于可以被vmap转换的函数类型有一些限制。
# 最适合转换的函数是纯函数：其输出仅由输入决定，并且没有副作用（例如突变）。
# vmap无法处理任意Python数据结构的突变，但可以处理许多基于PyTorch的原地操作。
```