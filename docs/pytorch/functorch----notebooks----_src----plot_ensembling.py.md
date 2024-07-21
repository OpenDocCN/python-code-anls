# `.\pytorch\functorch\notebooks\_src\plot_ensembling.py`

```
"""
==========================
Model ensembling
==========================
This example illustrates how to vectorize model ensembling using vmap.

What is model ensembling?
--------------------------------------------------------------------
Model ensembling combines the predictions from multiple models together.
Traditionally this is done by running each model on some inputs separately
and then combining the predictions. However, if you're running models with
the same architecture, then it may be possible to combine them together
using ``vmap``. ``vmap`` is a function transform that maps functions across
dimensions of the input tensors. One of its use cases is eliminating
for-loops and speeding them up through vectorization.

Let's demonstrate how to do this using an ensemble of simple CNNs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# Here's a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Define fully connected layers
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Forward pass through the network
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # Log softmax activation for output
        output = F.log_softmax(x, dim=1)
        # Return the raw logits for potential further processing
        output = x
        return output


# Let's generate some dummy data. Pretend that we're working with an MNIST dataset
# where the images are 28 by 28.
# Furthermore, let's say we wish to combine the predictions from 10 different
# models.
device = "cuda"
num_models = 10
data = torch.randn(100, 64, 1, 28, 28, device=device)
targets = torch.randint(10, (6400,), device=device)
models = [SimpleCNN().to(device) for _ in range(num_models)]

# We have a couple of options for generating predictions. Maybe we want
# to give each model a different randomized minibatch of data, or maybe we
# want to run the same minibatch of data through each model (e.g. if we were
# testing the effect of different model initializations).

# Option 1: different minibatch for each model
minibatches = data[:num_models]
predictions1 = [model(minibatch) for model, minibatch in zip(models, minibatches)]

# Option 2: Same minibatch
minibatch = data[0]
predictions2 = [model(minibatch) for model in models]


######################################################################
# Using vmap to vectorize the ensemble
# --------------------------------------------------------------------
# Let's use ``vmap`` to speed up the for-loop. We must first prepare the models
# for use with ``vmap``.
#
# First, let's combine the states of the model together by stacking each parameter.
# For example, model[i].fc1.weight has shape [9216, 128]; we are going to stack the
# 导入函数 `combine_state_for_ensemble` 从 `functorch` 模块中，用于组合多个模型的状态。
# 它返回一个无状态的模型 `fmodel`，以及堆叠的参数 `params` 和缓冲区 `buffers`。
from functorch import combine_state_for_ensemble

# 将 `combine_state_for_ensemble` 返回的参数 `params` 中的每个参数设置为需要梯度。
fmodel, params, buffers = combine_state_for_ensemble(models)
[p.requires_grad_() for p in params]

# Option 1: 使用不同的小批量数据对每个模型进行预测。
# 默认情况下，`vmap` 将一个函数映射到所有输入的第一个维度。在 `combine_state_for_ensemble` 之后，
# `params` 和 `buffers` 的每个维度前面都有一个大小为 `num_models` 的额外维度；
# `minibatches` 的维度大小为 `num_models`。
print([p.size(0) for p in params])
assert minibatches.shape == (num_models, 64, 1, 28, 28)
from functorch import vmap

# 使用 `vmap(fmodel)` 对 `params`、`buffers` 和 `minibatches` 进行映射，得到预测结果。
predictions1_vmap = vmap(fmodel)(params, buffers, minibatches)
assert torch.allclose(
    predictions1_vmap, torch.stack(predictions1), atol=1e-6, rtol=1e-6
)

# Option 2: 使用相同的数据小批量进行预测
# `vmap` 有一个 `in_dims` 参数，用于指定要映射的维度。使用 `None` 表示我们希望所有的 10 个模型都使用相同的小批量。
predictions2_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, minibatch)
assert torch.allclose(
    predictions2_vmap, torch.stack(predictions2), atol=1e-6, rtol=1e-6
)

# 一个快速的注意事项：关于哪些类型的函数可以通过 `vmap` 转换存在限制。
# 最适合转换的函数是纯函数：输出仅由输入决定，没有副作用（例如突变）的函数。
# `vmap` 无法处理任意 Python 数据结构的突变，但能够处理许多原地的 PyTorch 操作。
```