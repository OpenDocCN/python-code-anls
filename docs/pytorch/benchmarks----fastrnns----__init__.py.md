# `.\pytorch\benchmarks\fastrnns\__init__.py`

```
# 导入.cells模块中的所有内容（忽略F403警告）
from .cells import *  # noqa: F403

# 导入.factory模块中的所有内容（忽略F403警告）
from .factory import *  # noqa: F403

# 定义用于循环神经网络的参数
# seqLength表示序列长度为100
seqLength = 100
# numLayers表示神经网络的层数为2
numLayers = 2
# inputSize表示输入数据的特征维度为512
inputSize = 512
# hiddenSize表示隐藏层的神经元数量为512
hiddenSize = 512
# miniBatch表示每个小批量数据的样本数为64
miniBatch = 64
```