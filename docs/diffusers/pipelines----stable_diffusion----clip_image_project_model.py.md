# `.\diffusers\pipelines\stable_diffusion\clip_image_project_model.py`

```py
# 版权所有 2024 GLIGEN 作者和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律规定或书面同意，依据许可证分发的软件是以“原样”基础进行的，
# 不提供任何形式的保证或条件，无论是明示的还是暗示的。
# 请参阅许可证以获取有关权限和限制的具体条款。

# 从 PyTorch 的 nn 模块导入神经网络相关功能
from torch import nn

# 从配置工具导入 ConfigMixin 和注册配置的装饰器
from ...configuration_utils import ConfigMixin, register_to_config
# 从模型工具导入 ModelMixin，用于模型相关功能
from ...models.modeling_utils import ModelMixin

# 定义一个 CLIP 图像投影类，继承 ModelMixin 和 ConfigMixin
class CLIPImageProjection(ModelMixin, ConfigMixin):
    # 使用装饰器注册配置，定义初始化方法
    @register_to_config
    def __init__(self, hidden_size: int = 768):
        # 调用父类的初始化方法
        super().__init__()
        # 设置隐藏层大小，默认为 768
        self.hidden_size = hidden_size
        # 定义一个线性层，用于投影，输入和输出维度均为隐藏层大小，不使用偏置
        self.project = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    # 定义前向传播方法，接受输入 x
    def forward(self, x):
        # 将输入 x 通过线性层进行投影并返回结果
        return self.project(x)
```