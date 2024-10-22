# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\wrappers.py`

```
# 导入 PyTorch 库及其神经网络模块
import torch
import torch.nn as nn
# 导入版本控制模块
from packaging import version

# 定义一个字符串，表示 OpenAI Wrapper 的路径
OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


# 定义一个身份包装器类，继承自 nn.Module
class IdentityWrapper(nn.Module):
    # 初始化方法，接受扩散模型、是否编译模型的标志和数据类型
    def __init__(self, diffusion_model, compile_model: bool = False, dtype: torch.dtype = torch.float32):
        # 调用父类的初始化方法
        super().__init__()
        # 根据 PyTorch 版本决定是否编译模型
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))  # 检查 PyTorch 版本
            and compile_model  # 仅当 compile_model 为 True 时
            else lambda x: x  # 否则返回原始模型
        )
        # 对扩散模型进行编译
        self.diffusion_model = compile(diffusion_model)
        # 保存数据类型
        self.dtype = dtype

    # 前向传播方法，接受任意数量的位置和关键字参数
    def forward(self, *args, **kwargs):
        # 调用扩散模型并返回结果
        return self.diffusion_model(*args, **kwargs)


# 定义 OpenAI Wrapper 类，继承自 IdentityWrapper
class OpenAIWrapper(IdentityWrapper):
    # 重写前向传播方法，接受输入张量、时间步、上下文字典和其他参数
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        # 将上下文字典中的每个值转换为指定的数据类型
        for key in c:
            c[key] = c[key].to(self.dtype)

        # 检查输入张量的形状是否为 3 维
        if len(x.shape) == 3:
            # 在最后一个维度拼接上下文中的 "concat" 数据
            x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=-1)
        else:
            # 在第一个维度拼接上下文中的 "concat" 数据
            x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)

        # 调用扩散模型进行前向传播，并返回结果
        return self.diffusion_model(
            x,  # 输入张量
            timesteps=t,  # 时间步
            context=c.get("crossattn", None),  # 上下文中的交叉注意力
            y=c.get("vector", None),  # 上下文中的向量
            **kwargs,  # 其他参数
        )
```