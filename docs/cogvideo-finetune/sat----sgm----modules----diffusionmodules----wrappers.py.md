# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\wrappers.py`

```py
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 从 packaging 库导入 version 模块以便进行版本比较
from packaging import version

# 定义 OpenAIWrapper 的模块路径
OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


# 定义身份包装器类，继承自 nn.Module
class IdentityWrapper(nn.Module):
    # 初始化函数，接收扩散模型、是否编译模型的标志和数据类型
    def __init__(self, diffusion_model, compile_model: bool = False, dtype: torch.dtype = torch.float32):
        # 调用父类的初始化函数
        super().__init__()
        # 判断 PyTorch 版本，选择是否编译模型
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0")) and compile_model
            else lambda x: x  # 如果不满足条件，返回原始输入
        )
        # 编译扩散模型并赋值给实例变量
        self.diffusion_model = compile(diffusion_model)
        # 设置数据类型
        self.dtype = dtype

    # 前向传播函数，接收任意数量的位置参数和关键字参数
    def forward(self, *args, **kwargs):
        # 调用扩散模型的前向传播，并返回结果
        return self.diffusion_model(*args, **kwargs)


# 定义 OpenAIWrapper 类，继承自 IdentityWrapper
class OpenAIWrapper(IdentityWrapper):
    # 重写前向传播函数，接收输入张量 x、时间步 t、上下文字典 c 以及其他关键字参数
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        # 将上下文字典中的每个张量转换为指定的数据类型
        for key in c:
            c[key] = c[key].to(self.dtype)

        # 如果输入张量是 4 维，按维度 1 拼接上下文中的 "concat" 张量
        if x.dim() == 4:
            x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        # 如果输入张量是 5 维，按维度 2 拼接上下文中的 "concat" 张量
        elif x.dim() == 5:
            x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=2)
        # 如果输入张量的维度不符合要求，抛出值错误
        else:
            raise ValueError("Input tensor must be 4D or 5D")

        # 调用扩散模型的前向传播，传入处理后的张量、时间步和上下文等
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,
        )
```