# `.\cogvideo-finetune\sat\vae_modules\ema.py`

```py
# 导入 PyTorch 库和神经网络模块
import torch
from torch import nn

# 定义 LitEma 类，继承自 nn.Module
class LitEma(nn.Module):
    # 初始化方法，接收模型、衰减率和是否使用更新计数
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        # 调用父类的初始化方法
        super().__init__()
        # 检查衰减率是否在有效范围内
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        # 创建一个空字典用于存储模型参数名称到阴影参数名称的映射
        self.m_name2s_name = {}
        # 注册衰减率的缓冲区
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        # 根据是否使用更新计数注册相应的缓冲区
        self.register_buffer(
            "num_updates",
            torch.tensor(0, dtype=torch.int) if use_num_upates else torch.tensor(-1, dtype=torch.int),
        )

        # 遍历模型的命名参数
        for name, p in model.named_parameters():
            # 如果参数需要梯度
            if p.requires_grad:
                # 将名称中的 '.' 替换为 ''
                s_name = name.replace(".", "")
                # 更新模型参数名称到阴影参数名称的映射
                self.m_name2s_name.update({name: s_name})
                # 注册克隆的参数数据为缓冲区
                self.register_buffer(s_name, p.clone().detach().data)

        # 初始化收集的参数列表
        self.collected_params = []

    # 重置更新计数的方法
    def reset_num_updates(self):
        # 删除当前的更新计数缓冲区
        del self.num_updates
        # 注册一个新的更新计数缓冲区，初始值为 0
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

    # 前向传播方法，接收一个模型作为输入
    def forward(self, model):
        # 获取当前的衰减率
        decay = self.decay

        # 如果更新计数为非负
        if self.num_updates >= 0:
            # 更新计数加 1
            self.num_updates += 1
            # 更新衰减率
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        # 计算 1 减去衰减率
        one_minus_decay = 1.0 - decay

        # 在无梯度计算的上下文中执行
        with torch.no_grad():
            # 获取模型的参数字典
            m_param = dict(model.named_parameters())
            # 获取阴影参数的字典
            shadow_params = dict(self.named_buffers())

            # 遍历模型的参数
            for key in m_param:
                # 如果参数需要梯度
                if m_param[key].requires_grad:
                    # 获取对应的阴影参数名称
                    sname = self.m_name2s_name[key]
                    # 将阴影参数转换为与模型参数相同的类型
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    # 更新阴影参数
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    # 确保该参数不在映射中
                    assert not key in self.m_name2s_name

    # 将当前的阴影参数复制到模型参数的方法
    def copy_to(self, model):
        # 获取模型的参数字典
        m_param = dict(model.named_parameters())
        # 获取阴影参数的字典
        shadow_params = dict(self.named_buffers())
        # 遍历模型的参数
        for key in m_param:
            # 如果参数需要梯度
            if m_param[key].requires_grad:
                # 将阴影参数的数据复制到模型参数
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                # 确保该参数不在映射中
                assert not key in self.m_name2s_name

    # 存储当前参数以备后续恢复的方法
    def store(self, parameters):
        """
        保存当前参数以便稍后恢复。
        参数:
          parameters: 可迭代的 `torch.nn.Parameter`；需要临时存储的参数。
        """
        # 克隆参数并存储在 collected_params 列表中
        self.collected_params = [param.clone() for param in parameters]
    # 定义恢复方法，接受参数以恢复存储的模型参数
    def restore(self, parameters):
        # 文档字符串，说明此方法的作用和参数
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        # 遍历收集到的参数和传入的参数，并将收集到的参数数据复制到对应的参数
        for c_param, param in zip(self.collected_params, parameters):
            # 将收集到的参数数据复制到当前参数的数据
            param.data.copy_(c_param.data)
```