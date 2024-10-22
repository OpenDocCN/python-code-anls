# `.\cogvideo-finetune\sat\sgm\modules\ema.py`

```py
# 导入 PyTorch 库
import torch
# 从 PyTorch 导入神经网络模块
from torch import nn


# 定义一个名为 LitEma 的类，继承自 nn.Module
class LitEma(nn.Module):
    # 初始化函数，接受模型、衰减因子和更新次数的使用标志
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        # 调用父类构造函数
        super().__init__()
        # 检查衰减因子是否在有效范围内
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        # 初始化模型参数名称到阴影参数名称的映射字典
        self.m_name2s_name = {}
        # 注册衰减因子为一个缓冲区
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        # 根据是否使用更新次数注册相应的缓冲区
        self.register_buffer(
            "num_updates",
            torch.tensor(0, dtype=torch.int) if use_num_upates else torch.tensor(-1, dtype=torch.int),
        )

        # 遍历模型的每个命名参数
        for name, p in model.named_parameters():
            # 只处理需要梯度的参数
            if p.requires_grad:
                # 将参数名称中的点替换为字符，以便注册为缓冲区
                s_name = name.replace(".", "")
                # 更新名称映射字典
                self.m_name2s_name.update({name: s_name})
                # 注册参数的副本为缓冲区
                self.register_buffer(s_name, p.clone().detach().data)

        # 初始化存储的参数列表
        self.collected_params = []

    # 重置更新次数的函数
    def reset_num_updates(self):
        # 删除当前的更新次数缓冲区
        del self.num_updates
        # 注册新的更新次数缓冲区，初始值为 0
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

    # 前向传播函数，更新阴影参数
    def forward(self, model):
        # 获取当前的衰减因子
        decay = self.decay

        # 如果更新次数有效，更新衰减因子
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        # 计算 1 减去衰减因子
        one_minus_decay = 1.0 - decay

        # 在不计算梯度的情况下执行以下操作
        with torch.no_grad():
            # 获取模型的参数字典
            m_param = dict(model.named_parameters())
            # 获取当前缓冲区中的阴影参数字典
            shadow_params = dict(self.named_buffers())

            # 遍历模型参数
            for key in m_param:
                # 只处理需要梯度的参数
                if m_param[key].requires_grad:
                    # 获取阴影参数名称
                    sname = self.m_name2s_name[key]
                    # 将阴影参数转换为与模型参数相同的类型
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    # 更新阴影参数
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    # 确保该参数不在名称映射中
                    assert not key in self.m_name2s_name

    # 将阴影参数复制回模型的函数
    def copy_to(self, model):
        # 获取模型参数和阴影参数的字典
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        # 遍历模型参数
        for key in m_param:
            # 只处理需要梯度的参数
            if m_param[key].requires_grad:
                # 将阴影参数的数据复制到模型参数
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                # 确保该参数不在名称映射中
                assert not key in self.m_name2s_name

    # 存储当前参数以便稍后恢复的函数
    def store(self, parameters):
        """
        保存当前参数以便稍后恢复。
        参数:
          parameters: 可迭代的 `torch.nn.Parameter`；要临时存储的参数。
        """
        # 将当前参数的副本存储在列表中
        self.collected_params = [param.clone() for param in parameters]
    # 定义一个恢复参数的方法
    def restore(self, parameters):
        # 文档字符串，说明此方法的作用及参数
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
        # 遍历收集的参数和传入的参数，进行一一对应
        for c_param, param in zip(self.collected_params, parameters):
            # 将收集的参数数据复制到当前参数
            param.data.copy_(c_param.data)
```