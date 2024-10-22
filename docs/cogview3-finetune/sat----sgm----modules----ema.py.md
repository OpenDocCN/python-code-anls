# `.\cogview3-finetune\sat\sgm\modules\ema.py`

```
# 导入 PyTorch 库
import torch
# 从 PyTorch 导入神经网络模块
from torch import nn


# 定义一个继承自 nn.Module 的类 LitEma
class LitEma(nn.Module):
    # 初始化函数，接收模型、衰减率和是否使用更新计数
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        # 调用父类初始化
        super().__init__()
        # 检查衰减率是否在 0 到 1 之间
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        # 初始化模型参数名到阴影参数名的映射
        self.m_name2s_name = {}
        # 注册衰减率的缓冲区
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        # 根据是否使用更新计数注册更新次数的缓冲区
        self.register_buffer(
            "num_updates",
            torch.tensor(0, dtype=torch.int)
            if use_num_upates
            else torch.tensor(-1, dtype=torch.int),
        )

        # 遍历模型的命名参数
        for name, p in model.named_parameters():
            # 如果参数需要梯度更新
            if p.requires_grad:
                # 移除参数名中的 '.' 字符
                s_name = name.replace(".", "")
                # 更新模型参数名到阴影参数名的映射
                self.m_name2s_name.update({name: s_name})
                # 注册参数的缓冲区
                self.register_buffer(s_name, p.clone().detach().data)

        # 初始化收集的参数列表
        self.collected_params = []

    # 重置更新计数的方法
    def reset_num_updates(self):
        # 删除 num_updates 的缓冲区
        del self.num_updates
        # 注册更新计数的缓冲区，初始化为 0
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

    # 前向传播方法，接收模型作为输入
    def forward(self, model):
        # 获取当前衰减率
        decay = self.decay

        # 如果更新计数有效
        if self.num_updates >= 0:
            # 增加更新计数
            self.num_updates += 1
            # 计算新的衰减率
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        # 计算 1 减去衰减率
        one_minus_decay = 1.0 - decay

        # 在不跟踪梯度的情况下执行操作
        with torch.no_grad():
            # 获取模型的命名参数字典
            m_param = dict(model.named_parameters())
            # 获取阴影参数的命名缓冲区字典
            shadow_params = dict(self.named_buffers())

            # 遍历模型参数字典
            for key in m_param:
                # 如果参数需要梯度更新
                if m_param[key].requires_grad:
                    # 获取对应的阴影参数名
                    sname = self.m_name2s_name[key]
                    # 将阴影参数转换为与模型参数相同的数据类型
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    # 更新阴影参数的值
                    shadow_params[sname].sub_(
                        one_minus_decay * (shadow_params[sname] - m_param[key])
                    )
                else:
                    # 确保此参数不在映射中
                    assert not key in self.m_name2s_name

    # 将阴影参数复制到模型参数的方法
    def copy_to(self, model):
        # 获取模型的命名参数字典
        m_param = dict(model.named_parameters())
        # 获取阴影参数的命名缓冲区字典
        shadow_params = dict(self.named_buffers())
        # 遍历模型参数字典
        for key in m_param:
            # 如果参数需要梯度更新
            if m_param[key].requires_grad:
                # 复制阴影参数的数据到模型参数
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                # 确保此参数不在映射中
                assert not key in self.m_name2s_name

    # 存储当前参数以供稍后恢复的方法
    def store(self, parameters):
        """
        保存当前参数以供稍后恢复。
        参数：
          parameters: 可迭代的 `torch.nn.Parameter`；需要临时存储的参数。
        """
        # 克隆参数并存储在收集的参数列表中
        self.collected_params = [param.clone() for param in parameters]
    # 定义一个恢复方法，用于恢复存储的参数
    def restore(self, parameters):
        """
        恢复通过 `store` 方法存储的参数。
        这对于在不影响原始优化过程的情况下使用 EMA 参数验证模型很有用。
        在调用 `copy_to` 方法之前存储参数。
        验证（或保存模型）后，使用此方法恢复先前的参数。
        Args:
          parameters: 可迭代的 `torch.nn.Parameter`；需要用存储的参数更新的参数。
        """
        # 遍历已收集的参数和输入参数，成对处理
        for c_param, param in zip(self.collected_params, parameters):
            # 将已收集参数的数据复制到输入参数的数据中
            param.data.copy_(c_param.data)
```