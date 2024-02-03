# `stable-diffusion-webui\extensions-builtin\Lora\extra_networks_lora.py`

```
# 导入额外的网络模块和共享模块
from modules import extra_networks, shared
# 导入网络模块

class ExtraNetworkLora(extra_networks.ExtraNetwork):
    # 初始化函数，继承自ExtraNetwork类
    def __init__(self):
        # 调用父类的初始化函数，传入'lora'参数
        super().__init__('lora')

        # 初始化errors字典，用于记录网络操作中的错误次数
        self.errors = {}
        """mapping of network names to the number of errors the network had during operation"""

    # 停用网络函数，接收参数p
    def deactivate(self, p):
        # 如果errors字典不为空
        if self.errors:
            # 在控制台输出带有错误的网络名称和错误次数的信息
            p.comment("Networks with errors: " + ", ".join(f"{k} ({v})" for k, v in self.errors.items()))

            # 清空errors字典
            self.errors.clear()
```