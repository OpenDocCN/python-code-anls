# `stable-diffusion-webui\extensions-builtin\Lora\network_norm.py`

```
# 导入 network 模块
import network

# 定义 ModuleTypeNorm 类，继承自 network.ModuleType
class ModuleTypeNorm(network.ModuleType):
    # 创建模块的方法，接受网络和权重作为参数
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        # 检查权重中是否包含"w_norm"和"b_norm"，如果都包含则创建 NetworkModuleNorm 实例
        if all(x in weights.w for x in ["w_norm", "b_norm"]):
            return NetworkModuleNorm(net, weights)

        # 如果不包含则返回 None
        return None

# 定义 NetworkModuleNorm 类，继承自 network.NetworkModule
class NetworkModuleNorm(network.NetworkModule):
    # 初始化方法，接受网络和权重作为参数
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        # 调用父类的初始化方法
        super().__init__(net, weights)

        # 从权重中获取"w_norm"和"b_norm"，并赋值给实例变量
        self.w_norm = weights.w.get("w_norm")
        self.b_norm = weights.w.get("b_norm")

    # 计算上下行的方法，接受原始权重作为参数
    def calc_updown(self, orig_weight):
        # 获取"w_norm"的形状
        output_shape = self.w_norm.shape
        # 将"w_norm"转换为原始权重的设备和数据类型
        updown = self.w_norm.to(orig_weight.device, dtype=orig_weight.dtype)

        # 如果"b_norm"不为 None，则将其转换为原始权重的设备和数据类型
        if self.b_norm is not None:
            ex_bias = self.b_norm.to(orig_weight.device, dtype=orig_weight.dtype)
        else:
            ex_bias = None

        # 调用 finalize_updown 方法，返回最终的上下行权重
        return self.finalize_updown(updown, orig_weight, output_shape, ex_bias)
```