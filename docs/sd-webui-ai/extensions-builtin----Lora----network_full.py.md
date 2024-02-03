# `stable-diffusion-webui\extensions-builtin\Lora\network_full.py`

```py
# 导入 network 模块
import network

# 定义 ModuleTypeFull 类，继承自 network.ModuleType
class ModuleTypeFull(network.ModuleType):
    # 创建模块的方法，接受网络和权重作为参数
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        # 检查权重中是否包含 "diff" 键
        if all(x in weights.w for x in ["diff"]):
            # 如果包含，则返回 NetworkModuleFull 对象
            return NetworkModuleFull(net, weights)

        # 如果不包含，则返回 None
        return None

# 定义 NetworkModuleFull 类，继承自 network.NetworkModule
class NetworkModuleFull(network.NetworkModule):
    # 初始化方法，接受网络和权重作为参数
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        # 调用父类的初始化方法
        super().__init__(net, weights)

        # 获取权重中的 "diff" 键对应的值，并赋给 self.weight
        self.weight = weights.w.get("diff")
        # 获取权重中的 "diff_b" 键对应的值，并赋给 self.ex_bias
        self.ex_bias = weights.w.get("diff_b")

    # 计算上下行的方法，接受原始权重作为参数
    def calc_updown(self, orig_weight):
        # 获取输出形状
        output_shape = self.weight.shape
        # 将权重转换为原始权重的设备和数据类型
        updown = self.weight.to(orig_weight.device, dtype=orig_weight.dtype)
        # 如果存在额外的偏置项
        if self.ex_bias is not None:
            # 将额外的偏置项转换为原始权重的设备和数据类型
            ex_bias = self.ex_bias.to(orig_weight.device, dtype=orig_weight.dtype)
        else:
            ex_bias = None

        # 调用 finalize_updown 方法，返回最终的上下行结果
        return self.finalize_updown(updown, orig_weight, output_shape, ex_bias)
```