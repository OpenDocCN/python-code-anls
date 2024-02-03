# `stable-diffusion-webui\extensions-builtin\Lora\network_ia3.py`

```
# 导入 network 模块
import network

# 定义 ModuleTypeIa3 类，继承自 network.ModuleType
class ModuleTypeIa3(network.ModuleType):
    # 创建模块的方法，接受网络和权重作为参数
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        # 检查权重中是否包含 "weight" 字段
        if all(x in weights.w for x in ["weight"]):
            # 如果包含，则返回 NetworkModuleIa3 对象
            return NetworkModuleIa3(net, weights)

        # 如果不包含，则返回 None
        return None

# 定义 NetworkModuleIa3 类，继承自 network.NetworkModule
class NetworkModuleIa3(network.NetworkModule):
    # 初始化方法，接受网络和权重作为参数
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        # 调用父类的初始化方法
        super().__init__(net, weights)

        # 从权重中获取 "weight" 字段的值，并赋给 self.w
        self.w = weights.w["weight"]
        # 从权重中获取 "on_input" 字段的值，并转换为标量
        self.on_input = weights.w["on_input"].item()

    # 计算上下行权重的方法，接受原始权重作为参数
    def calc_updown(self, orig_weight):
        # 将 self.w 转换为与原始权重相同的设备和数据类型
        w = self.w.to(orig_weight.device, dtype=orig_weight.dtype)

        # 根据权重的维度确定输出形状
        output_shape = [w.size(0), orig_weight.size(1)]
        # 如果 on_input 为真，则反转输出形状
        if self.on_input:
            output_shape.reverse()
        else:
            # 如果 on_input 为假，则将 w 重塑为一列
            w = w.reshape(-1, 1)

        # 计算上下行权重
        updown = orig_weight * w

        # 调用 finalize_updown 方法，返回最终的上下行权重和输出形状
        return self.finalize_updown(updown, orig_weight, output_shape)
```