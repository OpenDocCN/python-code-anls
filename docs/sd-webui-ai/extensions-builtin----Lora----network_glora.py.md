# `stable-diffusion-webui\extensions-builtin\Lora\network_glora.py`

```py
# 导入 network 模块
import network

# 定义 ModuleTypeGLora 类，继承自 network.ModuleType
class ModuleTypeGLora(network.ModuleType):
    # 创建模块的方法，接受网络和权重作为参数
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        # 检查权重中是否包含指定的键
        if all(x in weights.w for x in ["a1.weight", "a2.weight", "alpha", "b1.weight", "b2.weight"]):
            # 如果包含指定的键，则返回 NetworkModuleGLora 对象
            return NetworkModuleGLora(net, weights)

        # 如果不包含指定的键，则返回 None
        return None

# 从 https://github.com/KohakuBlueleaf/LyCORIS 改编而来
# 定义 NetworkModuleGLora 类，继承自 network.NetworkModule
class NetworkModuleGLora(network.NetworkModule):
    # 初始化方法，接受网络和权重作为参数
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        # 调用父类的初始化方法
        super().__init__(net, weights)

        # 如果 self.sd_module 中有 'weight' 属性
        if hasattr(self.sd_module, 'weight'):
            # 获取权重的形状
            self.shape = self.sd_module.weight.shape

        # 获取指定键对应的权重值
        self.w1a = weights.w["a1.weight"]
        self.w1b = weights.w["b1.weight"]
        self.w2a = weights.w["a2.weight"]
        self.w2b = weights.w["b2.weight"]

    # 计算上下文方法，接受原始权重作为参数
    def calc_updown(self, orig_weight):
        # 将权重转换为指定设备和数据类型
        w1a = self.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
        w1b = self.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
        w2a = self.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
        w2b = self.w2b.to(orig_weight.device, dtype=orig_weight.dtype)

        # 计算输出形状
        output_shape = [w1a.size(0), w1b.size(1)]
        # 计算上下文
        updown = ((w2b @ w1b) + ((orig_weight @ w2a) @ w1a))

        # 返回最终的上下文结果
        return self.finalize_updown(updown, orig_weight, output_shape)
```