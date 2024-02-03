# `stable-diffusion-webui\extensions-builtin\Lora\network_hada.py`

```py
# 导入 lyco_helpers 模块和 network 模块
import lyco_helpers
import network

# 定义 ModuleTypeHada 类，继承自 network.ModuleType
class ModuleTypeHada(network.ModuleType):
    # 创建模块的方法，接受网络和权重作为参数
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        # 检查权重中是否包含指定的键
        if all(x in weights.w for x in ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"]):
            # 如果包含指定的键，则返回 NetworkModuleHada 类的实例
            return NetworkModuleHada(net, weights)

        # 如果不包含指定的键，则返回 None
        return None

# 定义 NetworkModuleHada 类，继承自 network.NetworkModule
class NetworkModuleHada(network.NetworkModule):
    # 初始化方法，接受网络和权重作为参数
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        # 调用父类的初始化方法
        super().__init__(net, weights)

        # 如果 self.sd_module 中存在 'weight' 属性，则获取其形状
        if hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape

        # 获取权重中指定键对应的值
        self.w1a = weights.w["hada_w1_a"]
        self.w1b = weights.w["hada_w1_b"]
        self.dim = self.w1b.shape[0]
        self.w2a = weights.w["hada_w2_a"]
        self.w2b = weights.w["hada_w2_b"]

        # 获取权重中指定键对应的值，如果键不存在则返回 None
        self.t1 = weights.w.get("hada_t1")
        self.t2 = weights.w.get("hada_t2")
    # 计算上下游权重
    def calc_updown(self, orig_weight):
        # 将 self.w1a 转换为与 orig_weight 相同的设备和数据类型
        w1a = self.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
        # 将 self.w1b 转换为与 orig_weight 相同的设备和数据类型
        w1b = self.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
        # 将 self.w2a 转换为与 orig_weight 相同的设备和数据类型
        w2a = self.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
        # 将 self.w2b 转换为与 orig_weight 相同的设备和数据类型
        w2b = self.w2b.to(orig_weight.device, dtype=orig_weight.dtype)

        # 初始化输出形状为 [w1a 的行数, w1b 的列数]
        output_shape = [w1a.size(0), w1b.size(1)]

        # 如果 self.t1 不为 None
        if self.t1 is not None:
            # 更新输出形状为 [w1a 的列数, w1b 的列数]
            output_shape = [w1a.size(1), w1b.size(1)]
            # 将 self.t1 转换为与 orig_weight 相同的设备和数据类型
            t1 = self.t1.to(orig_weight.device, dtype=orig_weight.dtype)
            # 使用 lyco_helpers.make_weight_cp 创建 updown1
            updown1 = lyco_helpers.make_weight_cp(t1, w1a, w1b)
            # 更新输出形状为原输出形状加上 t1 的形状
            output_shape += t1.shape[2:]
        else:
            # 如果 w1b 的维度为 4
            if len(w1b.shape) == 4:
                # 更新输出形状为原输出形状加上 w1b 的形状
                output_shape += w1b.shape[2:]
            # 使用 lyco_helpers.rebuild_conventional 重建 updown1
            updown1 = lyco_helpers.rebuild_conventional(w1a, w1b, output_shape)

        # 如果 self.t2 不为 None
        if self.t2 is not None:
            # 将 self.t2 转换为与 orig_weight 相同的设备和数据类型
            t2 = self.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            # 使用 lyco_helpers.make_weight_cp 创建 updown2
            updown2 = lyco_helpers.make_weight_cp(t2, w2a, w2b)
        else:
            # 使用 lyco_helpers.rebuild_conventional 重建 updown2
            updown2 = lyco_helpers.rebuild_conventional(w2a, w2b, output_shape)

        # 计算 updown 为 updown1 和 updown2 的乘积
        updown = updown1 * updown2

        # 返回最终的上下游权重
        return self.finalize_updown(updown, orig_weight, output_shape)
```