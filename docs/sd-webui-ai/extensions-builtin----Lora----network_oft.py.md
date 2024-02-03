# `stable-diffusion-webui\extensions-builtin\Lora\network_oft.py`

```py
# 导入 torch 库
import torch
# 导入 network 模块
import network
# 从 lyco_helpers 模块中导入 factorization 函数
from lyco_helpers import factorization
# 从 einops 模块中导入 rearrange 函数

# 定义 ModuleTypeOFT 类，继承自 network.ModuleType 类
class ModuleTypeOFT(network.ModuleType):
    # 创建模块的方法
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        # 如果权重中包含 "oft_blocks" 或 "oft_diag"，则返回 NetworkModuleOFT 对象
        if all(x in weights.w for x in ["oft_blocks"]) or all(x in weights.w for x in ["oft_diag"]):
            return NetworkModuleOFT(net, weights)

        return None

# 支持 kohya-ss 实现的 COFT https://github.com/kohya-ss/sd-scripts/blob/main/networks/oft.py
# 和 KohakuBlueleaf 实现的 OFT/COFT https://github.com/KohakuBlueleaf/LyCORIS/blob/dev/lycoris/modules/diag_oft.py
class NetworkModuleOFT(network.NetworkModule):
    # 初始化函数，接受网络和权重作为参数
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):

        # 调用父类的初始化函数
        super().__init__(net, weights)

        # 初始化线性模块和原始模块列表
        self.lin_module = None
        self.org_module: list[torch.Module] = [self.sd_module]

        # 初始化缩放因子
        self.scale = 1.0

        # 判断是否为 kohya-ss 模型
        if "oft_blocks" in weights.w.keys():
            self.is_kohya = True
            # 获取权重中的 oft_blocks 和 alpha
            self.oft_blocks = weights.w["oft_blocks"] # (num_blocks, block_size, block_size)
            self.alpha = weights.w["alpha"] # alpha is constraint
            self.dim = self.oft_blocks.shape[0] # lora dim
        # LyCORIS 模型
        elif "oft_diag" in weights.w.keys():
            self.is_kohya = False
            # 获取权重中的 oft_diag
            self.oft_blocks = weights.w["oft_diag"]
            # self.alpha 未使用
            self.dim = self.oft_blocks.shape[1] # (num_blocks, block_size, block_size)

        # 判断模块类型
        is_linear = type(self.sd_module) in [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear]
        is_conv = type(self.sd_module) in [torch.nn.Conv2d]
        is_other_linear = type(self.sd_module) in [torch.nn.MultiheadAttention] # 不支持的模块类型

        # 根据模块类型设置输出维度
        if is_linear:
            self.out_dim = self.sd_module.out_features
        elif is_conv:
            self.out_dim = self.sd_module.out_channels
        elif is_other_linear:
            self.out_dim = self.sd_module.embed_dim

        # 如果是 kohya-ss 模型
        if self.is_kohya:
            # 设置约束和块数、块大小
            self.constraint = self.alpha * self.out_dim
            self.num_blocks = self.dim
            self.block_size = self.out_dim // self.dim
        else:
            # 计算块大小和块数
            self.constraint = None
            self.block_size, self.num_blocks = factorization(self.out_dim, self.dim)
    # 计算权重的上下游变化
    def calc_updown(self, orig_weight):
        # 将 self.oft_blocks 转移到 orig_weight 所在设备上，并且保持数据类型一致
        oft_blocks = self.oft_blocks.to(orig_weight.device, dtype=orig_weight.dtype)
        # 创建一个单位矩阵，设备为 self.oft_blocks 所在设备
        eye = torch.eye(self.block_size, device=self.oft_blocks.device)

        # 如果是 Kohya 模式
        if self.is_kohya:
            # 计算 oft_blocks 与其转置的差，确保是一个斜对称正交矩阵
            block_Q = oft_blocks - oft_blocks.transpose(1, 2)
            # 计算 block_Q 的范数
            norm_Q = torch.norm(block_Q.flatten())
            # 将 norm_Q 限制在 constraint 以内
            new_norm_Q = torch.clamp(norm_Q, max=self.constraint)
            # 对 block_Q 进行缩放
            block_Q = block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))
            # 计算 oft_blocks 的新值
            oft_blocks = torch.matmul(eye + block_Q, (eye - block_Q).float().inverse())

        # 将 oft_blocks 转移到 orig_weight 所在设备上，并且保持数据类型一致
        R = oft_blocks.to(orig_weight.device, dtype=orig_weight.dtype)

        # 对 orig_weight 进行重排列，以便进行矩阵乘法
        merged_weight = rearrange(orig_weight, '(k n) ... -> k n ...', k=self.num_blocks, n=self.block_size)
        # 使用 einsum 进行矩阵乘法
        merged_weight = torch.einsum(
            'k n m, k n ... -> k m ...',
            R,
            merged_weight
        )
        # 重新排列 merged_weight
        merged_weight = rearrange(merged_weight, 'k m ... -> (k m) ...')

        # 计算上下游变化
        updown = merged_weight.to(orig_weight.device, dtype=orig_weight.dtype) - orig_weight
        # 获取原始权重的形状
        output_shape = orig_weight.shape
        # 调用 finalize_updown 方法处理上下游变化
        return self.finalize_updown(updown, orig_weight, output_shape)
```