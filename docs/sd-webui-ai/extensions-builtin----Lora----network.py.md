# `stable-diffusion-webui\extensions-builtin\Lora\network.py`

```py
# 导入必要的模块
from __future__ import annotations
import os
from collections import namedtuple
import enum

# 从自定义模块中导入指定内容
from modules import sd_models, cache, errors, hashes, shared

# 创建一个具名元组，用于表示网络权重
NetworkWeights = namedtuple('NetworkWeights', ['network_key', 'sd_key', 'w', 'sd_module'])

# 定义元数据标签的顺序
metadata_tags_order = {"ss_sd_model_name": 1, "ss_resolution": 2, "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}

# 定义枚举类 SdVersion
class SdVersion(enum.Enum):
    Unknown = 1
    SD1 = 2
    SD2 = 3
    SDXL = 4

# 定义 NetworkOnDisk 类
class NetworkOnDisk:
    # 初始化方法
    def __init__(self, name, filename):
        # 设置属性值
        self.name = name
        self.filename = filename
        self.metadata = {}
        # 判断文件是否为 .safetensors 格式
        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        # 定义读取元数据的函数
        def read_metadata():
            # 从 .safetensors 文件中读取元数据
            metadata = sd_models.read_metadata_from_safetensors(filename)
            # 移除不需要在 UI 中显示的封面图片
            metadata.pop('ssmd_cover_images', None)

            return metadata

        # 如果文件为 .safetensors 格式，则尝试从缓存中读取元数据
        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file('safetensors-metadata', "lora/" + self.name, filename, read_metadata)
            except Exception as e:
                # 处理读取元数据时的异常
                errors.display(e, f"reading lora {filename}")

        # 对元数据按照指定顺序进行排序
        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v

            self.metadata = m

        # 设置别名属性
        self.alias = self.metadata.get('ss_output_name', self.name)

        # 初始化哈希值和短哈希值
        self.hash = None
        self.shorthash = None
        # 设置哈希值
        self.set_hash(
            self.metadata.get('sshs_model_hash') or
            hashes.sha256_from_cache(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or
            ''
        )

        # 检测 SD 版本
        self.sd_version = self.detect_version()
    # 检测模型版本，根据 metadata 中的信息判断模型版本
    def detect_version(self):
        # 如果 metadata 中的 ss_base_model_version 以 "sdxl_" 开头，则返回 SdVersion.SDXL
        if str(self.metadata.get('ss_base_model_version', "")).startswith("sdxl_"):
            return SdVersion.SDXL
        # 如果 metadata 中的 ss_v2 为 "True"，则返回 SdVersion.SD2
        elif str(self.metadata.get('ss_v2', "")) == "True":
            return SdVersion.SD2
        # 如果 metadata 不为空，则返回 SdVersion.SD1
        elif len(self.metadata):
            return SdVersion.SD1

        # 如果以上条件都不满足，则返回 SdVersion.Unknown

    # 设置哈希值，根据给定的值设置哈希值和短哈希值
    def set_hash(self, v):
        # 设置哈希值为给定值
        self.hash = v
        # 设置短哈希值为哈希值的前12位
        self.shorthash = self.hash[0:12]

        # 如果存在短哈希值
        if self.shorthash:
            # 导入 networks 模块
            import networks
            # 将当前对象添加到可用网络哈希查找表中
            networks.available_network_hash_lookup[self.shorthash] = self

    # 读取哈希值，如果当前对象没有哈希值，则根据文件名和其他信息计算哈希值
    def read_hash(self):
        # 如果当前对象没有哈希值
        if not self.hash:
            # 根据文件名和其他信息计算哈希值，并设置为当前对象的哈希值
            self.set_hash(hashes.sha256(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or '')

    # 获取别名，根据设置和网络模块的限制返回别名或名称
    def get_alias(self):
        # 导入 networks 模块
        import networks
        # 如果设置为使用文件名作为首选名称，或者别名在网络模块的禁止别名列表中，则返回名称
        if shared.opts.lora_preferred_name == "Filename" or self.alias.lower() in networks.forbidden_network_aliases:
            return self.name
        # 否则返回别名
        else:
            return self.alias
class Network:  # LoraModule
    # 定义 Network 类，表示网络模块
    def __init__(self, name, network_on_disk: NetworkOnDisk):
        # 初始化 Network 对象，设置名称和磁盘上的网络
        self.name = name
        self.network_on_disk = network_on_disk
        self.te_multiplier = 1.0
        self.unet_multiplier = 1.0
        self.dyn_dim = None
        self.modules = {}
        self.bundle_embeddings = {}
        self.mtime = None

        self.mentioned_name = None
        """the text that was used to add the network to prompt - can be either name or an alias"""


class ModuleType:
    # 定义 ModuleType 类
    def create_module(self, net: Network, weights: NetworkWeights) -> Network | None:
        # 创建模块的方法，接受网络和权重作为参数，返回 Network 对象或 None
        return None


class NetworkModule:
    # 定义 NetworkModule 类
    def __init__(self, net: Network, weights: NetworkWeights):
        # 初始化 NetworkModule 对象，接受网络和权重作为参数
        self.network = net
        self.network_key = weights.network_key
        self.sd_key = weights.sd_key
        self.sd_module = weights.sd_module

        if hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape

        self.dim = None
        self.bias = weights.w.get("bias")
        self.alpha = weights.w["alpha"].item() if "alpha" in weights.w else None
        self.scale = weights.w["scale"].item() if "scale" in weights.w else None

    def multiplier(self):
        # 计算模块的乘数
        if 'transformer' in self.sd_key[:20]:
            return self.network.te_multiplier
        else:
            return self.network.unet_multiplier

    def calc_scale(self):
        # 计算模块的缩放比例
        if self.scale is not None:
            return self.scale
        if self.dim is not None and self.alpha is not None:
            return self.alpha / self.dim

        return 1.0
    # 定义一个方法，用于处理反向传播的梯度计算和参数更新
    def finalize_updown(self, updown, orig_weight, output_shape, ex_bias=None):
        # 如果存在偏置项
        if self.bias is not None:
            # 将梯度重塑为与偏置项相同的形状
            updown = updown.reshape(self.bias.shape)
            # 将偏置项添加到梯度中
            updown += self.bias.to(orig_weight.device, dtype=orig_weight.dtype)
            # 将梯度重塑为输出形状
            updown = updown.reshape(output_shape)

        # 如果输出形状是四维的
        if len(output_shape) == 4:
            # 将梯度重塑为输出形状
            updown = updown.reshape(output_shape)

        # 如果原始权重的元素数量与梯度的元素数量相同
        if orig_weight.size().numel() == updown.size().numel():
            # 将梯度重塑为原始权重的形状
            updown = updown.reshape(orig_weight.shape)

        # 如果存在额外的偏置项
        if ex_bias is not None:
            # 将额外的偏置项乘以乘数
            ex_bias = ex_bias * self.multiplier()

        # 返回更新后的梯度和额外的偏置项
        return updown * self.calc_scale() * self.multiplier(), ex_bias

    # 定义一个方法，用于计算梯度
    def calc_updown(self, target):
        # 抛出未实现错误
        raise NotImplementedError()

    # 定义一个方法，用于前向传播
    def forward(self, x, y):
        # 抛出未实现错误
        raise NotImplementedError()
```