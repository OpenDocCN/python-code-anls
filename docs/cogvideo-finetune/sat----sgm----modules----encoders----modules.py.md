# `.\cogvideo-finetune\sat\sgm\modules\encoders\modules.py`

```py
# 导入数学库
import math
# 导入上下文管理器的空上下文
from contextlib import nullcontext
# 导入部分函数的工具
from functools import partial
# 导入类型提示相关的类型
from typing import Dict, List, Optional, Tuple, Union

# 导入 Kornia 图像处理库
import kornia
# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 einops 库的重排列和重复函数
from einops import rearrange, repeat
# 导入 OmegaConf 库的 ListConfig
from omegaconf import ListConfig
# 导入 PyTorch 的检查点工具
from torch.utils.checkpoint import checkpoint
# 导入 Hugging Face 的 T5 模型和分词器
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
)

# 从自定义工具模块导入多个函数
from ...util import (
    append_dims,
    autocast,
    count_params,
    default,
    disabled_train,
    expand_dims_like,
    instantiate_from_config,
)


# 定义一个抽象的嵌入模型类，继承自 PyTorch 的 nn.Module
class AbstractEmbModel(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类初始化方法
        super().__init__()
        # 初始化是否可训练的标志
        self._is_trainable = None
        # 初始化 UCG 速率
        self._ucg_rate = None
        # 初始化输入键
        self._input_key = None

    # 返回是否可训练的属性
    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    # 返回 UCG 速率属性，可能是浮点数或张量
    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    # 返回输入键属性
    @property
    def input_key(self) -> str:
        return self._input_key

    # 设置是否可训练的属性
    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    # 设置 UCG 速率属性
    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    # 设置输入键属性
    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    # 删除是否可训练的属性
    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    # 删除 UCG 速率属性
    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    # 删除输入键属性
    @input_key.deleter
    def input_key(self):
        del self._input_key


# 定义一个通用条件器类，继承自 PyTorch 的 nn.Module
class GeneralConditioner(nn.Module):
    # 定义输出维度到键的映射
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    # 定义键到拼接维度的映射
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}
    # 初始化方法，接受嵌入模型配置及相关参数
        def __init__(self, emb_models: Union[List, ListConfig], cor_embs=[], cor_p=[]):
            # 调用父类初始化方法
            super().__init__()
            # 用于存储嵌入模型实例的列表
            embedders = []
            # 遍历每个嵌入模型的配置
            for n, embconfig in enumerate(emb_models):
                # 根据配置实例化嵌入模型
                embedder = instantiate_from_config(embconfig)
                # 确保实例是 AbstractEmbModel 的子类
                assert isinstance(
                    embedder, AbstractEmbModel
                ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
                # 设置嵌入模型是否可训练
                embedder.is_trainable = embconfig.get("is_trainable", False)
                # 设置嵌入模型的 ucg_rate 参数
                embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
                # 如果模型不可训练
                if not embedder.is_trainable:
                    # 禁用训练方法
                    embedder.train = disabled_train
                    # 将模型参数的 requires_grad 属性设为 False
                    for param in embedder.parameters():
                        param.requires_grad = False
                    # 将模型设置为评估模式
                    embedder.eval()
                # 打印嵌入模型的初始化信息
                print(
                    f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                    f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
                )
    
                # 检查配置中是否有 input_key，并赋值给嵌入模型
                if "input_key" in embconfig:
                    embedder.input_key = embconfig["input_key"]
                # 检查配置中是否有 input_keys，并赋值给嵌入模型
                elif "input_keys" in embconfig:
                    embedder.input_keys = embconfig["input_keys"]
                # 如果都没有，抛出 KeyError
                else:
                    raise KeyError(f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}")
    
                # 设置嵌入模型的 legacy_ucg_value 参数
                embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
                # 如果有 legacy_ucg_val，则初始化随机状态
                if embedder.legacy_ucg_val is not None:
                    embedder.ucg_prng = np.random.RandomState()
    
                # 将嵌入模型添加到列表中
                embedders.append(embedder)
            # 将嵌入模型列表转换为 nn.ModuleList
            self.embedders = nn.ModuleList(embedders)
    
            # 如果有 cor_embs，确保 cor_p 的长度正确
            if len(cor_embs) > 0:
                assert len(cor_p) == 2 ** len(cor_embs)
            # 设置相关嵌入和参数
            self.cor_embs = cor_embs
            self.cor_p = cor_p
    
        # 获取 UCG 值的方法，可能会基于概率进行赋值
        def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
            # 确保 legacy_ucg_val 不为 None
            assert embedder.legacy_ucg_val is not None
            # 获取 ucg_rate 参数
            p = embedder.ucg_rate
            # 获取 legacy_ucg_val 值
            val = embedder.legacy_ucg_val
            # 遍历 batch 中的输入数据
            for i in range(len(batch[embedder.input_key])):
                # 根据概率选择是否替换为 legacy_ucg_val
                if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[embedder.input_key][i] = val
            # 返回更新后的 batch
            return batch
    
        # 获取 UCG 值的方法，基于条件进行赋值
        def surely_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict, cond_or_not) -> Dict:
            # 确保 legacy_ucg_val 不为 None
            assert embedder.legacy_ucg_val is not None
            # 获取 legacy_ucg_val 值
            val = embedder.legacy_ucg_val
            # 遍历 batch 中的输入数据
            for i in range(len(batch[embedder.input_key])):
                # 如果条件为真，则替换为 legacy_ucg_val
                if cond_or_not[i]:
                    batch[embedder.input_key][i] = val
            # 返回更新后的 batch
            return batch
    
        # 获取单个嵌入的方法
        def get_single_embedding(
            self,
            embedder,
            batch,
            output,
            cond_or_not: Optional[np.ndarray] = None,
            force_zero_embeddings: Optional[List] = None,
    ):
        # 根据 embedder 是否可训练选择适当的上下文管理器
        embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
        # 进入上下文管理器以控制梯度计算
        with embedding_context():
            # 检查 embedder 是否有输入键属性并且不为 None
            if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                # 检查 embedder 的遗留 UCG 值是否不为 None
                if embedder.legacy_ucg_val is not None:
                    # 如果条件为 None，获取可能的 UCG 值
                    if cond_or_not is None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    # 否则，确保获取 UCG 值
                    else:
                        batch = self.surely_get_ucg_val(embedder, batch, cond_or_not)
                # 使用指定的输入键从 batch 中获取嵌入输出
                emb_out = embedder(batch[embedder.input_key])
            # 检查 embedder 是否有输入键列表
            elif hasattr(embedder, "input_keys"):
                # 解包 batch 中的输入键以获取嵌入输出
                emb_out = embedder(*[batch[k] for k in embedder.input_keys])
        # 确保嵌入输出是张量或序列类型
        assert isinstance(
            emb_out, (torch.Tensor, list, tuple)
        ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
        # 如果嵌入输出不是列表或元组，则将其转换为列表
        if not isinstance(emb_out, (list, tuple)):
            emb_out = [emb_out]
        # 遍历嵌入输出中的每个嵌入
        for emb in emb_out:
            # 根据嵌入的维度获取对应的输出键
            out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
            # 如果 UCG 率大于 0 且没有遗留 UCG 值
            if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                # 如果条件为 None，随机应用 UCG 率
                if cond_or_not is None:
                    emb = (
                        expand_dims_like(
                            torch.bernoulli((1.0 - embedder.ucg_rate) * torch.ones(emb.shape[0], device=emb.device)),
                            emb,
                        )
                        * emb
                    )
                # 否则，基于条件应用 UCG
                else:
                    emb = (
                        expand_dims_like(
                            torch.tensor(1 - cond_or_not, dtype=emb.dtype, device=emb.device),
                            emb,
                        )
                        * emb
                    )
            # 如果输入键在强制零嵌入列表中，将嵌入置为零
            if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
                emb = torch.zeros_like(emb)
            # 如果输出中已有该输出键，则拼接嵌入
            if out_key in output:
                output[out_key] = torch.cat((output[out_key], emb), self.KEY2CATDIM[out_key])
            # 否则，直接保存嵌入
            else:
                output[out_key] = emb
        # 返回最终输出字典
        return output
    # 定义一个前向传播函数，接收批次数据和强制零嵌入列表
        def forward(self, batch: Dict, force_zero_embeddings: Optional[List] = None) -> Dict:
            # 初始化输出字典
            output = dict()
            # 如果没有强制零嵌入，初始化为空列表
            if force_zero_embeddings is None:
                force_zero_embeddings = []
    
            # 如果相关嵌入列表不为空
            if len(self.cor_embs) > 0:
                # 获取批次大小
                batch_size = len(batch[list(batch.keys())[0]])
                # 根据相关概率随机选择索引
                rand_idx = np.random.choice(len(self.cor_p), size=(batch_size,), p=self.cor_p)
                # 遍历相关嵌入索引
                for emb_idx in self.cor_embs:
                    # 计算条件是否满足
                    cond_or_not = rand_idx % 2
                    # 更新随机索引
                    rand_idx //= 2
                    # 获取单个嵌入并更新输出字典
                    output = self.get_single_embedding(
                        self.embedders[emb_idx],
                        batch,
                        output=output,
                        cond_or_not=cond_or_not,
                        force_zero_embeddings=force_zero_embeddings,
                    )
    
            # 遍历所有嵌入
            for i, embedder in enumerate(self.embedders):
                # 如果当前索引在相关嵌入列表中，则跳过
                if i in self.cor_embs:
                    continue
                # 获取单个嵌入并更新输出字典
                output = self.get_single_embedding(
                    embedder, batch, output=output, force_zero_embeddings=force_zero_embeddings
                )
            # 返回最终输出字典
            return output
    
        # 定义获取无条件条件的函数
        def get_unconditional_conditioning(self, batch_c, batch_uc=None, force_uc_zero_embeddings=None):
            # 如果没有强制无条件嵌入，初始化为空列表
            if force_uc_zero_embeddings is None:
                force_uc_zero_embeddings = []
            # 初始化无条件生成率列表
            ucg_rates = list()
            # 遍历所有嵌入，保存其生成率并将生成率设置为零
            for embedder in self.embedders:
                ucg_rates.append(embedder.ucg_rate)
                embedder.ucg_rate = 0.0
            # 保存当前相关嵌入和概率
            cor_embs = self.cor_embs
            cor_p = self.cor_p
            # 清空相关嵌入和概率
            self.cor_embs = []
            self.cor_p = []
    
            # 计算条件输出
            c = self(batch_c)
            # 计算无条件输出
            uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)
    
            # 恢复每个嵌入的生成率
            for embedder, rate in zip(self.embedders, ucg_rates):
                embedder.ucg_rate = rate
            # 恢复相关嵌入和概率
            self.cor_embs = cor_embs
            self.cor_p = cor_p
    
            # 返回条件输出和无条件输出
            return c, uc
# 定义一个名为 FrozenT5Embedder 的类，继承自 AbstractEmbModel
class FrozenT5Embedder(AbstractEmbModel):
    """使用 T5 变换器编码器处理文本"""

    # 初始化方法，设置模型的基本参数
    def __init__(
        self,
        model_dir="google/t5-v1_1-xxl",  # 模型目录，默认是 T5 模型
        device="cuda",                   # 设备设置，默认使用 GPU
        max_length=77,                   # 输入文本的最大长度
        freeze=True,                     # 是否冻结模型参数，默认是冻结
        cache_dir=None,                  # 缓存目录，默认无
    ):
        super().__init__()               # 调用父类的初始化方法
        # 检查模型目录是否为默认 T5 模型
        if model_dir is not "google/t5-v1_1-xxl":
            # 从指定目录加载 tokenizer 和 transformer 模型
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir)
        else:
            # 从指定目录加载 tokenizer 和 transformer 模型，同时指定缓存目录
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir, cache_dir=cache_dir)
        # 设置设备
        self.device = device
        # 设置最大输入长度
        self.max_length = max_length
        # 如果需要冻结模型参数，调用 freeze 方法
        if freeze:
            self.freeze()

    # 定义冻结方法
    def freeze(self):
        # 将 transformer 设置为评估模式
        self.transformer = self.transformer.eval()

        # 遍历模型参数，设置为不需要梯度更新
        for param in self.parameters():
            param.requires_grad = False

    # @autocast 注解（注释掉的），用于自动混合精度计算
    def forward(self, text):
        # 使用 tokenizer 对输入文本进行编码，返回批处理编码结果
        batch_encoding = self.tokenizer(
            text,
            truncation=True,                # 超出最大长度时进行截断
            max_length=self.max_length,     # 设置最大长度
            return_length=True,             # 返回编码长度
            return_overflowing_tokens=False, # 不返回溢出的 tokens
            padding="max_length",           # 填充到最大长度
            return_tensors="pt",           # 返回 PyTorch 张量
        )
        # 将输入 id 转移到指定设备
        tokens = batch_encoding["input_ids"].to(self.device)
        # 在禁用自动混合精度的上下文中进行前向传播
        with torch.autocast("cuda", enabled=False):
            # 使用 transformer 进行前向传播，获取输出
            outputs = self.transformer(input_ids=tokens)
        # 获取 transformer 输出的最后隐藏状态
        z = outputs.last_hidden_state
        # 返回最后隐藏状态
        return z

    # 定义编码方法，直接调用 forward 方法
    def encode(self, text):
        return self(text)  # 将输入文本传递给 forward 方法进行编码
```