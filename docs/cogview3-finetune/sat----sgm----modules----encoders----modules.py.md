# `.\cogview3-finetune\sat\sgm\modules\encoders\modules.py`

```py
# 导入数学库，用于数学计算
import math
# 从上下文管理库导入 nullcontext，用于创建一个不执行任何操作的上下文管理器
from contextlib import nullcontext
# 从 functools 导入 partial，用于创建偏函数
from functools import partial
# 从 typing 导入各种类型注解，用于类型检查
from typing import Dict, List, Optional, Tuple, Union

# 导入 kornia 库，用于计算机视觉的操作
import kornia
# 导入 numpy 库，用于数组和数值计算
import numpy as np
# 导入 open_clip 库，用于处理 CLIP 模型
import open_clip
# 导入 PyTorch 库，深度学习框架
import torch
# 导入 PyTorch 的分布式模块
import torch.distributed
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 从 einops 导入 rearrange 和 repeat，用于重排和复制张量
from einops import rearrange, repeat
# 从 omegaconf 导入 ListConfig，用于处理配置文件
from omegaconf import ListConfig
# 从 torch.utils.checkpoint 导入 checkpoint，用于节省内存的检查点机制
from torch.utils.checkpoint import checkpoint
# 从 transformers 导入各种模型和分词器
from transformers import (
    ByT5Tokenizer,  # 导入 ByT5 的分词器
    CLIPTextModel,  # 导入 CLIP 的文本模型
    CLIPTokenizer,  # 导入 CLIP 的分词器
    T5EncoderModel,  # 导入 T5 的编码器模型
    T5Tokenizer,  # 导入 T5 的分词器
    AutoModel,  # 导入自动模型类，用于加载预训练模型
    AutoTokenizer  # 导入自动分词器类，用于加载预训练分词器
)

# 从模块中导入正则化器、编码器和时间步等工具
from ...modules.autoencoding.regularizers import DiagonalGaussianRegularizer
from ...modules.diffusionmodules.model import Encoder
from ...modules.diffusionmodules.openaimodel import Timestep
from ...modules.diffusionmodules.util import extract_into_tensor, make_beta_schedule
from ...modules.distributions.distributions import DiagonalGaussianDistribution
from ...util import (
    append_dims,  # 导入函数，用于向张量追加维度
    autocast,  # 导入函数，用于自动混合精度训练
    count_params,  # 导入函数，用于计算模型参数数量
    default,  # 导入函数，用于获取默认值
    disabled_train,  # 导入函数，用于禁用训练模式
    expand_dims_like,  # 导入函数，用于扩展张量维度以匹配另一个张量
    instantiate_from_config,  # 导入函数，从配置实例化对象
)


# 定义一个抽象的嵌入模型类，继承自 nn.Module
class AbstractEmbModel(nn.Module):
    # 初始化方法
    def __init__(self):
        super().__init__()  # 调用父类构造函数
        self._is_trainable = None  # 初始化可训练标志
        self._ucg_rate = None  # 初始化 UCG 率
        self._input_key = None  # 初始化输入键

    # 定义 is_trainable 属性的 getter 方法
    @property
    def is_trainable(self) -> bool:
        return self._is_trainable  # 返回可训练标志

    # 定义 ucg_rate 属性的 getter 方法
    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate  # 返回 UCG 率

    # 定义 input_key 属性的 getter 方法
    @property
    def input_key(self) -> str:
        return self._input_key  # 返回输入键

    # 定义 is_trainable 属性的 setter 方法
    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value  # 设置可训练标志

    # 定义 ucg_rate 属性的 setter 方法
    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value  # 设置 UCG 率

    # 定义 input_key 属性的 setter 方法
    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value  # 设置输入键

    # 定义 is_trainable 属性的 deleter 方法
    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable  # 删除可训练标志

    # 定义 ucg_rate 属性的 deleter 方法
    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate  # 删除 UCG 率

    # 定义 input_key 属性的 deleter 方法
    @input_key.deleter
    def input_key(self):
        del self._input_key  # 删除输入键


# 定义通用条件器类，继承自 nn.Module
class GeneralConditioner(nn.Module):
    # 输出维度到键的映射
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    # 键到拼接维度的映射
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}
    # 初始化函数，接收嵌入模型配置及其他参数
        def __init__(self, emb_models: Union[List, ListConfig], cor_embs=[], cor_p=[]):
            # 调用父类的初始化方法
            super().__init__()
            # 存储嵌入模型的列表
            embedders = []
            # 遍历嵌入模型配置
            for n, embconfig in enumerate(emb_models):
                # 从配置中实例化嵌入模型
                embedder = instantiate_from_config(embconfig)
                # 确保嵌入模型继承自 AbstractEmbModel
                assert isinstance(
                    embedder, AbstractEmbModel
                ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
                # 获取是否可训练的标志，默认为 False
                embedder.is_trainable = embconfig.get("is_trainable", False)
                # 获取 UCG 比率，默认为 0.0
                embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
                # 如果不可训练，禁用训练
                if not embedder.is_trainable:
                    embedder.train = disabled_train
                    # 将模型参数的梯度要求设为 False
                    for param in embedder.parameters():
                        param.requires_grad = False
                    # 将模型设置为评估模式
                    embedder.eval()
                # print(
                #     f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                #     f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
                # )
    
                # 检查是否有单一输入键
                if "input_key" in embconfig:
                    embedder.input_key = embconfig["input_key"]
                # 检查是否有多个输入键
                elif "input_keys" in embconfig:
                    embedder.input_keys = embconfig["input_keys"]
                # 如果没有输入键，则引发异常
                else:
                    raise KeyError(
                        f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}"
                    )
    
                # 获取遗留 UCG 值
                embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
                # 如果遗留 UCG 值存在，初始化随机状态
                if embedder.legacy_ucg_val is not None:
                    embedder.ucg_prng = np.random.RandomState()
    
                # 将嵌入模型添加到列表中
                embedders.append(embedder)
            # 将嵌入模型列表存储为模块列表
            self.embedders = nn.ModuleList(embedders)
    
            # 如果存在条件嵌入，确保条件概率长度匹配
            if len(cor_embs) > 0:
                assert len(cor_p) == 2**len(cor_embs)
            # 存储条件嵌入和概率
            self.cor_embs = cor_embs
            self.cor_p = cor_p
    
        # 根据嵌入模型和批量数据获取 UCG 值（可能）
        def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
            # 确保遗留 UCG 值存在
            assert embedder.legacy_ucg_val is not None
            # 获取 UCG 比率
            p = embedder.ucg_rate
            # 获取遗留 UCG 值
            val = embedder.legacy_ucg_val
            # 遍历批量数据的输入键
            for i in range(len(batch[embedder.input_key])):
                # 根据概率选择是否替换值
                if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[embedder.input_key][i] = val
            # 返回更新后的批量数据
            return batch
        
        # 根据嵌入模型和条件获取 UCG 值（必定）
        def surely_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict, cond_or_not) -> Dict:
            # 确保遗留 UCG 值存在
            assert embedder.legacy_ucg_val is not None
            # 获取遗留 UCG 值
            val = embedder.legacy_ucg_val
            # 遍历批量数据的输入键
            for i in range(len(batch[embedder.input_key])):
                # 如果条件满足，替换值
                if cond_or_not[i]:
                    batch[embedder.input_key][i] = val
            # 返回更新后的批量数据
            return batch
    # 定义获取单个嵌入的方法
    def get_single_embedding(self, embedder, batch, output, cond_or_not: Optional[np.ndarray] = None, force_zero_embeddings: Optional[List] = None):
        # 根据嵌入器是否可训练选择上下文管理器
        embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
        # 使用选定的上下文管理器
        with embedding_context():
            # 检查嵌入器是否有输入键，并且输入键不为 None
            if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                # 如果嵌入器的 legacy_ucg_val 不为 None
                if embedder.legacy_ucg_val is not None:
                    # 如果条件不为 None
                    if cond_or_not is None:
                        # 可能获取 ucg_val 的值
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    else:
                        # 确定获取 ucg_val 的值
                        batch = self.surely_get_ucg_val(embedder, batch, cond_or_not)
                # 从批次中获取嵌入输出
                emb_out = embedder(batch[embedder.input_key])
            # 如果嵌入器有多个输入键
            elif hasattr(embedder, "input_keys"):
                # 从批次中解包并获取嵌入输出
                emb_out = embedder(*[batch[k] for k in embedder.input_keys])
        # 确保嵌入输出是张量、列表或元组
        assert isinstance(
            emb_out, (torch.Tensor, list, tuple)
        ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
        # 如果嵌入输出不是列表或元组，将其转为列表
        if not isinstance(emb_out, (list, tuple)):
            emb_out = [emb_out]    
        # 遍历嵌入输出
        for emb in emb_out:
            # 获取输出键
            out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
            # 如果嵌入器的 ucg_rate 大于 0 且 legacy_ucg_val 为 None
            if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                # 如果条件不为 None
                if cond_or_not is None:
                    # 扩展嵌入维度并应用伯努利分布
                    emb = (
                        expand_dims_like(
                            torch.bernoulli(
                                (1.0 - embedder.ucg_rate)
                                * torch.ones(emb.shape[0], device=emb.device)
                            ),
                            emb,
                        )
                        * emb
                    )
                else:
                    # 根据条件扩展嵌入维度
                    emb = (
                        expand_dims_like(
                            torch.tensor(1-cond_or_not, dtype=emb.dtype, device=emb.device),
                            emb,
                        )
                        * emb
                    )
            # 如果嵌入器有输入键且在强制零嵌入列表中
            if (
                hasattr(embedder, "input_key")
                and embedder.input_key in force_zero_embeddings
            ):
                # 将嵌入设置为全零
                emb = torch.zeros_like(emb)
            # 如果输出中已有该键
            if out_key in output:
                # 将新的嵌入与已有输出拼接
                output[out_key] = torch.cat(
                    (output[out_key], emb), self.KEY2CATDIM[out_key]
                )
            else:
                # 否则，直接赋值新的嵌入
                output[out_key] = emb
        # 返回更新后的输出
        return output
    
    # 定义前向传播的方法
    def forward(
        self, batch: Dict, force_zero_embeddings: Optional[List] = None
    ) -> Dict:  # 定义函数的返回类型为字典
        output = dict()  # 初始化一个空字典用于存储输出结果
        if force_zero_embeddings is None:  # 检查是否提供强制零嵌入参数
            force_zero_embeddings = []  # 如果没有，初始化为空列表

        if len(self.cor_embs) > 0:  # 如果相关嵌入存在
            batch_size = len(batch[list(batch.keys())[0]])  # 获取批次中第一个键的大小
            rand_idx = np.random.choice(len(self.cor_p), size=(batch_size,), p=self.cor_p)  # 根据相关概率随机选择索引
            for emb_idx in self.cor_embs:  # 遍历相关嵌入索引
                cond_or_not = rand_idx % 2  # 计算条件标志（0或1）
                rand_idx //= 2  # 更新随机索引
                embedder = self.embedders[emb_idx]  # 获取对应的嵌入器
                output = self.get_single_embedding(self.embedders[emb_idx], batch, output=output, cond_or_not=cond_or_not, force_zero_embeddings=force_zero_embeddings)  # 获取单个嵌入并更新输出

        for i, embedder in enumerate(self.embedders):  # 遍历所有嵌入器及其索引
            if i in self.cor_embs:  # 如果索引在相关嵌入中，则跳过
                continue  # 继续下一个循环
            output = self.get_single_embedding(embedder, batch, output=output, force_zero_embeddings=force_zero_embeddings)  # 获取单个嵌入并更新输出
        return output  # 返回最终的输出字典

    def get_unconditional_conditioning(  # 定义获取无条件调节的函数
        self, batch_c, batch_uc=None, force_uc_zero_embeddings=None  # 输入批次及可选参数
    ):
        if force_uc_zero_embeddings is None:  # 检查强制无条件零嵌入参数
            force_uc_zero_embeddings = []  # 如果没有，初始化为空列表
        ucg_rates = list()  # 初始化列表用于存储原有的无条件生成率
        for embedder in self.embedders:  # 遍历所有嵌入器
            ucg_rates.append(embedder.ucg_rate)  # 保存当前的无条件生成率
            embedder.ucg_rate = 0.0  # 将无条件生成率设置为0

        cor_embs = self.cor_embs  # 保存当前相关嵌入
        cor_p = self.cor_p  # 保存当前相关概率
        self.cor_embs = []  # 清空相关嵌入
        self.cor_p = []  # 清空相关概率

        c = self(batch_c)  # 计算输入批次的输出
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)  # 计算无条件输出

        for embedder, rate in zip(self.embedders, ucg_rates):  # 恢复每个嵌入器的无条件生成率
            embedder.ucg_rate = rate  # 将原有的无条件生成率重新赋值
        self.cor_embs = cor_embs  # 恢复相关嵌入
        self.cor_p = cor_p  # 恢复相关概率

        return c, uc  # 返回有条件和无条件的输出
# 定义一个名为 InceptionV3 的类，继承自 nn.Module
class InceptionV3(nn.Module):
    """对 https://github.com/mseitzer/pytorch-fid 的 Inception 
    端口进行包装，并在末尾增加一个 squeeze 操作"""

    # 初始化函数，接受 normalize_input 参数和其他可选参数
    def __init__(self, normalize_input=False, **kwargs):
        # 调用父类的初始化函数
        super().__init__()
        # 从 pytorch_fid 导入 inception 模块
        from pytorch_fid import inception

        # 设置输入调整标志为 True
        kwargs["resize_input"] = True
        # 创建 InceptionV3 模型实例，并传入参数
        self.model = inception.InceptionV3(normalize_input=normalize_input, **kwargs)

    # 前向传播函数，接受输入张量 inp
    def forward(self, inp):
        # 对输入进行尺寸调整（已注释）
        # inp = kornia.geometry.resize(inp, (299, 299),
        #                              interpolation='bicubic',
        #                              align_corners=False,
        #                              antialias=True)
        # 将输入值限制在 -1 到 1 之间（已注释）
        # inp = inp.clamp(min=-1, max=1)

        # 使用模型对输入进行处理，获得输出
        outp = self.model(inp)

        # 如果输出只有一个元素，去掉维度并返回
        if len(outp) == 1:
            return outp[0].squeeze()

        # 返回原始输出
        return outp


# 定义一个名为 IdentityEncoder 的类，继承自 AbstractEmbModel
class IdentityEncoder(AbstractEmbModel):
    # 编码函数，直接返回输入
    def encode(self, x):
        return x

    # 前向传播函数，直接返回输入
    def forward(self, x):
        return x


# 定义一个名为 ClassEmbedder 的类，继承自 AbstractEmbModel
class ClassEmbedder(AbstractEmbModel):
    # 初始化函数，接受嵌入维度、类数和是否添加序列维度的参数
    def __init__(self, embed_dim, n_classes=1000, add_sequence_dim=False):
        # 调用父类的初始化函数
        super().__init__()
        # 创建嵌入层，映射类到嵌入维度
        self.embedding = nn.Embedding(n_classes, embed_dim)
        # 保存类数和是否添加序列维度的标志
        self.n_classes = n_classes
        self.add_sequence_dim = add_sequence_dim

    # 前向传播函数，接受类的输入
    def forward(self, c):
        # 获取类的嵌入表示
        c = self.embedding(c)
        # 如果需要，添加序列维度
        if self.add_sequence_dim:
            c = c[:, None, :]
        # 返回嵌入表示
        return c

    # 获取无条件的条件信息
    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = (
            self.n_classes - 1
        )  # 1000 类 --> 0 ... 999，额外的类用于无条件生成
        # 创建一个全为 uc_class 的张量
        uc = torch.ones((bs,), device=device) * uc_class
        # 将类信息包装成字典
        uc = {self.key: uc.long()}
        # 返回字典
        return uc


# 定义一个名为 ClassEmbedderForMultiCond 的类，继承自 ClassEmbedder
class ClassEmbedderForMultiCond(ClassEmbedder):
    # 前向传播函数，接受批量数据、键和是否禁用丢弃的标志
    def forward(self, batch, key=None, disable_dropout=False):
        # 将输出初始化为输入批次
        out = batch
        # 如果未提供键，使用默认键
        key = default(key, self.key)
        # 检查批次中的值是否为列表
        islist = isinstance(batch[key], list)
        # 如果是列表，则取第一个元素
        if islist:
            batch[key] = batch[key][0]
        # 调用父类的前向传播
        c_out = super().forward(batch, key, disable_dropout)
        # 根据是否为列表，更新输出
        out[key] = [c_out] if islist else c_out
        # 返回更新后的输出
        return out


# 定义一个名为 FrozenT5Embedder 的类，继承自 AbstractEmbModel
class FrozenT5Embedder(AbstractEmbModel):
    """使用 T5 转换器编码器进行文本处理"""

    # 初始化函数，接受模型目录、设备、最大长度、是否冻结和缓存目录的参数
    def __init__(
        self,
        model_dir="google/t5-v1_1-xxl",
        device="cuda",
        max_length=77,
        freeze=True,
        cache_dir=None,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 如果模型目录不是默认的，加载相应的分词器和模型
        if model_dir is not "google/t5-v1_1-xxl":
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir)
        else:
            # 否则，使用缓存目录加载分词器和模型
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir, cache_dir=cache_dir)
        # 保存设备信息
        self.device = device
        # 保存最大长度
        self.max_length = max_length
        # 如果需要冻结，调用冻结函数
        if freeze:
            self.freeze()
    # 定义冻结模型参数的方法
        def freeze(self):
            # 将转换器设置为评估模式，以禁用训练时的行为（如 Dropout）
            self.transformer = self.transformer.eval()
    
            # 遍历所有模型参数
            for param in self.parameters():
                # 禁用参数的梯度计算，以减少内存使用和提高推理速度
                param.requires_grad = False
    
        # @autocast  # 可选装饰器，用于自动混合精度
        def forward(self, text):
            # 使用分词器对输入文本进行编码，返回编码后的批次信息
            batch_encoding = self.tokenizer(
                text,
                # 截断超出最大长度的文本
                truncation=True,
                # 设置最大长度
                max_length=self.max_length,
                # 返回编码后的文本长度
                return_length=True,
                # 不返回溢出的令牌
                return_overflowing_tokens=False,
                # 填充到最大长度
                padding="max_length",
                # 返回 PyTorch 张量
                return_tensors="pt",
            )
            # 将输入ID移动到指定的设备（CPU或GPU）
            tokens = batch_encoding["input_ids"].to(self.device)
            # 使用上下文管理器禁用混合精度计算
            with torch.autocast("cuda", enabled=False):
                # 将令牌输入到转换器中，获取输出
                outputs = self.transformer(input_ids=tokens)
            # 获取最后一个隐藏状态，作为编码的表示
            z = outputs.last_hidden_state
            # 返回编码结果
            return z
    
        # 定义编码文本的方法
        def encode(self, text):
            # 调用当前对象的 forward 方法进行编码
            return self(text)
# 定义一个名为 FrozenByT5Embedder 的类，继承自 AbstractEmbModel
class FrozenByT5Embedder(AbstractEmbModel):
    """
    使用 ByT5 转换器编码器处理文本，具备字符意识。
    """

    # 初始化方法，设置模型的版本、设备、最大长度和是否冻结参数
    def __init__(
        self, version="google/byt5-base", device="cuda", max_length=77, freeze=True
    ):  # 其他可用版本为 google/t5-v1_1-xl 和 google/t5-v1_1-xxl
        # 调用父类构造函数
        super().__init__()
        # 加载预训练的 ByT5 分词器
        self.tokenizer = ByT5Tokenizer.from_pretrained(version)
        # 加载预训练的 T5 编码器模型
        self.transformer = T5EncoderModel.from_pretrained(version)
        # 设置设备类型（如 CUDA）
        self.device = device
        # 设置输入文本的最大长度
        self.max_length = max_length
        # 如果需要冻结参数，则调用冻结方法
        if freeze:
            self.freeze()

    # 冻结模型的参数，以避免训练时更新
    def freeze(self):
        # 将变换器设置为评估模式
        self.transformer = self.transformer.eval()
        # 遍历所有参数并设置为不可更新
        for param in self.parameters():
            param.requires_grad = False

    # 定义前向传播方法
    def forward(self, text):
        # 对输入文本进行编码，返回批次编码
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        # 将输入的 token 移动到指定设备
        tokens = batch_encoding["input_ids"].to(self.device)
        # 在不启用自动混合精度的情况下进行前向传播
        with torch.autocast("cuda", enabled=False):
            # 获取模型的输出
            outputs = self.transformer(input_ids=tokens)
        # 取出最后一层的隐藏状态
        z = outputs.last_hidden_state
        # 返回最后的隐藏状态
        return z

    # 定义编码方法，直接调用前向传播
    def encode(self, text):
        return self(text)


# 定义一个名为 FrozenCLIPEmbedder 的类，继承自 AbstractEmbModel
class FrozenCLIPEmbedder(AbstractEmbModel):
    """使用 CLIP 转换器编码器处理文本（来自 huggingface）"""

    # 定义可用的层类型
    LAYERS = ["last", "pooled", "hidden"]

    # 初始化方法，设置模型的版本、设备、最大长度、冻结状态和层类型
    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
        always_return_pooled=False,
    ):  # clip-vit-base-patch32
        # 调用父类构造函数
        super().__init__()
        # 确保层类型在可用层中
        assert layer in self.LAYERS
        # 加载预训练的 CLIP 分词器
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        # 加载预训练的 CLIP 文本模型
        self.transformer = CLIPTextModel.from_pretrained(version)
        # 设置设备类型（如 CUDA）
        self.device = device
        # 设置输入文本的最大长度
        self.max_length = max_length
        # 如果需要冻结参数，则调用冻结方法
        if freeze:
            self.freeze()
        # 设置所用层的类型
        self.layer = layer
        # 设置所用层的索引
        self.layer_idx = layer_idx
        # 设置是否总是返回池化结果
        self.return_pooled = always_return_pooled
        # 如果层为隐藏层，确保层索引有效
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    # 冻结模型的参数，以避免训练时更新
    def freeze(self):
        # 将变换器设置为评估模式
        self.transformer = self.transformer.eval()
        # 遍历所有参数并设置为不可更新
        for param in self.parameters():
            param.requires_grad = False

    # 这里缺少方法体，可能是注释或未完成代码
    @autocast
    # 定义前向传播函数，接收文本输入
    def forward(self, text):
        # 对输入文本进行编码，生成批量编码，设置各种参数以控制编码行为
        batch_encoding = self.tokenizer(
            text,
            truncation=True,  # 超出最大长度时截断文本
            max_length=self.max_length,  # 最大长度限制
            return_length=True,  # 返回编码后每个文本的长度
            return_overflowing_tokens=False,  # 不返回溢出的标记
            padding="max_length",  # 填充到最大长度
            return_tensors="pt",  # 返回 PyTorch 张量格式
        )
        # 获取编码后的输入标记，并将其移动到指定设备上
        tokens = batch_encoding["input_ids"].to(self.device)
        # 使用 transformer 模型进行前向传播，获取输出
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=self.layer == "hidden"  # 根据条件决定是否返回隐藏状态
        )
        # 根据层级选择相应的输出
        if self.layer == "last":
            # 选择最后一层的隐藏状态
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            # 选择池化后的输出，并增加一个维度
            z = outputs.pooler_output[:, None, :]
        else:
            # 选择指定索引的隐藏状态
            z = outputs.hidden_states[self.layer_idx]
        # 根据是否需要池化输出，返回相应结果
        if self.return_pooled:
            return z, outputs.pooler_output  # 返回输出和池化结果
        return z  # 返回仅隐藏状态

    # 定义编码函数，简化调用前向传播
    def encode(self, text):
        # 调用前向传播函数并返回结果
        return self(text)
# 定义一个名为 FrozenOpenCLIPEmbedder2 的类，继承自 AbstractEmbModel
class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    使用 OpenCLIP 变换器编码器进行文本处理
    """

    # 定义可用的层名称
    LAYERS = ["pooled", "last", "penultimate"]

    # 初始化方法，设置模型的基本参数
    def __init__(
        self,
        arch="ViT-H-14",  # 模型架构
        version="laion2b_s32b_b79k",  # 版本信息
        device="cuda",  # 设备类型
        max_length=77,  # 最大输入长度
        freeze=True,  # 是否冻结模型参数
        layer="last",  # 选择的层
        always_return_pooled=False,  # 是否始终返回池化结果
        legacy=True,  # 是否使用遗留模式
    ):
        super().__init__()  # 调用父类构造函数
        assert layer in self.LAYERS  # 确保指定的层有效
        # 创建模型和转换器，并将其移动到 CPU
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
        )
        del model.visual  # 删除视觉部分
        self.model = model  # 保存模型

        self.device = device  # 设置设备
        self.max_length = max_length  # 设置最大长度
        self.return_pooled = always_return_pooled  # 设置是否返回池化
        if freeze:  # 如果需要冻结模型
            self.freeze()  # 调用冻结方法
        self.layer = layer  # 设置层
        # 根据选择的层更新层索引
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()  # 不支持的层选择
        self.legacy = legacy  # 设置遗留模式

    # 冻结模型参数的方法
    def freeze(self):
        self.model = self.model.eval()  # 设置模型为评估模式
        for param in self.parameters():  # 遍历所有参数
            param.requires_grad = False  # 禁止梯度更新

    # 前向传播的方法，处理输入文本
    @autocast
    def forward(self, text):
        tokens = open_clip.tokenize(text)  # 将文本转换为标记
        z = self.encode_with_transformer(tokens.to(self.device))  # 编码处理
        # 根据条件返回不同结果
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            assert not self.legacy  # 确保不在遗留模式下
            return z[self.layer], z["pooled"]  # 返回选定层和池化结果
        return z[self.layer]  # 返回选定层结果

    # 使用变换器进行编码的方法
    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # 获取标记嵌入
        x = x + self.model.positional_embedding  # 加入位置嵌入
        x = x.permute(1, 0, 2)  # 转换维度顺序
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)  # 通过变换器前向传播
        if self.legacy:  # 如果在遗留模式
            x = x[self.layer]  # 获取选定层结果
            x = self.model.ln_final(x)  # 最终归一化
            return x  # 返回结果
        else:
            # x 为字典，将保持为字典
            o = x["last"]  # 获取最后一层输出
            o = self.model.ln_final(o)  # 最终归一化
            pooled = self.pool(o, text)  # 进行池化处理
            x["pooled"] = pooled  # 将池化结果存入字典
            return x  # 返回字典

    # 池化处理的方法
    def pool(self, x, text):
        # 从 eot 嵌入中获取特征（eot_token 为每个序列中的最大值）
        x = (
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)]  # 获取 eot 嵌入
            @ self.model.text_projection  # 应用文本投影
        )
        return x  # 返回池化结果
    # 定义文本转换器的前向传播方法，接受输入张量 x 和可选的注意力掩码
        def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
            # 创建一个空字典以存储输出
            outputs = {}
            # 遍历转换器的每个残差块
            for i, r in enumerate(self.model.transformer.resblocks):
                # 如果是最后一个残差块，将输入张量进行维度变换并存储
                if i == len(self.model.transformer.resblocks) - 1:
                    outputs["penultimate"] = x.permute(1, 0, 2)  # 将维度从 LND 转换为 NLD
                # 如果启用了梯度检查点并且不是脚本模式
                if (
                    self.model.transformer.grad_checkpointing
                    and not torch.jit.is_scripting()
                ):
                    # 使用检查点技术进行前向传播以节省内存
                    x = checkpoint(r, x, attn_mask)
                else:
                    # 正常执行残差块的前向传播
                    x = r(x, attn_mask=attn_mask)
            # 将最后输出张量的维度进行转换并存储
            outputs["last"] = x.permute(1, 0, 2)  # 将维度从 LND 转换为 NLD
            # 返回包含倒数第二层和最后一层输出的字典
            return outputs
    
        # 定义编码方法，接受文本输入
        def encode(self, text):
            # 调用文本转换器的前向传播方法并返回结果
            return self(text)
# 定义一个名为 FrozenOpenCLIPEmbedder 的类，继承自 AbstractEmbModel
class FrozenOpenCLIPEmbedder(AbstractEmbModel):
    # 定义一个类属性 LAYERS，包含模型中可用的层
    LAYERS = [
        # "pooled",  # 注释掉的层选项
        "last",  # 最后一个层
        "penultimate",  # 倒数第二个层
    ]

    # 初始化方法，用于设置实例的基本参数
    def __init__(
        self,
        arch="ViT-H-14",  # 模型架构
        version="laion2b_s32b_b79k",  # 预训练模型版本
        device="cuda",  # 设备类型，默认为 GPU
        max_length=77,  # 最大输入文本长度
        freeze=True,  # 是否冻结模型参数
        layer="last",  # 选择使用的层
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 确保选择的层在可用层中
        assert layer in self.LAYERS
        # 创建模型及其转换，使用指定的架构和设备
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device("cpu"), pretrained=version
        )
        # 删除视觉部分以冻结模型
        del model.visual
        # 将模型赋值给实例变量
        self.model = model

        # 设置设备属性
        self.device = device
        # 设置最大长度属性
        self.max_length = max_length
        # 如果需要冻结，则调用冻结方法
        if freeze:
            self.freeze()
        # 设置所选层
        self.layer = layer
        # 根据所选层设置层索引
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            # 如果层不在可用选项中，则抛出异常
            raise NotImplementedError()

    # 冻结模型参数的方法
    def freeze(self):
        # 将模型设置为评估模式
        self.model = self.model.eval()
        # 将所有参数的 requires_grad 属性设置为 False，停止梯度计算
        for param in self.parameters():
            param.requires_grad = False

    # 前向传播方法，接受文本输入
    def forward(self, text):
        # 对文本进行分词处理
        tokens = open_clip.tokenize(text)
        # 使用变换器进行编码，并将结果传送到设备
        z = self.encode_with_transformer(tokens.to(self.device))
        # 返回编码结果
        return z

    # 使用变换器编码文本的方法
    def encode_with_transformer(self, text):
        # 获取文本的嵌入表示，形状为 [batch_size, n_ctx, d_model]
        x = self.model.token_embedding(text)  
        # 加上位置嵌入
        x = x + self.model.positional_embedding
        # 重新排列维度，将形状从 NLD 转换为 LND
        x = x.permute(1, 0, 2)  
        # 执行变换器前向传播
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        # 重新排列维度回到 NLD
        x = x.permute(1, 0, 2)  
        # 通过最终层归一化
        x = self.model.ln_final(x)
        # 返回处理后的结果
        return x

    # 执行文本变换器前向传播的方法
    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        # 遍历变换器的残差块
        for i, r in enumerate(self.model.transformer.resblocks):
            # 如果达到所需层索引则停止
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            # 检查是否使用梯度检查点
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                # 使用检查点方式更新输入
                x = checkpoint(r, x, attn_mask)
            else:
                # 否则直接通过残差块处理输入
                x = r(x, attn_mask=attn_mask)
        # 返回变换后的结果
        return x

    # 编码文本的简化方法，直接调用前向传播
    def encode(self, text):
        return self(text)


# 定义一个名为 FrozenOpenCLIPImageEmbedder 的类，继承自 AbstractEmbModel
class FrozenOpenCLIPImageEmbedder(AbstractEmbModel):
    """
    使用 OpenCLIP 视觉变换器编码器处理图像
    """

    # 初始化方法，用于设置实例的基本参数
    def __init__(
        self,
        arch="ViT-H-14",  # 模型架构
        version="laion2b_s32b_b79k",  # 预训练模型版本
        device="cuda",  # 设备类型，默认为 GPU
        max_length=77,  # 最大输入图像长度
        freeze=True,  # 是否冻结模型参数
        antialias=True,  # 是否使用抗锯齿
        ucg_rate=0.0,  # 用户定义的裁剪率
        unsqueeze_dim=False,  # 是否增加维度
        repeat_to_max_len=False,  # 是否重复到最大长度
        num_image_crops=0,  # 图像裁剪数量
        output_tokens=False,  # 是否输出 tokens
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 创建模型和转换器，使用指定的架构、设备和预训练版本
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),  # 使用 CPU 作为计算设备
            pretrained=version,  # 指定预训练版本
        )
        # 删除模型中的变换器部分
        del model.transformer
        # 将创建的模型赋值给实例变量
        self.model = model
        # 设置最大图像裁剪数量
        self.max_crops = num_image_crops
        # 检查是否需要填充到最大长度
        self.pad_to_max_len = self.max_crops > 0
        # 检查是否需要重复到最大长度
        self.repeat_to_max_len = repeat_to_max_len and (not self.pad_to_max_len)
        # 设置设备类型
        self.device = device
        # 设置最大长度
        self.max_length = max_length
        # 如果需要冻结模型参数，则调用冻结方法
        if freeze:
            self.freeze()

        # 设置抗锯齿参数
        self.antialias = antialias

        # 注册均值张量作为缓冲区，设置为非持久性
        self.register_buffer(
            "mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        # 注册标准差张量作为缓冲区，设置为非持久性
        self.register_buffer(
            "std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )
        # 设置 UCG 速率
        self.ucg_rate = ucg_rate
        # 设置需要扩展的维度
        self.unsqueeze_dim = unsqueeze_dim
        # 存储的批次初始化为 None
        self.stored_batch = None
        # 设置视觉模型的输出标记
        self.model.visual.output_tokens = output_tokens
        # 保存输出标记的状态
        self.output_tokens = output_tokens

    def preprocess(self, x):
        # 将输入归一化到 [0,1] 范围
        x = kornia.geometry.resize(
            x,
            (224, 224),  # 将图像大小调整为 224x224
            interpolation="bicubic",  # 使用双三次插值
            align_corners=True,  # 对齐角点
            antialias=self.antialias,  # 使用抗锯齿
        )
        # 将图像数据从 [-1,1] 范围转换到 [0,1]
        x = (x + 1.0) / 2.0
        # 根据 CLIP 模型的均值和标准差重新归一化图像
        x = kornia.enhance.normalize(x, self.mean, self.std)
        # 返回处理后的图像
        return x

    def freeze(self):
        # 将模型设置为评估模式
        self.model = self.model.eval()
        # 禁用所有参数的梯度计算
        for param in self.parameters():
            param.requires_grad = False

    @autocast  # 启用自动混合精度计算
    # 前向传播方法，处理输入图像并返回特征或标记
        def forward(self, image, no_dropout=False):
            # 使用视觉变换器对输入图像进行编码，得到特征 z
            z = self.encode_with_vision_transformer(image)
            # 初始化 tokens 为 None
            tokens = None
            # 如果输出标记为真，分离特征和标记
            if self.output_tokens:
                z, tokens = z[0], z[1]
            # 将特征 z 转换为与图像相同的数据类型
            z = z.to(image.dtype)
            # 如果 ucg_rate 大于 0，且没有进行 dropout，且没有最大裁剪
            if self.ucg_rate > 0.0 and not no_dropout and not (self.max_crops > 0):
                # 根据 Bernoulli 分布随机丢弃特征
                z = (
                    torch.bernoulli(
                        (1.0 - self.ucg_rate) * torch.ones(z.shape[0], device=z.device)
                    )[:, None]
                    * z
                )
                # 如果 tokens 不为 None，应用相同的丢弃逻辑
                if tokens is not None:
                    tokens = (
                        expand_dims_like(
                            torch.bernoulli(
                                (1.0 - self.ucg_rate)
                                * torch.ones(tokens.shape[0], device=tokens.device)
                            ),
                            tokens,
                        )
                        * tokens
                    )
            # 如果需要扩展维度，将特征 z 变为三维
            if self.unsqueeze_dim:
                z = z[:, None, :]
            # 如果输出标记为真，检查标记和特征的重复与填充条件
            if self.output_tokens:
                assert not self.repeat_to_max_len
                assert not self.pad_to_max_len
                # 返回标记和特征
                return tokens, z
            # 如果需要重复到最大长度
            if self.repeat_to_max_len:
                # 将二维特征扩展为三维
                if z.dim() == 2:
                    z_ = z[:, None, :]
                else:
                    z_ = z
                # 返回重复的特征
                return repeat(z_, "b 1 d -> b n d", n=self.max_length), z
            # 如果需要填充到最大长度
            elif self.pad_to_max_len:
                # 确保特征是三维的
                assert z.dim() == 3
                # 在特征后面填充零
                z_pad = torch.cat(
                    (
                        z,
                        torch.zeros(
                            z.shape[0],
                            self.max_length - z.shape[1],
                            z.shape[2],
                            device=z.device,
                        ),
                    ),
                    1,
                )
                # 返回填充后的特征和第一个时间步的特征
                return z_pad, z_pad[:, 0, ...]
            # 默认返回特征 z
            return z
    # 使用视觉变换器对图像进行编码
    def encode_with_vision_transformer(self, img):
        # 如果最大裁剪数大于0，则对图像进行裁剪预处理
        # if self.max_crops > 0:
        #    img = self.preprocess_by_cropping(img)
        # 检查图像维度是否为5
        if img.dim() == 5:
            # 确保最大裁剪数与图像的第二维度匹配
            assert self.max_crops == img.shape[1]
            # 重排图像维度，将其从 (b n) c h w 变为 (b n) c h w
            img = rearrange(img, "b n c h w -> (b n) c h w")
        # 对图像进行预处理
        img = self.preprocess(img)
        # 如果不需要输出tokens
        if not self.output_tokens:
            # 确保模型不输出tokens
            assert not self.model.visual.output_tokens
            # 将图像传入模型进行处理
            x = self.model.visual(img)
            tokens = None
        else:
            # 确保模型输出tokens
            assert self.model.visual.output_tokens
            # 将图像传入模型并获取输出和tokens
            x, tokens = self.model.visual(img)
        # 如果最大裁剪数大于0
        if self.max_crops > 0:
            # 重排输出，将其从 (b n) d 变为 b n d
            x = rearrange(x, "(b n) d -> b n d", n=self.max_crops)
            # 在序列轴上进行drop out，控制一定比例的输出
            x = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate)
                    * torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
                )
                * x
            )
            # 如果tokens不为None
            if tokens is not None:
                # 重排tokens，将其从 (b n) t d 变为 b t (n d)
                tokens = rearrange(tokens, "(b n) t d -> b t (n d)", n=self.max_crops)
                # 输出实验性提示信息
                print(
                    f"You are running very experimental token-concat in {self.__class__.__name__}. "
                    f"Check what you are doing, and then remove this message."
                )
        # 如果需要输出tokens，则返回
        if self.output_tokens:
            return x, tokens
        # 返回处理后的图像
        return x
    
    # 对输入文本进行编码
    def encode(self, text):
        # 调用自身对文本进行处理
        return self(text)
# 定义一个继承自 AbstractEmbModel 的类，名为 FrozenCLIPT5Encoder
class FrozenCLIPT5Encoder(AbstractEmbModel):
    # 构造函数，初始化模型的参数
    def __init__(
        self,
        clip_version="openai/clip-vit-large-patch14",  # CLIP 模型的版本
        t5_version="google/t5-v1_1-xl",  # T5 模型的版本
        device="cuda",  # 指定使用的设备
        clip_max_length=77,  # CLIP 模型的最大输入长度
        t5_max_length=77,  # T5 模型的最大输入长度
    ):
        super().__init__()  # 调用父类的构造函数
        # 创建 CLIP 嵌入模型实例
        self.clip_encoder = FrozenCLIPEmbedder(
            clip_version, device, max_length=clip_max_length
        )
        # 创建 T5 嵌入模型实例
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        # 打印 CLIP 和 T5 模型的参数数量
        print(
            f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder) * 1.e-6:.2f} M parameters, "
            f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder) * 1.e-6:.2f} M params."
        )

    # 定义编码函数，调用前向传播
    def encode(self, text):
        return self(text)

    # 定义前向传播函数
    def forward(self, text):
        # 使用 CLIP 编码器对文本进行编码
        clip_z = self.clip_encoder.encode(text)
        # 使用 T5 编码器对文本进行编码
        t5_z = self.t5_encoder.encode(text)
        # 返回 CLIP 和 T5 的编码结果
        return [clip_z, t5_z]


# 定义一个继承自 nn.Module 的类，名为 SpatialRescaler
class SpatialRescaler(nn.Module):
    # 构造函数，初始化空间重缩放器的参数
    def __init__(
        self,
        n_stages=1,  # 重缩放的阶段数
        method="bilinear",  # 插值方法
        multiplier=0.5,  # 缩放因子
        in_channels=3,  # 输入通道数
        out_channels=None,  # 输出通道数
        bias=False,  # 是否使用偏置
        wrap_video=False,  # 是否处理视频数据
        kernel_size=1,  # 卷积核大小
        remap_output=False,  # 是否重映射输出通道
    ):
        super().__init__()  # 调用父类的构造函数
        self.n_stages = n_stages  # 保存阶段数
        assert self.n_stages >= 0  # 确保阶段数非负
        # 验证插值方法是否在支持的范围内
        assert method in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "bicubic",
            "area",
        ]
        self.multiplier = multiplier  # 保存缩放因子
        # 创建部分应用的插值函数
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        # 判断是否需要重映射输出通道
        self.remap_output = out_channels is not None or remap_output
        # 如果需要重映射输出通道，创建卷积层
        if self.remap_output:
            print(
                f"Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing."
            )
            self.channel_mapper = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=kernel_size // 2,
            )
        self.wrap_video = wrap_video  # 保存是否处理视频的标志

    # 定义前向传播函数
    def forward(self, x):
        # 如果处理视频数据且输入是五维张量，进行维度调整
        if self.wrap_video and x.ndim == 5:
            B, C, T, H, W = x.shape  # 解包维度
            x = rearrange(x, "b c t h w -> b t c h w")  # 调整维度顺序
            x = rearrange(x, "b t c h w -> (b t) c h w")  # 合并批次和时间维度

        # 进行指定阶段的重缩放操作
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        # 如果处理视频数据，恢复维度顺序
        if self.wrap_video:
            x = rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T, c=C)  # 恢复维度
            x = rearrange(x, "b t c h w -> b c t h w")  # 再次调整维度
        # 如果需要重映射输出，应用卷积层
        if self.remap_output:
            x = self.channel_mapper(x)
        return x  # 返回处理后的张量

    # 定义编码函数，调用前向传播
    def encode(self, x):
        return self(x)


# 定义一个继承自 nn.Module 的类，名为 LowScaleEncoder
class LowScaleEncoder(nn.Module):
    # 构造函数，初始化低缩放编码器的参数
    def __init__(
        self,
        model_config,  # 模型配置
        linear_start,  # 线性起始值
        linear_end,  # 线性结束值
        timesteps=1000,  # 时间步数
        max_noise_level=250,  # 最大噪声水平
        output_size=64,  # 输出大小
        scale_factor=1.0,  # 缩放因子
    # 定义一个类，继承自父类
    def __init__(self, max_noise_level, model_config, timesteps, linear_start, linear_end, output_size, scale_factor):
        # 调用父类的初始化方法
        super().__init__()
        # 设置最大噪声级别
        self.max_noise_level = max_noise_level
        # 根据配置实例化模型
        self.model = instantiate_from_config(model_config)
        # 注册一个调度表，用于控制噪声的变化
        self.augmentation_schedule = self.register_schedule(
            timesteps=timesteps, linear_start=linear_start, linear_end=linear_end
        )
        # 设置输出大小
        self.out_size = output_size
        # 设置缩放因子
        self.scale_factor = scale_factor
    
    # 注册一个调度表，用于控制噪声的变化
    def register_schedule(self, beta_schedule, timesteps, linear_start, linear_end, cosine_s):
        # 根据给定的参数生成 beta 调度表
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        # 根据 betas 计算 alphas
        alphas = 1.0 - betas
        # 计算 alphas 的累积乘积
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # 计算 alphas 的累积乘积的前一个值
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    
        # 获取 betas 的形状
        (timesteps,) = betas.shape
        # 将 timesteps 转换为整数
        self.num_timesteps = int(timesteps)
        # 设置线性起始值
        self.linear_start = linear_start
        # 设置线性结束值
        self.linear_end = linear_end
        # 判断 alphas_cumprod 的形状是否与 num_timesteps 相同
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"
    
        # 创建一个偏函数，用于将数组转换为 torch.tensor
        to_torch = partial(torch.tensor, dtype=torch.float32)
    
        # 注册缓冲区，存储 betas
        self.register_buffer("betas", to_torch(betas))
        # 注册缓冲区，存储 alphas_cumprod
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        # 注册缓冲区，存储 alphas_cumprod_prev
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
    
        # 计算扩散 q(x_t | x_{t-1}) 和其他参数
        # 注册缓冲区，存储 sqrt_alphas_cumprod
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        # 注册缓冲区，存储 sqrt_one_minus_alphas_cumprod
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        # 注册缓冲区，存储 log_one_minus_alphas_cumprod
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        # 注册缓冲区，存储 sqrt_recip_alphas_cumprod
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        # 注册缓冲区，存储 sqrt_recipm1_alphas_cumprod
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )
    
    # 从初始值 x_start 和时间步 t 生成噪声样本
    def q_sample(self, x_start, t, noise):
        # 如果没有传入噪声，则生成一个与 x_start 形状相同的随机噪声
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 根据噪声和调度表生成噪声样本
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    # 定义前向传播函数，接收输入 x
    def forward(self, x):
        # 使用模型对输入 x 进行编码，得到潜在表示 z
        z = self.model.encode(x)
        # 检查 z 是否为对角高斯分布类型
        if isinstance(z, DiagonalGaussianDistribution):
            # 从高斯分布中采样，更新 z
            z = z.sample()
        # 将 z 乘以缩放因子，调整其大小
        z = z * self.scale_factor
        # 随机生成噪声水平，范围从 0 到 max_noise_level，形状与批大小相同
        noise_level = torch.randint(
            0, self.max_noise_level, (x.shape[0],), device=x.device
        ).long()
        # 对 z 应用 q_sample 函数，根据噪声水平生成样本
        z = self.q_sample(z, noise_level)
        # 如果指定了输出大小，则调整 z 的尺寸
        if self.out_size is not None:
            z = torch.nn.functional.interpolate(z, size=self.out_size, mode="nearest")
        # z = z.repeat_interleave(2, -2).repeat_interleave(2, -1)  # 注释掉的代码：可能用于调整 z 的形状
        # 返回处理后的 z 和噪声水平
        return z, noise_level

    # 定义解码函数，接收潜在表示 z
    def decode(self, z):
        # 将 z 除以缩放因子，恢复其原始尺度
        z = z / self.scale_factor
        # 使用模型对 z 进行解码，返回解码后的结果
        return self.model.decode(z)
# 定义一个多维时间步嵌入模型类，继承自抽象嵌入模型
class ConcatTimestepEmbedderND(AbstractEmbModel):
    """嵌入每个维度并独立拼接它们"""

    # 初始化方法，接受输出维度参数
    def __init__(self, outdim):
        # 调用父类的初始化方法
        super().__init__()
        # 创建时间步嵌入对象
        self.timestep = Timestep(outdim)
        # 保存输出维度
        self.outdim = outdim

    # 前向传播方法，处理输入数据
    def forward(self, x):
        # 如果输入是1维，则增加一个维度
        if x.ndim == 1:
            x = x[:, None]
        # 确保输入为2维
        assert len(x.shape) == 2
        # 获取批大小和维度数量
        b, dims = x.shape[0], x.shape[1]
        # 重排输入数据为一维
        x = rearrange(x, "b d -> (b d)")
        # 获取时间步嵌入
        emb = self.timestep(x)
        # 重排嵌入为批大小和输出维度格式
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        # 返回最终的嵌入
        return emb


# 定义一个高斯编码器类，继承自编码器和抽象嵌入模型
class GaussianEncoder(Encoder, AbstractEmbModel):
    # 初始化方法，接受权重和是否扁平化输出参数
    def __init__(
        self, weight: float = 1.0, flatten_output: bool = True, *args, **kwargs
    ):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 创建对角高斯正则化器
        self.posterior = DiagonalGaussianRegularizer()
        # 保存权重
        self.weight = weight
        # 保存是否扁平化输出标志
        self.flatten_output = flatten_output

    # 前向传播方法，处理输入数据
    def forward(self, x) -> Tuple[Dict, torch.Tensor]:
        # 调用父类的前向传播，获取潜变量
        z = super().forward(x)
        # 通过正则化器处理潜变量
        z, log = self.posterior(z)
        # 记录损失和权重
        log["loss"] = log["kl_loss"]
        log["weight"] = self.weight
        # 如果需要，扁平化输出
        if self.flatten_output:
            z = rearrange(z, "b c h w -> b (h w ) c")
        # 返回日志和潜变量
        return log, z


# 定义一个冻结的 OpenCLIP 图像预测嵌入模型类
class FrozenOpenCLIPImagePredictionEmbedder(AbstractEmbModel):
    # 初始化方法，接受配置、条件帧数和副本数
    def __init__(
        self,
        open_clip_embedding_config: Dict,
        n_cond_frames: int,
        n_copies: int,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 保存条件帧数
        self.n_cond_frames = n_cond_frames
        # 保存副本数
        self.n_copies = n_copies
        # 实例化 OpenCLIP 嵌入对象
        self.open_clip = instantiate_from_config(open_clip_embedding_config)

    # 前向传播方法，处理视频输入
    def forward(self, vid):
        # 通过 OpenCLIP 嵌入视频数据
        vid = self.open_clip(vid)
        # 重排视频数据为批大小和时间步格式
        vid = rearrange(vid, "(b t) d -> b t d", t=self.n_cond_frames)
        # 重复视频数据以匹配副本数
        vid = repeat(vid, "b t d -> (b s) t d", s=self.n_copies)

        # 返回处理后的视频数据
        return vid


# 定义一个原始图像嵌入模型类，继承自抽象嵌入模型
class RawImageEmbedder(AbstractEmbModel):
    """
    在 Instructpix2pix 中将原始图像作为条件
    """
    
    # 前向传播方法，直接返回输入图像
    def forward(self, image):
        return image
```