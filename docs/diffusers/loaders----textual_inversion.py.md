# `.\diffusers\loaders\textual_inversion.py`

```py
# 版权声明，表示该文件的所有权及使用条款
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 许可证信息，指明该文件遵循的开源许可证
# Licensed under the Apache License, Version 2.0 (the "License");
# 使用本文件需遵循许可证的规定
# you may not use this file except in compliance with the License.
# 获取许可证的链接
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 免责声明，表示在法律允许的范围内不承担任何责任
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.
# 从 typing 模块导入所需类型，便于类型注解
from typing import Dict, List, Optional, Union

# 导入 safetensors 库，可能用于安全的张量处理
import safetensors
# 导入 PyTorch 库，便于深度学习模型的构建与训练
import torch
# 导入验证 Hugging Face Hub 参数的函数
from huggingface_hub.utils import validate_hf_hub_args
# 从 PyTorch 导入神经网络模块
from torch import nn

# 根据可用性导入 transformers 模块的预训练模型与分词器
from ..models.modeling_utils import load_state_dict
# 导入工具函数，处理模型文件、检查依赖等
from ..utils import _get_model_file, is_accelerate_available, is_transformers_available, logging

# 检查 transformers 库是否可用，如果可用则导入相关类
if is_transformers_available():
    from transformers import PreTrainedModel, PreTrainedTokenizer

# 检查 accelerate 库是否可用，如果可用则导入相关钩子
if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义文本反转所需的文件名
TEXT_INVERSION_NAME = "learned_embeds.bin"
# 定义安全版本的文本反转文件名
TEXT_INVERSION_NAME_SAFE = "learned_embeds.safetensors"

# 装饰器，用于验证 Hugging Face Hub 的参数
@validate_hf_hub_args
# 定义加载文本反转状态字典的函数
def load_textual_inversion_state_dicts(pretrained_model_name_or_paths, **kwargs):
    # 从关键字参数中获取缓存目录，默认为 None
    cache_dir = kwargs.pop("cache_dir", None)
    # 从关键字参数中获取是否强制下载的标志，默认为 False
    force_download = kwargs.pop("force_download", False)
    # 从关键字参数中获取代理设置，默认为 None
    proxies = kwargs.pop("proxies", None)
    # 从关键字参数中获取本地文件是否仅使用的标志，默认为 None
    local_files_only = kwargs.pop("local_files_only", None)
    # 从关键字参数中获取访问令牌，默认为 None
    token = kwargs.pop("token", None)
    # 从关键字参数中获取版本号，默认为 None
    revision = kwargs.pop("revision", None)
    # 从关键字参数中获取子文件夹名称，默认为 None
    subfolder = kwargs.pop("subfolder", None)
    # 从关键字参数中获取权重文件名，默认为 None
    weight_name = kwargs.pop("weight_name", None)
    # 从关键字参数中获取是否使用 safetensors 的标志，默认为 None
    use_safetensors = kwargs.pop("use_safetensors", None)

    # 设置允许使用 pickle 的标志为 False
    allow_pickle = False
    # 如果未指定使用 safetensors，则默认启用，并允许使用 pickle
    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True

    # 设置用户代理信息，用于标识请求的类型和框架
    user_agent = {
        "file_type": "text_inversion",
        "framework": "pytorch",
    }
    # 初始化状态字典列表
    state_dicts = []
    # 遍历预训练模型名称或路径列表
    for pretrained_model_name_or_path in pretrained_model_name_or_paths:
        # 检查当前项是否不是字典或张量
        if not isinstance(pretrained_model_name_or_path, (dict, torch.Tensor)):
            # 初始化模型文件为 None
            model_file = None

            # 尝试加载 .safetensors 权重
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    # 获取模型文件，提供相关参数
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=weight_name or TEXT_INVERSION_NAME_SAFE,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    # 从文件中加载状态字典到 CPU 上
                    state_dict = safetensors.torch.load_file(model_file, device="cpu")
                except Exception as e:
                    # 如果不允许 pickle，抛出异常
                    if not allow_pickle:
                        raise e

                    # 如果加载失败，设置模型文件为 None
                    model_file = None

            # 如果模型文件仍然是 None，则尝试加载其他格式
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=weight_name or TEXT_INVERSION_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                # 从文件中加载状态字典
                state_dict = load_state_dict(model_file)
        else:
            # 如果当前项是字典或张量，直接使用它作为状态字典
            state_dict = pretrained_model_name_or_path

        # 将状态字典添加到列表中
        state_dicts.append(state_dict)

    # 返回状态字典列表
    return state_dicts
# 定义一个混合类，用于加载文本反转的标记和嵌入到分词器和文本编码器中
class TextualInversionLoaderMixin:
    r"""
    加载文本反转标记和嵌入到分词器和文本编码器中。
    """

    # 定义一个方法，根据输入的提示和分词器可能进行转换
    def maybe_convert_prompt(self, prompt: Union[str, List[str]], tokenizer: "PreTrainedTokenizer"):  # noqa: F821
        r"""
        处理包含特殊标记的提示，这些标记对应于多向量文本反转嵌入，将其替换为多个
        特殊标记，每个对应一个向量。如果提示没有文本反转标记或文本反转标记是单个向量，
        则返回输入提示。

        参数:
            prompt (`str` 或 list of `str`):
                引导图像生成的提示。
            tokenizer (`PreTrainedTokenizer`):
                负责将提示编码为输入标记的分词器。

        返回:
            `str` 或 list of `str`: 转换后的提示
        """
        # 检查输入提示是否为列表，如果不是则将其转换为列表
        if not isinstance(prompt, List):
            prompts = [prompt]
        else:
            prompts = prompt

        # 对每个提示应用可能的转换
        prompts = [self._maybe_convert_prompt(p, tokenizer) for p in prompts]

        # 如果输入提示不是列表，则返回第一个转换后的提示
        if not isinstance(prompt, List):
            return prompts[0]

        # 返回转换后的提示列表
        return prompts

    # 定义一个私有方法，可能将提示转换为“多向量”兼容提示
    def _maybe_convert_prompt(self, prompt: str, tokenizer: "PreTrainedTokenizer"):  # noqa: F821
        r"""
        可能将提示转换为“多向量”兼容提示。如果提示包含一个与多向量文本反转嵌入
        对应的标记，该函数将处理提示，使特殊标记被多个特殊标记替换，每个对应一个向量。
        如果提示没有文本反转标记或文本反转标记是单个向量，则简单返回输入提示。

        参数:
            prompt (`str`):
                引导图像生成的提示。
            tokenizer (`PreTrainedTokenizer`):
                负责将提示编码为输入标记的分词器。

        返回:
            `str`: 转换后的提示
        """
        # 使用分词器对提示进行分词
        tokens = tokenizer.tokenize(prompt)
        # 创建一个唯一标记的集合
        unique_tokens = set(tokens)
        # 遍历唯一标记
        for token in unique_tokens:
            # 检查标记是否在添加的标记编码器中
            if token in tokenizer.added_tokens_encoder:
                replacement = token  # 初始化替换变量为当前标记
                i = 1  # 初始化计数器
                # 生成替换标记，直到没有更多的标记存在
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"  # 添加计数到替换变量
                    i += 1  # 增加计数器

                # 在提示中替换原始标记为生成的替换标记
                prompt = prompt.replace(token, replacement)

        # 返回最终的提示
        return prompt
    # 定义检查文本反转输入参数的私有方法
    def _check_text_inv_inputs(self, tokenizer, text_encoder, pretrained_model_name_or_paths, tokens):
        # 检查传入的 tokenizer 是否为 None，如果是，则抛出 ValueError
        if tokenizer is None:
            raise ValueError(
                # 报错信息，说明需要提供 tokenizer 参数
                f"{self.__class__.__name__} requires `self.tokenizer` or passing a `tokenizer` of type `PreTrainedTokenizer` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )

        # 检查传入的 text_encoder 是否为 None，如果是，则抛出 ValueError
        if text_encoder is None:
            raise ValueError(
                # 报错信息，说明需要提供 text_encoder 参数
                f"{self.__class__.__name__} requires `self.text_encoder` or passing a `text_encoder` of type `PreTrainedModel` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )

        # 检查预训练模型名称列表的长度与 tokens 列表的长度是否一致
        if len(pretrained_model_name_or_paths) > 1 and len(pretrained_model_name_or_paths) != len(tokens):
            raise ValueError(
                # 报错信息，说明模型列表与 tokens 列表的长度不匹配
                f"You have passed a list of models of length {len(pretrained_model_name_or_paths)}, and list of tokens of length {len(tokens)} "
                f"Make sure both lists have the same length."
            )

        # 过滤出有效的 tokens，即不为 None 的 tokens
        valid_tokens = [t for t in tokens if t is not None]
        # 检查有效 tokens 的集合长度是否小于有效 tokens 的列表长度，如果是，则说明有重复
        if len(set(valid_tokens)) < len(valid_tokens):
            raise ValueError(f"You have passed a list of tokens that contains duplicates: {tokens}")

    # 定义一个静态方法
    @staticmethod
    # 定义一个私有方法，用于检索 tokens 和 embeddings
        def _retrieve_tokens_and_embeddings(tokens, state_dicts, tokenizer):
            # 初始化空列表以存储所有 tokens 和 embeddings
            all_tokens = []
            all_embeddings = []
            # 同时遍历状态字典和 tokens
            for state_dict, token in zip(state_dicts, tokens):
                # 检查状态字典是否为 PyTorch 张量
                if isinstance(state_dict, torch.Tensor):
                    # 如果 token 为 None，抛出错误
                    if token is None:
                        raise ValueError(
                            "You are trying to load a textual inversion embedding that has been saved as a PyTorch tensor. Make sure to pass the name of the corresponding token in this case: `token=...`."
                        )
                    # 加载 token 和 embedding
                    loaded_token = token
                    embedding = state_dict
                # 检查状态字典是否只包含一个键
                elif len(state_dict) == 1:
                    # 处理 diffusers 格式
                    loaded_token, embedding = next(iter(state_dict.items()))
                # 检查状态字典是否包含 "string_to_param" 键
                elif "string_to_param" in state_dict:
                    # 处理 A1111 格式
                    loaded_token = state_dict["name"]
                    embedding = state_dict["string_to_param"]["*"]
                else:
                    # 抛出状态字典格式错误的错误
                    raise ValueError(
                        f"Loaded state dictionary is incorrect: {state_dict}. \n\n"
                        "Please verify that the loaded state dictionary of the textual embedding either only has a single key or includes the `string_to_param`"
                        " input key."
                    )
    
                # 如果 token 不为 None 且加载的 token 与当前 token 不同，记录日志
                if token is not None and loaded_token != token:
                    logger.info(f"The loaded token: {loaded_token} is overwritten by the passed token {token}.")
                else:
                    # 将加载的 token 赋值给当前 token
                    token = loaded_token
    
                # 检查 token 是否已经在 tokenizer 的词汇表中
                if token in tokenizer.get_vocab():
                    # 如果已存在，抛出错误
                    raise ValueError(
                        f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
                    )
    
                # 将 token 和 embedding 添加到对应列表中
                all_tokens.append(token)
                all_embeddings.append(embedding)
    
            # 返回所有的 tokens 和 embeddings
            return all_tokens, all_embeddings
    
        # 声明该方法为静态方法
        @staticmethod
    # 扩展给定的令牌和嵌入，将多向量令牌和其嵌入整合到一起
    def _extend_tokens_and_embeddings(tokens, embeddings, tokenizer):
        # 初始化一个空列表以存储所有令牌
        all_tokens = []
        # 初始化一个空列表以存储所有嵌入
        all_embeddings = []
    
        # 遍历嵌入和令牌的配对
        for embedding, token in zip(embeddings, tokens):
            # 检查令牌是否已经在词汇表中
            if f"{token}_1" in tokenizer.get_vocab():
                # 如果令牌已经存在，初始化多向量令牌列表
                multi_vector_tokens = [token]
                # 初始化索引
                i = 1
                # 检查是否有后续的多向量令牌
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    # 将多向量令牌添加到列表中
                    multi_vector_tokens.append(f"{token}_{i}")
                    # 递增索引
                    i += 1
    
                # 抛出异常，提示多向量令牌已经存在
                raise ValueError(
                    f"Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder."
                )
    
            # 判断当前嵌入是否为多维向量
            is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1
            if is_multi_vector:
                # 如果是多维向量，将令牌及其索引添加到列表中
                all_tokens += [token] + [f"{token}_{i}" for i in range(1, embedding.shape[0])]
                # 添加对应的所有嵌入到列表中
                all_embeddings += [e for e in embedding]  # noqa: C416
            else:
                # 如果不是多维向量，仅添加当前令牌
                all_tokens += [token]
                # 根据嵌入的维度添加嵌入
                all_embeddings += [embedding[0]] if len(embedding.shape) > 1 else [embedding]
    
        # 返回所有令牌和嵌入的列表
        return all_tokens, all_embeddings
    
    # 装饰器，用于验证 Hugging Face Hub 参数
    @validate_hf_hub_args
    # 加载文本反转（Textual Inversion）模型
    def load_textual_inversion(
        # 预训练模型的名称或路径，支持多种格式
        pretrained_model_name_or_path: Union[str, List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        # 可选的令牌
        token: Optional[Union[str, List[str]]] = None,
        # 可选的预训练分词器
        tokenizer: Optional["PreTrainedTokenizer"] = None,  # noqa: F821
        # 可选的文本编码器
        text_encoder: Optional["PreTrainedModel"] = None,  # noqa: F821
        # 其他关键字参数
        **kwargs,
    ):
        # 省略具体实现
    
    # 卸载文本反转（Textual Inversion）模型
    def unload_textual_inversion(
        # 可选的令牌
        tokens: Optional[Union[str, List[str]]] = None,
        # 可选的预训练分词器
        tokenizer: Optional["PreTrainedTokenizer"] = None,
        # 可选的文本编码器
        text_encoder: Optional["PreTrainedModel"] = None,
    ):
        # 省略具体实现
```