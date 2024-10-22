# `.\chatglm4v-9b\tokenization_chatglm.py`

```
# 导入所需的库和模块
import regex as re  # 正则表达式库
import base64  # 用于处理 Base64 编码
import os  # 提供与操作系统交互的功能
import json  # 处理 JSON 数据
import tiktoken  # 处理文本编码
import torch  # 深度学习框架
from torch import TensorType  # 从 torch 导入 TensorType 类型
from typing import List, Optional, Union, Dict, Any  # 类型提示
from torchvision import transforms  # 计算机视觉的转换工具
from transformers import PreTrainedTokenizer  # 导入预训练的分词器基类
from transformers.utils import logging, PaddingStrategy  # 导入日志和填充策略
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding  # 导入编码输入和批处理编码类


# 定义 ChatGLM4Tokenizer 类，继承自 PreTrainedTokenizer
class ChatGLM4Tokenizer(PreTrainedTokenizer):
    # 词汇文件名称
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    # 初始化方法
    def __init__(
            self,
            vocab_file,  # 词汇文件路径
            padding_side="left",  # 填充方向
            clean_up_tokenization_spaces=False,  # 是否清理标记化空格
            encode_special_tokens=False,  # 是否编码特殊标记
            image_size=None,  # 图像大小
            **kwargs  # 其他关键字参数
    ):
        self.name = "GLM4Tokenizer"  # 设置分词器名称
        self.vocab_file = vocab_file  # 设置词汇文件
        # 定义正则表达式模式字符串
        pat_str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        self.pat_str = re.compile(pat_str)  # 编译正则表达式
        self.encode_special_tokens = encode_special_tokens  # 设置特殊标记编码标志
        self.image_size = image_size  # 设置图像大小

        mergeable_ranks = {}  # 初始化可合并的排名字典
        with open(vocab_file) as f:  # 打开词汇文件
            for line in f:  # 遍历文件每一行
                token, rank = line.strip().split()  # 拆分标记和排名
                rank = int(rank)  # 将排名转换为整数
                token = base64.b64decode(token)  # 解码 Base64 标记
                mergeable_ranks[token] = rank  # 将标记和排名添加到字典中

        self.mergeable_ranks = mergeable_ranks  # 保存可合并的排名字典

        # 初始化 tiktoken 编码器
        self.tokenizer = tiktoken.Encoding(
            name="my_tokenizer",  # 设置编码器名称
            pat_str=pat_str,  # 设置正则表达式模式
            mergeable_ranks=mergeable_ranks,  # 设置可合并的排名
            special_tokens={}  # 初始化特殊标记为空字典
        )
        self.decoder = {rank: token for token, rank in mergeable_ranks.items()}  # 反转可合并排名字典为解码器
        self.n_words = len(self.decoder)  # 计算单词数量

        # 调用父类初始化方法
        super().__init__(
            padding_side=padding_side,  # 设置填充方向
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,  # 设置空格清理标志
            **kwargs  # 传递其他参数
        )

    # 词汇大小属性
    @property
    def vocab_size(self):
        return self.n_words  # 返回单词数量

    # 获取词汇的方法
    def get_vocab(self):
        """ Returns vocab as a dict """
        # 创建一个从标记 ID 到标记的字典
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)  # 更新词汇字典，添加额外的标记
        return vocab  # 返回词汇字典

    # 将标记转换为字符串的方法
    def convert_tokens_to_string(self, tokens: List[Union[bytes, str, int]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""  # 初始化文本字符串
        temp = b""  # 初始化临时字节串
        for t in tokens:  # 遍历标记列表
            if isinstance(t, int):  # 如果标记是整数
                t = chr(t)  # 将其转换为字符
            if isinstance(t, str):  # 如果标记是字符串
                if temp:  # 如果临时字节串不为空
                    text += temp.decode("utf-8", errors="replace")  # 解码并添加到文本中
            elif isinstance(t, bytes):  # 如果标记是字节
                temp += t  # 将字节添加到临时字节串
            else:  # 如果标记类型不匹配
                raise TypeError("token should only be of type int, bytes or str")  # 抛出类型错误
        if temp:  # 如果临时字节串不为空
            text += temp.decode("utf-8", errors="replace")  # 解码并添加到文本中
        return text  # 返回最终文本字符串
    # 定义一个私有方法，用于对输入文本进行分词
        def _tokenize(self, text, **kwargs):
            # 初始化一个空列表用于存放分词结果
            tokens = []
            # 使用分词器将输入文本编码为 ID 列表
            ids = self.tokenizer.encode(text)
            # 遍历 ID 列表，将每个 ID 转换为对应的词汇
            for t in ids:
                tokens.append(self.decoder[t])
            # 返回分词结果
            return tokens
    
        # 定义一个私有方法，将给定的词转换为其对应的 ID
        def _convert_token_to_id(self, token):
            """ 将词（字符串）转换为词汇表中的 ID。 """
            # 返回词的可合并排名作为其 ID
            return self.mergeable_ranks[token]
    
        # 定义一个私有方法，将给定的索引转换为对应的词
        def _convert_id_to_token(self, index):
            """ 将索引（整数）转换为词（字符串），使用词汇表。 """
            # 返回对应索引的词，若不存在则返回空字符串
            return self.decoder.get(index, "")
    
        # 定义一个方法，将词汇表和特殊标记文件保存到指定目录
        def save_vocabulary(self, save_directory, filename_prefix=None):
            """
            保存词汇表和特殊标记文件到一个目录。
    
            参数：
                save_directory (`str`):
                    要保存词汇表的目录。
                filename_prefix (`str`, *可选*):
                    要添加到保存文件名称的可选前缀。
    
            返回：
                `Tuple(str)`: 保存文件的路径。
            """
            # 检查指定目录是否存在
            if os.path.isdir(save_directory):
                # 如果存在，构造词汇文件的完整路径
                vocab_file = os.path.join(
                    save_directory, self.vocab_files_names["vocab_file"]
                )
            else:
                # 如果不存在，使用指定的保存目录作为文件路径
                vocab_file = save_directory
    
            # 以二进制模式打开词汇文件进行读取
            with open(self.vocab_file, 'rb') as fin:
                # 读取词汇文件内容
                proto_str = fin.read()
    
            # 以二进制模式打开目标文件进行写入
            with open(vocab_file, "wb") as writer:
                # 将读取的内容写入目标文件
                writer.write(proto_str)
    
            # 返回保存的文件路径元组
            return (vocab_file,)
    
        # 定义一个方法获取前缀标记
        def get_prefix_tokens(self):
            # 返回特定的前缀标记 ID 列表
            prefix_tokens = [self.convert_tokens_to_ids("[gMASK]"), self.convert_tokens_to_ids("<sop>")]
            return prefix_tokens
    
        # 定义一个方法构建单条消息
        def build_single_message(self, role, metadata, message, tokenize=True, message_prefix=None):
            # 断言角色在预定义的角色列表中
            assert role in ["system", "user", "assistant", "observation"], role
            # 如果需要分词
            if tokenize:
                # 将角色转换为 ID，并编码元数据为 ID 列表
                role_tokens = [self.convert_tokens_to_ids(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n",
                                                                                                  disallowed_special=())
                # 编码消息为 ID 列表
                message_tokens = self.tokenizer.encode(message, disallowed_special=())
                # 如果提供了消息前缀，将其添加到消息 ID 列表中
                if message_prefix is not None:
                    message_tokens = message_prefix + message_tokens
                # 返回角色 ID 和消息 ID 的组合
                tokens = role_tokens + message_tokens
                return tokens
            else:
                # 如果不需要分词，返回格式化的消息字符串
                return str(f"<|{role}|>{metadata}\n{message}")
    
        # 定义一个方法应用聊天模板
        def apply_chat_template(
                self,
                conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]], "Conversation"],
                add_generation_prompt: bool = False,
                tokenize: bool = True,
                padding: bool = False,
                truncation: bool = False,
                max_length: Optional[int] = None,
                return_tensors: Optional[Union[str, TensorType]] = None,
                return_dict: bool = False,
                tokenizer_kwargs: Optional[Dict[str, Any]] = None,
                add_special_tokens: bool = True,
                **kwargs,
    # 构建包含特殊标记的模型输入，用于序列分类任务
    def build_inputs_with_special_tokens(
                self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
            """
            从单个或成对序列构建模型输入，通过连接并添加特殊标记。BERT 序列格式如下：
    
            - 单序列: `[CLS] X [SEP]`
            - 序列对: `[CLS] A [SEP] B [SEP]`
    
            参数:
                token_ids_0 (`List[int]`):
                    将添加特殊标记的 ID 列表。
                token_ids_1 (`List[int]`, *可选*):
                    可选的第二个 ID 列表，用于序列对。
    
            返回:
                `List[int]`: 包含适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
            """
            # 获取前缀标记
            prefix_tokens = self.get_prefix_tokens()
            # 在 token_ids_0 前添加前缀标记
            token_ids_0 = prefix_tokens + token_ids_0
            # 如果存在 token_ids_1，则进行连接并添加结束标记
            if token_ids_1 is not None:
                token_ids_0 = token_ids_0 + token_ids_1 + [self.convert_tokens_to_ids("<eos>")]
            # 返回构建好的 token_ids_0
            return token_ids_0
    
        # 填充输入以匹配最大长度
        def _pad(
                self,
                encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
                max_length: Optional[int] = None,
                padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
                pad_to_multiple_of: Optional[int] = None,
                return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        对编码后的输入进行填充（左侧/右侧填充，并达到预定义长度或批次中的最大长度）

        Args:
            encoded_inputs:
                词元化输入的字典（`List[int]`）或词元化输入的批次（`List[List[int]]`）。
            max_length: 返回列表的最大长度，选项上也为填充长度（见下文）。
                将根据特殊标记进行截断。
            padding_strategy: 用于填充的 PaddingStrategy。

                - PaddingStrategy.LONGEST: 填充到批次中最长的序列
                - PaddingStrategy.MAX_LENGTH: 填充到最大长度（默认）
                - PaddingStrategy.DO_NOT_PAD: 不进行填充
                填充的方向由 self.padding_side 定义：

                    - 'left': 在序列的左侧进行填充
                    - 'right': 在序列的右侧进行填充
            pad_to_multiple_of: （可选）如果设置，将序列填充到所提供值的倍数。
                这在启用计算能力 `>= 7.5`（Volta）上的 NVIDIA 硬件的 Tensor Core 时尤其有用。
            return_attention_mask:
                （可选）设置为 False 以避免返回注意力掩码（默认：根据模型具体情况设置）
        """
        # 从模型默认值加载
        assert self.padding_side == "left"  # 确保填充方向为左侧

        required_input = encoded_inputs[self.model_input_names[0]]  # 获取必需的输入
        seq_length = len(required_input)  # 计算输入序列的长度

        if padding_strategy == PaddingStrategy.LONGEST:  # 如果填充策略为最长填充
            max_length = len(required_input)  # 设置最大长度为输入序列的长度

        # 如果最大长度和倍数填充值都不为 None，且最大长度不是倍数填充值的倍数
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            # 调整最大长度为下一个倍数填充值
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        # 检查是否需要填充
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # 如果没有注意力掩码，初始化它
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length  # 初始化注意力掩码为全1

        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))  # 初始化位置ID为0到序列长度-1

        if needs_to_be_padded:  # 如果需要填充
            difference = max_length - len(required_input)  # 计算需要填充的数量

            if "attention_mask" in encoded_inputs:  # 如果存在注意力掩码
                # 在注意力掩码的左侧添加0以匹配最大长度
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            if "position_ids" in encoded_inputs:  # 如果存在位置ID
                # 在位置ID的左侧添加0以匹配最大长度
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
            # 在必需输入的左侧添加填充标记以匹配最大长度
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs  # 返回处理后的编码输入
```