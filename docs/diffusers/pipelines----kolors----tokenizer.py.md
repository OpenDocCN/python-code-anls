# `.\diffusers\pipelines\kolors\tokenizer.py`

```py
# Copyright 2024 ChatGLM3-6B Model Team, Kwai-Kolors Team and The HuggingFace Team. All rights reserved.
#
# 许可信息，声明版权和许可证条款
# Licensed under the Apache License, Version 2.0 (the "License");
# 在遵守许可证的前提下才能使用此文件
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 在没有适用的法律或书面协议情况下，软件以“按现状”方式分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以获取具体的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入所需的库
import json
import os
import re
from typing import Dict, List, Optional, Union

# 从 SentencePiece 导入处理器
from sentencepiece import SentencePieceProcessor
# 从 transformers 导入预训练的 tokenizer
from transformers import PreTrainedTokenizer
# 导入批处理编码和编码输入的工具
from transformers.tokenization_utils_base import BatchEncoding, EncodedInput
# 导入填充策略
from transformers.utils import PaddingStrategy

# 定义 SPTokenizer 类
class SPTokenizer:
    # 初始化函数，接收模型路径
    def __init__(self, model_path: str):
        # 断言模型文件存在
        assert os.path.isfile(model_path), model_path
        # 通过模型文件加载 SentencePiece 处理器
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # 获取 BOS / EOS token 的 ID
        self.n_words: int = self.sp_model.vocab_size()  # 词汇表大小
        self.bos_id: int = self.sp_model.bos_id()      # BOS token ID
        self.eos_id: int = self.sp_model.eos_id()      # EOS token ID
        self.pad_id: int = self.sp_model.unk_id()      # PAD token ID
        # 确保词汇表大小与片段大小相同
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        # 定义角色特定的特殊 tokens
        role_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]
        # 定义其他特殊 tokens
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"] + role_special_tokens
        # 初始化特殊 tokens 的字典和索引
        self.special_tokens = {}
        self.index_special_tokens = {}
        # 为每个特殊 token 分配唯一的 ID
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1
        # 将角色特殊 tokens 组成正则表达式
        self.role_special_token_expression = "|".join([re.escape(token) for token in role_special_tokens])

    # 对输入字符串进行分词
    def tokenize(self, s: str, encode_special_tokens=False):
        # 如果需要编码特殊 tokens
        if encode_special_tokens:
            last_index = 0
            t = []
            # 查找匹配的角色特殊 tokens
            for match in re.finditer(self.role_special_token_expression, s):
                # 如果有普通文本，先编码它
                if last_index < match.start():
                    t.extend(self.sp_model.EncodeAsPieces(s[last_index : match.start()]))
                # 添加匹配的特殊 token
                t.append(s[match.start() : match.end()])
                last_index = match.end()
            # 编码最后一段普通文本
            if last_index < len(s):
                t.extend(self.sp_model.EncodeAsPieces(s[last_index:]))
            return t
        else:
            # 如果不需要编码特殊 tokens，直接编码整个字符串
            return self.sp_model.EncodeAsPieces(s)

    # 对输入字符串进行编码，返回 token ID 列表
    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        # 断言输入为字符串
        assert isinstance(s, str)
        # 使用 SentencePiece 编码字符串
        t = self.sp_model.encode(s)
        # 如果需要，添加 BOS token ID
        if bos:
            t = [self.bos_id] + t
        # 如果需要，添加 EOS token ID
        if eos:
            t = t + [self.eos_id]
        return t
    # 定义解码函数，将一组整数标记转换为字符串
    def decode(self, t: List[int]) -> str:
        # 初始化解码后的文本和一个缓冲区列表
        text, buffer = "", []
        # 遍历每个标记
        for token in t:
            # 检查当前标记是否为特殊标记
            if token in self.index_special_tokens:
                # 如果缓冲区不为空，解码缓冲区中的标记并添加到文本中
                if buffer:
                    text += self.sp_model.decode(buffer)
                    # 清空缓冲区
                    buffer = []
                # 将特殊标记对应的文本添加到解码文本中
                text += self.index_special_tokens[token]
            else:
                # 将普通标记添加到缓冲区
                buffer.append(token)
        # 如果缓冲区仍然有标记，解码缓冲区中的标记并添加到文本中
        if buffer:
            text += self.sp_model.decode(buffer)
        # 返回解码后的文本
        return text
    
    # 定义将标记列表解码为字符串的函数
    def decode_tokens(self, tokens: List[str]) -> str:
        # 使用 sp_model 解码标记列表，返回解码结果
        text = self.sp_model.DecodePieces(tokens)
        # 返回解码后的文本
        return text
    
    # 定义将标记（字符串）转换为 ID 的函数
    def convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 如果标记是特殊标记，则返回其对应的 ID
        if token in self.special_tokens:
            return self.special_tokens[token]
        # 否则，使用 sp_model 将标记转换为 ID
        return self.sp_model.PieceToId(token)
    
    # 定义将索引（整数）转换为标记（字符串）的函数
    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 如果索引是特殊标记的索引，返回对应的标记
        if index in self.index_special_tokens:
            return self.index_special_tokens[index]
        # 如果索引是结束标记、开始标记、填充标记，或小于 0，返回空字符串
        if index in [self.eos_id, self.bos_id, self.pad_id] or index < 0:
            return ""
        # 否则，使用 sp_model 将索引转换为标记
        return self.sp_model.IdToPiece(index)
# 定义一个名为 ChatGLMTokenizer 的类，继承自 PreTrainedTokenizer
class ChatGLMTokenizer(PreTrainedTokenizer):
    # 定义词汇文件名称，指定 tokenizer.model 为 vocab_file
    vocab_files_names = {"vocab_file": "tokenizer.model"}

    # 定义模型输入的名称，包括输入ID、注意力掩码和位置ID
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    # 初始化方法，接收词汇文件及其他可选参数
    def __init__(
        self,
        vocab_file,
        padding_side="left",  # 默认填充方向为左侧
        clean_up_tokenization_spaces=False,  # 是否清理标记化空间的选项
        encode_special_tokens=False,  # 是否编码特殊标记的选项
        **kwargs,  # 其他额外的关键字参数
    ):
        # 设置 tokenizer 的名称
        self.name = "GLMTokenizer"

        # 保存词汇文件的路径
        self.vocab_file = vocab_file
        # 使用词汇文件初始化 SPTokenizer
        self.tokenizer = SPTokenizer(vocab_file)
        # 定义特殊标记及其对应的ID
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,  # 句首标记
            "<eos>": self.tokenizer.eos_id,  # 句尾标记
            "<pad>": self.tokenizer.pad_id,  # 填充标记
        }
        # 保存是否编码特殊标记的选项
        self.encode_special_tokens = encode_special_tokens
        # 调用父类的初始化方法
        super().__init__(
            padding_side=padding_side,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            encode_special_tokens=encode_special_tokens,
            **kwargs,
        )

    # 根据传入的标记获取相应的命令ID
    def get_command(self, token):
        # 如果标记在特殊标记字典中，返回对应的ID
        if token in self.special_tokens:
            return self.special_tokens[token]
        # 确保传入的标记是有效的特殊标记
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        # 返回 tokenizer 中对应的特殊标记ID
        return self.tokenizer.special_tokens[token]

    # 属性，返回未知标记的字符串
    @property
    def unk_token(self) -> str:
        return "<unk>"

    # 设置未知标记的字符串
    @unk_token.setter
    def unk_token(self, value: str):
        self._unk_token = value

    # 属性，返回填充标记的字符串
    @property
    def pad_token(self) -> str:
        return "<unk>"

    # 设置填充标记的字符串
    @pad_token.setter
    def pad_token(self, value: str):
        self._pad_token = value

    # 属性，返回填充标记的ID
    @property
    def pad_token_id(self):
        return self.get_command("<pad>")

    # 属性，返回结束标记的字符串
    @property
    def eos_token(self) -> str:
        return "</s>"

    # 设置结束标记的字符串
    @eos_token.setter
    def eos_token(self, value: str):
        self._eos_token = value

    # 属性，返回结束标记的ID
    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    # 属性，返回词汇表的大小
    @property
    def vocab_size(self):
        return self.tokenizer.n_words

    # 获取词汇表并返回为字典
    def get_vocab(self):
        """Returns vocab as a dict"""
        # 创建一个字典，将词汇ID映射到对应的标记
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        # 更新字典，包含添加的标记
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 对输入文本进行标记化
    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text, encode_special_tokens=self.encode_special_tokens)

    # 将标记字符串转换为对应的ID
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.tokenizer.convert_token_to_id(token)

    # 将ID转换为对应的标记字符串
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.tokenizer.convert_id_to_token(index)

    # 将标记列表转换为字符串
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.decode_tokens(tokens)
    # 定义保存词汇和特殊标记文件的方法
    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        保存词汇和特殊标记文件到指定目录。

        参数:
            save_directory (`str`):
                要保存词汇的目录。
            filename_prefix (`str`, *可选*):
                保存文件名时添加的可选前缀。

        返回:
            `Tuple(str)`: 保存的文件路径。
        """
        # 检查保存目录是否存在
        if os.path.isdir(save_directory):
            # 如果目录存在，构建词汇文件的完整路径
            vocab_file = os.path.join(save_directory, self.vocab_files_names["vocab_file"])
        else:
            # 如果目录不存在，使用提供的保存目录作为词汇文件路径
            vocab_file = save_directory

        # 以二进制读取模式打开当前的词汇文件
        with open(self.vocab_file, "rb") as fin:
            # 读取文件内容并存储为字节串
            proto_str = fin.read()

        # 以二进制写入模式打开目标词汇文件
        with open(vocab_file, "wb") as writer:
            # 将读取的内容写入到目标词汇文件
            writer.write(proto_str)

        # 返回保存的词汇文件路径
        return (vocab_file,)

    # 定义获取前缀标记的方法
    def get_prefix_tokens(self):
        # 获取特殊前缀标记
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        # 返回前缀标记列表
        return prefix_tokens

    # 定义构建单个消息的方法
    def build_single_message(self, role, metadata, message):
        # 确保角色是有效的选项之一
        assert role in ["system", "user", "assistant", "observation"], role
        # 根据角色构建角色标记和元数据的编码
        role_tokens = [self.get_command(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n")
        # 编码消息内容
        message_tokens = self.tokenizer.encode(message)
        # 合并角色标记和消息标记
        tokens = role_tokens + message_tokens
        # 返回合并后的标记
        return tokens

    # 定义构建聊天输入的方法
    def build_chat_input(self, query, history=None, role="user"):
        # 如果历史记录为空，初始化为空列表
        if history is None:
            history = []
        # 初始化输入标识符列表
        input_ids = []
        # 遍历历史记录
        for item in history:
            # 获取内容
            content = item["content"]
            # 如果角色是系统并且有工具信息，将其添加到内容中
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
            # 将构建的单个消息标记扩展到输入标识符列表中
            input_ids.extend(self.build_single_message(item["role"], item.get("metadata", ""), content))
        # 将当前查询的消息标记添加到输入标识符列表中
        input_ids.extend(self.build_single_message(role, "", query))
        # 添加结束标记
        input_ids.extend([self.get_command("<|assistant|>")])
        # 返回经过批量编码后的输入标识符
        return self.batch_encode_plus([input_ids], return_tensors="pt", is_split_into_words=True)

    # 定义构建带特殊标记的输入的方法
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 返回一个整数列表，构建序列分类任务的模型输入
    ) -> List[int]:
        # 文档字符串，说明该函数的作用和输入输出格式
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
    
        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`
    
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
    
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 获取前缀特殊令牌
        prefix_tokens = self.get_prefix_tokens()
        # 将前缀令牌添加到第一个序列的 ID 列表中
        token_ids_0 = prefix_tokens + token_ids_0
        # 如果第二个序列存在，则将其添加到第一个序列中
        if token_ids_1 is not None:
            # 合并两个序列，并添加结束符令牌
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
        # 返回包含特殊令牌的 ID 列表
        return token_ids_0
    
        # 定义一个私有函数，用于填充编码后的输入
        def _pad(
            self,
            # 编码输入的字典或批处理编码
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            # 最大长度，默认值为 None
            max_length: Optional[int] = None,
            # 填充策略，默认不填充
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            # 填充到的倍数，默认值为 None
            pad_to_multiple_of: Optional[int] = None,
            # 是否返回注意力掩码，默认值为 None
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        对编码后的输入进行填充（左右填充以及根据预定义长度或批次中的最大长度进行填充）

        参数：
            encoded_inputs:
                标记化输入的字典（`List[int]`）或标记化输入的批次（`List[List[int]]`）。
            max_length: 返回列表的最大长度以及可选的填充长度（见下文）。
                将通过考虑特殊标记来截断。
            padding_strategy: 填充策略，用于填充。

                - PaddingStrategy.LONGEST 填充到批次中最长的序列
                - PaddingStrategy.MAX_LENGTH: 填充到最大长度（默认）
                - PaddingStrategy.DO_NOT_PAD: 不进行填充
                标记器的填充方向由 self.padding_side 定义：

                    - 'left': 在序列的左侧进行填充
                    - 'right': 在序列的右侧进行填充
            pad_to_multiple_of: （可选）如果设置，将序列填充到提供值的倍数。
                这在启用 NVIDIA 硬件的 Tensor Core 使用时尤其有用，计算能力 `>= 7.5`（Volta）。
            return_attention_mask:
                （可选）设置为 False 以避免返回注意力掩码（默认值：根据模型具体情况设置）
        """
        # 从模型默认值加载
        assert self.padding_side == "left"  # 确保填充方向为左侧

        required_input = encoded_inputs[self.model_input_names[0]]  # 获取所需的输入数据
        seq_length = len(required_input)  # 计算输入序列的长度

        if padding_strategy == PaddingStrategy.LONGEST:  # 如果填充策略为最长
            max_length = len(required_input)  # 设置最大长度为输入的长度

        # 如果 max_length 和 pad_to_multiple_of 都被定义且 max_length 不是 pad_to_multiple_of 的倍数
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            # 将 max_length 调整为 pad_to_multiple_of 的下一个倍数
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        # 判断是否需要填充：填充策略不为不填充且输入长度不等于最大长度
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # 如果没有注意力掩码，则初始化注意力掩码
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length  # 填充为1，表示有效的输入

        # 如果没有位置 ID，则初始化位置 ID
        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))  # 填充为从0到序列长度的范围

        # 如果需要填充
        if needs_to_be_padded:
            difference = max_length - len(required_input)  # 计算需要填充的长度

            # 如果存在注意力掩码，则在前面填充0
            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            # 如果存在位置 ID，则在前面填充0
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
            # 在输入数据前面填充 pad_token_id
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs  # 返回填充后的输入数据
```