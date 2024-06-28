# `.\models\realm\retrieval_realm.py`

```py
# coding=utf-8
# Copyright 2022 The REALM authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""REALM Retriever model implementation."""

import os
from typing import Optional, Union

import numpy as np
from huggingface_hub import hf_hub_download

from ... import AutoTokenizer
from ...utils import logging


_REALM_BLOCK_RECORDS_FILENAME = "block_records.npy"


logger = logging.get_logger(__name__)


class ScaNNSearcher:
    """Note that ScaNNSearcher cannot currently be used within the model. In future versions, it might however be included."""

    def __init__(
        self,
        db,
        num_neighbors,
        dimensions_per_block=2,
        num_leaves=1000,
        num_leaves_to_search=100,
        training_sample_size=100000,
    ):
        """Build scann searcher."""
        
        # Import the necessary modules for constructing a SCANN searcher
        from scann.scann_ops.py.scann_ops_pybind import builder as Builder
        
        # Initialize the builder with database and search parameters
        builder = Builder(db=db, num_neighbors=num_neighbors, distance_measure="dot_product")
        
        # Configure the tree parameters
        builder = builder.tree(
            num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=training_sample_size
        )
        
        # Configure scoring parameters
        builder = builder.score_ah(dimensions_per_block=dimensions_per_block)
        
        # Build the searcher object
        self.searcher = builder.build()

    def search_batched(self, question_projection):
        """Perform batched search using the constructed SCANN searcher."""
        
        # Perform batched search and retrieve block IDs
        retrieved_block_ids, _ = self.searcher.search_batched(question_projection.detach().cpu())
        
        # Return retrieved block IDs as int64
        return retrieved_block_ids.astype("int64")


class RealmRetriever:
    """The retriever of REALM outputting the retrieved evidence block and whether the block has answers as well as answer
    positions."

        Parameters:
            block_records (`np.ndarray`):
                A numpy array which cantains evidence texts.
            tokenizer ([`RealmTokenizer`]):
                The tokenizer to encode retrieved texts.
    """

    def __init__(self, block_records, tokenizer):
        """Initialize RealmRetriever with block records and tokenizer."""
        
        # Initialize superclass
        super().__init__()
        
        # Store the provided block records
        self.block_records = block_records
        
        # Store the provided tokenizer
        self.tokenizer = tokenizer
    # 定义类的实例方法，用于生成压缩块的输入
    def __call__(self, retrieved_block_ids, question_input_ids, answer_ids, max_length=None, return_tensors="pt"):
        # 从 self.block_records 中按索引提取检索到的块
        retrieved_blocks = np.take(self.block_records, indices=retrieved_block_ids, axis=0)

        # 根据问题输入的 token IDs 解码出文本问题
        question = self.tokenizer.decode(question_input_ids[0], skip_special_tokens=True)

        # 初始化文本列表
        text = []
        text_pair = []

        # 遍历每个检索到的块
        for retrieved_block in retrieved_blocks:
            # 将问题文本添加到 text 列表
            text.append(question)
            # 将检索到的块解码并添加到 text_pair 列表
            text_pair.append(retrieved_block.decode())

        # 使用 tokenizer 处理 text 和 text_pair，进行拼接和填充等预处理
        concat_inputs = self.tokenizer(
            text, text_pair, padding=True, truncation=True, return_special_tokens_mask=True, max_length=max_length
        )

        # 将处理后的输入转换为张量
        concat_inputs_tensors = concat_inputs.convert_to_tensors(return_tensors)

        # 如果提供了答案 IDs，则调用 block_has_answer 方法计算答案和返回拼接输入的张量
        if answer_ids is not None:
            return self.block_has_answer(concat_inputs, answer_ids) + (concat_inputs_tensors,)
        else:
            # 否则返回空元组和拼接输入的张量
            return (None, None, None, concat_inputs_tensors)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *init_inputs, **kwargs):
        # 如果预训练模型路径是一个目录，则拼接出块记录文件的路径
        if os.path.isdir(pretrained_model_name_or_path):
            block_records_path = os.path.join(pretrained_model_name_or_path, _REALM_BLOCK_RECORDS_FILENAME)
        else:
            # 否则从 Hugging Face Hub 下载模型文件并指定块记录文件名
            block_records_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=_REALM_BLOCK_RECORDS_FILENAME, **kwargs
            )
        # 加载块记录文件为 numpy 数组
        block_records = np.load(block_records_path, allow_pickle=True)

        # 从预训练模型加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)

        # 返回当前类的实例，初始化时传入加载的块记录和 tokenizer
        return cls(block_records, tokenizer)

    # 实例方法，用于将块记录和 tokenizer 保存到指定目录
    def save_pretrained(self, save_directory):
        # 保存块记录文件为 numpy 格式
        np.save(os.path.join(save_directory, _REALM_BLOCK_RECORDS_FILENAME), self.block_records)
        # 保存 tokenizer 到指定目录
        self.tokenizer.save_pretrained(save_directory)
    # 检查给定的拼接输入中是否包含答案
    def block_has_answer(self, concat_inputs, answer_ids):
        """check if retrieved_blocks has answers."""
        # 用于存储每个拼接输入是否含有答案的布尔列表
        has_answers = []
        # 用于存储每个拼接输入中所有答案起始位置的列表
        start_pos = []
        # 用于存储每个拼接输入中所有答案结束位置的列表
        end_pos = []
        # 记录每个拼接输入中最多的答案数
        max_answers = 0

        # 遍历每个拼接输入的input_ids
        for input_id in concat_inputs.input_ids:
            # 将input_id转换为Python列表
            input_id_list = input_id.tolist()
            # 查找第一个[SEP]标记的索引位置
            first_sep_idx = input_id_list.index(self.tokenizer.sep_token_id)
            # 查找第二个[SEP]标记的索引位置，限定搜索范围从第一个[SEP]之后开始
            second_sep_idx = first_sep_idx + 1 + input_id_list[first_sep_idx + 1:].index(self.tokenizer.sep_token_id)

            # 初始化存储当前拼接输入答案起始和结束位置的列表
            start_pos.append([])
            end_pos.append([])
            # 遍历每个答案id列表中的答案
            for answer in answer_ids:
                # 在第一个和第二个[SEP]之间查找答案的起始位置
                for idx in range(first_sep_idx + 1, second_sep_idx):
                    if answer[0] == input_id_list[idx]:
                        # 检查是否在当前位置开始的连续序列与答案匹配
                        if input_id_list[idx: idx + len(answer)] == answer:
                            # 将找到的答案起始和结束位置添加到列表中
                            start_pos[-1].append(idx)
                            end_pos[-1].append(idx + len(answer) - 1)

            # 如果当前拼接输入没有找到答案，则记录为False，否则记录为True
            if len(start_pos[-1]) == 0:
                has_answers.append(False)
            else:
                has_answers.append(True)
                # 更新当前拼接输入中最大答案数量
                if len(start_pos[-1]) > max_answers:
                    max_answers = len(start_pos[-1])

        # 对于没有答案的拼接输入，在start_pos和end_pos中填充-1以对齐最大答案数量
        for start_pos_, end_pos_ in zip(start_pos, end_pos):
            if len(start_pos_) < max_answers:
                padded = [-1] * (max_answers - len(start_pos_))
                start_pos_ += padded
                end_pos_ += padded

        # 返回结果：每个拼接输入是否含有答案的布尔列表，每个拼接输入中答案起始和结束位置的列表
        return has_answers, start_pos, end_pos
```