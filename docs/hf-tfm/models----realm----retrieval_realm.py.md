# `.\transformers\models\realm\retrieval_realm.py`

```
# 设置文件编码为 UTF-8
# 版权声明，作者REALM和HuggingFace Inc.团队
# 根据Apache许可证版本2.0许可使用此文件
# 只有在符合许可证的情况下才能使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则按原样分发的软件
# 基于“按原样”提供，没有任何担保或条件
# 无论是明示的还是暗示的，也没有任何保证或条件
# 详细内容请参阅许可证，规定可能的语言以及
# 许可证限制。
"""REALM Retriever model implementation."""

# 导入所需的模块
import os
from typing import Optional, Union
import numpy as np
from huggingface_hub import hf_hub_download
from ... import AutoTokenizer
from ...utils import logging

# 定义一个全局变量
# 用于存储块记录的文件名
_REALM_BLOCK_RECORDS_FILENAME = "block_records.npy"

# 获取logger
logger = logging.get_logger(__name__)

# 将TensorFlow记录转换为numpy数组
def convert_tfrecord_to_np(block_records_path: str, num_block_records: int) -> np.ndarray:
    import tensorflow.compat.v1 as tf

    # 读取TFRecord文件中的数据集
    blocks_dataset = tf.data.TFRecordDataset(block_records_path, buffer_size=512 * 1024 * 1024)
    # 将数据集分成指定大小的批次
    blocks_dataset = blocks_dataset.batch(num_block_records, drop_remainder=True)
    # 将第一个批次转换为numpy数组
    np_record = next(blocks_dataset.take(1).as_numpy_iterator())

    return np_record

# 定义一个ScaNNSearcher类
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

        from scann.scann_ops.py.scann_ops_pybind import builder as Builder

        # 使用scann库构建搜索器
        builder = Builder(db=db, num_neighbors=num_neighbors, distance_measure="dot_product")
        builder = builder.tree(
            num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=training_sample_size
        )
        builder = builder.score_ah(dimensions_per_block=dimensions_per_block)

        # 构建搜索器实例
        self.searcher = builder.build()

    def search_batched(self, question_projection):
        # 使用搜索器查询输入问题
        retrieved_block_ids, _ = self.searcher.search_batched(question_projection.detach().cpu())
        # 将检索到的块ID转换为int64类型
        return retrieved_block_ids.astype("int64")

# 定义一个RealmRetriever类
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
        super().__init__()
        # 初始化实例属性
        self.block_records = block_records
        self.tokenizer = tokenizer
    # 调用函数，输入检索到的块的索引、问题输入的 ID、答案 ID、最大长度和返回的张量格式
    def __call__(self, retrieved_block_ids, question_input_ids, answer_ids, max_length=None, return_tensors="pt"):
        # 从块记录中取出检索到的块
        retrieved_blocks = np.take(self.block_records, indices=retrieved_block_ids, axis=0)

        # 解码问题输入的 ID，得到问题文本
        question = self.tokenizer.decode(question_input_ids[0], skip_special_tokens=True)

        text = []
        text_pair = []
        # 遍历检索到的块，将问题文本和块文本组成文本对
        for retrieved_block in retrieved_blocks:
            text.append(question)
            text_pair.append(retrieved_block.decode())

        # 使用 tokenizer 将文本对编码成输入形式
        concat_inputs = self.tokenizer(
            text, text_pair, padding=True, truncation=True, return_special_tokens_mask=True, max_length=max_length
        )
        # 将编码后的输入转换为张量格式
        concat_inputs_tensors = concat_inputs.convert_to_tensors(return_tensors)

        # 如果有答案 ID，调用 block_has_answer 函数，并返回结果与张量
        if answer_ids is not None:
            return self.block_has_answer(concat_inputs, answer_ids) + (concat_inputs_tensors,)
        # 如果没有答案 ID，返回空值和张量
        else:
            return (None, None, None, concat_inputs_tensors)

    @classmethod
    # 从预训练模型中加载实例
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *init_inputs, **kwargs):
        # 判断预训练模型路径是目录还是文件
        if os.path.isdir(pretrained_model_name_or_path):
            block_records_path = os.path.join(pretrained_model_name_or_path, _REALM_BLOCK_RECORDS_FILENAME)
        else:
            # 如果是文件，则下载块记录文件
            block_records_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=_REALM_BLOCK_RECORDS_FILENAME, **kwargs
            )
        # 加载块记录
        block_records = np.load(block_records_path, allow_pickle=True)

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)

        return cls(block_records, tokenizer)

    # 保存预训练模型
    def save_pretrained(self, save_directory):
        # 保存块记录
        np.save(os.path.join(save_directory, _REALM_BLOCK_RECORDS_FILENAME), self.block_records)
        # 保存 tokenizer
        self.tokenizer.save_pretrained(save_directory)
    # 检查检索到的块是否包含答案
    def block_has_answer(self, concat_inputs, answer_ids):
        """check if retrieved_blocks has answers."""
        # 保存检索到的块是否包含答案的结果
        has_answers = []
        # 保存每个检索到的块中答案的起始位置
        start_pos = []
        # 保存每个检索到的块中答案的结束位置
        end_pos = []
        # 保存最大答案数量
        max_answers = 0

        # 遍历每个输入 ID
        for input_id in concat_inputs.input_ids:
            # 将 input_id 转换为列表
            input_id_list = input_id.tolist()
            # 查找第一个 [SEP] 标记的索引位置
            first_sep_idx = input_id_list.index(self.tokenizer.sep_token_id)
            # 查找第二个 [SEP] 标记的索引位置
            second_sep_idx = first_sep_idx + 1 + input_id_list[first_sep_idx + 1 :].index(self.tokenizer.sep_token_id)

            # 初始化答案起始位置列表
            start_pos.append([])
            # 初始化答案结束位置列表
            end_pos.append([])
            # 遍历答案列表
            for answer in answer_ids:
                for idx in range(first_sep_idx + 1, second_sep_idx):
                    # 如果答案的首个词在输入列表中
                    if answer[0] == input_id_list[idx]:
                        # 如果答案在输入列表中存在且和答案相同
                        if input_id_list[idx : idx + len(answer)] == answer:
                            # 添加答案的起始位置和结束位置
                            start_pos[-1].append(idx)
                            end_pos[-1].append(idx + len(answer) - 1)

            # 如果没有找到答案，则添加 False 到 has_answers 中
            if len(start_pos[-1]) == 0:
                has_answers.append(False)
            # 如果找到答案，则添加 True 到 has_answers 中
            else:
                has_answers.append(True)
                # 更新最大答案数量
                if len(start_pos[-1]) > max_answers:
                    max_answers = len(start_pos[-1])

        # 对于每个 start_pos_ 和 end_pos_ 列表，将 -1 添加到列表中，使其长度等于 max_answers
        for start_pos_, end_pos_ in zip(start_pos, end_pos):
            if len(start_pos_) < max_answers:
                padded = [-1] * (max_answers - len(start_pos_))
                start_pos_ += padded
                end_pos_ += padded
        # 返回结果
        return has_answers, start_pos, end_pos
```