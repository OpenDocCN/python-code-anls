# `jieba\jieba\lac_small\utils.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证，版本 2.0 进行许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”分发的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
"""
# 导入所需的库
from __future__ import print_function
import os
import sys
import numpy as np
import paddle.fluid as fluid
import io

# 定义一个函数，将字符串转换为布尔值
def str2bool(v):
    """
    argparse 不支持在 Python 中使用 True 或 False
    """
    return v.lower() in ("true", "t", "1")

# 解析结果的函数
def parse_result(words, crf_decode, dataset):
    """ parse result """
    # 获取偏移列表
    offset_list = (crf_decode.lod())[0]
    # 将输入的单词转换为数组
    words = np.array(words)
    # 将 CRF 解码结果转换为数组
    crf_decode = np.array(crf_decode)
    # 获取批处理大小
    batch_size = len(offset_list) - 1
    # 遍历每个句子的索引，范围是 batch_size
    for sent_index in range(batch_size):
        # 获取当前句子在单词列表中的起始和结束位置
        begin, end = offset_list[sent_index], offset_list[sent_index + 1]
        # 初始化句子列表
        sent=[]
        # 遍历当前句子的单词列表
        for id in words[begin:end]:
            # 如果单词是OOV（Out of Vocabulary），则添加空格到句子中
            if dataset.id2word_dict[str(id[0])]=='OOV':
                sent.append(' ')
            else:
                # 否则添加单词到句子中
                sent.append(dataset.id2word_dict[str(id[0])])
        # 获取当前句子的标签列表
        tags = [
            dataset.id2label_dict[str(id[0])] for id in crf_decode[begin:end]
        ]

        # 初始化输出句子和标签列表
        sent_out = []
        tags_out = []
        parital_word = ""
        # 遍历标签和单词，将连续的单词合并为一个词
        for ind, tag in enumerate(tags):
            # 对于第一个单词
            if parital_word == "":
                parital_word = sent[ind]
                tags_out.append(tag.split('-')[0])
                continue

            # 对于词的开头
            if tag.endswith("-B") or (tag == "O" and tags[ind - 1] != "O"):
                sent_out.append(parital_word)
                tags_out.append(tag.split('-')[0])
                parital_word = sent[ind]
                continue

            parital_word += sent[ind]

        # 添加最后一个词，除非标签列表为空
        if len(sent_out) < len(tags_out):
            sent_out.append(parital_word)
    # 返回合并后的句子和标签列表
    return sent_out,tags_out
# 解析填充结果
def parse_padding_result(words, crf_decode, seq_lens, dataset):
    # 压缩 words 数组，去除多余的维度
    words = np.squeeze(words)
    # 获取批处理大小
    batch_size = len(seq_lens)

    # 存储批处理结果
    batch_out = []
    for sent_index in range(batch_size):

        # 存储句子
        sent=[]
        for id in words[begin:end]:
            # 如果是未登录词，则添加空格
            if dataset.id2word_dict[str(id[0])]=='OOV':
                sent.append(' ')
            else:
                sent.append(dataset.id2word_dict[str(id[0])]
        # 获取标签
        tags = [
            dataset.id2label_dict[str(id)]
            for id in crf_decode[sent_index][1:seq_lens[sent_index] - 1]
        ]

        sent_out = []
        tags_out = []
        parital_word = ""
        for ind, tag in enumerate(tags):
            # 对于第一个词
            if parital_word == "":
                parital_word = sent[ind]
                tags_out.append(tag.split('-')[0])
                continue

            # 对于词的开头
            if tag.endswith("-B") or (tag == "O" and tags[ind - 1] != "O"):
                sent_out.append(parital_word)
                tags_out.append(tag.split('-')[0])
                parital_word = sent[ind]
                continue

            parital_word += sent[ind]

        # 添加最后一个词，除非 len(tags)=0
        if len(sent_out) < len(tags_out):
            sent_out.append(parital_word)

        batch_out.append([sent_out, tags_out])
    return batch_out


# 初始化检查点
def init_checkpoint(exe, init_checkpoint_path, main_program):
    # 检查检查点路径是否存在
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        # 检查是否存在持久化变量
        if not fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    # 加载变量
    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)
```