# `jieba\jieba\lac_small\reader_small.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权使用该文件；
# 除非符合许可证的规定，否则不得使用该文件；
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
"""
# 导入所需的库
import os
import __future__
import io
import paddle
import paddle.fluid as fluid

# 从文件中加载键值字典
def load_kv_dict(dict_path,
                 reverse=False,
                 delimiter="\t",
                 key_func=None,
                 value_func=None):
    """
    Load key-value dict from file
    """
    # 初始化结果字典
    result_dict = {}
    # 逐行读取文件内容
    for line in io.open(dict_path, "r", encoding='utf8'):
        # 去除换行符并按分隔符拆分行内容
        terms = line.strip("\n").split(delimiter)
        # 如果拆分后的内容不是键值对，则跳过
        if len(terms) != 2:
            continue
        # 根据 reverse 参数确定键值对的顺序
        if reverse:
            value, key = terms
        else:
            key, value = terms
        # 如果键已存在于字典中，则抛出 KeyError
        if key in result_dict:
            raise KeyError("key duplicated with [%s]" % (key))
        # 如果存在键处理函数，则对键进行处理
        if key_func:
            key = key_func(key)
        # 如果存在值处理函数，则对值进行处理
        if value_func:
            value = value_func(value)
        # 将处理后的键值对添加到结果字典中
        result_dict[key] = value
    # 返回结果字典
    return result_dict

# 定义数据集类
class Dataset(object):
    """data reader"""
    # 初始化方法，用于读取字典文件并加载到内存中
    def __init__(self):
        # 获取当前文件的绝对路径
        basepath = os.path.abspath(__file__)
        # 获取当前文件所在文件夹路径
        folder = os.path.dirname(basepath)
        # 拼接得到字典文件的路径
        word_dict_path = os.path.join(folder, "word.dic")
        label_dict_path = os.path.join(folder, "tag.dic")
        # 加载字典文件内容到内存中，构建单词到ID的映射字典和ID到单词的映射字典
        self.word2id_dict = load_kv_dict(
            word_dict_path, reverse=True, value_func=int)
        self.id2word_dict = load_kv_dict(word_dict_path)
        # 加载字典文件内容到内存中，构建标签到ID的映射字典和ID到标签的映射字典
        self.label2id_dict = load_kv_dict(
            label_dict_path, reverse=True, value_func=int)
        self.id2label_dict = load_kv_dict(label_dict_path)
    
    @property
    def vocab_size(self):
        """获取词汇表大小"""
        return max(self.word2id_dict.values()) + 1
    
    @property
    def num_labels(self):
        """获取标签数量"""
        return max(self.label2id_dict.values()) + 1

    def word_to_ids(self, words):
        """将单词转换为单词索引"""
        word_ids = []
        for word in words:
            # 如果单词不在单词到ID的映射字典中，则用"OOV"代替
            if word not in self.word2id_dict:
                word = "OOV"
            # 获取单词对应的ID并添加到列表中
            word_id = self.word2id_dict[word]
            word_ids.append(word_id)
        return word_ids
   
    def label_to_ids(self, labels):
        """将标签转换为标签索引"""
        label_ids = []
        for label in labels:
            # 如果标签不在标签到ID的映射字典中，则用"O"代替
            if label not in self.label2id_dict:
                label = "O"
            # 获取标签对应的ID并添加到列表中
            label_id = self.label2id_dict[label]
            label_ids.append(label_id)
        return label_ids

    def get_vars(self,str1):
        # 去除字符串两端的空格
        words = str1.strip()
        # 将单词转换为单词索引
        word_ids = self.word_to_ids(words)
        return word_ids
```