# `.\PaddleOCR\ppocr\data\imaug\vqa\token\vqa_token_chunk.py`

```py
# 版权声明
#
# 版权所有 (c) 2021 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”基础分发的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言下的权限和限制。
#

# 导入 defaultdict 类
from collections import defaultdict

# 定义 VQASerTokenChunk 类
class VQASerTokenChunk(object):
    # 初始化函数
    def __init__(self, max_seq_len=512, infer_mode=False, **kwargs):
        # 设置最大序列长度和推断模式
        self.max_seq_len = max_seq_len
        self.infer_mode = infer_mode

    # 调用函数
    def __call__(self, data):
        # 初始化编码后的输入列表
        encoded_inputs_all = []
        # 获取输入数据的序列长度
        seq_len = len(data['input_ids'])
        # 根据最大序列长度对数据进行分块处理
        for index in range(0, seq_len, self.max_seq_len):
            chunk_beg = index
            chunk_end = min(index + self.max_seq_len, seq_len)
            # 初始化编码后的单个示例
            encoded_inputs_example = {}
            # 遍历数据的键
            for key in data:
                # 如果键在指定列表中
                if key in [
                        'label', 'input_ids', 'labels', 'token_type_ids',
                        'bbox', 'attention_mask'
                ]:
                    # 根据推断模式和键的值进行处理
                    if self.infer_mode and key == 'labels':
                        encoded_inputs_example[key] = data[key]
                    else:
                        encoded_inputs_example[key] = data[key][chunk_beg:
                                                                chunk_end]
                else:
                    encoded_inputs_example[key] = data[key]

            # 将处理后的示例添加到编码后的输入列表中
            encoded_inputs_all.append(encoded_inputs_example)
        
        # 如果编码后的输入列表为空，则返回 None
        if len(encoded_inputs_all) == 0:
            return None
        # 返回第一个编码后的示例
        return encoded_inputs_all[0]

# 定义 VQAReTokenChunk 类
class VQAReTokenChunk(object):
    # 初始化方法，设置最大序列长度、实体标签和推断模式
    def __init__(self,
                 max_seq_len=512,
                 entities_labels=None,
                 infer_mode=False,
                 **kwargs):
        # 设置最大序列长度
        self.max_seq_len = max_seq_len
        # 如果实体标签为空，则使用默认的实体标签字典
        self.entities_labels = {
            'HEADER': 0,
            'QUESTION': 1,
            'ANSWER': 2
        } if entities_labels is None else entities_labels
        # 设置推断模式
        self.infer_mode = infer_mode
    
    # 重新格式化数据，将数据重新组织为字典形式
    def reformat(self, data):
        # 创建一个默认值为列表的字典
        new_data = defaultdict(list)
        # 遍历数据中的每个项
        for item in data:
            # 遍历每个项中的键值对
            for k, v in item.items():
                # 将值按键分类存储到新的字典中
                new_data[k].append(v)
        # 返回重新组织后的数据字典
        return new_data
```