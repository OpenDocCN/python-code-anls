# `.\PaddleOCR\ppocr\data\imaug\vqa\token\vqa_re_convert.py`

```
# 版权声明
# 版权所有 (c) 2022 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本:
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言管理权限和限制。

# 导入 NumPy 库
import numpy as np

# 定义一个类 TensorizeEntitiesRelations
class TensorizeEntitiesRelations(object):
    # 初始化方法，设置最大序列长度和推断模式
    def __init__(self, max_seq_len=512, infer_mode=False, **kwargs):
        self.max_seq_len = max_seq_len
        self.infer_mode = infer_mode
    # 定义一个类的方法，用于处理数据
    def __call__(self, data):
        # 从数据中获取实体信息和关系信息
        entities = data['entities']
        relations = data['relations']

        # 创建一个新的实体数组，初始化为-1
        entities_new = np.full(
            shape=[self.max_seq_len + 1, 3], fill_value=-1, dtype='int64')
        # 将实体数量信息存储到数组中
        entities_new[0, 0] = len(entities['start'])
        entities_new[0, 1] = len(entities['end'])
        entities_new[0, 2] = len(entities['label'])
        # 将实体的起始位置、结束位置和标签信息存储到数组中
        entities_new[1:len(entities['start']) + 1, 0] = np.array(entities[
            'start'])
        entities_new[1:len(entities['end']) + 1, 1] = np.array(entities['end'])
        entities_new[1:len(entities['label']) + 1, 2] = np.array(entities[
            'label'])

        # 创建一个新的关系数组，初始化为-1
        relations_new = np.full(
            shape=[self.max_seq_len * self.max_seq_len + 1, 2],
            fill_value=-1,
            dtype='int64')
        # 将关系数量信息存储到数组中
        relations_new[0, 0] = len(relations['head'])
        relations_new[0, 1] = len(relations['tail'])
        # 将关系的头部和尾部信息存储到数组中
        relations_new[1:len(relations['head']) + 1, 0] = np.array(relations[
            'head'])
        relations_new[1:len(relations['tail']) + 1, 1] = np.array(relations[
            'tail'])

        # 更新数据中的实体和关系信息为新的数组
        data['entities'] = entities_new
        data['relations'] = relations_new
        # 返回更新后的数据
        return data
```