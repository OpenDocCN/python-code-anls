# `.\PaddleOCR\ppocr\postprocess\table_postprocess.py`

```
# 导入必要的库
import numpy as np
import paddle

# 导入自定义的rec_postprocess模块中的AttnLabelDecode类
from .rec_postprocess import AttnLabelDecode

# 定义一个TableLabelDecode类，继承自AttnLabelDecode类
class TableLabelDecode(AttnLabelDecode):
    """  """

    # 初始化方法，接收字符字典路径和是否合并无跨度结构等参数
    def __init__(self,
                 character_dict_path,
                 merge_no_span_structure=False,
                 **kwargs):
        # 初始化一个空列表来存储字符字典
        dict_character = []
        # 打开字符字典文件，读取每一行并添加到dict_character列表中
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                dict_character.append(line)

        # 如果需要合并无跨度结构
        if merge_no_span_structure:
            # 如果"<td></td>"不在字符字典中，则添加进去
            if "<td></td>" not in dict_character:
                dict_character.append("<td></td>")
            # 如果"<td>"在字符字典中，则移除
            if "<td>" in dict_character:
                dict_character.remove("<td>")

        # 添加特殊字符到字符字典中
        dict_character = self.add_special_char(dict_character)
        # 初始化一个空字典来存储字符到索引的映射关系
        self.dict = {}
        # 遍历字符字典，将字符和对应的索引添加到self.dict中
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        # 将字符字典和特殊token保存到类属性中
        self.character = dict_character
        self.td_token = ['<td>', '<td', '<td></td>']
    # 定义一个类的调用方法，接受预测结果和批次数据作为参数
    def __call__(self, preds, batch=None):
        # 获取预测结果中的结构概率和边界框预测
        structure_probs = preds['structure_probs']
        bbox_preds = preds['loc_preds']
        # 如果结构概率是 PaddlePaddle 的张量，则转换为 NumPy 数组
        if isinstance(structure_probs, paddle.Tensor):
            structure_probs = structure_probs.numpy()
        # 如果边界框预测是 PaddlePaddle 的张量，则转换为 NumPy 数组
        if isinstance(bbox_preds, paddle.Tensor):
            bbox_preds = bbox_preds.numpy()
        # 获取批次数据中的形状列表
        shape_list = batch[-1]
        # 调用解码方法，传入结构概率、边界框预测和形状列表，得到解码结果
        result = self.decode(structure_probs, bbox_preds, shape_list)
        # 如果批次数据只包含形状信息，则直接返回解码结果
        if len(batch) == 1:  # only contains shape
            return result

        # 调用解码标签方法，传入批次数据，得到标签解码结果
        label_decode_result = self.decode_label(batch)
        # 返回解码结果和标签解码结果
        return result, label_decode_result
    # 将文本标签转换为文本索引
    def decode(self, structure_probs, bbox_preds, shape_list):
        """convert text-label into text-index.
        """
        # 获取需要忽略的标记
        ignored_tokens = self.get_ignored_tokens()
        # 获取结束标记的索引
        end_idx = self.dict[self.end_str]

        # 获取结构概率最大值的索引
        structure_idx = structure_probs.argmax(axis=2)
        # 获取结构概率的最大值
        structure_probs = structure_probs.max(axis=2)

        # 初始化结构列表和边界框列表
        structure_batch_list = []
        bbox_batch_list = []
        # 获取批处理大小
        batch_size = len(structure_idx)
        # 遍历每个批次
        for batch_idx in range(batch_size):
            structure_list = []
            bbox_list = []
            score_list = []
            # 遍历每个索引
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                # 如果索引大于0且字符索引为结束索引，则跳出循环
                if idx > 0 and char_idx == end_idx:
                    break
                # 如果字符索引在忽略的标记中，则继续下一次循环
                if char_idx in ignored_tokens:
                    continue
                # 获取字符对应的文本
                text = self.character[char_idx]
                # 如果文本在特殊标记中
                if text in self.td_token:
                    # 获取边界框预测值
                    bbox = bbox_preds[batch_idx, idx]
                    # 解码边界框
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)
                structure_list.append(text)
                score_list.append(structure_probs[batch_idx, idx])
            structure_batch_list.append([structure_list, np.mean(score_list)])
            bbox_batch_list.append(np.array(bbox_list))
        # 返回结果字典
        result = {
            'bbox_batch_list': bbox_batch_list,
            'structure_batch_list': structure_batch_list,
        }
        return result
    # 将文本标签转换为文本索引
    def decode_label(self, batch):
        """convert text-label into text-index.
        """
        # 获取结构索引
        structure_idx = batch[1]
        # 获取真实边界框列表
        gt_bbox_list = batch[2]
        # 获取形状列表
        shape_list = batch[-1]
        # 获取需要忽略的标记
        ignored_tokens = self.get_ignored_tokens()
        # 获取结束索引
        end_idx = self.dict[self.end_str]

        # 初始化结构批次列表和边界框批次列表
        structure_batch_list = []
        bbox_batch_list = []
        # 获取批次大小
        batch_size = len(structure_idx)
        # 遍历每个批次
        for batch_idx in range(batch_size):
            # 初始化结构列表和边界框列表
            structure_list = []
            bbox_list = []
            # 遍历每个结构索引
            for idx in range(len(structure_idx[batch_idx])):
                # 获取字符索引
                char_idx = int(structure_idx[batch_idx][idx])
                # 如果不是第一个字符且字符索引为结束索引，则跳出循环
                if idx > 0 and char_idx == end_idx:
                    break
                # 如果字符索引在忽略标记中，则继续下一个循环
                if char_idx in ignored_tokens:
                    continue
                # 将字符索引转换为字符并添加到结构列表中
                structure_list.append(self.character[char_idx])

                # 获取当前字符对应的边界框
                bbox = gt_bbox_list[batch_idx][idx]
                # 如果边界框不全为0
                if bbox.sum() != 0:
                    # 解码边界框并添加到边界框列表中
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)
            # 将当前批次的结构列表和边界框列表添加到对应的批次列表中
            structure_batch_list.append(structure_list)
            bbox_batch_list.append(bbox_list)
        # 返回结果字典
        result = {
            'bbox_batch_list': bbox_batch_list,
            'structure_batch_list': structure_batch_list,
        }
        return result

    # 解码边界框
    def _bbox_decode(self, bbox, shape):
        # 获取形状参数
        h, w, ratio_h, ratio_w, pad_h, pad_w = shape
        # 根据形状参数对边界框进行解码
        bbox[0::2] *= w
        bbox[1::2] *= h
        return bbox
class TableMasterLabelDecode(TableLabelDecode):
    """ 表示一个用于处理表格标签解码的类，继承自TableLabelDecode类 """

    def __init__(self,
                 character_dict_path,
                 box_shape='ori',
                 merge_no_span_structure=True,
                 **kwargs):
        # 初始化方法，接受字符字典路径、框形状、是否合并无跨度结构等参数
        super(TableMasterLabelDecode, self).__init__(character_dict_path,
                                                     merge_no_span_structure)
        # 调用父类的初始化方法
        self.box_shape = box_shape
        # 设置框形状属性
        assert box_shape in [
            'ori', 'pad'
        ], 'The shape used for box normalization must be ori or pad'
        # 断言框形状只能是'ori'或'pad'

    def add_special_char(self, dict_character):
        # 添加特殊字符方法，接受字符字典作为参数
        self.beg_str = '<SOS>'
        self.end_str = '<EOS>'
        self.unknown_str = '<UKN>'
        self.pad_str = '<PAD>'
        # 设置特殊字符属性
        dict_character = dict_character
        # 将参数赋值给局部变量
        dict_character = dict_character + [
            self.unknown_str, self.beg_str, self.end_str, self.pad_str
        ]
        # 将特殊字符添加到字符字典中
        return dict_character
        # 返回更新后的字符字典

    def get_ignored_tokens(self):
        # 获取被忽略的标记方法
        pad_idx = self.dict[self.pad_str]
        start_idx = self.dict[self.beg_str]
        end_idx = self.dict[self.end_str]
        unknown_idx = self.dict[self.unknown_str]
        # 获取特殊字符的索引
        return [start_idx, end_idx, pad_idx, unknown_idx]
        # 返回被忽略的标记列表

    def _bbox_decode(self, bbox, shape):
        # 框解码方法，接受框和形状作为参数
        h, w, ratio_h, ratio_w, pad_h, pad_w = shape
        # 解包形状参数
        if self.box_shape == 'pad':
            h, w = pad_h, pad_w
        # 如果框形状是'pad'，则更新高度和宽度
        bbox[0::2] *= w
        bbox[1::2] *= h
        bbox[0::2] /= ratio_w
        bbox[1::2] /= ratio_h
        # 根据比率和形状更新框的坐标
        x, y, w, h = bbox
        x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
        # 计算框的四个顶点坐标
        bbox = np.array([x1, y1, x2, y2])
        # 将四个顶点坐标组成数组
        return bbox
        # 返回更新后的框
```