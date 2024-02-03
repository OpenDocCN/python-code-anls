# `.\PaddleOCR\ppocr\data\imaug\label_ops.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import numpy as np
import string
from shapely.geometry import LineString, Point, Polygon
import json
import copy
from random import sample

# 导入日志记录器
from ppocr.utils.logging import get_logger
# 导入图像增强相关模块
from ppocr.data.imaug.vqa.augment import order_by_tbyx

# 定义一个类，用于将分类标签编码为数字
class ClsLabelEncode(object):
    def __init__(self, label_list, **kwargs):
        self.label_list = label_list

    def __call__(self, data):
        label = data['label']
        # 如果标签不在标签列表中，则返回空
        if label not in self.label_list:
            return None
        # 将标签转换为对应的索引
        label = self.label_list.index(label)
        data['label'] = label
        return data

# 定义一个类，用于处理检测任务的标签编码
class DetLabelEncode(object):
    def __init__(self, **kwargs):
        pass
    # 定义一个方法，用于处理输入的数据
    def __call__(self, data):
        # 从数据中获取标签信息
        label = data['label']
        # 将标签信息转换为 JSON 格式
        label = json.loads(label)
        # 获取标签中包含的框的数量
        nBox = len(label)
        # 初始化空列表用于存储框的坐标、文本和文本标签
        boxes, txts, txt_tags = [], [], []
        # 遍历每个框
        for bno in range(0, nBox):
            # 获取框的坐标信息
            box = label[bno]['points']
            # 获取框的文本信息
            txt = label[bno]['transcription']
            # 将框的坐标和文本分别存储到对应的列表中
            boxes.append(box)
            txts.append(txt)
            # 判断文本是否为特定值，如果是则将对应位置的标签设置为 True，否则设置为 False
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        # 如果没有框信息，则返回 None
        if len(boxes) == 0:
            return None
        # 对框的坐标进行扩展，使其具有相同的点数
        boxes = self.expand_points_num(boxes)
        # 将框的坐标和文本标签转换为 NumPy 数组
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool_)

        # 将处理后的数据存储到输入数据中的指定字段中
        data['polys'] = boxes
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        # 返回处理后的数据
        return data

    # 定义一个方法，用于按顺时针顺序排列给定的点
    def order_points_clockwise(self, pts):
        # 初始化一个包含四个点的矩形
        rect = np.zeros((4, 2), dtype="float32")
        # 计算点的和
        s = pts.sum(axis=1)
        # 找到和最小的点和和最大的点，分别作为矩形的两个对角点
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # 删除最小和最大的点，计算剩余两个点的差值
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        # 找到差值最小和最大的点，作为矩形的另外两个对角点
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        # 返回排列好的点
        return rect

    # 定义一个方法，用于扩展框的点数使其具有相同的点数
    def expand_points_num(self, boxes):
        # 初始化最大点数为 0
        max_points_num = 0
        # 遍历每个框，找到最大的点数
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        # 初始化空列表用于存储扩展后的框
        ex_boxes = []
        # 遍历每个框，将其扩展为具有相同点数的形式
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        # 返回扩展后的框列表
        return ex_boxes
class BaseRecLabelEncode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 lower=False):

        # 初始化函数，设置最大文本长度、起始字符串、结束字符串和是否转换为小写
        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = lower

        # 如果字符字典路径为空，则使用默认字符集合
        if character_dict_path is None:
            logger = get_logger()
            logger.warning(
                "The character_dict_path is None, model can only recognize number and lower letters"
            )
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True
        else:
            # 从字符字典文件中读取字符集合
            self.character_str = []
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            # 如果使用空格字符，则添加空格到字符集合中
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
        # 添加特殊字符到字符集合中
        dict_character = self.add_special_char(dict_character)
        # 创建字符到索引的映射字典
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    # 添加特殊字符到字符集合中的方法
    def add_special_char(self, dict_character):
        return dict_character
    # 将文本标签转换为文本索引
    # 输入：
    #     text: 每个图像的文本标签。[batch_size]
    # 输出：
    #     text: 用于CTCLoss的文本索引串联。[sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
    #     length: 每个文本的长度。[batch_size]
    def encode(self, text):
        # 如果文本长度为0或大于最大文本长度，则返回None
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        # 如果需要转换为小写，则将文本转换为小写
        if self.lower:
            text = text.lower()
        text_list = []
        # 遍历文本中的每个字符
        for char in text:
            # 如果字符不在字典中，则跳过
            if char not in self.dict:
                # 记录警告信息
                # logger = get_logger()
                # logger.warning('{} is not in dict'.format(char))
                continue
            # 将字符对应的索引添加到文本列表中
            text_list.append(self.dict[char])
        # 如果文本列表为空，则返回None
        if len(text_list) == 0:
            return None
        # 返回文本列表
        return text_list
# 定义一个类 CTCLabelEncode，用于文本标签和文本索引之间的转换
class CTCLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    # 初始化方法，接受最大文本长度、字符字典路径、是否使用空格字符等参数
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(CTCLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    # 定义 __call__ 方法，用于对数据进行处理
    def __call__(self, data):
        # 获取数据中的标签文本
        text = data['label']
        # 将文本编码成索引
        text = self.encode(text)
        # 如果文本为空，则返回空
        if text is None:
            return None
        # 将文本长度存储到数据中
        data['length'] = np.array(len(text))
        # 将文本补齐到最大文本长度
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text)

        # 初始化一个全零列表，用于存储标签
        label = [0] * len(self.character)
        # 统计每个索引出现的次数
        for x in text:
            label[x] += 1
        data['label_ace'] = np.array(label)
        return data

    # 定义添加特殊字符的方法
    def add_special_char(self, dict_character):
        # 在字符字典中添加一个特殊字符 'blank'
        dict_character = ['blank'] + dict_character
        return dict_character


# 定义一个类 E2ELabelEncodeTest，继承自 BaseRecLabelEncode
class E2ELabelEncodeTest(BaseRecLabelEncode):
    # 初始化方法，接受最大文本长度、字符字典路径、是否使用空格字符等参数
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(E2ELabelEncodeTest, self).__init__(
            max_text_length, character_dict_path, use_space_char)
    # 定义一个类的方法，用于处理输入的数据
    def __call__(self, data):
        # 导入json模块，用于处理JSON格式的数据
        import json
        # 获取字典的长度
        padnum = len(self.dict)
        # 获取数据中的标签
        label = data['label']
        # 将标签转换为JSON格式
        label = json.loads(label)
        # 获取标签中包含的框的数量
        nBox = len(label)
        # 初始化空列表用于存储框的坐标、文本和文本标签
        boxes, txts, txt_tags = [], [], []
        # 遍历每个框
        for bno in range(0, nBox):
            # 获取框的坐标
            box = label[bno]['points']
            # 获取框的文本
            txt = label[bno]['transcription']
            # 将框的坐标和文本分别存储到对应的列表中
            boxes.append(box)
            txts.append(txt)
            # 判断文本是否为特定值，如果是则将对应位置的文本标签设置为True，否则设置为False
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        # 将框的坐标转换为NumPy数组，并指定数据类型为float32
        boxes = np.array(boxes, dtype=np.float32)
        # 将文本标签转换为NumPy数组，并指定数据类型为bool_
        txt_tags = np.array(txt_tags, dtype=np.bool_)
        # 将框的坐标存储到数据中的'polys'键
        data['polys'] = boxes
        # 将文本标签存储到数据中的'ignore_tags'键
        data['ignore_tags'] = txt_tags
        # 初始化临时文本列表
        temp_texts = []
        # 遍历每个文本
        for text in txts:
            # 将文本转换为小写
            text = text.lower()
            # 使用类中的encode方法对文本进行编码
            text = self.encode(text)
            # 如果编码结果为None，则返回None
            if text is None:
                return None
            # 将文本进行填充，使其长度达到self.max_text_len，填充值为padnum
            text = text + [padnum] * (self.max_text_len - len(text))  # use 36 to pad
            temp_texts.append(text)
        # 将填充后的文本列表转换为NumPy数组，并存储到数据中的'texts'键
        data['texts'] = np.array(temp_texts)
        # 返回处理后的数据
        return data
# 定义一个类 E2ELabelEncodeTrain
class E2ELabelEncodeTrain(object):
    # 初始化方法，接受任意关键字参数
    def __init__(self, **kwargs):
        # pass 表示不执行任何操作
        pass

    # 定义一个调用方法，接受参数 data
    def __call__(self, data):
        # 导入 json 模块
        import json
        # 从 data 中获取 'label' 字段
        label = data['label']
        # 将 'label' 字段的值解析为 JSON 格式
        label = json.loads(label)
        # 获取 label 中的盒子数量
        nBox = len(label)
        # 初始化空列表 boxes, txts, txt_tags
        boxes, txts, txt_tags = [], [], []
        # 遍历盒子数量范围
        for bno in range(0, nBox):
            # 获取当前盒子的坐标
            box = label[bno]['points']
            # 获取当前盒子的文本
            txt = label[bno]['transcription']
            # 将盒子坐标和文本添加到对应列表中
            boxes.append(box)
            txts.append(txt)
            # 如果文本为 '*' 或 '###'，则将 True 添加到 txt_tags 中，否则添加 False
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        # 将 boxes 转换为 numpy 数组，数据类型为 float32
        boxes = np.array(boxes, dtype=np.float32)
        # 将 txt_tags 转换为 numpy 数组，数据类型为 bool
        txt_tags = np.array(txt_tags, dtype=np.bool_)

        # 将处理后的数据更新到 data 中
        data['polys'] = boxes
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        # 返回处理后的 data
        return data

# 定义一个类 KieLabelEncode
class KieLabelEncode(object):
    # 初始化方法，接受字符字典路径、类别路径、norm、directed 和任意关键字参数
    def __init__(self,
                 character_dict_path,
                 class_path,
                 norm=10,
                 directed=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(KieLabelEncode, self).__init__()
        # 初始化空字典 dict，初始键值对为 {'' : 0}
        self.dict = dict({'': 0})
        # 初始化空字典 label2classid_map
        self.label2classid_map = dict()
        # 打开字符字典文件，读取内容并处理
        with open(character_dict_path, 'r', encoding='utf-8') as fr:
            idx = 1
            for line in fr:
                char = line.strip()
                self.dict[char] = idx
                idx += 1
        # 打开类别文件，读取内容并处理
        with open(class_path, "r") as fin:
            lines = fin.readlines()
            for idx, line in enumerate(lines):
                line = line.strip("\n")
                self.label2classid_map[line] = idx
        # 初始化 norm、directed 属性
        self.norm = norm
        self.directed = directed
    # 计算每两个框之间的关系
    def compute_relation(self, boxes):
        """Compute relation between every two boxes."""
        # 提取框的左上角和右下角坐标
        x1s, y1s = boxes[:, 0:1], boxes[:, 1:2]
        x2s, y2s = boxes[:, 4:5], boxes[:, 5:6]
        # 计算宽度和高度
        ws, hs = x2s - x1s + 1, np.maximum(y2s - y1s + 1, 1)
        # 计算相对于第一个框的偏移量
        dxs = (x1s[:, 0][None] - x1s) / self.norm
        dys = (y1s[:, 0][None] - y1s) / self.norm
        # 计算宽高比和高度比
        xhhs, xwhs = hs[:, 0][None] / hs, ws[:, 0][None] / hs
        whs = ws / hs + np.zeros_like(xhhs)
        # 组合关系信息
        relations = np.stack([dxs, dys, whs, xhhs, xwhs], -1)
        # 组合框的坐标信息
        bboxes = np.concatenate([x1s, y1s, x2s, y2s], -1).astype(np.float32)
        return relations, bboxes

    # 填充文本索引到相同长度
    def pad_text_indices(self, text_inds):
        """Pad text index to same length."""
        # 设置最大长度为300
        max_len = 300
        # 计算所有文本索引中最长的长度
        recoder_len = max([len(text_ind) for text_ind in text_inds])
        # 创建一个填充后的文本索引数组
        padded_text_inds = -np.ones((len(text_inds), max_len), np.int32)
        # 遍历文本索引，将其填充到指定长度
        for idx, text_ind in enumerate(text_inds):
            padded_text_inds[idx, :len(text_ind)] = np.array(text_ind)
        return padded_text_inds, recoder_len
    def list_to_numpy(self, ann_infos):
        """Convert bboxes, relations, texts and labels to ndarray."""
        # 从注释信息中获取边界框和文本索引
        boxes, text_inds = ann_infos['points'], ann_infos['text_inds']
        # 将边界框转换为 numpy 数组，数据类型为 np.int32
        boxes = np.array(boxes, np.int32)
        # 计算关系和边界框之间的关系
        relations, bboxes = self.compute_relation(boxes)

        # 获取标签信息
        labels = ann_infos.get('labels', None)
        if labels is not None:
            # 将标签转换为 numpy 数组，数据类型为 np.int32
            labels = np.array(labels, np.int32)
            # 获取边缘信息
            edges = ann_infos.get('edges', None)
            if edges is not None:
                # 将标签扩展为二维数组
                labels = labels[:, None]
                # 将边缘转换为 numpy 数组
                edges = np.array(edges)
                # 将边缘转换为布尔类型的数组
                edges = (edges[:, None] == edges[None, :]).astype(np.int32)
                if self.directed:
                    # 如果是有向图，则将边缘与标签进行逻辑与操作
                    edges = (edges & labels == 1).astype(np.int32)
                # 将对角线元素填充为-1
                np.fill_diagonal(edges, -1)
                # 将标签和边缘连接在一起
                labels = np.concatenate([labels, edges], -1)
        # 对文本索引进行填充
        padded_text_inds, recoder_len = self.pad_text_indices(text_inds)
        max_num = 300
        # 创建临时边界框数组
        temp_bboxes = np.zeros([max_num, 4])
        h, _ = bboxes.shape
        temp_bboxes[:h, :] = bboxes

        # 创建临时关系数组
        temp_relations = np.zeros([max_num, max_num, 5])
        temp_relations[:h, :h, :] = relations

        # 创建临时填充文本索引数组
        temp_padded_text_inds = np.zeros([max_num, max_num])
        temp_padded_text_inds[:h, :] = padded_text_inds

        # 创建临时标签数组
        temp_labels = np.zeros([max_num, max_num])
        temp_labels[:h, :h + 1] = labels

        # 创建标签数组的标记
        tag = np.array([h, recoder_len])
        # 返回包含转换后数据的字典
        return dict(
            image=ann_infos['image'],
            points=temp_bboxes,
            relations=temp_relations,
            texts=temp_padded_text_inds,
            labels=temp_labels,
            tag=tag)
    # 将传入的四个 x 坐标和四个 y 坐标转换为 Point 对象列表
    def convert_canonical(self, points_x, points_y):

        # 断言传入的 x 坐标列表长度为 4
        assert len(points_x) == 4
        # 断言传入的 y 坐标列表长度为 4
        assert len(points_y) == 4

        # 根据传入的 x 和 y 坐标创建 Point 对象列表
        points = [Point(points_x[i], points_y[i]) for i in range(4)]

        # 根据 Point 对象列表创建多边形对象
        polygon = Polygon([(p.x, p.y) for p in points])
        # 获取多边形的最小 x、最小 y 坐标
        min_x, min_y, _, _ = polygon.bounds
        # 计算每个点到左上角点的线段
        points_to_lefttop = [
            LineString([points[i], Point(min_x, min_y)]) for i in range(4)
        ]
        # 计算每个线段的长度
        distances = np.array([line.length for line in points_to_lefttop])
        # 对线段长度进行排序，获取最小长度的索引
        sort_dist_idx = np.argsort(distances)
        lefttop_idx = sort_dist_idx[0]

        # 根据左上角点的索引确定点的顺序
        if lefttop_idx == 0:
            point_orders = [0, 1, 2, 3]
        elif lefttop_idx == 1:
            point_orders = [1, 2, 3, 0]
        elif lefttop_idx == 2:
            point_orders = [2, 3, 0, 1]
        else:
            point_orders = [3, 0, 1, 2]

        # 根据点的顺序重新排序 x 和 y 坐标列表
        sorted_points_x = [points_x[i] for i in point_orders]
        sorted_points_y = [points_y[j] for j in point_orders]

        # 返回重新排序后的 x 和 y 坐标列表
        return sorted_points_x, sorted_points_y

    # 对传入的四个 x 坐标和四个 y 坐标进行排序
    def sort_vertex(self, points_x, points_y):

        # 断言传入的 x 坐标列表长度为 4
        assert len(points_x) == 4
        # 断言传入的 y 坐标列表长度为 4
        assert len(points_y) == 4

        # 将 x 和 y 坐标列表转换为 numpy 数组
        x = np.array(points_x)
        y = np.array(points_y)
        # 计算 x 和 y 坐标的中心点坐标
        center_x = np.sum(x) * 0.25
        center_y = np.sum(y) * 0.25

        # 计算相对中心点的 x 和 y 坐标数组
        x_arr = np.array(x - center_x)
        y_arr = np.array(y - center_y)

        # 计算每个点相对中心点的角度
        angle = np.arctan2(y_arr, x_arr) * 180.0 / np.pi
        # 对角度进行排序，获取排序后的索引
        sort_idx = np.argsort(angle)

        sorted_points_x, sorted_points_y = [], []
        # 根据排序后的索引重新排序 x 和 y 坐标列表
        for i in range(4):
            sorted_points_x.append(points_x[sort_idx[i]])
            sorted_points_y.append(points_y[sort_idx[i]])

        # 调用 convert_canonical 方法，将重新排序后的坐标列表转换为规范形式
        return self.convert_canonical(sorted_points_x, sorted_points_y)
    # 定义一个类的调用方法，接受数据作为参数
    def __call__(self, data):
        # 导入 json 模块
        import json
        # 从数据中获取标签信息
        label = data['label']
        # 将标签信息解析为 JSON 格式
        annotations = json.loads(label)
        # 初始化空列表用于存储边界框、文本、文本索引、标签和边信息
        boxes, texts, text_inds, labels, edges = [], [], [], [], []
        # 遍历每个标注信息
        for ann in annotations:
            # 获取边界框的顶点坐标
            box = ann['points']
            # 提取顶点坐标的 x 和 y 值
            x_list = [box[i][0] for i in range(4)]
            y_list = [box[i][1] for i in range(4)]
            # 对顶点坐标进行排序
            sorted_x_list, sorted_y_list = self.sort_vertex(x_list, y_list)
            sorted_box = []
            # 将排序后的顶点坐标添加到列表中
            for x, y in zip(sorted_x_list, sorted_y_list):
                sorted_box.append(x)
                sorted_box.append(y)
            boxes.append(sorted_box)
            # 获取文本信息并添加到列表中
            text = ann['transcription']
            texts.append(ann['transcription'])
            # 将文本转换为索引并添加到列表中
            text_ind = [self.dict[c] for c in text if c in self.dict]
            text_inds.append(text_ind)
            # 根据不同情况获取标签信息并添加到列表中
            if 'label' in ann.keys():
                labels.append(self.label2classid_map[ann['label']])
            elif 'key_cls' in ann.keys():
                labels.append(ann['key_cls'])
            else:
                raise ValueError(
                    "Cannot found 'key_cls' in ann.keys(), please check your training annotation."
                )
            # 获取边信息并添加到列表中
            edges.append(ann.get('edge', 0))
        # 将所有信息整合为字典
        ann_infos = dict(
            image=data['image'],
            points=boxes,
            texts=texts,
            text_inds=text_inds,
            edges=edges,
            labels=labels)
        # 将字典转换为 numpy 数组并返回
        return self.list_to_numpy(ann_infos)
class AttnLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    # 初始化方法，设置最大文本长度、字符字典路径和是否使用空格字符等参数
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(AttnLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    # 添加特殊字符到字符字典中
    def add_special_char(self, dict_character):
        # 设置起始和结束标记字符串
        self.beg_str = "sos"
        self.end_str = "eos"
        # 在字符字典中添加起始和结束标记字符串
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    # 对数据进行编码处理
    def __call__(self, data):
        # 获取数据中的标签文本
        text = data['label']
        # 对标签文本进行编码处理
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        # 设置数据中的长度属性
        data['length'] = np.array(len(text))
        # 在标签文本前后添加特殊标记，并填充到最大文本长度
        text = [0] + text + [len(self.character) - 1] + [0] * (self.max_text_len
                                                               - len(text) - 2)
        data['label'] = np.array(text)
        return data

    # 获取被忽略的标记
    def get_ignored_tokens(self):
        # 获取起始和结束标记的索引
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    # 获取起始或结束标记的索引
    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "Unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class RFLLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    # 初始化方法，设置最大文本长度、字符字典路径和是否使用空格字符等参数
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(RFLLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)
    # 向字符字典中添加特殊字符
    def add_special_char(self, dict_character):
        # 初始化起始字符和结束字符
        self.beg_str = "sos"
        self.end_str = "eos"
        # 在字符字典中添加起始字符、原字符集合、结束字符
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    # 对文本进行字符计数编码
    def encode_cnt(self, text):
        # 初始化字符计数列表
        cnt_label = [0.0] * len(self.character)
        # 统计文本中每个字符出现的次数
        for char_ in text:
            cnt_label[char_] += 1
        return np.array(cnt_label)

    # 对数据进行处理
    def __call__(self, data):
        # 获取标签文本
        text = data['label']
        # 对标签文本进行编码
        text = self.encode(text)
        # 如果编码结果为空，则返回空
        if text is None:
            return None
        # 如果编码后的文本长度超过最大文本长度，则返回空
        if len(text) >= self.max_text_len:
            return None
        # 对编码后的文本进行字符计数编码
        cnt_label = self.encode_cnt(text)
        # 记录文本长度
        data['length'] = np.array(len(text))
        # 在文本前后添加特殊字符，并补齐到最大文本长度
        text = [0] + text + [len(self.character) - 1] + [0] * (self.max_text_len
                                                               - len(text) - 2)
        # 如果文本长度不等于最大文本长度，则返回空
        if len(text) != self.max_text_len:
            return None
        # 更新数据中的标签和字符计数编码
        data['label'] = np.array(text)
        data['cnt_label'] = cnt_label
        return data

    # 获取被忽略的特殊字符索引
    def get_ignored_tokens(self):
        # 获取起始和结束标志的索引
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    # 获取起始或结束标志的索引
    def get_beg_end_flag_idx(self, beg_or_end):
        # 根据参数选择起始或结束标志的索引
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            # 如果参数不是"beg"或"end"，则抛出异常
            assert False, "Unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx
# 定义 SEEDLabelEncode 类，用于将文本标签和文本索引之间进行转换
class SEEDLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    # 初始化方法，设置最大文本长度、字符字典路径、是否使用空格字符等参数
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(SEEDLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    # 添加特殊字符的方法
    def add_special_char(self, dict_character):
        # 设置填充字符、结束字符、未知字符
        self.padding = "padding"
        self.end_str = "eos"
        self.unknown = "unknown"
        # 将特殊字符添加到字符字典中
        dict_character = dict_character + [
            self.end_str, self.padding, self.unknown
        ]
        return dict_character

    # 调用对象时执行的方法
    def __call__(self, data):
        # 获取数据中的标签文本
        text = data['label']
        # 将标签文本编码成索引
        text = self.encode(text)
        # 如果编码结果为空，则返回空
        if text is None:
            return None
        # 如果编码后的文本长度超过最大文本长度，则返回空
        if len(text) >= self.max_text_len:
            return None
        # 计算文本长度并加上结束符
        data['length'] = np.array(len(text)) + 1  # conclude eos
        # 将文本补齐到最大文本长度
        text = text + [len(self.character) - 3] + [len(self.character) - 2] * (
            self.max_text_len - len(text) - 1)
        # 更新数据中的标签为编码后的索引
        data['label'] = np.array(text)
        return data


# 定义 SRNLabelEncode 类，用于将文本标签和文本索引之间进行转换
class SRNLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    # 初始化方法，设置最大文本长度、字符字典路径、是否使用空格字符等参数
    def __init__(self,
                 max_text_length=25,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(SRNLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    # 添加特殊字符的方法
    def add_special_char(self, dict_character):
        # 将开始字符和结束字符添加到字符字典中
        dict_character = dict_character + [self.beg_str, self.end_str]
        return dict_character
    # 定义一个方法，用于处理输入数据，将标签编码后返回
    def __call__(self, data):
        # 从输入数据中获取标签文本
        text = data['label']
        # 对标签文本进行编码处理
        text = self.encode(text)
        # 获取字符集的长度
        char_num = len(self.character)
        # 如果标签文本为空，则返回空
        if text is None:
            return None
        # 如果标签文本长度超过设定的最大长度，则返回空
        if len(text) > self.max_text_len:
            return None
        # 将标签文本的长度存储到数据中
        data['length'] = np.array(len(text))
        # 如果标签文本长度不足最大长度，则用字符集中最后一个字符填充
        text = text + [char_num - 1] * (self.max_text_len - len(text))
        # 将处理后的标签文本存储到数据中
        data['label'] = np.array(text)
        # 返回处理后的数据
        return data

    # 获取被忽略的标记
    def get_ignored_tokens(self):
        # 获取起始标记和结束标记的索引
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        # 返回起始标记和结束标记的索引
        return [beg_idx, end_idx]

    # 获取起始或结束标记的索引
    def get_beg_end_flag_idx(self, beg_or_end):
        # 如果是起始标记
        if beg_or_end == "beg":
            # 获取起始标记的索引
            idx = np.array(self.dict[self.beg_str])
        # 如果是结束标记
        elif beg_or_end == "end":
            # 获取结束标记的索引
            idx = np.array(self.dict[self.end_str])
        else:
            # 抛出异常，表示不支持的类型
            assert False, "Unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        # 返回标记的索引
        return idx
# 定义一个类 TableLabelEncode，继承自 AttnLabelEncode 类
class TableLabelEncode(AttnLabelEncode):
    """ Convert between text-label and text-index """
    # 定义 _max_text_len 属性，返回 self.max_text_len + 2 的值
    @property
    def _max_text_len(self):
        return self.max_text_len + 2
    # 定义一个调用函数，接受数据作为参数
    def __call__(self, data):
        # 从数据中获取单元格和结构信息
        cells = data['cells']
        structure = data['structure']
        # 如果需要合并没有跨度的结构，则调用_merge_no_span_structure方法
        if self.merge_no_span_structure:
            structure = self._merge_no_span_structure(structure)
        # 如果需要替换空单元格标记，则调用_replace_empty_cell_token方法
        if self.replace_empty_cell_token:
            structure = self._replace_empty_cell_token(structure, cells)
        # 移除空标记并在跨度标记前添加空格
        new_structure = []
        for token in structure:
            if token != '':
                if 'span' in token and token[0] != ' ':
                    token = ' ' + token
                new_structure.append(token)
        # 编码结构信息
        structure = self.encode(new_structure)
        if structure is None:
            return None

        # 添加起始和结束索引
        structure = [self.start_idx] + structure + [self.end_idx]
        # 填充结构信息
        structure = structure + [self.pad_idx] * (self._max_text_len - len(structure))
        structure = np.array(structure)
        data['structure'] = structure

        # 如果结构信息长度超过最大文本长度，则返回None
        if len(structure) > self._max_text_len:
            return None

        # 编码框信息
        bboxes = np.zeros((self._max_text_len, self.loc_reg_num), dtype=np.float32)
        bbox_masks = np.zeros((self._max_text_len, 1), dtype=np.float32)

        bbox_idx = 0

        # 遍历结构信息，根据条件填充框信息
        for i, token in enumerate(structure):
            if self.idx2char[token] in self.td_token:
                if 'bbox' in cells[bbox_idx] and len(cells[bbox_idx]['tokens']) > 0:
                    bbox = cells[bbox_idx]['bbox'].copy()
                    bbox = np.array(bbox, dtype=np.float32).reshape(-1)
                    bboxes[i] = bbox
                    bbox_masks[i] = 1.0
                if self.learn_empty_box:
                    bbox_masks[i] = 1.0
                bbox_idx += 1
        data['bboxes'] = bboxes
        data['bbox_masks'] = bbox_masks
        return data
    # 合并结构中的空单元格标记，将 '<td>' 替换为 '<td></td>'
    def _merge_no_span_structure(self, structure):
        """
        This code is refer from:
        https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/table_recognition/data_preprocess.py
        """
        # 创建一个新的结构列表
        new_structure = []
        i = 0
        # 遍历结构列表
        while i < len(structure):
            token = structure[i]
            # 如果当前 token 是 '<td>'，则将其替换为 '<td></td>'
            if token == '<td>':
                token = '<td></td>'
                i += 1
            # 将 token 添加到新的结构列表中
            new_structure.append(token)
            i += 1
        # 返回处理后的新结构列表
        return new_structure

    # 替换空单元格标记
    def _replace_empty_cell_token(self, token_list, cells):
        """
        This fun code is refer from:
        https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/table_recognition/data_preprocess.py
        """

        bbox_idx = 0
        add_empty_bbox_token_list = []
        # 遍历 token 列表
        for token in token_list:
            # 如果 token 是空单元格标记，则替换为对应的内容
            if token in ['<td></td>', '<td', '<td>']:
                # 如果单元格中没有 bbox 信息，则使用空单元格标记字典中的内容替换
                if 'bbox' not in cells[bbox_idx].keys():
                    content = str(cells[bbox_idx]['tokens'])
                    token = self.empty_bbox_token_dict[content]
                # 将替换后的 token 添加到新列表中
                add_empty_bbox_token_list.append(token)
                bbox_idx += 1
            else:
                # 如果不是空单元格标记，则直接添加到新列表中
                add_empty_bbox_token_list.append(token)
        # 返回处理后的新 token 列表
        return add_empty_bbox_token_list
# 定义一个类 TableMasterLabelEncode，继承自 TableLabelEncode，用于文本标签和索引之间的转换
class TableMasterLabelEncode(TableLabelEncode):
    """ Convert between text-label and text-index """

    # 初始化方法，接受多个参数
    def __init__(self,
                 max_text_length,
                 character_dict_path,
                 replace_empty_cell_token=False,
                 merge_no_span_structure=False,
                 learn_empty_box=False,
                 loc_reg_num=4,
                 **kwargs):
        # 调用父类的初始化方法
        super(TableMasterLabelEncode, self).__init__(
            max_text_length, character_dict_path, replace_empty_cell_token,
            merge_no_span_structure, learn_empty_box, loc_reg_num, **kwargs)
        # 设置 pad_idx 为 pad_str 对应的索引
        self.pad_idx = self.dict[self.pad_str]
        # 设置 unknown_idx 为 unknown_str 对应的索引
        self.unknown_idx = self.dict[self.unknown_str]

    # 定义属性 _max_text_len，返回 max_text_len 的值
    @property
    def _max_text_len(self):
        return self.max_text_len

    # 添加特殊字符到字典中
    def add_special_char(self, dict_character):
        # 设置 beg_str 为 '<SOS>'
        self.beg_str = '<SOS>'
        # 设置 end_str 为 '<EOS>'
        self.end_str = '<EOS>'
        # 设置 unknown_str 为 '<UKN>'
        self.unknown_str = '<UKN>'
        # 设置 pad_str 为 '<PAD>'
        self.pad_str = '<PAD>'
        # 将特殊字符添加到 dict_character 中
        dict_character = dict_character
        dict_character = dict_character + [
            self.unknown_str, self.beg_str, self.end_str, self.pad_str
        ]
        return dict_character


# 定义一个类 TableBoxEncode，用于处理表格框的格式转换
class TableBoxEncode(object):
    # 初始化方法，接受输入框格式和输出框格式两个参数
    def __init__(self, in_box_format='xyxy', out_box_format='xyxy', **kwargs):
        # 断言输出框格式只能是 'xywh', 'xyxy', 'xyxyxyxy' 中的一种
        assert out_box_format in ['xywh', 'xyxy', 'xyxyxyxy']
        # 设置输入框格式和输出框格式
        self.in_box_format = in_box_format
        self.out_box_format = out_box_format

    # 调用对象时执行的方法，用于处理数据中的表格框
    def __call__(self, data):
        # 获取图像的高度和宽度
        img_height, img_width = data['image'].shape[:2]
        # 获取表格框的坐标
        bboxes = data['bboxes']
        # 如果输入框格式和输出框格式不一致
        if self.in_box_format != self.out_box_format:
            # 如果输出框格式是 'xywh'
            if self.out_box_format == 'xywh':
                # 如果输入框格式是 'xyxyxyxy'
                if self.in_box_format == 'xyxyxyxy':
                    # 将输入框格式转换为 'xywh'
                    bboxes = self.xyxyxyxy2xywh(bboxes)
                # 如果输入框格式是 'xyxy'
                elif self.in_box_format == 'xyxy':
                    # 将输入框格式转换为 'xywh'
                    bboxes = self.xyxy2xywh(bboxes)

        # 将表格框的 x 和 y 坐标归一化
        bboxes[:, 0::2] /= img_width
        bboxes[:, 1::2] /= img_height
        # 更新数据中的表格框坐标
        data['bboxes'] = bboxes
        return data
    # 将包含左上角和右下角坐标的边界框转换为包含中心点坐标和宽高的边界框
    def xyxyxyxy2xywh(self, boxes):
        # 创建一个与输入边界框相同形状的全零数组
        new_bboxes = np.zeros([len(bboxes), 4])
        # 计算新边界框的左上角 x 坐标
        new_bboxes[:, 0] = bboxes[:, 0::2].min()  # x1
        # 计算新边界框的左上角 y 坐标
        new_bboxes[:, 1] = bboxes[:, 1::2].min()  # y1
        # 计算新边界框的宽度
        new_bboxes[:, 2] = bboxes[:, 0::2].max() - new_bboxes[:, 0]  # w
        # 计算新边界框的高度
        new_bboxes[:, 3] = bboxes[:, 1::2].max() - new_bboxes[:, 1]  # h
        return new_bboxes

    # 将包含左上角和右下角坐标的边界框转换为包含中心点坐标和宽高的边界框
    def xyxy2xywh(self, bboxes):
        # 创建一个与输入边界框相同形状的空数组
        new_bboxes = np.empty_like(bboxes)
        # 计算新边界框的中心点 x 坐标
        new_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2  # x center
        # 计算新边界框的中心点 y 坐标
        new_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2  # y center
        # 计算新边界框的宽度
        new_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # width
        # 计算新边界框的高度
        new_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # height
        return new_bboxes
# 定义 SARLabelEncode 类，用于将文本标签和文本索引之间进行转换
class SARLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    # 初始化方法，设置最大文本长度、字符字典路径、是否使用空格字符等参数
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(SARLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    # 添加特殊字符到字符字典中
    def add_special_char(self, dict_character):
        # 定义特殊字符
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        # 将未知字符添加到字符字典中
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        # 将起始和结束字符添加到字符字典中
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        # 将填充字符添加到字符字典中
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1

        return dict_character

    # 对数据进行处理，将文本转换为索引表示
    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len - 1:
            return None
        data['length'] = np.array(len(text))
        target = [self.start_idx] + text + [self.end_idx]
        padded_text = [self.padding_idx for _ in range(self.max_text_len)]

        padded_text[:len(target)] = target
        data['label'] = np.array(padded_text)
        return data

    # 获取被忽略的标记
    def get_ignored_tokens(self):
        return [self.padding_idx]


# 定义 SATRNLabelEncode 类，用于将文本标签和文本索引之间进行转换
class SATRNLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    # 初始化方法，设置最大文本长度、字符字典路径、是否使用空格字符、是否转换为小写等参数
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 lower=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(SATRNLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)
        self.lower = lower
    # 向字符字典中添加特殊字符
    def add_special_char(self, dict_character):
        # 定义起始和结束字符
        beg_end_str = "<BOS/EOS>"
        # 定义未知字符
        unknown_str = "<UKN>"
        # 定义填充字符
        padding_str = "<PAD>"
        # 将未知字符添加到字符字典中
        dict_character = dict_character + [unknown_str]
        # 记录未知字符的索引
        self.unknown_idx = len(dict_character) - 1
        # 将起始和结束字符添加到字符字典中
        dict_character = dict_character + [beg_end_str]
        # 记录起始和结束字符的索引
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        # 将填充字符添加到字符字典中
        dict_character = dict_character + [padding_str]
        # 记录填充字符的索引
        self.padding_idx = len(dict_character) - 1

        return dict_character

    # 将文本编码为索引列表
    def encode(self, text):
        # 如果需要转换为小写，则将文本转换为小写
        if self.lower:
            text = text.lower()
        text_list = []
        # 遍历文本中的每个字符，将其转换为对应的索引
        for char in text:
            text_list.append(self.dict.get(char, self.unknown_idx))
        # 如果文本为空，则返回 None
        if len(text_list) == 0:
            return None
        return text_list

    # 对数据进行处理
    def __call__(self, data):
        # 获取数据中的文本
        text = data['label']
        # 将文本编码为索引列表
        text = self.encode(text)
        # 如果文本为空，则返回 None
        if text is None:
            return None
        # 记录文本的长度
        data['length'] = np.array(len(text))
        # 在文本前后添加起始和结束字符的索引
        target = [self.start_idx] + text + [self.end_idx]
        # 创建一个填充文本列表
        padded_text = [self.padding_idx for _ in range(self.max_text_len)]
        # 如果目标文本长度超过最大文本长度，则截取前部分
        if len(target) > self.max_text_len:
            padded_text = target[:self.max_text_len]
        else:
            # 否则将目标文本复制到填充文本列表中
            padded_text[:len(target)] = target
        # 更新数据中的标签为填充后的文本
        data['label'] = np.array(padded_text)
        return data

    # 获取被忽略的标记
    def get_ignored_tokens(self):
        return [self.padding_idx]
class PRENLabelEncode(BaseRecLabelEncode):
    # 定义 PRENLabelEncode 类，继承自 BaseRecLabelEncode 类
    def __init__(self,
                 max_text_length,
                 character_dict_path,
                 use_space_char=False,
                 **kwargs):
        # 初始化方法，接受最大文本长度、字符字典路径和是否使用空格字符等参数
        super(PRENLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        # 添加特殊字符方法
        padding_str = '<PAD>'  # 0 
        end_str = '<EOS>'  # 1
        unknown_str = '<UNK>'  # 2

        dict_character = [padding_str, end_str, unknown_str] + dict_character
        # 将特殊字符添加到字符字典中
        self.padding_idx = 0
        self.end_idx = 1
        self.unknown_idx = 2

        return dict_character

    def encode(self, text):
        # 编码方法，将文本转换为编码列表
        if len(text) == 0 or len(text) >= self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                text_list.append(self.unknown_idx)
            else:
                text_list.append(self.dict[char])
        text_list.append(self.end_idx)
        if len(text_list) < self.max_text_len:
            text_list += [self.padding_idx] * (
                self.max_text_len - len(text_list))
        return text_list

    def __call__(self, data):
        # 调用方法，对数据进行编码
        text = data['label']
        encoded_text = self.encode(text)
        if encoded_text is None:
            return None
        data['label'] = np.array(encoded_text)
        return data


class VQATokenLabelEncode(object):
    """
    Label encode for NLP VQA methods
    """
    # 初始化函数，接受多个参数
    def __init__(self,
                 class_path,
                 contains_re=False,
                 add_special_ids=False,
                 algorithm='LayoutXLM',
                 use_textline_bbox_info=True,
                 order_method=None,
                 infer_mode=False,
                 ocr_engine=None,
                 **kwargs):
        # 调用父类的初始化函数
        super(VQATokenLabelEncode, self).__init__()
        # 导入所需的模块和函数
        from paddlenlp.transformers import LayoutXLMTokenizer, LayoutLMTokenizer, LayoutLMv2Tokenizer
        from ppocr.utils.utility import load_vqa_bio_label_maps
        # 定义不同算法对应的tokenizer类和预训练模型
        tokenizer_dict = {
            'LayoutXLM': {
                'class': LayoutXLMTokenizer,
                'pretrained_model': 'layoutxlm-base-uncased'
            },
            'LayoutLM': {
                'class': LayoutLMTokenizer,
                'pretrained_model': 'layoutlm-base-uncased'
            },
            'LayoutLMv2': {
                'class': LayoutLMv2Tokenizer,
                'pretrained_model': 'layoutlmv2-base-uncased'
            }
        }
        # 设置是否包含正则表达式的标志
        self.contains_re = contains_re
        # 根据算法选择对应的tokenizer配置
        tokenizer_config = tokenizer_dict[algorithm]
        # 根据tokenizer配置初始化tokenizer对象
        self.tokenizer = tokenizer_config['class'].from_pretrained(
            tokenizer_config['pretrained_model'])
        # 加载VQA的标签映射
        self.label2id_map, id2label_map = load_vqa_bio_label_maps(class_path)
        # 设置是否添加特殊id的标志
        self.add_special_ids = add_special_ids
        # 设置推理模式的标志
        self.infer_mode = infer_mode
        # 设置OCR引擎
        self.ocr_engine = ocr_engine
        # 设置是否使用文本行边界框信息的标志
        self.use_textline_bbox_info = use_textline_bbox_info
        # 设置排序方法，必须为None或者"tb-yx"
        self.order_method = order_method
        assert self.order_method in [None, "tb-yx"]
    # 根据给定的边界框和文本内容，将文本按空格分割成单词
    def split_bbox(self, bbox, text, tokenizer):
        # 将文本按空格分割成单词
        words = text.split()
        # 存储每个单词对应的边界框
        token_bboxes = []
        # 初始化当前单词索引和边界框坐标
        curr_word_idx = 0
        x1, y1, x2, y2 = bbox
        # 计算每个字符的宽度
        unit_w = (x2 - x1) / len(text)
        # 遍历每个单词及其对应的边界框
        for idx, word in enumerate(words):
            # 计算当前单词的宽度
            curr_w = len(word) * unit_w
            # 根据当前单词的宽度更新边界框坐标
            word_bbox = [x1, y1, x1 + curr_w, y2]
            # 将当前单词的边界框添加到列表中
            token_bboxes.extend([word_bbox] * len(tokenizer.tokenize(word)))
            # 更新 x1 以便处理下一个单词
            x1 += (len(word) + 1) * unit_w
        # 返回所有单词的边界框列表
        return token_bboxes

    # 过滤掉 OCR 信息中的空内容和相关链接
    def filter_empty_contents(self, ocr_info):
        """
        find out the empty texts and remove the links
        """
        # 存储非空内容的 OCR 信息
        new_ocr_info = []
        # 存储空内容的索引
        empty_index = []
        # 遍历 OCR 信息
        for idx, info in enumerate(ocr_info):
            # 如果转录内容不为空，则将该信息添加到新的 OCR 信息列表中
            if len(info["transcription"]) > 0:
                new_ocr_info.append(copy.deepcopy(info))
            else:
                # 否则将空内容的索引添加到列表中
                empty_index.append(info["id"])

        # 遍历新的 OCR 信息列表
        for idx, info in enumerate(new_ocr_info):
            new_link = []
            # 遍历链接信息
            for link in info["linking"]:
                # 如果链接的起始或结束索引在空内容索引列表中，则跳过该链接
                if link[0] in empty_index or link[1] in empty_index:
                    continue
                # 否则将链接添加到新的链接列表中
                new_link.append(link)
            # 更新 OCR 信息中的链接信息
            new_ocr_info[idx]["linking"] = new_link
        # 返回过滤后的 OCR 信息
        return new_ocr_info

    # 将多边形坐标转换为边界框坐标
    def trans_poly_to_bbox(self, poly):
        # 计算多边形的最小和最大 x、y 坐标
        x1 = int(np.min([p[0] for p in poly]))
        x2 = int(np.max([p[0] for p in poly]))
        y1 = int(np.min([p[1] for p in poly]))
        y2 = int(np.max([p[1] for p in poly]))
        # 返回转换后的边界框坐标
        return [x1, y1, x2, y2]
    # 加载 OCR 信息，根据推断模式返回 OCR 结果或者读取标签信息
    def _load_ocr_info(self, data):
        # 如果处于推断模式
        if self.infer_mode:
            # 使用 OCR 引擎对图像进行 OCR 识别，获取结果
            ocr_result = self.ocr_engine.ocr(data['image'], cls=False)[0]
            ocr_info = []
            # 遍历 OCR 结果，将结果转换为指定格式
            for res in ocr_result:
                ocr_info.append({
                    "transcription": res[1][0],
                    "bbox": self.trans_poly_to_bbox(res[0]),
                    "points": res[0],
                })
            return ocr_info
        else:
            # 读取标签信息并转换为字典格式
            info = data['label']
            info_dict = json.loads(info)
            return info_dict

    # 平滑边界框坐标，将坐标值转换为相对于图像宽高的百分比
    def _smooth_box(self, bboxes, height, width):
        bboxes = np.array(bboxes)
        bboxes[:, 0] = bboxes[:, 0] * 1000 / width
        bboxes[:, 2] = bboxes[:, 2] * 1000 / width
        bboxes[:, 1] = bboxes[:, 1] * 1000 / height
        bboxes[:, 3] = bboxes[:, 3] * 1000 / height
        bboxes = bboxes.astype("int64").tolist()
        return bboxes

    # 解析标签，根据标签内容和编码结果生成标签列表
    def _parse_label(self, label, encode_res):
        gt_label = []
        # 如果标签为其他或忽略，则将标签列表填充为0
        if label.lower() in ["other", "others", "ignore"]:
            gt_label.extend([0] * len(encode_res["input_ids"]))
        else:
            # 否则根据标签映射表生成标签列表
            gt_label.append(self.label2id_map[("b-" + label).upper()])
            gt_label.extend([self.label2id_map[("i-" + label).upper()]] *
                            (len(encode_res["input_ids"]) - 1))
        return gt_label
# 定义一个类 MultiLabelEncode，继承自 BaseRecLabelEncode
class MultiLabelEncode(BaseRecLabelEncode):
    # 初始化方法，接受最大文本长度、字符字典路径、是否使用空格字符、gtc_encode 等参数
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 gtc_encode=None,
                 **kwargs):
        # 调用父类的初始化方法
        super(MultiLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

        # 创建 CTCLabelEncode 对象
        self.ctc_encode = CTCLabelEncode(max_text_length, character_dict_path,
                                         use_space_char, **kwargs)
        # 设置 gtc_encode_type 属性
        self.gtc_encode_type = gtc_encode
        # 如果 gtc_encode 为 None，则创建 SARLabelEncode 对象
        if gtc_encode is None:
            self.gtc_encode = SARLabelEncode(
                max_text_length, character_dict_path, use_space_char, **kwargs)
        # 否则，根据 gtc_encode 字符串创建对应对象
        else:
            self.gtc_encode = eval(gtc_encode)(
                max_text_length, character_dict_path, use_space_char, **kwargs)

    # 定义 __call__ 方法，用于处理数据
    def __call__(self, data):
        # 深拷贝数据
        data_ctc = copy.deepcopy(data)
        data_gtc = copy.deepcopy(data)
        data_out = dict()
        data_out['img_path'] = data.get('img_path', None)
        data_out['image'] = data['image']
        # 调用 ctc_encode 处理数据
        ctc = self.ctc_encode.__call__(data_ctc)
        # 调用 gtc_encode 处理数据
        gtc = self.gtc_encode.__call__(data_gtc)
        # 如果 ctc 或 gtc 为 None，则返回 None
        if ctc is None or gtc is None:
            return None
        data_out['label_ctc'] = ctc['label']
        # 如果 gtc_encode_type 不为 None，则设置 label_gtc，否则设置 label_sar
        if self.gtc_encode_type is not None:
            data_out['label_gtc'] = gtc['label']
        else:
            data_out['label_sar'] = gtc['label']
        data_out['length'] = ctc['length']
        return data_out


# 定义一个类 NRTRLabelEncode，用于文本标签和文本索引之间的转换
class NRTRLabelEncode(BaseRecLabelEncode):
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(NRTRLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)
    # 定义一个类的调用方法，接受数据字典作为参数
    def __call__(self, data):
        # 从数据字典中获取标签文本
        text = data['label']
        # 使用类中的编码方法对文本进行编码
        text = self.encode(text)
        # 如果编码结果为空，则返回空
        if text is None:
            return None
        # 如果编码后的文本长度大于等于最大文本长度减1，则返回空
        if len(text) >= self.max_text_len - 1:
            return None
        # 将文本长度存储到数据字典中
        data['length'] = np.array(len(text))
        # 在文本开头插入特殊字符2
        text.insert(0, 2)
        # 在文本末尾添加特殊字符3
        text.append(3)
        # 如果文本长度小于最大文本长度，则用0填充至最大文本长度
        text = text + [0] * (self.max_text_len - len(text))
        # 将处理后的文本存储回数据字典中
        data['label'] = np.array(text)
        # 返回处理后的数据字典
        return data

    # 定义一个方法，用于向字符字典中添加特殊字符
    def add_special_char(self, dict_character):
        # 在字符字典开头添加特殊字符'blank', '<unk>', '<s>', '</s>'
        dict_character = ['blank', '<unk>', '<s>', '</s>'] + dict_character
        # 返回添加特殊字符后的字符字典
        return dict_character
# 定义一个类 ViTSTRLabelEncode，用于文本标签和文本索引之间的转换
class ViTSTRLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    # 初始化方法，设置最大文本长度、字符字典路径、是否使用空格字符、忽略索引等参数
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 ignore_index=0,
                 **kwargs):

        # 调用父类的初始化方法
        super(ViTSTRLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)
        # 设置忽略索引
        self.ignore_index = ignore_index

    # 定义 __call__ 方法，用于处理数据
    def __call__(self, data):
        # 获取数据中的标签文本
        text = data['label']
        # 将文本编码成索引
        text = self.encode(text)
        # 如果文本为空，则返回 None
        if text is None:
            return None
        # 如果文本长度超过最大文本长度，则返回 None
        if len(text) >= self.max_text_len:
            return None
        # 将文本长度添加到数据中
        data['length'] = np.array(len(text))
        # 在文本开头插入忽略索引
        text.insert(0, self.ignore_index)
        # 在文本末尾添加索引 1
        text.append(1)
        # 在文本末尾添加忽略索引，使文本长度达到最大文本长度加 2
        text = text + [self.ignore_index] * (self.max_text_len + 2 - len(text))
        # 更新数据中的标签
        data['label'] = np.array(text)
        # 返回处理后的数据
        return data

    # 添加特殊字符到字符字典中
    def add_special_char(self, dict_character):
        # 在字符字典开头添加特殊字符 '<s>' 和 '</s>'
        dict_character = ['<s>', '</s>'] + dict_character
        return dict_character


# 定义一个类 ABINetLabelEncode，用于文本标签和文本索引之间的转换
class ABINetLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    # 初始化方法，设置最大文本长度、字符字典路径、是否使用空格字符、忽略索引等参数
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 ignore_index=100,
                 **kwargs):

        # 调用父类的初始化方法
        super(ABINetLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)
        # 设置忽略索引
        self.ignore_index = ignore_index

    # 定义 __call__ 方法，用于处理数据
    def __call__(self, data):
        # 获取数据中的标签文本
        text = data['label']
        # 将文本编码成索引
        text = self.encode(text)
        # 如果文本为空，则返回 None
        if text is None:
            return None
        # 如果文本长度超过最大文本长度，则返回 None
        if len(text) >= self.max_text_len:
            return None
        # 将文本长度添加到数据中
        data['length'] = np.array(len(text))
        # 在文本末尾添加索引 0
        text.append(0)
        # 在文本末尾添加忽略索引，使文本长度达到最大文本长度加 1
        text = text + [self.ignore_index] * (self.max_text_len + 1 - len(text))
        # 更新数据中的标签
        data['label'] = np.array(text)
        # 返回处理后的数据
        return data
    # 在字符字典开头添加特殊字符'</s>'，并返回更新后的字符字典
    def add_special_char(self, dict_character):
        dict_character = ['</s>'] + dict_character
        return dict_character
# 定义一个类 SRLabelEncode，继承自 BaseRecLabelEncode
class SRLabelEncode(BaseRecLabelEncode):
    # 初始化方法，接受最大文本长度、字符字典路径、是否使用空格字符等参数
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(SRLabelEncode, self).__init__(max_text_length,
                                            character_dict_path, use_space_char)
        # 初始化一个空字典
        self.dic = {}
        # 打开字符字典文件，读取每一行
        with open(character_dict_path, 'r') as fin:
            for line in fin.readlines():
                # 去除行两端的空格
                line = line.strip()
                # 将字符和序列分开
                character, sequence = line.split()
                # 将字符和序列添加到字典中
                self.dic[character] = sequence
        # 定义英文笔画字母表
        english_stroke_alphabet = '0123456789'
        # 初始化一个英文笔画字典
        self.english_stroke_dict = {}
        # 遍历英文笔画字母表
        for index in range(len(english_stroke_alphabet)):
            # 将字母和索引添加到字典中
            self.english_stroke_dict[english_stroke_alphabet[index]] = index

    # 定义一个编码方法，接受标签作为参数
    def encode(self, label):
        # 初始化笔画序列为空字符串
        stroke_sequence = ''
        # 遍历标签中的每个字符
        for character in label:
            # 如果字符不在字典中，则跳过
            if character not in self.dic:
                continue
            else:
                # 否则将字符对应的序列添加到笔画序列中
                stroke_sequence += self.dic[character]
        # 在笔画序列末尾添加 '0'
        stroke_sequence += '0'
        # 将笔画序列赋值给标签
        label = stroke_sequence

        # 计算标签的长度
        length = len(label)

        # 初始化一个全零数组作为输入张量
        input_tensor = np.zeros(self.max_text_len).astype("int64")
        # 遍历标签中的每个字符
        for j in range(length - 1):
            # 将字符对应的索引添加到输入张量中
            input_tensor[j + 1] = self.english_stroke_dict[label[j]]

        # 返回标签长度和输入张量
        return length, input_tensor

    # 定义一个调用方法，接受数据作为参数
    def __call__(self, data):
        # 获取数据中的标签
        text = data['label']
        # 调用编码方法，得到标签长度和输入张量
        length, input_tensor = self.encode(text)

        # 将标签长度和输入张量添加到数据中
        data["length"] = length
        data["input_tensor"] = input_tensor
        # 如果标签为空，则返回空
        if text is None:
            return None
        # 否则返回数据
        return data

# 定义一个类 SPINLabelEncode，继承自 AttnLabelEncode
class SPINLabelEncode(AttnLabelEncode):
    """ Convert between text-label and text-index """
    # 初始化函数，设置最大文本长度、字符字典路径、是否使用空格字符、是否转换为小写等参数
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 lower=True,
                 **kwargs):
        # 调用父类的初始化函数，传入最大文本长度、字符字典路径、是否使用空格字符等参数
        super(SPINLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)
        # 设置是否转换为小写的参数
        self.lower = lower

    # 添加特殊字符到字符字典中
    def add_special_char(self, dict_character):
        # 设置起始字符和结束字符
        self.beg_str = "sos"
        self.end_str = "eos"
        # 将起始字符、结束字符和原字符字典合并
        dict_character = [self.beg_str] + [self.end_str] + dict_character
        return dict_character

    # 对数据进行编码处理
    def __call__(self, data):
        # 获取数据中的标签文本
        text = data['label']
        # 对标签文本进行编码
        text = self.encode(text)
        # 如果编码结果为空，则返回空
        if text is None:
            return None
        # 如果编码后的文本长度超过最大文本长度，则返回空
        if len(text) > self.max_text_len:
            return None
        # 将文本长度存储到数据中
        data['length'] = np.array(len(text))
        # 在文本前后添加起始和结束标记
        target = [0] + text + [1]
        # 创建一个与最大文本长度相等的全零列表
        padded_text = [0 for _ in range(self.max_text_len + 2)]

        # 将标记文本填充到全零列表中
        padded_text[:len(target)] = target
        # 将填充后的标记文本存储到数据的标签中
        data['label'] = np.array(padded_text)
        return data
# 定义一个类 VLLabelEncode，继承自 BaseRecLabelEncode 类，用于文本标签和文本索引之间的转换
class VLLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """
    
    # 初始化方法，接受最大文本长度、字符字典路径、是否使用空格字符等参数
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法，传入最大文本长度、字符字典路径、是否使用空格字符等参数
        super(VLLabelEncode, self).__init__(max_text_length,
                                            character_dict_path, use_space_char)
        # 创建一个空字典用于存储字符和对应的索引
        self.dict = {}
        # 遍历字符列表，将字符和对应的索引存储到字典中
        for i, char in enumerate(self.character):
            self.dict[char] = i
    # 定义一个类的方法，用于处理输入数据
    def __call__(self, data):
        # 获取原始字符串
        text = data['label']  # original string
        # 生成遮挡后的文本
        len_str = len(text)
        # 如果字符串长度小于等于0，则返回空
        if len_str <= 0:
            return None
        # 设置要改变的字符数量为1
        change_num = 1
        # 生成字符位置的列表
        order = list(range(len_str))
        # 随机选择一个字符位置进行遮挡
        change_id = sample(order, change_num)[0]
        # 获取被遮挡的字符
        label_sub = text[change_id]
        # 根据遮挡字符的位置生成遮挡后的文本
        if change_id == (len_str - 1):
            label_res = text[:change_id]
        elif change_id == 0:
            label_res = text[1:]
        else:
            label_res = text[:change_id] + text[change_id + 1:]

        # 将遮挡后的文本、遮挡字符和字符位置添加到数据中
        data['label_res'] = label_res  # remaining string
        data['label_sub'] = label_sub  # occluded character
        data['label_id'] = change_id  # character index
        # 对原始字符串进行编码
        text = self.encode(text)
        # 如果编码结果为空，则返回空
        if text is None:
            return None
        # 将编码后的文本转换为数组形式
        text = [i + 1 for i in text]
        data['length'] = np.array(len(text))
        # 将文本填充到指定长度
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text)
        # 对遮挡后的文本进行编码
        label_res = self.encode(label_res)
        label_sub = self.encode(label_sub)
        # 如果遮挡后的文本为空，则设置为空列表
        if label_res is None:
            label_res = []
        else:
            label_res = [i + 1 for i in label_res]
        # 如果遮挡字符为空，则设置为空列表
        if label_sub is None:
            label_sub = []
        else:
            label_sub = [i + 1 for i in label_sub]
        data['length_res'] = np.array(len(label_res))
        data['length_sub'] = np.array(len(label_sub))
        # 将遮挡后的文本填充到指定长度
        label_res = label_res + [0] * (self.max_text_len - len(label_res))
        label_sub = label_sub + [0] * (self.max_text_len - len(label_sub))
        data['label_res'] = np.array(label_res)
        data['label_sub'] = np.array(label_sub)
        # 返回处理后的数据
        return data
class CTLabelEncode(object):
    # 定义 CTLabelEncode 类
    def __init__(self, **kwargs):
        # 初始化函数，接受任意关键字参数
        pass

    def __call__(self, data):
        # 定义 __call__ 方法，接受数据作为参数
        label = data['label']
        # 从数据中获取标签信息

        label = json.loads(label)
        # 将标签信息转换为 JSON 格式
        nBox = len(label)
        # 获取标签中盒子的数量
        boxes, txts = [], []
        # 初始化盒子和文本列表
        for bno in range(0, nBox):
            # 遍历盒子数量
            box = label[bno]['points']
            # 获取盒子的坐标信息
            box = np.array(box)
            # 将盒子坐标信息转换为 NumPy 数组

            boxes.append(box)
            # 将盒子坐标信息添加到盒子列表中
            txt = label[bno]['transcription']
            # 获取盒子对应的文本信息
            txts.append(txt)
            # 将文本信息添加到文本列表中

        if len(boxes) == 0:
            # 如果盒子列表为空
            return None
            # 返回空值

        data['polys'] = boxes
        # 将盒子列表添加到数据中的 'polys' 键
        data['texts'] = txts
        # 将文本列表添加到数据中的 'texts' 键
        return data
        # 返回处理后的数据


class CANLabelEncode(BaseRecLabelEncode):
    # 定义 CANLabelEncode 类，继承自 BaseRecLabelEncode 类
    def __init__(self,
                 character_dict_path,
                 max_text_length=100,
                 use_space_char=False,
                 lower=True,
                 **kwargs):
        # 初始化函数，接受字符字典路径、最大文本长度、是否使用空格字符、是否转换为小写等参数
        super(CANLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char, lower)
        # 调用父类的初始化函数，传入参数

    def encode(self, text_seq):
        # 定义 encode 方法，接受文本序列作为参数
        text_seq_encoded = []
        # 初始化编码后的文本序列列表
        for text in text_seq:
            # 遍历文本序列
            if text not in self.character:
                # 如果文本不在字符集中
                continue
                # 继续下一次循环
            text_seq_encoded.append(self.dict.get(text))
            # 将文本编码后的值添加到编码后的文本序列列表中
        if len(text_seq_encoded) == 0:
            # 如果编码后的文本序列列表为空
            return None
            # 返回空值
        return text_seq_encoded
        # 返回编码后的文本序列列表

    def __call__(self, data):
        # 定义 __call__ 方法，接受数据作为参数
        label = data['label']
        # 从数据中获取标签信息
        if isinstance(label, str):
            # 如果标签信息是字符串类型
            label = label.strip().split()
            # 去除首尾空格并按空格分割
        label.append(self.end_str)
        # 将结束字符串添加到标签信息中
        data['label'] = self.encode(label)
        # 将编码后的标签信息添加到数据中的 'label' 键
        return data
        # 返回处理后的数据
```