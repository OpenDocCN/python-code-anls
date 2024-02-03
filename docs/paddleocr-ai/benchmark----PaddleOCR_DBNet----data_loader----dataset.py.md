# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\data_loader\dataset.py`

```py
# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:54
# @Author  : zhoujun
# 导入所需的库
import pathlib
import os
import cv2
import numpy as np
import scipy.io as sio
from tqdm.auto import tqdm

# 导入自定义的类和函数
from base import BaseDataSet
from utils import order_points_clockwise, get_datalist, load, expand_polygon

# 定义一个名为ICDAR2015Dataset的类，继承自BaseDataSet类
class ICDAR2015Dataset(BaseDataSet):
    # 初始化函数，接收数据路径、图像模式、预处理方法、过滤关键字、忽略标签、变换方法等参数
    def __init__(self,
                 data_path: str,
                 img_mode,
                 pre_processes,
                 filter_keys,
                 ignore_tags,
                 transform=None,
                 **kwargs):
        # 调用父类的初始化函数
        super().__init__(data_path, img_mode, pre_processes, filter_keys,
                         ignore_tags, transform)

    # 加载数据的方法，接收数据路径，返回数据列表
    def load_data(self, data_path: str) -> list:
        # 获取数据列表
        data_list = get_datalist(data_path)
        # 初始化一个空列表用于存储处理后的数据
        t_data_list = []
        # 遍历数据列表中的每个图像路径和标签路径
        for img_path, label_path in data_list:
            # 获取标签数据
            data = self._get_annotation(label_path)
            # 如果标签数据中存在文本多边形
            if len(data['text_polys']) > 0:
                # 创建一个字典，包含图像路径和图像名称
                item = {
                    'img_path': img_path,
                    'img_name': pathlib.Path(img_path).stem
                }
                # 将标签数据添加到字典中
                item.update(data)
                # 将字典添加到处理后的数据列表中
                t_data_list.append(item)
            else:
                # 如果标签数据中不存在合适的边界框，打印提示信息
                print('there is no suit bbox in {}'.format(label_path))
        # 返回处理后的数据列表
        return t_data_list
    # 获取标签文件中的注释信息，返回一个字典
    def _get_annotation(self, label_path: str) -> dict:
        # 初始化空列表，用于存储边界框、文本和忽略标签
        boxes = []
        texts = []
        ignores = []
        # 打开标签文件，按照 UTF-8 编码读取内容
        with open(label_path, encoding='utf-8', mode='r') as f:
            # 逐行读取文件内容
            for line in f.readlines():
                # 去除每行两端的空格和特殊字符，并按逗号分割参数
                params = line.strip().strip('\ufeff').strip(
                    '\xef\xbb\xbf').split(',')
                try:
                    # 将参数转换为浮点数，并按照顺时针顺序排列，形成边界框
                    box = order_points_clockwise(
                        np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    # 如果边界框的面积大于0，则将其添加到列表中
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        # 获取标签信息，并添加到文本列表中
                        label = params[8]
                        texts.append(label)
                        # 判断标签是否在忽略标签列表中，并添加到忽略标签列表中
                        ignores.append(label in self.ignore_tags)
                except:
                    # 捕获异常，打印加载标签失败的信息
                    print('load label failed on {}'.format(label_path))
        # 将边界框、文本和忽略标签组成的字典返回
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data
class DetDataset(BaseDataSet):
    # 定义一个名为DetDataset的类，继承自BaseDataSet类
    def __init__(self,
                 data_path: str,
                 img_mode,
                 pre_processes,
                 filter_keys,
                 ignore_tags,
                 transform=None,
                 **kwargs):
        # 初始化函数，接受data_path、img_mode、pre_processes、filter_keys、ignore_tags、transform等参数
        self.load_char_annotation = kwargs['load_char_annotation']
        # 从kwargs参数中获取load_char_annotation值，赋给load_char_annotation属性
        self.expand_one_char = kwargs['expand_one_char']
        # 从kwargs参数中获取expand_one_char值，赋给expand_one_char属性
        super().__init__(data_path, img_mode, pre_processes, filter_keys,
                         ignore_tags, transform)
        # 调用父类BaseDataSet的初始化函数，传入相应参数

class SynthTextDataset(BaseDataSet):
    # 定义一个名为SynthTextDataset的类，继承自BaseDataSet类
    def __init__(self,
                 data_path: str,
                 img_mode,
                 pre_processes,
                 filter_keys,
                 transform=None,
                 **kwargs):
        # 初始化函数，接受data_path、img_mode、pre_processes、filter_keys、transform等参数
        self.transform = transform
        # 将transform参数赋给transform属性
        self.dataRoot = pathlib.Path(data_path)
        # 将data_path转换为Path对象，赋给dataRoot属性
        if not self.dataRoot.exists():
            raise FileNotFoundError('Dataset folder is not exist.')
            # 如果dataRoot路径不存在，则抛出FileNotFoundError异常
        self.targetFilePath = self.dataRoot / 'gt.mat'
        # 将dataRoot路径与'gt.mat'拼接成新路径，赋给targetFilePath属性
        if not self.targetFilePath.exists():
            raise FileExistsError('Target file is not exist.')
            # 如果targetFilePath路径不存在，则抛出FileExistsError异常
        targets = {}
        # 创建一个空字典targets
        sio.loadmat(
            self.targetFilePath,
            targets,
            squeeze_me=True,
            struct_as_record=False,
            variable_names=['imnames', 'wordBB', 'txt'])
        # 加载targetFilePath路径下的.mat文件内容到targets字典中
        self.imageNames = targets['imnames']
        # 从targets字典中获取'imnames'对应的值，赋给imageNames属性
        self.wordBBoxes = targets['wordBB']
        # 从targets字典中获取'wordBB'对应的值，赋给wordBBoxes属性
        self.transcripts = targets['txt']
        # 从targets字典中获取'txt'对应的值，赋给transcripts属性
        super().__init__(data_path, img_mode, pre_processes, filter_keys,
                         transform)
        # 调用父类BaseDataSet的初始化函数，传入相应参数
    # 加载数据的方法，接受数据路径参数，返回数据列表
    def load_data(self, data_path: str) -> list:
        # 初始化空列表用于存储数据
        t_data_list = []
        # 遍历三个列表的元素，分别为图片名称、文字边界框、文本
        for imageName, wordBBoxes, texts in zip(
                self.imageNames, self.wordBBoxes, self.transcripts):
            # 创建空字典用于存储当前数据项
            item = {}
            # 如果文字边界框的维度为2，则在第二个维度上扩展为1
            wordBBoxes = np.expand_dims(
                wordBBoxes, axis=2) if (wordBBoxes.ndim == 2) else wordBBoxes
            # 获取文字边界框的形状信息
            _, _, numOfWords = wordBBoxes.shape
            # 将文字边界框重塑为8行numOfWords列的形式，按列优先顺序
            text_polys = wordBBoxes.reshape(
                [8, numOfWords], order='F').T  # num_words * 8
            # 将文字边界框重塑为numOfWords行4列2通道的形式
            text_polys = text_polys.reshape(numOfWords, 4,
                                            2)  # num_of_words * 4 * 2
            # 将文本按空格分隔为单词列表
            transcripts = [word for line in texts for word in line.split()]
            # 如果单词数量与文字边界框数量不匹配，则跳过当前数据项
            if numOfWords != len(transcripts):
                continue
            # 添加图片路径到当前数据项
            item['img_path'] = str(self.dataRoot / imageName)
            # 添加图片名称到当前数据项
            item['img_name'] = (self.dataRoot / imageName).stem
            # 添加文字边界框到当前数据项
            item['text_polys'] = text_polys
            # 添加文本列表到当前数据项
            item['texts'] = transcripts
            # 添加忽略标签列表到当前数据项，判断每个单词是否在忽略标签中
            item['ignore_tags'] = [x in self.ignore_tags for x in transcripts]
            # 将当前数据项添加到数据列表中
            t_data_list.append(item)
        # 返回数据列表
        return t_data_list
```