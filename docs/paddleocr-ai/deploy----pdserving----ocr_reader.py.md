# `.\PaddleOCR\deploy\pdserving\ocr_reader.py`

```
# 导入所需的库
import cv2
import copy
import numpy as np
import math
import re
import sys
import argparse
import string
from copy import deepcopy

# 定义一个用于测试的图像尺寸调整类
class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        # 如果参数中包含'image_shape'，则使用指定的图像形状
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
        # 如果参数中包含'limit_side_len'，则根据指定的边长限制进行调整
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        # 如果参数中包含'resize_short'，则使用默认的边长限制进行调整
        elif 'resize_short' in kwargs:
            self.limit_side_len = 736
            self.limit_type = 'min'
        # 否则，根据指定的长边长度进行调整
        else:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)

    # 调整图像尺寸的方法
    def __call__(self, data):
        # 复制输入的图像数据
        img = deepcopy(data)
        # 获取原始图像的高度和宽度
        src_h, src_w, _ = img.shape

        # 根据不同的调整类型进行图像尺寸调整
        if self.resize_type == 0:
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)

        # 返回调整后的图像数据
        return img
    def resize_image_type1(self, img):
        # 获取目标图片的高度和宽度
        resize_h, resize_w = self.image_shape
        # 获取原始图片的高度和宽度
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        # 计算高度和宽度的缩放比例
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        # 调整图片大小
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # 返回调整后的图片和缩放比例
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        # 获取限制边长
        limit_side_len = self.limit_side_len
        h, w, _ = img.shape

        # 限制最大边长
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        else:
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        # 将调整后的高度和宽度调整为32的倍数
        resize_h = int(round(resize_h / 32) * 32)
        resize_w = int(round(resize_w / 32) * 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            # 调整图片大小
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        # 返回调整后的图片和高度、宽度的缩放比例
        return img, [ratio_h, ratio_w]
    # 定义一个方法用于调整图像大小，接受一个图像作为参数
    def resize_image_type2(self, img):
        # 获取图像的高度、宽度和通道数
        h, w, _ = img.shape

        # 初始化调整后的宽度和高度为原始宽度和高度
        resize_w = w
        resize_h = h

        # 如果高度大于宽度，则以长边为基准计算缩放比例
        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        # 根据缩放比例调整高度和宽度
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio

        # 设置最大步长为128，将调整后的高度和宽度调整为128的倍数
        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride

        # 使用OpenCV的resize函数调整图像大小
        img = cv2.resize(img, (int(resize_w), int(resize_h)))

        # 计算高度和宽度的缩放比例
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        # 返回调整后的图像和高度、宽度的缩放比例
        return img, [ratio_h, ratio_w]
# 定义一个基础的文本标签解码类，用于文本标签和文本索引之间的转换
class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 支持的字符类型列表
        support_character_type = [
            'ch', 'en', 'EN_symbol', 'french', 'german', 'japan', 'korean',
            'it', 'xi', 'pu', 'ru', 'ar', 'ta', 'ug', 'fa', 'ur', 'rs', 'oc',
            'rsc', 'bg', 'uk', 'be', 'te', 'ka', 'chinese_cht', 'hi', 'mr',
            'ne', 'EN'
        ]
        # 从配置参数中获取字符类型
        character_type = config['character_type']
        # 从配置参数中获取字符字典路径
        character_dict_path = config['character_dict_path']
        # 是否使用空格字符
        use_space_char = True
        # 断言字符类型在支持的字符类型列表中，否则抛出异常
        assert character_type in support_character_type, "Only {} are supported now but get {}".format(
            support_character_type, character_type)

        # 初始化起始字符串和结束字符串
        self.beg_str = "sos"
        self.end_str = "eos"

        # 根据字符类型设置字符集
        if character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type == "EN_symbol":
            # 与ASTER设置相同（使用94个字符）
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        elif character_type in support_character_type:
            self.character_str = ""
            # 断言字符字典路径不为空，否则抛出异常
            assert character_dict_path is not None, "character_dict_path should not be None when character_type is {}".format(
                character_type)
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)

        else:
            # 抛出未实现异常
            raise NotImplementedError
        # 设置字符类型
        self.character_type = character_type
        # 添加特殊字符
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        # 构建字符到索引的映射字典
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        # 设置字符集
        self.character = dict_character
    # 添加特殊字符到字符字典中
    def add_special_char(self, dict_character):
        return dict_character

    # 将文本索引转换为文本标签
    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        # 初始化结果列表
        result_list = []
        # 获取需要忽略的标记
        ignored_tokens = self.get_ignored_tokens()
        # 获取批处理大小
        batch_size = len(text_index)
        # 遍历每个批次
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            # 遍历每个索引
            for idx in range(len(text_index[batch_idx])):
                # 如果索引在忽略标记中，则跳过
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                # 如果需要去除重复字符
                if is_remove_duplicate:
                    # 仅用于预测
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                # 将字符添加到字符列表中
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                # 如果存在文本概率，则添加到置信度列表中
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            # 将字符列表转换为文本
            text = ''.join(char_list)
            # 计算置信度的平均值，并添加到结果列表中
            result_list.append((text, np.mean(conf_list)))
        return result_list

    # 获取需要忽略的标记
    def get_ignored_tokens(self):
        return [0]  # for ctc blank
# 定义一个类 CTCLabelDecode，用于文本标签和文本索引之间的转换
class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    # 初始化方法，接受配置参数
    def __init__(
            self,
            config,
            #character_dict_path=None,
            #character_type='ch',
            #use_space_char=False,
            **kwargs):
        # 调用父类的初始化方法
        super(CTCLabelDecode, self).__init__(config)

    # 定义 __call__ 方法，用于执行预测结果的解码操作
    def __call__(self, preds, label=None, *args, **kwargs):
        # 获取预测结果中概率最大的索引
        preds_idx = preds.argmax(axis=2)
        # 获取预测结果中概率最大的值
        preds_prob = preds.max(axis=2)
        # 解码预测结果，去除重复字符
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        # 如果没有标签，则返回解码后的文本
        if label is None:
            return text
        # 解码标签
        label = self.decode(label)
        # 返回解码后的文本和标签
        return text, label

    # 定义添加特殊字符的方法
    def add_special_char(self, dict_character):
        # 在字符字典中添加特殊字符 'blank'
        dict_character = ['blank'] + dict_character
        return dict_character


# 定义一个类 CharacterOps，用于文本标签和文本索引之间的转换
class CharacterOps(object):
    """ Convert between text-label and text-index """
    # 初始化方法，根据配置参数设置字符类型和损失类型
    def __init__(self, config):
        # 从配置中获取字符类型和损失类型
        self.character_type = config['character_type']
        self.loss_type = config['loss_type']
        # 如果字符类型为英文
        if self.character_type == "en":
            # 设置字符集为数字和小写字母
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        # 如果字符类型为中文
        elif self.character_type == "ch":
            # 从配置中获取中文字符字典路径
            character_dict_path = config['character_dict_path']
            self.character_str = ""
            # 读取中文字符字典文件内容
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    # 解码并去除换行符，拼接成字符集
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            dict_character = list(self.character_str)
        # 如果字符类型为英文敏感字符
        elif self.character_type == "en_sensitive":
            # 设置字符集为可打印字符减去6个特殊字符
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        else:
            self.character_str = None
        # 断言字符集不为空
        assert self.character_str is not None, \
            "Nonsupport type of the character: {}".format(self.character_str)
        # 设置起始和结束标记
        self.beg_str = "sos"
        self.end_str = "eos"
        # 如果损失类型为注意力机制
        if self.loss_type == "attention":
            # 在字符集前加入起始和结束标记
            dict_character = [self.beg_str, self.end_str] + dict_character
        # 创建字符到索引的映射字典
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        # 设置字符集
        self.character = dict_character
    # 将文本标签转换为文本索引
    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        # 如果字符类型是英文，将文本转换为小写
        if self.character_type == "en":
            text = text.lower()

        text_list = []
        # 遍历文本中的每个字符
        for char in text:
            # 如果字符不在字典中，跳过
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        text = np.array(text_list)
        return text

    # 将文本索引转换为文本标签
    def decode(self, text_index, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        char_list = []
        char_num = self.get_char_num()

        # 根据损失类型设置忽略的标记
        if self.loss_type == "attention":
            beg_idx = self.get_beg_end_flag_idx("beg")
            end_idx = self.get_beg_end_flag_idx("end")
            ignored_tokens = [beg_idx, end_idx]
        else:
            ignored_tokens = [char_num]

        # 遍历文本索引
        for idx in range(len(text_index)):
            # 如果索引在忽略的标记中，跳过
            if text_index[idx] in ignored_tokens:
                continue
            # 如果需要去除重复字符，并且当前字符与前一个字符相同，跳过
            if is_remove_duplicate:
                if idx > 0 and text_index[idx - 1] == text_index[idx]:
                    continue
            char_list.append(self.character[text_index[idx]])
        text = ''.join(char_list)
        return text

    # 获取字符数量
    def get_char_num(self):
        return len(self.character)
    # 获取起始或结束标志的索引
    def get_beg_end_flag_idx(self, beg_or_end):
        # 如果损失类型是"attention"
        if self.loss_type == "attention":
            # 如果是起始标志
            if beg_or_end == "beg":
                # 获取起始标志的索引
                idx = np.array(self.dict[self.beg_str])
            # 如果是结束标志
            elif beg_or_end == "end":
                # 获取结束标志的索引
                idx = np.array(self.dict[self.end_str])
            else:
                # 抛出异常，不支持的类型
                assert False, "Unsupport type %s in get_beg_end_flag_idx"\
                    % beg_or_end
            # 返回索引
            return idx
        else:
            # 如果损失类型不是"attention"，抛出错误
            err = "error in get_beg_end_flag_idx when using the loss %s"\
                % (self.loss_type)
            assert False, err
class OCRReader(object):
    # 定义 OCRReader 类
    def __init__(self,
                 algorithm="CRNN",
                 image_shape=[3, 48, 320],
                 char_type="ch",
                 batch_num=1,
                 char_dict_path="./ppocr_keys_v1.txt"):
        # 初始化 OCRReader 对象，设置默认参数
        self.rec_image_shape = image_shape
        self.character_type = char_type
        self.rec_batch_num = batch_num
        char_ops_params = {}
        char_ops_params["character_type"] = char_type
        char_ops_params["character_dict_path"] = char_dict_path
        char_ops_params['loss_type'] = 'ctc'
        # 设置字符操作参数
        self.char_ops = CharacterOps(char_ops_params)
        self.label_ops = CTCLabelDecode(char_ops_params)

    def resize_norm_img(self, img, max_wh_ratio):
        # 定义 resize_norm_img 方法，用于调整和标准化图像
        imgC, imgH, imgW = self.rec_image_shape
        # 获取图像的通道数、高度和宽度
        if self.character_type == "ch":
            imgW = int(imgH * max_wh_ratio)
        # 如果字符类型是中文，调整图像宽度
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        # 计算图像宽高比
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        # 根据宽高比调整图像宽度
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        # 调整图像大小、类型和数值范围
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        # 创建全零填充的图像数组
        padding_im[:, :, 0:resized_w] = resized_image
        # 将调整后的图像放入填充数组
        return padding_im
    # 对输入的图像列表进行预处理，返回归一化后的图像批量
    def preprocess(self, img_list):
        # 获取图像列表的数量
        img_num = len(img_list)
        # 初始化归一化后的图像批量列表
        norm_img_batch = []
        # 设置最大宽高比
        max_wh_ratio = 320/48.
        # 遍历图像列表
        for ino in range(img_num):
            # 获取当前图像的高度和宽度
            h, w = img_list[ino].shape[0:2]
            # 计算当前图像的宽高比
            wh_ratio = w * 1.0 / h
            # 更新最大宽高比
            max_wh_ratio = max(max_wh_ratio, wh_ratio)

        # 再次遍历图像列表
        for ino in range(img_num):
            # 对当前图像进行大小调整和归一化处理
            norm_img = self.resize_norm_img(img_list[ino], max_wh_ratio)
            # 在第0维度上增加一个维度
            norm_img = norm_img[np.newaxis, :]
            # 将处理后的图像添加到归一化图像批量列表中
            norm_img_batch.append(norm_img)
        # 将归一化图像批量列表合并成一个数组
        norm_img_batch = np.concatenate(norm_img_batch)
        # 复制一份归一化图像批量
        norm_img_batch = norm_img_batch.copy()

        # 返回第一个归一化图像
        return norm_img_batch[0]

    # 对模型输出进行后处理，返回识别的文本
    def postprocess(self, outputs, with_score=False):
        # 获取模型输出的预测结果
        preds = list(outputs.values())[0]
        # 尝试将预测结果转换为 numpy 数组
        try:
            preds = preds.numpy()
        except:
            pass
        # 获取预测结果中概率最大的索引
        preds_idx = preds.argmax(axis=2)
        # 获取预测结果中概率最大的值
        preds_prob = preds.max(axis=2)
        # 解码预测结果，得到文本
        text = self.label_ops.decode(
            preds_idx, preds_prob, is_remove_duplicate=True)
        # 返回识别的文本
        return text
# 导入必要的模块
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import yaml

# 定义一个参数解析类，继承自ArgumentParser
class ArgsParser(ArgumentParser):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法，并指定formatter_class为RawDescriptionHelpFormatter
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        # 添加-c和-o参数
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")

    # 解析参数的方法
    def parse_args(self, argv=None):
        # 调用父类的parse_args方法解析参数
        args = super(ArgsParser, self).parse_args(argv)
        # 断言config参数不为空
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        # 解析配置文件并将结果存储在conf_dict中
        args.conf_dict = self._parse_opt(args.opt, args.config)
        # 打印解析结果
        print("args config:", args.conf_dict)
        return args

    # 辅助方法，用于解析配置值的类型
    def _parse_helper(self, v):
        if v.isnumeric():
            if "." in v:
                v = float(v)
            else:
                v = int(v)
        elif v == "True" or v == "False":
            v = (v == "True")
        return v

    # 解析配置文件的方法
    def _parse_opt(self, opts, conf_path):
        # 打开配置文件
        f = open(conf_path)
        # 使用yaml模块加载配置文件内容
        config = yaml.load(f, Loader=yaml.Loader)
        # 如果没有指定配置选项，则直接返回整个配置文件内容
        if not opts:
            return config
        # 遍历配置选项
        for s in opts:
            s = s.strip()
            # 按照等号分割键值对
            k, v = s.split('=')
            # 解析值的类型
            v = self._parse_helper(v)
            print(k, v, type(v))
            cur = config
            parent = cur
            # 根据键的层级结构更新配置文件内容
            for kk in k.split("."):
                if kk not in cur:
                    cur[kk] = {}
                    parent = cur
                    cur = cur[kk]
                else:
                    parent = cur
                    cur = cur[kk]
            parent[k.split(".")[-1]] = v
        return config
```