# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\utils\util.py`

```py
# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:59
# @Author  : zhoujun
# 导入所需的模块
import json
import pathlib
import time
import os
import glob
import cv2
import yaml
from typing import Mapping
import matplotlib.pyplot as plt
import numpy as np

# 导入 ArgumentParser 和 RawDescriptionHelpFormatter 类
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# 检查文件是否为图片文件
def _check_image_file(path):
    # 定义图片文件的后缀名集合
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    # 检查文件路径是否以图片文件后缀结尾
    return any([path.lower().endswith(e) for e in img_end])

# 获取图片文件列表
def get_image_file_list(img_file):
    # 初始化图片文件列表
    imgs_lists = []
    # 如果图片文件为空或不存在，则抛出异常
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    # 定义图片文件的后缀名集合
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    # 如果图片文件是文件且为图片文件，则将其添加到图片文件列表中
    if os.path.isfile(img_file) and _check_image_file(img_file):
        imgs_lists.append(img_file)
    # 如果图片文件是目录，则遍历目录下的文件，将图片文件添加到图片文件列表中
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                imgs_lists.append(file_path)
    # 如果图片文件列表为空，则抛出异常
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    # 对图片文件列表进行排序
    imgs_lists = sorted(imgs_lists)
    return imgs_lists

# 设置日志记录器
def setup_logger(log_file_path: str=None):
    import logging
    # 禁止警告输出到标准错误流
    logging._warn_preinit_stderr = 0
    # 获取名为 'DBNet.paddle' 的日志记录器
    logger = logging.getLogger('DBNet.paddle')
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s: %(message)s')
    # 创建控制台日志处理器
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # 如果指定了日志文件路径，则创建文件日志处理器
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    # 设置日志记录器的日志级别为 DEBUG
    logger.setLevel(logging.DEBUG)
    return logger

# 计算函数执行时间的装饰器
def exe_time(func):
    # 定义一个新的函数，接受任意数量的位置参数和关键字参数
    def newFunc(*args, **args2):
        # 记录函数开始执行的时间
        t0 = time.time()
        # 调用原始函数，并获取返回值
        back = func(*args, **args2)
        # 打印函数执行时间
        print("{} cost {:.3f}s".format(func.__name__, time.time() - t0))
        # 返回原始函数的返回值
        return back

    # 返回新定义的函数
    return newFunc
# 加载文件，根据文件后缀选择对应的加载函数
def load(file_path: str):
    # 将文件路径转换为 Path 对象
    file_path = pathlib.Path(file_path)
    # 定义文件后缀与加载函数的映射关系
    func_dict = {'.txt': _load_txt, '.json': _load_json, '.list': _load_txt}
    # 确保文件后缀在映射关系中
    assert file_path.suffix in func_dict
    # 调用对应的加载函数并返回结果
    return func_dict[file_path.suffix](file_path)


# 加载文本文件
def _load_txt(file_path: str):
    # 打开文本文件，读取内容并去除空白字符
    with open(file_path, 'r', encoding='utf8') as f:
        content = [
            x.strip().strip('\ufeff').strip('\xef\xbb\xbf')
            for x in f.readlines()
        ]
    return content


# 加载 JSON 文件
def _load_json(file_path: str):
    # 打开 JSON 文件，加载内容并返回
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


# 保存数据到文件
def save(data, file_path):
    # 将文件路径转换为 Path 对象
    file_path = pathlib.Path(file_path)
    # 定义文件后缀与保存函数的映射关系
    func_dict = {'.txt': _save_txt, '.json': _save_json}
    # 确保文件后缀在映射关系中
    assert file_path.suffix in func_dict
    # 调用对应的保存函数并返回结果
    return func_dict[file_path.suffix](data, file_path)


# 保存数据到文本文件
def _save_txt(data, file_path):
    """
    将一个list的数组写入txt文件里
    :param data:
    :param file_path:
    :return:
    """
    # 如果数据不是列表，则转换为列表
    if not isinstance(data, list):
        data = [data]
    # 打开文本文件，写入数据并换行
    with open(file_path, mode='w', encoding='utf8') as f:
        f.write('\n'.join(data))


# 保存数据到 JSON 文件
def _save_json(data, file_path):
    # 打开 JSON 文件，将数据以 JSON 格式写入文件
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


# 显示图像
def show_img(imgs: np.ndarray, title='img'):
    # 判断图像是否为彩色
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    # 将图像扩展一个维度
    imgs = np.expand_dims(imgs, axis=0)
    # 遍历图像并显示
    for i, img in enumerate(imgs):
        plt.figure()
        plt.title('{}_{}'.format(title, i))
        plt.imshow(img, cmap=None if color else 'gray')
    plt.show()


# 绘制边界框
def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    # 如果图像路径为字符串，则读取图像
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    # 复制图像
    img_path = img_path.copy()
    # 遍历边界框的点并绘制多边形
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path
# 计算文本分数
def cal_text_score(texts,
                   gt_texts,
                   training_masks,
                   running_metric_text,
                   thred=0.5):
    # 将训练掩码转换为 numpy 数组
    training_masks = training_masks.numpy()
    # 计算预测文本
    pred_text = texts.numpy() * training_masks
    pred_text[pred_text <= thred] = 0
    pred_text[pred_text > thred] = 1
    pred_text = pred_text.astype(np.int32)
    # 计算真实文本
    gt_text = gt_texts.numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    # 更新文本度量
    running_metric_text.update(gt_text, pred_text)
    # 获取文本分数
    score_text, _ = running_metric_text.get_scores()
    return score_text


# 按顺时针顺序排列点
def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 按顺时针顺序排列点列表
def order_points_clockwise_list(pts):
    pts = pts.tolist()
    pts.sort(key=lambda x: (x[1], x[0]))
    pts[:2] = sorted(pts[:2], key=lambda x: x[0])
    pts[2:] = sorted(pts[2:], key=lambda x: -x[0])
    pts = np.array(pts)
    return pts


# 获取数据列表
def get_datalist(train_data_path):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :return:
    """
    train_data = []
    for p in train_data_path:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                if len(line) > 1:
                    img_path = pathlib.Path(line[0].strip(' '))
                    label_path = pathlib.Path(line[1].strip(' '))
                    if img_path.exists() and img_path.stat(
                    ).st_size > 0 and label_path.exists() and label_path.stat(
                    ).st_size > 0:
                        train_data.append((str(img_path), str(label_path)))
    return train_data
# 保存检测结果到文件
def save_result(result_path, box_list, score_list, is_output_polygon):
    # 如果需要输出多边形结果
    if is_output_polygon:
        # 打开结果文件，写入每个框的坐标和得分
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                # 将框的坐标转换为字符串格式
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                score = score_list[i]
                # 写入结果文件
                res.write(result + ',' + str(score) + "\n")
    else:
        # 如果不需要输出多边形结果
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                score = score_list[i]
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                # 写入结果文件
                res.write(result + ',' + str(score) + "\n")

# 对只有一个字符的框进行扩充
def expand_polygon(polygon):
    # 获取最小外接矩形的信息
    (x, y), (w, h), angle = cv2.minAreaRect(np.float32(polygon))
    # 调整角度和宽高
    if angle < -45:
        w, h = h, w
        angle += 90
    new_w = w + h
    box = ((x, y), (new_w, h), angle)
    points = cv2.boxPoints(box)
    return order_points_clockwise(points)

# 递归合并字典
def _merge_dict(config, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    Args:
        config: dict onto which the merge is executed
        merge_dct: dct merged into config
    Returns: dct
    """
    for key, value in merge_dct.items():
        sub_keys = key.split('.')
        key = sub_keys[0]
        # 如果键存在且有子键
        if key in config and len(sub_keys) > 1:
            _merge_dict(config[key], {'.'.join(sub_keys[1:]): value})
        # 如果键存在且是字典类型
        elif key in config and isinstance(config[key], dict) and isinstance(
                value, Mapping):
            _merge_dict(config[key], value)
        else:
            config[key] = value
    return config

# 递归打印字典
def print_dict(cfg, print_func=print, delimiter=0):
    """
    Recursively visualize a dict and
    # 根据键的关系进行缩进
    """
    # 遍历配置字典中的键值对，按键排序
    for k, v in sorted(cfg.items()):
        # 如果值是字典类型，则递归打印字典内容
        if isinstance(v, dict):
            # 打印键，并在前面添加指定数量的分隔符空格
            print_func("{}{} : ".format(delimiter * " ", str(k)))
            # 递归打印字典内容，缩进增加4
            print_dict(v, print_func, delimiter + 4)
        # 如果值是列表类型且第一个元素是字典类型，则递归打印列表中的字典内容
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            # 打印键，并在前面添加指定数量的分隔符空格
            print_func("{}{} : ".format(delimiter * " ", str(k)))
            # 遍历列表中的字典，递归打印字典内容，缩进增加4
            for value in v:
                print_dict(value, print_func, delimiter + 4)
        # 否则直接打印键值对
        else:
            # 打印键值对，并在前面添加指定数量的分隔符空格
            print_func("{}{} : {}".format(delimiter * " ", k, v))
class Config(object):
    # 配置类，用于加载和保存配置信息
    def __init__(self, config_path, BASE_KEY='base'):
        # 初始化函数，设置基础键和加载配置信息
        self.BASE_KEY = BASE_KEY
        self.cfg = self._load_config_with_base(config_path)

    def _load_config_with_base(self, file_path):
        """
        Load config from file.
        Args:
            file_path (str): Path of the config file to be loaded.
        Returns: global config
        """
        # 加载配置文件并返回全局配置信息
        _, ext = os.path.splitext(file_path)
        assert ext in ['.yml', '.yaml'], "only support yaml files for now"

        with open(file_path) as f:
            file_cfg = yaml.load(f, Loader=yaml.Loader)

        # NOTE: cfgs outside have higher priority than cfgs in _BASE_
        # 注意：外部配置优先级高于基础配置
        if self.BASE_KEY in file_cfg:
            all_base_cfg = dict()
            base_ymls = list(file_cfg[self.BASE_KEY])
            for base_yml in base_ymls:
                with open(base_yml) as f:
                    base_cfg = self._load_config_with_base(base_yml)
                    all_base_cfg = _merge_dict(all_base_cfg, base_cfg)

            del file_cfg[self.BASE_KEY]
            file_cfg = _merge_dict(all_base_cfg, file_cfg)
        file_cfg['filename'] = os.path.splitext(os.path.split(file_path)[-1])[0]
        return file_cfg

    def merge_dict(self, args):
        # 合并字典
        self.cfg = _merge_dict(self.cfg, args)

    def print_cfg(self, print_func=print):
        """
        Recursively visualize a dict and
        indenting acrrording by the relationship of keys.
        """
        # 递归可视化字典，并根据键的关系进行缩进
        print_func('----------- Config -----------')
        print_dict(self.cfg, print_func)
        print_func('---------------------------------------------')

    def save(self, p):
        # 保存配置信息到文件
        with open(p, 'w') as f:
            yaml.dump(
                dict(self.cfg), f, default_flow_style=False, sort_keys=False)


class ArgsParser(ArgumentParser):
    # 初始化参数解析器，设置 formatter_class 为 RawDescriptionHelpFormatter
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        # 添加参数 -c 或 --config_file，用于指定配置文件
        self.add_argument(
            "-c", "--config_file", help="configuration file to use")
        # 添加参数 -o 或 --opt，允许多个值，用于设置配置选项
        self.add_argument(
            "-o", "--opt", nargs='*', help="set configuration options")
        # 添加参数 -p 或 --profiler_options，类型为字符串，默认为 None，用于设置性能分析器的选项
        self.add_argument(
            '-p',
            '--profiler_options',
            type=str,
            default=None,
            help='The option of profiler, which should be in format ' \
                 '\"key1=value1;key2=value2;key3=value3\".'
        )
    
    # 解析参数
    def parse_args(self, argv=None):
        # 调用父类的 parse_args 方法解析参数
        args = super(ArgsParser, self).parse_args(argv)
        # 断言 config_file 参数不为空
        assert args.config_file is not None, \
            "Please specify --config_file=configure_file_path."
        # 解析 opt 参数
        args.opt = self._parse_opt(args.opt)
        return args
    
    # 解析配置选项
    def _parse_opt(self, opts):
        config = {}
        # 如果 opts 为空，则返回空字典
        if not opts:
            return config
        # 遍历 opts
        for s in opts:
            # 去除空格
            s = s.strip()
            # 以等号分割键值对
            k, v = s.split('=', 1)
            # 如果键中不包含点，则直接加载值到配置字典中
            if '.' not in k:
                config[k] = yaml.load(v, Loader=yaml.Loader)
            else:
                keys = k.split('.')
                # 如果键的第一个部分不在配置字典中，则创建一个新的字典
                if keys[0] not in config:
                    config[keys[0]] = {}
                cur = config[keys[0]]
                # 遍历键的剩余部分
                for idx, key in enumerate(keys[1:]):
                    # 如果是最后一个键，则加载值到当前位置
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        # 否则创建一个新的字典，并移动到下一层
                        cur[key] = {}
                        cur = cur[key]
        return config
# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 创建一个形状为(1, 3, 640, 640)的全零数组
    img = np.zeros((1, 3, 640, 640))
    # 显示数组中第一个元素的第一个通道的图像
    show_img(img[0][0])
    # 显示图像
    plt.show()
```