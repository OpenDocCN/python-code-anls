# `.\PaddleOCR\ppocr\data\pubtab_dataset.py`

```
# 版权声明，版权所有 (c) 2021 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本:
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言下的权限和限制。
import numpy as np
import os
import random
from paddle.io import Dataset
import json
from copy import deepcopy

from .imaug import transform, create_operators

# 定义 PubTabDataSet 类，继承自 Dataset 类
class PubTabDataSet(Dataset):
    # PubTabDataSet 类的初始化方法，接受配置、模式、日志器和种子作为参数
    def __init__(self, config, mode, logger, seed=None):
        # 调用父类的初始化方法
        super(PubTabDataSet, self).__init__()
        # 将传入的日志器保存到实例变量中
        self.logger = logger

        # 获取全局配置、数据集配置和加载器配置
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        # 从数据集配置中弹出标签文件列表
        label_file_list = dataset_config.pop('label_file_list')
        # 获取标签文件列表的长度
        data_source_num = len(label_file_list)
        # 获取数据集配置中的比例列表，如果没有则默认为[1.0]
        ratio_list = dataset_config.get("ratio_list", [1.0])
        # 如果比例列表是单个数值，则将其转换为与文件列表长度相同的列表
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        # 断言比例列表的长度与文件列表长度相同
        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."

        # 保存数据集目录和是否打乱数据的标志
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        # 保存种子和模式，并输出日志信息
        self.seed = seed
        self.mode = mode.lower()
        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        
        # 获取图像信息列表，根据比例列表处理数据
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        # self.check(config['Global']['max_text_length'])

        # 如果是训练模式且需要打乱数据，则进行随机打乱
        if mode.lower() == "train" and self.do_shuffle:
            self.shuffle_data_random()
        
        # 创建数据集操作符，并判断是否需要重置数据
        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.need_reset = True in [x < 1 for x in ratio_list]

    # 获取图像信息列表的方法，根据文件列表和比例列表处理数据
    def get_image_info_list(self, file_list, ratio_list):
        # 如果文件列表是字符串，则转换为列表
        if isinstance(file_list, str):
            file_list = [file_list]
        # 初始化数据行列表
        data_lines = []
        # 遍历文件列表
        for idx, file in enumerate(file_list):
            # 打开文件并读取所有行
            with open(file, "rb") as f:
                lines = f.readlines()
                # 如果是训练模式或比例小于1.0，则根据比例随机采样数据
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx])
                                          )
                # 将处理后的数据行添加到数据行列表中
                data_lines.extend(lines)
        # 返回处理后的数据行列表
        return data_lines
    # 检查数据，筛选符合条件的数据行
    def check(self, max_text_length):
        # 存储符合条件的数据行
        data_lines = []
        # 遍历数据行
        for line in self.data_lines:
            # 将数据行解码为 UTF-8 格式并去除换行符
            data_line = line.decode('utf-8').strip("\n")
            # 将数据行转换为 JSON 格式
            info = json.loads(data_line)
            # 获取文件名
            file_name = info['filename']
            # 复制 HTML 中的单元格数据
            cells = info['html']['cells'].copy()
            # 复制 HTML 结构中的标记
            structure = info['html']['structure']['tokens'].copy()

            # 拼接图片路径
            img_path = os.path.join(self.data_dir, file_name)
            # 如果图片路径不存在，则记录警告信息并继续下一轮循环
            if not os.path.exists(img_path):
                self.logger.warning("{} does not exist!".format(img_path))
                continue
            # 如果结构为空或长度超过最大文本长度，则继续下一轮循环
            if len(structure) == 0 or len(structure) > max_text_length:
                continue
            # 将符合条件的数据行添加到列表中
            data_lines.append(line)
        # 更新数据行列表
        self.data_lines = data_lines

    # 随机打乱数据行顺序
    def shuffle_data_random(self):
        # 如果需要打乱数据行顺序
        if self.do_shuffle:
            # 设置随机种子
            random.seed(self.seed)
            # 随机打乱数据行顺序
            random.shuffle(self.data_lines)
        # 返回
        return
    # 重载索引操作符，用于获取指定索引位置的数据
    def __getitem__(self, idx):
        # 尝试获取指定索引位置的数据行
        try:
            data_line = self.data_lines[idx]
            # 将数据行解码为 UTF-8 格式并去除换行符
            data_line = data_line.decode('utf-8').strip("\n")
            # 将数据行解析为 JSON 格式
            info = json.loads(data_line)
            # 获取文件名
            file_name = info['filename']
            # 复制 cells 和 structure 数据
            cells = info['html']['cells'].copy()
            structure = info['html']['structure']['tokens'].copy()

            # 拼接图片路径
            img_path = os.path.join(self.data_dir, file_name)
            # 如果图片路径不存在，则抛出异常
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            # 构建数据字典
            data = {
                'img_path': img_path,
                'cells': cells,
                'structure': structure,
                'file_name': file_name
            }

            # 读取图片数据并存储在数据字典中
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            # 对数据进行转换操作
            outs = transform(data, self.ops)
        # 捕获异常
        except:
            import traceback
            err = traceback.format_exc()
            # 记录错误日志
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, err))
            outs = None
        # 如果输出为空，则根据模式选择随机索引或下一个索引
        if outs is None:
            rnd_idx = np.random.randint(self.__len__(
            )) if self.mode == "train" else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        return outs

    # 返回数据行数
    def __len__(self):
        return len(self.data_lines)
```