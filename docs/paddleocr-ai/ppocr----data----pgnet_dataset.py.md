# `.\PaddleOCR\ppocr\data\pgnet_dataset.py`

```py
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
# 请查看许可证以获取有关权限和限制的详细信息
import numpy as np
import os
from paddle.io import Dataset
from .imaug import transform, create_operators
import random

# 定义 PGDataSet 类，继承自 Dataset 类
class PGDataSet(Dataset):
    # 初始化 PGDataSet 类，接受配置、模式、日志器和种子作为参数
    def __init__(self, config, mode, logger, seed=None):
        # 调用父类 PGDataSet 的初始化方法
        super(PGDataSet, self).__init__()

        # 设置日志器和种子
        self.logger = logger
        self.seed = seed
        self.mode = mode
        # 获取全局配置、数据集配置和加载器配置
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        # 设置分隔符，默认为制表符
        self.delimiter = dataset_config.get('delimiter', '\t')
        # 获取标签文件列表
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        # 获取数据比例列表，默认为1.0
        ratio_list = dataset_config.get("ratio_list", [1.0])
        # 如果比例列表为单个数值，则复制为与文件列表长度相同的列表
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)
        # 检查比例列表长度与文件列表长度是否相同
        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."
        # 设置数据目录和是否打乱数据
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        # 记录初始化数据集索引信息
        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        # 获取图像信息列表
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        # 创建数据索引顺序列表
        self.data_idx_order_list = list(range(len(self.data_lines)))
        # 如果模式为训练，则随机打乱数据
        if mode.lower() == "train":
            self.shuffle_data_random()

        # 创建数据集操作符
        self.ops = create_operators(dataset_config['transforms'], global_config)

        # 检查是否需要重置数据
        self.need_reset = True in [x < 1 for x in ratio_list]

    # 随机打乱数据
    def shuffle_data_random(self):
        # 如果需要打乱数据
        if self.do_shuffle:
            # 设置随机种子并打乱数据行
            random.seed(self.seed)
            random.shuffle(self.data_lines)
        return
    # 获取图像信息列表，根据文件列表和比例列表
    def get_image_info_list(self, file_list, ratio_list):
        # 如果文件列表是字符串，则转换为列表
        if isinstance(file_list, str):
            file_list = [file_list]
        # 初始化数据行列表
        data_lines = []
        # 遍历文件列表
        for idx, file in enumerate(file_list):
            # 打开文件
            with open(file, "rb") as f:
                # 读取文件的所有行
                lines = f.readlines()
                # 如果模式是训练模式或者比例小于1.0
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    # 设置随机种子
                    random.seed(self.seed)
                    # 从文件行中随机抽样一定比例的行
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx])
                # 将抽样的行添加到数据行列表中
                data_lines.extend(lines)
        # 返回数据行列表
        return data_lines

    # 获取数据集中指定索引的数据
    def __getitem__(self, idx):
        # 获取文件索引
        file_idx = self.data_idx_order_list[idx]
        # 获取数据行
        data_line = self.data_lines[file_idx]
        # 初始化图像ID
        img_id = 0
        try:
            # 将数据行解码为UTF-8格式
            data_line = data_line.decode('utf-8')
            # 根据分隔符拆分数据行
            substr = data_line.strip("\n").split(self.delimiter)
            # 获取文件名和标签
            file_name = substr[0]
            label = substr[1]
            # 获取图像路径
            img_path = os.path.join(self.data_dir, file_name)
            # 如果模式是评估模式
            if self.mode.lower() == 'eval':
                try:
                    # 提取图像ID
                    img_id = int(data_line.split(".")[0][7:])
                except:
                    img_id = 0
            # 构建数据字典
            data = {'img_path': img_path, 'label': label, 'img_id': img_id}
            # 如果图像路径不存在，则抛出异常
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            # 打开图像文件并读取图像数据
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            # 对数据进行转换操作
            outs = transform(data, self.ops)
        except Exception as e:
            # 捕获异常并记录日志
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    self.data_idx_order_list[idx], e))
            outs = None
        # 如果处理结果为空，则随机选择一个索引重新获取数据
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        # 返回处理结果
        return outs

    # 获取数据集的长度
    def __len__(self):
        return len(self.data_idx_order_list)
```