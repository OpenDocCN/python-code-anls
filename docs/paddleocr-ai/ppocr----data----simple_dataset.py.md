# `.\PaddleOCR\ppocr\data\simple_dataset.py`

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
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制
# 导入所需的库
import numpy as np
import cv2
import math
import os
import json
import random
import traceback
# 从 PaddlePaddle 的 Dataset 模块中导入 Dataset 类
from paddle.io import Dataset
# 从当前目录下的 imaug 模块中导入 transform 和 create_operators 函数
from .imaug import transform, create_operators

# 定义一个名为 SimpleDataSet 的类，继承自 Dataset 类
class SimpleDataSet(Dataset):
    # 初始化 SimpleDataSet 类，传入配置、模式、日志器和种子
    def __init__(self, config, mode, logger, seed=None):
        # 调用父类的初始化方法
        super(SimpleDataSet, self).__init__()
        # 设置日志器和模式
        self.logger = logger
        self.mode = mode.lower()

        # 获取全局配置、数据集配置和加载器配置
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        # 设置分隔符，默认为制表符
        self.delimiter = dataset_config.get('delimiter', '\t')
        # 获取标签文件列表
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        # 获取数据源数量和比例列表
        ratio_list = dataset_config.get("ratio_list", 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        # 检查比例列表长度是否与文件列表长度相同
        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."
        
        # 设置数据目录、是否打乱数据和种子
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']
        self.seed = seed
        # 记录初始化数据集索引
        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        # 获取图像信息列表
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        # 创建数据索引顺序列表
        self.data_idx_order_list = list(range(len(self.data_lines)))
        # 如果是训练模式且需要打乱数据，则打乱数据
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()

        # 设置随机种子
        self.set_epoch_as_seed(self.seed, dataset_config)

        # 创建数据集操作符
        self.ops = create_operators(dataset_config['transforms'], global_config)
        # 获取外部操作变换索引，默认为2
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx",
                                                       2)
        # 检查是否需要重置
        self.need_reset = True in [x < 1 for x in ratio_list]
    # 将当前 epoch 作为随机种子设置，用于数据集配置
    def set_epoch_as_seed(self, seed, dataset_config):
        # 如果处于训练模式
        if self.mode == 'train':
            try:
                # 找到包含 'MakeBorderMap' 的字典在 transforms 列表中的索引
                border_map_id = [index
                    for index, dictionary in enumerate(dataset_config['transforms'])
                    if 'MakeBorderMap' in dictionary][0]
                # 找到包含 'MakeShrinkMap' 的字典在 transforms 列表中的索引
                shrink_map_id = [index
                    for index, dictionary in enumerate(dataset_config['transforms'])
                    if 'MakeShrinkMap' in dictionary][0]
                # 设置 'MakeBorderMap' 的 epoch 属性为当前 epoch 或默认值 0
                dataset_config['transforms'][border_map_id]['MakeBorderMap'][
                    'epoch'] = seed if seed is not None else 0
                # 设置 'MakeShrinkMap' 的 epoch 属性为当前 epoch 或默认值 0
                dataset_config['transforms'][shrink_map_id]['MakeShrinkMap'][
                    'epoch'] = seed if seed is not None else 0
            except Exception as E:
                # 捕获异常并打印错误信息
                print(E)
                return

    # 获取图像信息列表
    def get_image_info_list(self, file_list, ratio_list):
        # 如果 file_list 是字符串，则转换为列表
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        # 遍历文件列表
        for idx, file in enumerate(file_list):
            # 打开文件并读取所有行
            with open(file, "rb") as f:
                lines = f.readlines()
                # 如果处于训练模式或 ratio_list 中的值小于 1.0
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    # 设置随机种子并从 lines 中随机抽样一部分
                    random.seed(self.seed)
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx])
                                          )
                # 将抽样的行添加到 data_lines 中
                data_lines.extend(lines)
        return data_lines

    # 随机打乱数据
    def shuffle_data_random(self):
        # 设置随机种子并打乱数据行
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    # 尝试解析文件名列表
    def _try_parse_filename_list(self, file_name):
        # 如果文件名列表长度大于 0 且第一个字符是 '[', 则处理多个图像到一个 gt 标签的情况
        if len(file_name) > 0 and file_name[0] == "[":
            try:
                # 尝试将文件名列表解析为 JSON 格式，并随机选择一个文件名
                info = json.loads(file_name)
                file_name = random.choice(info)
            except:
                pass
        return file_name
    # 获取扩展数据
    def get_ext_data(self):
        # 初始化扩展数据数量为0
        ext_data_num = 0
        # 遍历操作列表
        for op in self.ops:
            # 如果操作对象有属性'ext_data_num'
            if hasattr(op, 'ext_data_num'):
                # 获取操作对象的'ext_data_num'属性值
                ext_data_num = getattr(op, 'ext_data_num')
                # 结束循环
                break
        # 获取加载数据操作列表
        load_data_ops = self.ops[:self.ext_op_transform_idx]
        # 初始化扩展数据列表
        ext_data = []

        # 当扩展数据列表长度小于扩展数据数量时循环
        while len(ext_data) < ext_data_num:
            # 随机选择一个数据索引
            file_idx = self.data_idx_order_list[np.random.randint(self.__len__())]
            # 获取数据行
            data_line = self.data_lines[file_idx]
            # 将数据行解码为utf-8格式
            data_line = data_line.decode('utf-8')
            # 根据分隔符拆分数据行
            substr = data_line.strip("\n").split(self.delimiter)
            # 获取文件名和标签
            file_name = substr[0]
            # 尝试解析文件名列表
            file_name = self._try_parse_filename_list(file_name)
            # 拼接图像路径
            img_path = os.path.join(self.data_dir, file_name)
            # 创建数据字典
            data = {'img_path': img_path, 'label': label}
            # 如果图像路径不存在则跳过
            if not os.path.exists(img_path):
                continue
            # 读取图像数据
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            # 对数据进行转换操作
            data = transform(data, load_data_ops)

            # 如果数据为空则跳过
            if data is None:
                continue
            # 如果数据字典中包含'polys'键
            if 'polys' in data.keys():
                # 如果'polys'的形状不为(1, 4)则跳过
                if data['polys'].shape[1] != 4:
                    continue
            # 将数据添加到扩展数据列表中
            ext_data.append(data)
        # 返回扩展数据列表
        return ext_data
    # 重载索引操作符，根据索引获取数据
    def __getitem__(self, idx):
        # 获取数据索引
        file_idx = self.data_idx_order_list[idx]
        # 获取数据行
        data_line = self.data_lines[file_idx]
        try:
            # 尝试将数据行解码为 UTF-8 格式
            data_line = data_line.decode('utf-8')
            # 去除换行符并根据分隔符拆分数据行
            substr = data_line.strip("\n").split(self.delimiter)
            # 获取文件名和标签
            file_name = substr[0]
            # 尝试解析文件名列表
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            # 拼接图像路径
            img_path = os.path.join(self.data_dir, file_name)
            # 构建数据字典
            data = {'img_path': img_path, 'label': label}
            # 如果图像路径不存在，则抛出异常
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            # 读取图像数据
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            # 获取额外数据
            data['ext_data'] = self.get_ext_data()
            # 对数据进行转换
            outs = transform(data, self.ops)
        except:
            # 捕获异常并记录日志
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            # 在评估过程中，应该修复索引以确保多次评估结果一致
            rnd_idx = np.random.randint(self.__len__(
            )) if self.mode == "train" else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        return outs

    # 返回数据集的长度
    def __len__(self):
        return len(self.data_idx_order_list)
# 定义一个继承自SimpleDataSet的MultiScaleDataSet类
class MultiScaleDataSet(SimpleDataSet):
    # 初始化方法，接受配置、模式、日志器和种子作为参数
    def __init__(self, config, mode, logger, seed=None):
        # 调用父类SimpleDataSet的初始化方法
        super(MultiScaleDataSet, self).__init__(config, mode, logger, seed)
        # 获取数据集宽度参数
        self.ds_width = config[mode]['dataset'].get('ds_width', False)
        # 如果数据集宽度参数存在，则调用wh_aware方法
        if self.ds_width:
            self.wh_aware()

    # 定义wh_aware方法
    def wh_aware(self):
        # 初始化新的数据行列表和宽高比列表
        data_line_new = []
        wh_ratio = []
        # 遍历数据行
        for lins in self.data_lines:
            # 将数据行添加到新的数据行列表中
            data_line_new.append(lins)
            # 将数据行解码为utf-8格式
            lins = lins.decode('utf-8')
            # 根据分隔符拆分数据行，获取名称、标签、宽度和高度
            name, label, w, h = lins.strip("\n").split(self.delimiter)
            # 计算宽高比并添加到宽高比列表中
            wh_ratio.append(float(w) / float(h))

        # 更新数据行列表和宽高比数组
        self.data_lines = data_line_new
        self.wh_ratio = np.array(wh_ratio)
        # 对宽高比数组进行排序并保存排序后的索引
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        # 生成数据行索引顺序列表
        self.data_idx_order_list = list(range(len(self.data_lines)))

    # 定义resize_norm_img方法，接受数据、图像宽度、图像高度和是否填充作为参数
    def resize_norm_img(self, data, imgW, imgH, padding=True):
        # 获取图像数据和尺寸
        img = data['image']
        h = img.shape[0]
        w = img.shape[1]
        # 如果不填充
        if not padding:
            # 调整图像大小
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
            resized_w = imgW
        else:
            # 计算宽高比
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        # 将调整后的图像转换为float32类型
        resized_image = resized_image.astype('float32')

        # 调整图像通道顺序并进行归一化
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        # 创建填充后的图像数组
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        # 计算有效比例并更新数据
        valid_ratio = min(1.0, float(resized_w / imgW))
        data['image'] = padding_im
        data['valid_ratio'] = valid_ratio
        return data
    # 重载索引操作符，用于获取指定属性的数据
    def __getitem__(self, properties):
        # properties 是一个元组，包含 (宽度，高度，索引)
        img_height = properties[1]
        idx = properties[2]
        # 如果数据集宽度存在且属性中的第四个元素不为空
        if self.ds_width and properties[3] is not None:
            # 获取宽高比
            wh_ratio = properties[3]
            # 计算图片宽度
            img_width = img_height * (1 if int(round(wh_ratio)) == 0 else
                                      int(round(wh_ratio)))
            # 获取文件索引
            file_idx = self.wh_ratio_sort[idx]
        else:
            # 获取文件索引
            file_idx = self.data_idx_order_list[idx]
            # 获取图片宽度
            img_width = properties[0]
            wh_ratio = None

        # 获取数据行
        data_line = self.data_lines[file_idx]
        try:
            # 尝试解码数据行
            data_line = data_line.decode('utf-8')
            # 去除换行符并根据分隔符拆分数据行
            substr = data_line.strip("\n").split(self.delimiter)
            # 获取文件名
            file_name = substr[0]
            # 尝试解析文件名列表
            file_name = self._try_parse_filename_list(file_name)
            # 获取标签
            label = substr[1]
            # 获取图片路径
            img_path = os.path.join(self.data_dir, file_name)
            # 构建数据字典
            data = {'img_path': img_path, 'label': label}
            # 如果图片路径不存在，则抛出异常
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            # 读取图片数据
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            # 获取额外数据
            data['ext_data'] = self.get_ext_data()
            # 对数据进行转换
            outs = transform(data, self.ops[:-1])
            # 如果转换结果不为空
            if outs is not None:
                # 调整图片大小并归一化
                outs = self.resize_norm_img(outs, img_width, img_height)
                # 继续对数据进行转换
                outs = transform(outs, self.ops[-1:])
        except:
            # 捕获异常并记录日志
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()))
            outs = None
        # 如果结果为空
        if outs is None:
            # 在评估过程中，应该修复索引以多次评估时获得相同结果
            rnd_idx = (idx + 1) % self.__len__()
            return self.__getitem__([img_width, img_height, rnd_idx, wh_ratio])
        # 返回结果
        return outs
```