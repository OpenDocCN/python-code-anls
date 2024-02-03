# `.\PaddleOCR\ppocr\data\lmdb_dataset.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
import numpy as np
import os
from paddle.io import Dataset
import lmdb
import cv2
import string
import six
import pickle
from PIL import Image

from .imaug import transform, create_operators

# 定义 LMDBDataSet 类，继承自 Dataset 类
class LMDBDataSet(Dataset):
    # 初始化方法，接受配置、模式、日志器和种子作为参数
    def __init__(self, config, mode, logger, seed=None):
        # 调用父类的初始化方法
        super(LMDBDataSet, self).__init__()

        # 获取全局配置、数据集配置和加载器配置
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        batch_size = loader_config['batch_size_per_card']
        data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        # 加载分层 LMDB 数据集
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir)
        logger.info("Initialize indexs of datasets:%s" % data_dir)
        
        # 遍历数据集，获取数据索引顺序列表
        self.data_idx_order_list = self.dataset_traversal()
        
        # 如果需要打乱数据集顺序，则进行打乱
        if self.do_shuffle:
            np.random.shuffle(self.data_idx_order_list)
        
        # 创建数据增强操作符
        self.ops = create_operators(dataset_config['transforms'], global_config)
        
        # 获取外部操作符的索引
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 1)

        # 获取比例列表，用于判断是否需要重置数据集
        ratio_list = dataset_config.get("ratio_list", [1.0])
        self.need_reset = True in [x < 1 for x in ratio_list]
    # 加载层次化 LMDB 数据集，返回数据集字典
    def load_hierarchical_lmdb_dataset(self, data_dir):
        # 初始化 LMDB 数据集字典
        lmdb_sets = {}
        # 初始化数据集索引
        dataset_idx = 0
        # 遍历指定目录下的所有文件夹
        for dirpath, dirnames, filenames in os.walk(data_dir + '/'):
            # 如果当前目录没有子文件夹
            if not dirnames:
                # 打开 LMDB 环境
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)
                # 开始 LMDB 事务
                txn = env.begin(write=False)
                # 获取当前 LMDB 数据集的样本数量
                num_samples = int(txn.get('num-samples'.encode()))
                # 将当前数据集信息存入数据集字典
                lmdb_sets[dataset_idx] = {"dirpath":dirpath, "env":env, \
                    "txn":txn, "num_samples":num_samples}
                # 更新数据集索引
                dataset_idx += 1
        # 返回 LMDB 数据集字典
        return lmdb_sets

    # 遍历数据集，返回数据索引顺序列表
    def dataset_traversal(self):
        # 获取 LMDB 数据集数量
        lmdb_num = len(self.lmdb_sets)
        # 初始化总样本数量
        total_sample_num = 0
        # 计算总样本数量
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['num_samples']
        # 初始化数据索引顺序列表
        data_idx_order_list = np.zeros((total_sample_num, 2))
        # 初始化起始索引
        beg_idx = 0
        # 遍历 LMDB 数据集
        for lno in range(lmdb_num):
            # 获取当前数据集的样本数量
            tmp_sample_num = self.lmdb_sets[lno]['num_samples']
            # 计算结束索引
            end_idx = beg_idx + tmp_sample_num
            # 填充数据索引顺序列表
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] \
                = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            # 更新起始索引
            beg_idx = beg_idx + tmp_sample_num
        # 返回数据索引顺序列表
        return data_idx_order_list

    # 获取图像数据
    def get_img_data(self, value):
        """get_img_data"""
        # 如果值为空，返回空
        if not value:
            return None
        # 将值转换为 numpy 数组
        imgdata = np.frombuffer(value, dtype='uint8')
        # 如果转换失败，返回空
        if imgdata is None:
            return None
        # 解码图像数据
        imgori = cv2.imdecode(imgdata, 1)
        # 如果解码失败，返回空
        if imgori is None:
            return None
        # 返回解码后的图像数据
        return imgori
    # 获取扩展数据
    def get_ext_data(self):
        # 初始化扩展数据数量为0
        ext_data_num = 0
        # 遍历操作列表
        for op in self.ops:
            # 如果操作对象具有'ext_data_num'属性
            if hasattr(op, 'ext_data_num'):
                # 获取'ext_data_num'属性的值
                ext_data_num = getattr(op, 'ext_data_num')
                break
        # 获取加载数据操作列表
        load_data_ops = self.ops[:self.ext_op_transform_idx]
        # 初始化扩展数据列表
        ext_data = []

        # 当扩展数据列表长度小于扩展数据数量时循环
        while len(ext_data) < ext_data_num:
            # 随机选择LMDB索引和文件索引
            lmdb_idx, file_idx = self.data_idx_order_list[np.random.randint(len(self))]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            # 获取LMDB样本信息
            sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'], file_idx)
            # 如果样本信息为空，则继续下一次循环
            if sample_info is None:
                continue
            img, label = sample_info
            data = {'image': img, 'label': label}
            # 对数据进行转换操作
            data = transform(data, load_data_ops)
            # 如果数据为空，则继续下一次循环
            if data is None:
                continue
            # 将数据添加到扩展数据列表中
            ext_data.append(data)
        # 返回扩展数据列表
        return ext_data

    # 获取LMDB样本信息
    def get_lmdb_sample_info(self, txn, index):
        # 构建标签键
        label_key = 'label-%09d'.encode() % index
        # 获取标签数据
        label = txn.get(label_key)
        # 如果标签为空，则返回None
        if label is None:
            return None
        # 解码标签数据
        label = label.decode('utf-8')
        # 构建图像键
        img_key = 'image-%09d'.encode() % index
        # 获取图像数据
        imgbuf = txn.get(img_key)
        # 返回图像数据和标签数据
        return imgbuf, label

    # 获取指定索引的数据
    def __getitem__(self, idx):
        # 获取LMDB索引和文件索引
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        # 获取LMDB样本信息
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'], file_idx)
        # 如果样本信息为空，则重新获取随机索引的数据
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        img, label = sample_info
        data = {'image': img, 'label': label}
        # 获取扩展数据并添加到数据中
        data['ext_data'] = self.get_ext_data()
        # 对数据进行转换操作
        outs = transform(data, self.ops)
        # 如果输出数据为空，则重新获取随机索引的数据
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        # 返回输出数据
        return outs
    # 返回数据索引顺序列表的长度作为对象的长度
    def __len__(self):
        return self.data_idx_order_list.shape[0]
class LMDBDataSetSR(LMDBDataSet):
    # 继承自LMDBDataSet的LMDBDataSetSR类

    def buf2PIL(self, txn, key, type='RGB'):
        # 将LMDB事务中的图像数据转换为PIL图像对象
        imgbuf = txn.get(key)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = Image.open(buf).convert(type)
        return im

    def str_filt(self, str_, voc_type):
        # 根据给定的词汇类型过滤字符串
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation
        }
        if voc_type == 'lower':
            str_ = str_.lower()
        for char in str_:
            if char not in alpha_dict[voc_type]:
                str_ = str_.replace(char, '')
        return str_

    def get_lmdb_sample_info(self, txn, index):
        # 获取LMDB数据集中指定索引的样本信息
        self.voc_type = 'upper'
        self.max_len = 100
        self.test = False
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = self.buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = self.buf2PIL(txn, img_lr_key, 'RGB')
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = self.str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str
    # 重载索引操作符，根据索引获取数据
    def __getitem__(self, idx):
        # 从数据索引顺序列表中获取 LMDB 索引和文件索引
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        # 将 LMDB 索引和文件索引转换为整数类型
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        # 获取 LMDB 数据库中指定索引的样本信息
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'],
                                                file_idx)
        # 如果样本信息为空，则随机选择一个索引重新获取数据
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        # 从样本信息中获取高分辨率图像、低分辨率图像和标签字符串
        img_HR, img_lr, label_str = sample_info
        # 将图像数据和标签字符串组成字典
        data = {'image_hr': img_HR, 'image_lr': img_lr, 'label': label_str}
        # 对数据进行转换操作
        outs = transform(data, self.ops)
        # 如果转换结果为空，则随机选择一个索引重新获取数据
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        # 返回转换后的数据
        return outs
class LMDBDataSetTableMaster(LMDBDataSet):
    # 定义一个继承自LMDBDataSet的LMDBDataSetTableMaster类

    def load_hierarchical_lmdb_dataset(self, data_dir):
        # 加载分层LMDB数据集
        lmdb_sets = {}
        # 初始化空字典lmdb_sets
        dataset_idx = 0
        # 初始化数据集索引为0
        env = lmdb.open(
            data_dir,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        # 打开LMDB环境，设置参数
        txn = env.begin(write=False)
        # 开始LMDB事务，只读模式
        num_samples = int(pickle.loads(txn.get(b"__len__")))
        # 从LMDB事务中获取数据集样本数量
        lmdb_sets[dataset_idx] = {"dirpath":data_dir, "env":env, \
            "txn":txn, "num_samples":num_samples}
        # 将数据集信息存入lmdb_sets字典
        return lmdb_sets
        # 返回lmdb_sets字典

    def get_img_data(self, value):
        """get_img_data"""
        # 获取图像数据
        if not value:
            return None
        # 如果值为空，则返回空
        imgdata = np.frombuffer(value, dtype='uint8')
        # 从值中获取图像数据，数据类型为uint8
        if imgdata is None:
            return None
        # 如果图像数据为空，则返回空
        imgori = cv2.imdecode(imgdata, 1)
        # 使用OpenCV解码图像数据
        if imgori is None:
            return None
        # 如果解码后的图像为空，则返回空
        return imgori
        # 返回解码后的图像数据
    # 从 LMDB 数据库中获取指定索引的样本信息
    def get_lmdb_sample_info(self, txn, index):
        # 定义一个函数，将字符串形式的边界框转换为整数列表
        def convert_bbox(bbox_str_list):
            bbox_list = []
            for bbox_str in bbox_str_list:
                bbox_list.append(int(bbox_str))
            return bbox_list

        try:
            # 从 LMDB 数据库中获取指定索引的数据，并反序列化
            data = pickle.loads(txn.get(str(index).encode('utf8')))
        except:
            return None

        # 从数据中获取文件名、字节流和信息行
        file_name = data[0]
        bytes = data[1]
        info_lines = data[2]  # TableMASTER 注释文件中的原始数据
        # 解析信息行
        raw_data = info_lines.strip().split('\n')
        raw_name, text = raw_data[0], raw_data[1]  # 不过滤超过最大序列长度的样本
        text = text.split(',')
        bbox_str_list = raw_data[2:]
        bbox_split = ','
        # 解析边界框数据
        bboxes = [{
            'bbox': convert_bbox(bsl.strip().split(bbox_split)),
            'tokens': ['1', '2']
        } for bsl in bbox_str_list]

        # 进一步解析边界框
        # import pdb;pdb.set_trace()

        # 构建包含样本信息的字典
        line_info = {}
        line_info['file_name'] = file_name
        line_info['structure'] = text
        line_info['cells'] = bboxes
        line_info['image'] = bytes
        return line_info

    # 获取指定索引的数据样本
    def __getitem__(self, idx):
        # 获取 LMDB 索引和文件索引
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        # 获取样本信息
        data = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'],
                                         file_idx)
        if data is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        # 对数据进行转换
        outs = transform(data, self.ops)
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs

    # 返回数据集的长度
    def __len__(self):
        return self.data_idx_order_list.shape[0]
```