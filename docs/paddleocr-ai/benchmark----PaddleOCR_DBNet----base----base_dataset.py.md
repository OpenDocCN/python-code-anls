# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\base\base_dataset.py`

```
# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 13:12
# @Author  : zhoujun
# 导入copy模块
import copy
# 从paddle.io中导入Dataset类
from paddle.io import Dataset
# 从data_loader.modules中导入所有模块
from data_loader.modules import *


# 定义BaseDataSet类，继承自Dataset类
class BaseDataSet(Dataset):
    # 初始化方法，接受数据路径、图像模式、预处理方法、过滤关键字、忽略标签、转换方法和目标转换方法
    def __init__(self,
                 data_path: str,
                 img_mode,
                 pre_processes,
                 filter_keys,
                 ignore_tags,
                 transform=None,
                 target_transform=None):
        # 断言图像模式为RGB、BRG或GRAY中的一种
        assert img_mode in ['RGB', 'BRG', 'GRAY']
        # 设置忽略标签和加载数据列表
        self.ignore_tags = ignore_tags
        self.data_list = self.load_data(data_path)
        # 确保加载的数据列表包含特定的项目键
        item_keys = [
            'img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags'
        ]
        for item in item_keys:
            assert item in self.data_list[
                0], 'data_list from load_data must contains {}'.format(
                    item_keys)
        # 设置图像模式、过滤关键字、转换方法和目标转换方法
        self.img_mode = img_mode
        self.filter_keys = filter_keys
        self.transform = transform
        self.target_transform = target_transform
        # 初始化预处理方法
        self._init_pre_processes(pre_processes)

    # 初始化预处理方法
    def _init_pre_processes(self, pre_processes):
        # 初始化增强方法列表
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                # 如果预处理方法中没有参数，则设置为空字典
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                # 根据参数类型创建对应的类实例
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                # 将类实例添加到增强方法列表中
                self.aug.append(cls)

    # 加载数据方法，需要在子类中实现
    def load_data(self, data_path: str) -> list:
        """
        把数据加载为一个list：
        :params data_path: 存储数据的文件夹或者文件
        return a dict ,包含了，'img_path','img_name','text_polys','texts','ignore_tags'
        """
        raise NotImplementedError

    # 应用预处理方法
    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data
    # 重载索引操作符，根据索引获取数据
    def __getitem__(self, index):
        try:
            # 深拷贝数据列表中的数据
            data = copy.deepcopy(self.data_list[index])
            # 读取图像数据
            im = cv2.imread(data['img_path'], 1
                            if self.img_mode != 'GRAY' else 0)
            # 如果图像模式为 RGB，则转换颜色空间
            if self.img_mode == 'RGB':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # 将图像数据和形状信息添加到数据中
            data['img'] = im
            data['shape'] = [im.shape[0], im.shape[1]]
            # 应用预处理操作
            data = self.apply_pre_processes(data)

            # 如果存在图像变换操作，则对图像进行变换
            if self.transform:
                data['img'] = self.transform(data['img'])
            # 将文本多边形坐标转换为列表形式
            data['text_polys'] = data['text_polys'].tolist()
            # 如果存在过滤键列表
            if len(self.filter_keys):
                data_dict = {}
                # 遍历数据字典，将不在过滤键列表中的键值对添加到新的字典中
                for k, v in data.items():
                    if k not in self.filter_keys:
                        data_dict[k] = v
                return data_dict
            else:
                return data
        except:
            # 如果出现异常，则随机选择索引重新获取数据
            return self.__getitem__(np.random.randint(self.__len__()))

    # 返回数据列表的长度
    def __len__(self):
        return len(self.data_list)
```