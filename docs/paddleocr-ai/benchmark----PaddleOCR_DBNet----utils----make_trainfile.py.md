# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\utils\make_trainfile.py`

```py
# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 12:06
# @Author  : zhoujun
# 导入必要的库
import os
import glob
import pathlib

# 设置数据路径
data_path = r'test'
# data_path/img 存放图片
# data_path/gt 存放标签文件

# 打开一个文件用于写入数据
f_w = open(os.path.join(data_path, 'test.txt'), 'w', encoding='utf8')

# 遍历 data_path/img 目录下的所有 jpg 图片文件
for img_path in glob.glob(data_path + '/img/*.jpg', recursive=True):
    # 获取图片路径的 Path 对象
    d = pathlib.Path(img_path)
    # 构建标签文件路径
    label_path = os.path.join(data_path, 'gt', ('gt_' + str(d.stem) + '.txt'))
    
    # 如果图片文件和对应的标签文件都存在，则打印它们的路径
    if os.path.exists(img_path) and os.path.exists(label_path):
        print(img_path, label_path)
    else:
        # 如果有文件不存在，则打印提示信息
        print('不存在', img_path, label_path)
    
    # 将图片路径和标签文件路径写入文件
    f_w.write('{}\t{}\n'.format(img_path, label_path))

# 关闭文件
f_w.close()
```