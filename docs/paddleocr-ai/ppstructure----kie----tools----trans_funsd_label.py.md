# `.\PaddleOCR\ppstructure\kie\tools\trans_funsd_label.py`

```py
# 版权声明
#
# 版权所有 (c) 2022 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言下的权限和限制。

import json
import os
import sys
import cv2
import numpy as np
from copy import deepcopy

# 将多边形转换为边界框
def trans_poly_to_bbox(poly):
    x1 = np.min([p[0] for p in poly])
    x2 = np.max([p[0] for p in poly])
    y1 = np.min([p[1] for p in poly])
    y2 = np.max([p[1] for p in poly])
    return [x1, y1, x2, y2]

# 获取外部多边形
def get_outer_poly(bbox_list):
    x1 = min([bbox[0] for bbox in bbox_list])
    y1 = min([bbox[1] for bbox in bbox_list])
    x2 = max([bbox[2] for bbox in bbox_list])
    y2 = max([bbox[3] for bbox in bbox_list])
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

# 加载 FUNSD 标签
def load_funsd_label(image_dir, anno_dir):
    imgs = os.listdir(image_dir)
    annos = os.listdir(anno_dir)

    imgs = [img.replace(".png", "") for img in imgs]
    annos = [anno.replace(".json", "") for anno in annos]

    fn_info_map = dict()
    return fn_info_map

# 主函数
def main():
    test_image_dir = "train_data/FUNSD/testing_data/images/"
    test_anno_dir = "train_data/FUNSD/testing_data/annotations/"
    test_output_dir = "train_data/FUNSD/test.json"

    fn_info_map = load_funsd_label(test_image_dir, test_anno_dir)
    with open(test_output_dir, "w") as fout:
        for fn in fn_info_map:
            fout.write(fn + ".png" + "\t" + json.dumps(
                fn_info_map[fn], ensure_ascii=False) + "\n")

    train_image_dir = "train_data/FUNSD/training_data/images/"
    # 训练数据的注释文件目录
    train_anno_dir = "train_data/FUNSD/training_data/annotations/"
    # 训练数据的输出目录
    train_output_dir = "train_data/FUNSD/train.json"
    
    # 载入 FUNSD 标签信息，返回文件名到信息的映射
    fn_info_map = load_funsd_label(train_image_dir, train_anno_dir)
    # 打开输出文件，写入文件名和对应信息到文件中
    with open(train_output_dir, "w") as fout:
        for fn in fn_info_map:
            fout.write(fn + ".png" + "\t" + json.dumps(
                fn_info_map[fn], ensure_ascii=False) + "\n")
    # 打印提示信息
    print("====ok====")
    # 返回函数
    return
# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```