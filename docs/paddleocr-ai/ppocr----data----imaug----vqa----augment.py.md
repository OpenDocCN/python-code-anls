# `.\PaddleOCR\ppocr\data\imaug\vqa\augment.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入所需的库
import os
import sys
import numpy as np
import random
from copy import deepcopy

# 根据文本框左上角坐标先按 y 坐标排序，再按 x 坐标排序的方式对 OCR 信息进行排序
def order_by_tbyx(ocr_info):
    # 根据指定的排序规则对 OCR 信息进行排序
    res = sorted(ocr_info, key=lambda r: (r["bbox"][1], r["bbox"][0]))
    # 遍历排序后的结果
    for i in range(len(res) - 1):
        # 从后往前遍历
        for j in range(i, 0, -1):
            # 如果两个文本框的 y 坐标差值小于 20 且 x 坐标不符合排序规则
            if abs(res[j + 1]["bbox"][1] - res[j]["bbox"][1]) < 20 and \
                    (res[j + 1]["bbox"][0] < res[j]["bbox"][0]):
                # 交换两个文本框的位置
                tmp = deepcopy(res[j])
                res[j] = deepcopy(res[j + 1])
                res[j + 1] = deepcopy(tmp)
            else:
                # 如果不符合条件，则跳出循环
                break
    # 返回排序后的结果
    return res
```