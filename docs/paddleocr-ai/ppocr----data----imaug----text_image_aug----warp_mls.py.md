# `.\PaddleOCR\ppocr\data\imaug\text_image_aug\warp_mls.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用来源
# https://github.com/RubanSeven/Text-Image-Augmentation-python/blob/master/warp_mls.py

import numpy as np

# 定义 WarpMLS 类
class WarpMLS:
    # 初始化函数，接收源图像、源点、目标点、目标宽度、目标高度和变换比例
    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.):
        self.src = src
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.trans_ratio = trans_ratio
        self.grid_size = 100
        self.rdx = np.zeros((self.dst_h, self.dst_w))
        self.rdy = np.zeros((self.dst_h, self.dst_w))

    # 静态方法，执行双线性插值
    @staticmethod
    def __bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x

    # 生成函数，计算 delta 并生成图像
    def generate(self):
        self.calc_delta()
        return self.gen_img()
```