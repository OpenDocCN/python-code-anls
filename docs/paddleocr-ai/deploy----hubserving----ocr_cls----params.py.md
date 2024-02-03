# `.\PaddleOCR\deploy\hubserving\ocr_cls\params.py`

```py
# 版权声明和许可信息
# 本代码版权归 PaddlePaddle 作者所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入未来的绝对导入、除法和打印功能
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 定义一个配置类
class Config(object):
    pass

# 读取参数函数
def read_params():
    # 创建一个配置对象
    cfg = Config()

    # 文本分类器的参数
    cfg.cls_model_dir = "./inference/ch_ppocr_mobile_v2.0_cls_infer/"
    cfg.cls_image_shape = "3, 48, 192"
    cfg.label_list = ['0', '180']
    cfg.cls_batch_num = 30
    cfg.cls_thresh = 0.9

    # 是否使用 PaddleServing
    cfg.use_pdserving = False
    # 是否使用 TensorRT
    cfg.use_tensorrt = False

    # 返回配置对象
    return cfg
```