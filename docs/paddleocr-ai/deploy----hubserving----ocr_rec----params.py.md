# `.\PaddleOCR\deploy\hubserving\ocr_rec\params.py`

```py
# 版权声明
# 2022年PaddlePaddle作者保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证，
# 包括但不限于特定目的的适用性和适销性。
# 有关特定语言的权限和限制，请参阅许可证。

# 导入未来的绝对导入、除法和打印功能
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 定义Config类
class Config(object):
    pass

# 读取参数函数
def read_params():
    # 创建Config对象
    cfg = Config()

    # 文本识别器的参数
    cfg.rec_algorithm = "CRNN"
    cfg.rec_model_dir = "./inference/ch_PP-OCRv3_rec_infer/"

    cfg.rec_image_shape = "3, 48, 320"
    cfg.rec_batch_num = 6
    cfg.max_text_length = 25

    cfg.rec_char_dict_path = "./ppocr/utils/ppocr_keys_v1.txt"
    cfg.use_space_char = True

    cfg.use_pdserving = False
    cfg.use_tensorrt = False

    # 返回配置对象
    return cfg
```