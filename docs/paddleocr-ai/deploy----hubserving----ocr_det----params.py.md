# `.\PaddleOCR\deploy\hubserving\ocr_det\params.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入未来的绝对导入、除法和打印功能
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 定义配置类
class Config(object):
    pass

# 读取参数函数
def read_params():
    # 创建配置对象
    cfg = Config()

    # 文本检测器参数
    cfg.det_algorithm = "DB"
    cfg.det_model_dir = "./inference/ch_PP-OCRv3_det_infer/"
    cfg.det_limit_side_len = 960
    cfg.det_limit_type = 'max'

    # DB 参数
    cfg.det_db_thresh = 0.3
    cfg.det_db_box_thresh = 0.6
    cfg.det_db_unclip_ratio = 1.5
    cfg.use_dilation = False
    cfg.det_db_score_mode = "fast"

    # #EAST 参数
    # cfg.det_east_score_thresh = 0.8
    # cfg.det_east_cover_thresh = 0.1
    # cfg.det_east_nms_thresh = 0.2

    cfg.use_pdserving = False
    cfg.use_tensorrt = False

    # 返回配置对象
    return cfg
```