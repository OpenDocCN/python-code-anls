# `.\PaddleOCR\deploy\hubserving\kie_ser\params.py`

```
# 版权声明和许可信息
# 该代码版权归 PaddlePaddle 作者所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权使用该文件；
# 除非符合许可证的规定，否则不得使用该文件。
# 可以在以下网址获取许可证的副本：
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取有关权限和限制的详细信息。

# 导入必要的模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 从 deploy.hubserving.ocr_system.params 模块中导入 read_params 函数
from deploy.hubserving.ocr_system.params import read_params as pp_ocr_read_params

# 定义 Config 类
class Config(object):
    pass

# 读取参数函数
def read_params():
    # 调用 pp_ocr_read_params 函数获取配置参数
    cfg = pp_ocr_read_params()

    # 设置 SER 参数
    cfg.kie_algorithm = "LayoutXLM"
    cfg.use_visual_backbone = False

    # 设置 SER 模型目录和字典路径
    cfg.ser_model_dir = "./inference/ser_vi_layoutxlm_xfund_infer"
    cfg.ser_dict_path = "train_data/XFUND/class_list_xfun.txt"
    cfg.vis_font_path = "./doc/fonts/simfang.ttf"
    cfg.ocr_order_method = "tb-yx"

    # 返回配置参数对象
    return cfg
```