# `.\PaddleOCR\deploy\hubserving\structure_table\params.py`

```py
# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 从指定路径导入 OCR 系统参数读取函数
from deploy.hubserving.ocr_system.params import read_params as pp_ocr_read_params

# 定义读取参数的函数
def read_params():
    # 调用 OCR 系统参数读取函数，获取参数配置
    cfg = pp_ocr_read_params()

    # 设置表格结构模型的参数
    cfg.table_max_len = 488
    cfg.table_model_dir = './inference/en_ppocr_mobile_v2.0_table_structure_infer/'
    cfg.table_char_dict_path = './ppocr/utils/dict/table_structure_dict.txt'
    cfg.show_log = False
    # 返回参数配置
    return cfg
```