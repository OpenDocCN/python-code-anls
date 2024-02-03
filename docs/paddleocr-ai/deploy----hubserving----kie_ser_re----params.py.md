# `.\PaddleOCR\deploy\hubserving\kie_ser_re\params.py`

```
# 版权声明和许可证信息
# 从未来的模块导入绝对导入、除法和打印函数
from deploy.hubserving.ocr_system.params import read_params as pp_ocr_read_params

# 定义一个空的配置类
class Config(object):
    pass

# 读取参数函数
def read_params():
    # 调用 OCR 参数读取函数，获取参数配置
    cfg = pp_ocr_read_params()

    # 设置 SER 参数
    cfg.kie_algorithm = "LayoutXLM"
    cfg.use_visual_backbone = False

    # 设置 SER 模型目录和 RE 模型目录
    cfg.ser_model_dir = "./inference/ser_vi_layoutxlm_xfund_infer"
    cfg.re_model_dir = "./inference/re_vi_layoutxlm_xfund_infer"

    # 设置 SER 字典路径、可视化字体路径和 OCR 排序方法
    cfg.ser_dict_path = "train_data/XFUND/class_list_xfun.txt"
    cfg.vis_font_path = "./doc/fonts/simfang.ttf"
    cfg.ocr_order_method = "tb-yx"

    # 返回配置对象
    return cfg
```