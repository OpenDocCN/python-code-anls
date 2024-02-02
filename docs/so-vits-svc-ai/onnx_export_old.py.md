# `so-vits-svc\onnx_export_old.py`

```py
# 导入 torch 模块
import torch

# 导入自定义的 utils 模块
import utils
# 从 onnxexport.model_onnx 模块中导入 SynthesizerTrn 类
from onnxexport.model_onnx import SynthesizerTrn

# 定义主函数 main，接受一个 NetExport 参数
def main(NetExport):
    # 设置路径变量为 "SoVits4.0"
    path = "SoVits4.0"
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 调用 main 函数，并传入 True 作为参数
    main(True)
```