# `d:/src/tocomm/Bert-VITS2\export_onnx.py`

```
# 从 onnx_modules 模块中导入 export_onnx 函数
from onnx_modules import export_onnx
# 导入 os 模块
import os

# 如果当前脚本是主程序
if __name__ == "__main__":
    # 设置导出路径
    export_path = "BertVits2.2PT"
    # 设置模型路径
    model_path = "model\\G_0.pth"
    # 设置配置文件路径
    config_path = "model\\config.json"
    # 设置 novq 变量为 False
    novq = False
    # 设置 dev 变量为 False
    dev = False
    # 如果 onnx 文件夹不存在，则创建该文件夹
    if not os.path.exists("onnx"):
        os.makedirs("onnx")
    # 如果导出路径对应的文件夹不存在，则创建该文件夹
    if not os.path.exists(f"onnx/{export_path}"):
        os.makedirs(f"onnx/{export_path}")
    # 调用 export_onnx 函数，将模型导出为 ONNX 格式
    export_onnx(export_path, model_path, config_path, novq, dev)
```