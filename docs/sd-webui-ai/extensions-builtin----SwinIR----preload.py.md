# `stable-diffusion-webui\extensions-builtin\SwinIR\preload.py`

```py
# 导入 os 模块
import os
# 从 modules 模块中导入 paths 模块
from modules import paths

# 预加载函数，用于添加命令行参数
def preload(parser):
    # 添加一个名为 "--swinir-models-path" 的命令行参数，类型为字符串，帮助信息为"Path to directory with SwinIR model file(s)."，默认值为 paths.models_path 下的 'SwinIR' 文件夹路径
    parser.add_argument("--swinir-models-path", type=str, help="Path to directory with SwinIR model file(s).", default=os.path.join(paths.models_path, 'SwinIR'))
```