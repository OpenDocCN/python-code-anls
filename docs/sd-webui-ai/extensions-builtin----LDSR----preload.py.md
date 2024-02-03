# `stable-diffusion-webui\extensions-builtin\LDSR\preload.py`

```
# 导入 os 模块
import os
# 从 modules 模块中导入 paths 模块
from modules import paths

# 预加载函数，用于解析参数
def preload(parser):
    # 添加一个名为 "--ldsr-models-path" 的参数，类型为字符串，帮助信息为"Path to directory with LDSR model file(s)."，默认值为 LDSR 模型文件所在目录的路径
    parser.add_argument("--ldsr-models-path", type=str, help="Path to directory with LDSR model file(s).", default=os.path.join(paths.models_path, 'LDSR'))
```