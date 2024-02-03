# `stable-diffusion-webui\extensions-builtin\Lora\preload.py`

```
# 导入 os 模块
import os
# 从 modules 模块中导入 paths 模块
from modules import paths

# 预加载函数，用于添加预加载参数
def preload(parser):
    # 添加名为 "--lora-dir" 的参数，类型为字符串，帮助信息为"Path to directory with Lora networks."，默认值为 paths.models_path 下的 'Lora' 文件夹路径
    parser.add_argument("--lora-dir", type=str, help="Path to directory with Lora networks.", default=os.path.join(paths.models_path, 'Lora'))
    # 添加名为 "--lyco-dir-backcompat" 的参数，类型为字符串，帮助信息为"Path to directory with LyCORIS networks (for backawards compatibility; can also use --lyco-dir)."，默认值为 paths.models_path 下的 'LyCORIS' 文件夹路径
    parser.add_argument("--lyco-dir-backcompat", type=str, help="Path to directory with LyCORIS networks (for backawards compatibility; can also use --lyco-dir).", default=os.path.join(paths.models_path, 'LyCORIS'))
```