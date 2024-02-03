# `stable-diffusion-webui\modules\codeformer_model.py`

```
# 导入所需的库
import os
import cv2
import torch

# 导入自定义模块
import modules.face_restoration
import modules.shared
from modules import shared, devices, modelloader, errors
from modules.paths import models_path

# codeformer 项目选择将修改后的 basicsr 库包含到他们的项目中，这使得无法与其他使用 basicsr 的库一起使用，比如 GFPGAN。
# 我选择包含 codeformer 中的一些文件来解决这个问题。
# 初始化模型目录和路径
model_dir = "Codeformer"
model_path = os.path.join(models_path, model_dir)
model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'

# 初始化 codeformer 变量
codeformer = None

# 设置模型函数
def setup_model(dirname):
    # 如果模型路径不存在，则创建
    os.makedirs(model_path, exist_ok=True)

    # 获取 CodeFormer 模型路径
    path = modules.paths.paths.get("CodeFormer", None)
    # 如果路径不存在，则返回
    if path is None:
        return

    # 捕获异常并报告错误
    except Exception:
        errors.report("Error setting up CodeFormer", exc_info=True)

    # 恢复系统路径
    # sys.path = stored_sys_path
```