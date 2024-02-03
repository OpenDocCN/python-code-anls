# `stable-diffusion-webui\modules\import_hook.py`

```
# 导入 sys 模块
import sys

# 如果命令行参数中不包含 "--xformers"，则将 xformers 模块设为 None，防止稳定扩散仓库尝试使用它
if "--xformers" not in "".join(sys.argv):
    sys.modules["xformers"] = None

# 修复 torchvision 0.17+ 中更改的导入，否则会破坏 basicsr；参考 https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985
try:
    # 尝试导入 torchvision.transforms.functional_tensor 模块
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    try:
        # 如果导入失败，尝试导入 torchvision.transforms.functional 模块
        import torchvision.transforms.functional as functional
        # 将 functional 模块设为 torchvision.transforms.functional_tensor 模块
        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        # 如果导入失败，什么也不做
        pass  # shrug...
```