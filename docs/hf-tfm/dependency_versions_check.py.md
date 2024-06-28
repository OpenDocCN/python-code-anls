# `.\dependency_versions_check.py`

```py
# 从依赖版本表导入依赖字典
from .dependency_versions_table import deps
# 从工具目录下的版本模块导入版本检查函数
from .utils.versions import require_version, require_version_core

# 定义需要在运行时检查的模块版本列表
# 通常包括在 setup.py 的 install_requires 中定义的模块
#
# 特定顺序的注意事项:
# - 必须在 tokenizers 之前检查 tqdm

pkgs_to_check_at_runtime = [
    "python",
    "tqdm",
    "regex",
    "requests",
    "packaging",
    "filelock",
    "numpy",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "accelerate",
    "pyyaml",
]

# 遍历需要在运行时检查的模块列表
for pkg in pkgs_to_check_at_runtime:
    # 如果依赖字典中存在该模块
    if pkg in deps:
        # 如果当前模块是 "tokenizers"
        if pkg == "tokenizers":
            # 必须在这里加载，否则 tqdm 的检查可能会失败
            from .utils import is_tokenizers_available

            # 如果 tokenizers 模块不可用，跳过检查版本，只在安装时检查
            if not is_tokenizers_available():
                continue
        # 如果当前模块是 "accelerate"
        elif pkg == "accelerate":
            # 必须在这里加载，否则 tqdm 的检查可能会失败
            from .utils import is_accelerate_available

            # 或许将来可以在这里切换为 is_torch_available，以便 Accelerate 成为 Transformers 与 PyTorch 的硬依赖
            if not is_accelerate_available():
                continue

        # 要求核心版本满足依赖字典中对应模块的要求
        require_version_core(deps[pkg])
    else:
        # 如果依赖字典中找不到当前模块，则抛出异常
        raise ValueError(f"can't find {pkg} in {deps.keys()}, check dependency_versions_table.py")


def dep_version_check(pkg, hint=None):
    # 要求满足依赖字典中对应模块的版本要求
    require_version(deps[pkg], hint)
```