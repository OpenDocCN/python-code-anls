# `.\transformers\dependency_versions_check.py`

```py
# 版权声明和许可证信息
#
# 从依赖版本表中导入依赖信息
from .dependency_versions_table import deps
# 从工具包中导入版本检查函数
from .utils.versions import require_version, require_version_core

# 定义我们始终希望在运行时检查的模块版本
# （通常是在 setup.py 的 `install_requires` 中定义的模块）
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

# 遍历需要在运行时检查的模块
for pkg in pkgs_to_check_at_runtime:
    # 如果模块在依赖中
    if pkg in deps:
        # 如果是 tokenizers 模块
        if pkg == "tokenizers":
            # 必须在此处加载，否则 tqdm 检查可能失败
            from .utils import is_tokenizers_available

            # 如果 tokenizers 模块不可用，则跳过
            if not is_tokenizers_available():
                continue  # 不是必需的，仅在安装时检查版本
        # 如果是 accelerate 模块
        elif pkg == "accelerate":
            # 必须在此处加载，否则 tqdm 检查可能失败
            from .utils import is_accelerate_available

            # 也许将来在这里切换到 is_torch_available，以便 Accelerate 是 Transformers 与 PyTorch 的硬依赖
            if not is_accelerate_available():
                continue  # 不是必需的，仅在安装时检查版本

        # 要求满足特定版本的模块
        require_version_core(deps[pkg])
    else:
        # 如果在依赖中找不到模块，则引发错误
        raise ValueError(f"can't find {pkg} in {deps.keys()}, check dependency_versions_table.py")

# 定义依赖版本检查函数
def dep_version_check(pkg, hint=None):
    require_version(deps[pkg], hint)
```