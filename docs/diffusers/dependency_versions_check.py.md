# `.\diffusers\dependency_versions_check.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵守该许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，
# 否则根据许可证分发的软件按“原样”提供，
# 不附带任何形式的明示或暗示的保证或条件。
# 请参阅许可证以获取特定语言管理权限和
# 限制的信息。

# 从当前包中导入依赖版本表
from .dependency_versions_table import deps
# 从当前包中导入版本检查工具
from .utils.versions import require_version, require_version_core


# 定义我们在运行时始终要检查的模块版本
# （通常是 setup.py 中定义的 `install_requires`）
#
# 特定顺序说明：
# - tqdm 必须在 tokenizers 之前检查

# 需要在运行时检查的包列表，使用空格分隔并拆分成列表
pkgs_to_check_at_runtime = "python requests filelock numpy".split()
# 遍历每个需要检查的包
for pkg in pkgs_to_check_at_runtime:
    # 如果包在依赖版本字典中
    if pkg in deps:
        # 检查该包的版本是否符合要求
        require_version_core(deps[pkg])
    # 如果包不在依赖版本字典中，抛出异常
    else:
        raise ValueError(f"can't find {pkg} in {deps.keys()}, check dependency_versions_table.py")


# 定义一个函数用于检查特定包的版本
def dep_version_check(pkg, hint=None):
    # 调用版本检查工具，检查指定包的版本
    require_version(deps[pkg], hint)
```