# `.\AutoGPT\autogpts\autogpt\scripts\check_requirements.py`

```py
# 导入必要的模块
import contextlib
import os
import sys
from importlib.metadata import version

# 尝试导入 poetry.factory 模块，如果导入失败则安装 poetry 包
try:
    import poetry.factory  # noqa
except ModuleNotFoundError:
    os.system(f"{sys.executable} -m pip install 'poetry>=1.6.1,<2.0.0'")

# 导入必要的类和函数
from poetry.core.constraints.version.version import Version
from poetry.factory import Factory

# 主函数
def main():
    # 创建一个 Poetry 项目对象
    poetry_project = Factory().create_poetry()
    # 获取主要依赖组
    dependency_group = poetry_project.package.dependency_group("main")

    # 存储缺失的包
    missing_packages = []
    # 遍历依赖组中的依赖
    for dep in dependency_group.dependencies:
        # 如果依赖是可选的，则跳过
        if dep.is_optional():
            continue
        # 尝试验证已安装的版本是否合适
        with contextlib.suppress(ModuleNotFoundError):
            installed_version = version(dep.name)  # 如果失败，则表示未安装
            if dep.constraint.allows(Version.parse(installed_version)):
                continue
        # 如果上述验证失败，则将包标记为缺失
        missing_packages.append(str(dep))

    # 如果有缺失的包，则打印出来并退出程序
    if missing_packages:
        print("Missing packages:")
        print(", ".join(missing_packages))
        sys.exit(1)

# 如果作为脚本运行，则执行主函数
if __name__ == "__main__":
    main()
```