# `kubehunter\setup.py`

```py
# 导入需要的模块
from subprocess import check_call
from pkg_resources import parse_requirements
from configparser import ConfigParser
from setuptools import setup, Command

# 自定义命令，用于列出依赖项
class ListDependenciesCommand(Command):
    """A custom command to list dependencies"""

    description = "list package dependencies"
    user_options = []

    # 初始化选项
    def initialize_options(self):
        pass

    # 完成选项
    def finalize_options(self):
        pass

    # 运行命令
    def run(self):
        # 读取配置文件
        cfg = ConfigParser()
        cfg.read("setup.cfg")
        # 获取安装依赖项
        requirements = cfg["options"]["install_requires"]
        # 打印依赖项
        print(requirements)

# 自定义命令，用于运行 PyInstaller 构建独立可执行文件
class PyInstallerCommand(Command):
    """A custom command to run PyInstaller to build standalone executable."""

    description = "run PyInstaller on kube-hunter entrypoint"
    user_options = []

    # 初始化选项
    def initialize_options(self):
        pass

    # 完成选项
    def finalize_options(self):
        pass

    # 运行命令
    def run(self):
        # 读取配置文件
        cfg = ConfigParser()
        cfg.read("setup.cfg")
        # 构建 PyInstaller 命令
        command = [
            "pyinstaller",
            "--clean",
            "--onefile",
            "--name",
            "kube-hunter",
        ]
        # 获取安装依赖项
        setup_cfg = cfg["options"]["install_requires"]
        requirements = parse_requirements(setup_cfg)
        # 添加隐藏导入的依赖项
        for r in requirements:
            command.extend(["--hidden-import", r.key])
        command.append("kube_hunter/__main__.py")
        # 打印构建命令
        print(" ".join(command))
        # 调用命令行执行构建命令
        check_call(command)

# 设置
setup(
    use_scm_version={"fallback_version": "noversion"},
    cmdclass={"dependencies": ListDependenciesCommand, "pyinstaller": PyInstallerCommand},
)
```