# `.\kubehunter\setup.py`

```

# 导入需要的模块
from subprocess import check_call
from pkg_resources import parse_requirements
from configparser import ConfigParser
from setuptools import setup, Command

# 自定义命令类，用于列出依赖项
class ListDependenciesCommand(Command):
    """A custom command to list dependencies"""

    description = "list package dependencies"  # 描述命令的作用
    user_options = []  # 用户选项为空列表

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # 读取 setup.cfg 文件
        cfg = ConfigParser()
        cfg.read("setup.cfg")
        # 获取安装依赖项
        requirements = cfg["options"]["install_requires"]
        # 打印依赖项
        print(requirements)

# 自定义命令类，用于运行 PyInstaller 构建独立可执行文件
class PyInstallerCommand(Command):
    """A custom command to run PyInstaller to build standalone executable."""

    description = "run PyInstaller on kube-hunter entrypoint"  # 描述命令的作用
    user_options = []  # 用户选项为空列表

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # 读取 setup.cfg 文件
        cfg = ConfigParser()
        cfg.read("setup.cfg")
        # 定义 PyInstaller 命令
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
        # 打印 PyInstaller 命令
        print(" ".join(command))
        # 调用 PyInstaller 命令
        check_call(command)

# 设置
setup(
    use_scm_version={"fallback_version": "noversion"},  # 使用 SCM 版本
    cmdclass={"dependencies": ListDependenciesCommand, "pyinstaller": PyInstallerCommand},  # 自定义命令类
)

```