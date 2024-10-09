# `.\MinerU\setup.py`

```
# 导入路径处理模块
from pathlib import Path
# 导入 setuptools 中的 setup 和 find_packages 方法
from setuptools import setup, find_packages
# 从版本模块中导入版本号
from magic_pdf.libs.version import __version__


# 定义解析依赖文件的函数
def parse_requirements(filename):
    # 打开指定的文件并读取内容
    with open(filename) as f:
        # 将文件内容按行分割成列表
        lines = f.read().splitlines()

    # 初始化一个空列表用于存储依赖项
    requires = []

    # 遍历每一行
    for line in lines:
        # 检查行中是否包含 "http"
        if "http" in line:
            # 提取 URL 前的包名并去除空格
            pkg_name_without_url = line.split('@')[0].strip()
            # 将包名添加到依赖项列表中
            requires.append(pkg_name_without_url)
        else:
            # 如果不包含 "http"，直接添加行内容到依赖项列表中
            requires.append(line)

    # 返回解析出的依赖项列表
    return requires


# 如果脚本是主程序，则执行以下代码
if __name__ == '__main__':
    # 打开 README.md 文件并读取其内容
    with Path(Path(__file__).parent,
              'README.md').open(encoding='utf-8') as file:
        long_description = file.read()
    # 调用 setup 函数进行包的配置
    setup(
        name="magic_pdf",  # 项目名
        version=__version__,  # 自动从tag中获取版本号
        packages=find_packages() + ["magic_pdf.resources"],  # 包含所有的包
        package_data={
            "magic_pdf.resources": ["**"],  # 包含magic_pdf.resources目录下的所有文件
        },
        install_requires=parse_requirements('requirements.txt'),  # 项目依赖的第三方库
        extras_require={
            "lite": ["paddleocr==2.7.3",
                     "paddlepaddle==3.0.0b1;platform_system=='Linux'",
                     "paddlepaddle==2.6.1;platform_system=='Windows' or platform_system=='Darwin'",
                     ],
            "full": ["unimernet==0.1.6",  # 0.1.6版本大幅裁剪依赖包范围，推荐使用此版本
                     "matplotlib<=3.9.0;platform_system=='Windows'",  # 3.9.1及之后不提供windows的预编译包，避免一些没有编译环境的windows设备安装失败
                     "matplotlib;platform_system=='Linux' or platform_system=='Darwin'",  # linux 和 macos 不应限制matplotlib的最高版本，以避免无法更新导致的一些bug
                     "ultralytics",  # yolov8,公式检测
                     "paddleocr==2.7.3",  # 2.8.0及2.8.1版本与detectron2有冲突，需锁定2.7.3
                     "paddlepaddle==3.0.0b1;platform_system=='Linux'",  # 解决linux的段异常问题
                     "paddlepaddle==2.6.1;platform_system=='Windows' or platform_system=='Darwin'",  # windows版本3.0.0b1效率下降，需锁定2.6.1
                     "pypandoc",  # 表格解析latex转html
                     "struct-eqtable==0.1.0",  # 表格解析
                     "detectron2"
                     ],
        },
        description="A practical tool for converting PDF to Markdown",  # 简短描述
        long_description=long_description,  # 详细描述
        long_description_content_type="text/markdown",  # 如果README是Markdown格式
        url="https://github.com/opendatalab/MinerU",  # 项目的 URL
        python_requires=">=3.9",  # 项目依赖的 Python 版本
        entry_points={
            "console_scripts": [
                "magic-pdf = magic_pdf.tools.cli:cli",  # 定义命令行接口
                "magic-pdf-dev = magic_pdf.tools.cli_dev:cli"  # 定义开发者命令行接口
            ],
        },  # 项目提供的可执行命令
        include_package_data=True,  # 是否包含非代码文件，如数据文件、配置文件等
        zip_safe=False,  # 是否使用 zip 文件格式打包，一般设为 False
    )
```