# `.\PaddleOCR\PPOCRLabel\setup.py`

```
# 导入设置安装包的函数
from setuptools import setup
# 导入打开文件的函数
from io import open

# 打开 requirements.txt 文件，读取其中的内容作为依赖项
with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()
    # 添加 'tqdm' 到依赖项列表中

# 定义读取 README 文件的函数
def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README

# 设置安装包的信息
setup(
    name='PPOCRLabel',  # 包名
    packages=['PPOCRLabel'],  # 包含的包
    package_data = {'PPOCRLabel': ['libs/*','resources/strings/*','resources/icons/*']},  # 包含的数据文件
    package_dir={'PPOCRLabel': ''},  # 包的目录
    include_package_data=True,  # 包含数据文件
    entry_points={"console_scripts": ["PPOCRLabel= PPOCRLabel.PPOCRLabel:main"]},  # 入口点
    version='2.1.3',  # 版本号
    install_requires=requirements,  # 安装依赖项
    license='Apache License 2.0',  # 许可证
    description='PPOCRLabelv2 is a semi-automatic graphic annotation tool suitable for OCR field, with built-in PP-OCR model to automatically detect and re-recognize data. It is written in Python3 and PyQT5, supporting rectangular box, table, irregular text and key information annotation modes. Annotations can be directly used for the training of PP-OCR detection and recognition models.',  # 描述
    long_description=readme(),  # 长描述
    long_description_content_type='text/markdown',  # 长描述内容类型
    url='https://github.com/PaddlePaddle/PaddleOCR',  # 项目链接
    download_url='https://github.com/PaddlePaddle/PaddleOCR.git',  # 下载链接
    # 定义关键字列表，用于描述项目的关键特性
    keywords=[
        'ocr textdetection textrecognition paddleocr crnn east star-net rosetta ocrlite db chineseocr chinesetextdetection chinesetextrecognition'
    ],
    # 定义分类列表，用于描述项目的属性
    classifiers=[
        'Intended Audience :: Developers', 'Operating System :: OS Independent',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7', 'Topic :: Utilities'
    ], )
```