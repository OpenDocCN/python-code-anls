# `arknights-mower\setup.py`

```py
# 导入 setuptools 模块
import setuptools
# 导入 arknights_mower 模块
import arknights_mower
# 从 pathlib 模块中导入 Path 类
from pathlib import Path

# 读取 README.md 文件的内容作为长描述
LONG_DESC = Path('README.md').read_text('utf8')
# 获取 arknights_mower 模块的版本号
VERSION = arknights_mower.__version__

# 设置包的信息和依赖关系
setuptools.setup(
    # 包名
    name='arknights_mower',
    # 版本号
    version=VERSION,
    # 作者
    author='Konano',
    # 作者邮箱
    author_email='w@nano.ac',
    # 描述
    description='Arknights Helper based on ADB and Python',
    # 长描述
    long_description=LONG_DESC,
    # 长描述的内容类型
    long_description_content_type='text/markdown',
    # 项目的 URL
    url='https://github.com/Konano/arknights-mower',
    # 查找所有包
    packages=setuptools.find_packages(),
    # 安装依赖
    install_requires=[
        'colorlog', 'opencv_python', 'matplotlib', 'numpy', 'scikit_image==0.18.3', 'scikit_learn>=1',
        'onnxruntime', 'pyclipper', 'shapely', 'tornado', 'requests', 'ruamel.yaml', 'schedule'
    ],
    # 包含包数据
    include_package_data=True,
    # 入口点
    entry_points={'console_scripts': [
        'arknights-mower=arknights_mower.__main__:main'
    ]},
    # 分类
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Games/Entertainment',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
```