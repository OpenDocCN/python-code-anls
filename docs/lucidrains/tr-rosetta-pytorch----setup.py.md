# `.\lucidrains\tr-rosetta-pytorch\setup.py`

```py
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的信息
setup(
  name = 'tr-rosetta-pytorch',  # 包的名称
  packages = find_packages(),  # 查找所有包
  include_package_data = True,  # 包含所有数据文件
  entry_points={  # 设置入口点
    'console_scripts': [  # 控制台脚本
      'tr_rosetta = tr_rosetta_pytorch.cli:predict',  # 脚本名称和执行函数
    ],
  },
  version = '0.0.3',  # 版本号
  license='MIT',  # 许可证
  description = 'trRosetta - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/tr-rosetta-pytorch',  # 项目链接
  keywords = [  # 关键词
    'artificial intelligence',
    'protein folding',
    'protein design'
  ],
  install_requires=[  # 安装依赖
    'einops>=0.3',
    'fire',
    'numpy',
    'torch>=1.6'
  ],
  classifiers=[  # 分类器
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```