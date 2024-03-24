# `.\lucidrains\omninet-pytorch\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'omninet-pytorch', # 包的名称
  packages = find_packages(), # 查找并包含所有包
  version = '0.0.6', # 版本号
  license='MIT', # 许可证
  description = 'Omninet - Pytorch', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  url = 'https://github.com/lucidrains/omninet-pytorch', # 项目链接
  keywords = [ # 关键词列表
    'artificial intelligence',
    'deep learning',
    'transformer',
    'attention mechanism'
  ],
  install_requires=[ # 安装依赖
    'einops>=0.3',
    'torch>=1.6',
    'performer-pytorch'
  ],
  classifiers=[ # 分类器
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```