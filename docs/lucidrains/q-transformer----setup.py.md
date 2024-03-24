# `.\lucidrains\q-transformer\setup.py`

```py
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'q-transformer', # 包的名称
  packages = find_packages(exclude=[]), # 查找所有包
  version = '0.1.14', # 版本号
  license='MIT', # 许可证
  description = 'Q-Transformer', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  long_description_content_type = 'text/markdown', # 长描述内容类型
  url = 'https://github.com/lucidrains/q-transformer', # URL
  keywords = [ # 关键词
    'artificial intelligence',
    'deep learning',
    'attention mechanisms',
    'transformers',
    'q-learning'
  ],
  install_requires=[ # 安装依赖
    'accelerate',
    'beartype',
    'classifier-free-guidance-pytorch>=0.4.2',
    'einops>=0.7.0',
    'ema-pytorch>=0.3.1',
    'numpy',
    'torchtyping',
    'torch>=2.0'
  ],
  classifiers=[ # 分类
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```