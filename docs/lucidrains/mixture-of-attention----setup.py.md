# `.\lucidrains\mixture-of-attention\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'mixture-of-attention', # 包的名称
  packages = find_packages(exclude=[]), # 查找所有包
  version = '0.0.24', # 版本号
  license='MIT', # 许可证
  description = 'Mixture of Attention', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  long_description_content_type = 'text/markdown', # 长描述内容类型
  url = 'https://github.com/lucidrains/mixture-of-attention', # URL
  keywords = [ # 关键词列表
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'mixture-of-experts',
    'routed attention'
  ],
  install_requires=[ # 安装依赖
    'colt5-attention>=0.10.14',
    'einops>=0.6.1',
    'local-attention>=1.8.6',
    'torch>=1.6',
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