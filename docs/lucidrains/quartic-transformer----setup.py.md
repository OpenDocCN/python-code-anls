# `.\lucidrains\quartic-transformer\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'quartic-transformer', # 包的名称
  packages = find_packages(exclude=[]), # 查找所有包
  version = '0.0.12', # 版本号
  license='MIT', # 许可证
  description = 'Quartic Transformer', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  long_description_content_type = 'text/markdown', # 长描述内容类型
  url = 'https://github.com/lucidrains/quartic-transformer', # URL
  keywords = [ # 关键词列表
    'artificial intelligence',
    'deep learning',
    'transformer',
    'attention'
  ],
  install_requires=[ # 安装依赖
    'colt5-attention',
    'einops>=0.7.0',
    'einx[torch]>=0.1.3',
    'taylor-series-linear-attention',
    'torch>=2.0',
    'x-transformers'
  ],
  classifiers=[ # 分类器列表
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```