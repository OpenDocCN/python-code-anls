# `.\lucidrains\iTransformer\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的信息
setup(
  name = 'iTransformer', # 包名
  packages = find_packages(exclude=[]), # 查找包
  version = '0.5.5', # 版本号
  license='MIT', # 许可证
  description = 'iTransformer - Inverted Transformer Are Effective for Time Series Forecasting', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  long_description_content_type = 'text/markdown', # 长描述内容类型
  url = 'https://github.com/lucidrains/iTransformer', # URL
  keywords = [ # 关键词
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'time series forecasting'
  ],
  install_requires=[ # 安装依赖
    'beartype',
    'einops>=0.7.0',
    'gateloop-transformer>=0.2.3',
    'rotary-embedding-torch',
    'torch>=2.1',
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