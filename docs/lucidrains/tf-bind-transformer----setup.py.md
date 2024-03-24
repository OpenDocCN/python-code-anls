# `.\lucidrains\tf-bind-transformer\setup.py`

```py
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'tf-bind-transformer',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.0.118',  # 版本号
  license='MIT',  # 许可证
  description = 'Transformer for Transcription Factor Binding',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/tf-bind-transformer',  # 项目链接
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',
    'attention mechanism',
    'transformers',
    'transcription factors',
    'gene expression'
  ],
  install_requires=[  # 安装依赖列表
    'bidirectional-cross-attention',
    'biopython',
    'click',
    'einops>=0.3',
    'enformer-pytorch>=0.5',
    'fair-esm',
    'logavgexp-pytorch',
    'polars',
    'python-dotenv',
    'sentencepiece',
    'torch>=1.6',
    'transformers>=4.0',
    'tqdm'
  ],
  classifiers=[  # 分类器列表
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```