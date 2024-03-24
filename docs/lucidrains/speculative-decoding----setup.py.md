# `.\lucidrains\speculative-decoding\setup.py`

```py
# 导入设置安装和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'speculative-decoding', # 包的名称
  packages = find_packages(exclude=[]), # 查找所有包，不排除任何包
  version = '0.1.2', # 版本号
  license='MIT', # 许可证
  description = 'Speculative Decoding', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  long_description_content_type = 'text/markdown', # 长描述内容类型
  url = 'https://github.com/lucidrains/speculative-decoding', # 项目链接
  keywords = [ # 关键词列表
    'artificial intelligence',
    'deep learning',
    'transformers',
    'efficient decoding'
  ],
  install_requires=[ # 安装依赖
    'beartype',
    'einops>=0.6.1',
    'torch>=1.12',
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