# `.\lucidrains\simple-hierarchical-transformer\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'simple-hierarchical-transformer',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找并包含所有包
  version = '0.2.0',  # 版本号
  license='MIT',  # 许可证
  description = 'Simple Hierarchical Transformer',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/simple-hierarchical-transformer',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'hierarchical'
  ],
  install_requires=[  # 依赖的包列表
    'accelerate',
    'einops>=0.7.0',
    'local-attention',
    'torch>=2.0'
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