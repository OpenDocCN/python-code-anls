# `.\lucidrains\flash-genomics-model\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'flash-genomics-model',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.0.1',  # 版本号
  license='MIT',  # 许可证
  description = 'Flash Genomics Model (FGM)',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/flash-genomics-model',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'long context',
    'genomics',
    'pre-training'
  ],
  install_requires=[  # 安装依赖
    'einops>=0.6.1',
    'MEGABYTE-pytorch',
    'torch>=1.6',
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