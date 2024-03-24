# `.\lucidrains\MEGABYTE-pytorch\setup.py`

```py
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'MEGABYTE-pytorch',  # 包的名称
  packages = find_packages(),  # 查找所有包
  version = '0.2.1',  # 版本号
  license='MIT',  # 许可证
  description = 'MEGABYTE - Pytorch',  # 描述
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/MEGABYTE-pytorch',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'attention mechanism',
    'transformers'
  ],
  install_requires=[  # 安装依赖
    'beartype',
    'einops>=0.6.1',
    'torch>=1.10',
    'tqdm'
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