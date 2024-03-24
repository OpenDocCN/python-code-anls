# `.\lucidrains\evolutionary-design-molecules\setup.py`

```
# 导入设置工具和查找包
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'evolutionary-design-molecules',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找并包含所有包
  version = '0.0.1',  # 版本号
  license='MIT',  # 许可证
  description = 'Evolutionary Design of Molecules',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/evolutionary-design-molecules',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',
    'evolutionary algorithms'
  ],
  install_requires=[  # 安装依赖
    'beartype',
    'einops>=0.7.0',
    'torch>=2.0',
    'vector-quantize-pytorch>=1.12.1'
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