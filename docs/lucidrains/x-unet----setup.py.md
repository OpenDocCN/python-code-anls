# `.\lucidrains\x-unet\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'x-unet',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.3.1',  # 版本号
  license='MIT',  # 许可证
  description = 'X-Unet',  # 描述
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/x-unet',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',
    'biomedical segmentation',
    'medical deep learning',
    'unets',
  ],
  install_requires=[  # 安装依赖
    'beartype',
    'einops>=0.4',
    'torch>=1.6',
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