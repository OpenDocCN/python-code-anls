# `.\lucidrains\perfusion-pytorch\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'perfusion-pytorch',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.1.23',  # 版本号
  license='MIT',  # 许可证
  description = 'Perfusion - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/perfusion-pytorch',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',
    'memory editing',
    'text-to-image'
  ],
  install_requires=[  # 安装依赖
    'beartype',
    'einops>=0.6.1',
    'open-clip-torch',
    'opt-einsum',
    'torch>=2.0'
  ],
  include_package_data = True,  # 包含数据文件
  classifiers=[  # 分类器列表
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```