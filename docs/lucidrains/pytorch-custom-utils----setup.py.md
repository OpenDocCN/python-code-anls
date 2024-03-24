# `.\lucidrains\pytorch-custom-utils\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'pytorch-custom-utils',  # 包名
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.0.18',  # 版本号
  license='MIT',  # 许可证
  description = 'Pytorch Custom Utils',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/pytorch-custom-utils',  # URL
  keywords = [
    'pytorch',  # 关键字
    'accelerate'  # 关键字
  ],
  install_requires=[
    'accelerate',  # 安装依赖
    'optree',  # 安装依赖
    'pytorch-warmup',  # 安装依赖
    'torch>=2.0'  # 安装依赖
  ],
  classifiers=[
    'Development Status :: 4 - Beta',  # 分类
    'Intended Audience :: Developers',  # 分类
    'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 分类
    'License :: OSI Approved :: MIT License',  # 分类
    'Programming Language :: Python :: 3.6',  # 分类
  ],
)
```