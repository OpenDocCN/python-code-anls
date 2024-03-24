# `.\lucidrains\ema-pytorch\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'ema-pytorch',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.4.3',  # 版本号
  license='MIT',  # 许可证
  description = 'Easy way to keep track of exponential moving average version of your pytorch module',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/ema-pytorch',  # URL
  keywords = [
    'artificial intelligence',  # 关键词
    'deep learning',  # 关键词
    'exponential moving average'  # 关键词
  ],
  install_requires=[
    'beartype',  # 安装依赖
    'torch>=1.6',  # 安装依赖
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