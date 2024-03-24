# `.\lucidrains\electra-pytorch\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'electra-pytorch',  # 包的名称
  packages = find_packages(),  # 查找所有包
  version = '0.1.2',  # 版本号
  license='MIT',  # 许可证
  description = 'Electra - Pytorch',  # 描述
  author = 'Erik Nijkamp, Phil Wang',  # 作者
  author_email = 'erik.nijkamp@gmail.com, lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/electra-pytorch',  # 项目链接
  keywords = [
    'transformers',  # 关键词
    'artificial intelligence',  # 关键词
    'pretraining'  # 关键词
  ],
  install_requires=[
    'torch>=1.6.0',  # 安装依赖
    'transformers==3.0.2',  # 安装依赖
    'scipy',  # 安装依赖
    'sklearn'  # 安装依赖
  ],
  setup_requires=[
    'pytest-runner'  # 安装依赖
  ],
  tests_require=[
    'pytest',  # 测试依赖
    'reformer-pytorch'  # 测试依赖
  ],
  classifiers=[
    'Development Status :: 4 - Beta',  # 分类
    'Intended Audience :: Developers',  # 分类
    'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 分类
    'License :: OSI Approved :: MIT License',  # 分类
    'Programming Language :: Python :: 3.7',  # 分类
  ],
)
```