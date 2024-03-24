# `.\lucidrains\g-mlp-gpt\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的信息
setup(
  # 包的名称
  name = 'g-mlp-gpt',
  # 查找所有包
  packages = find_packages(),
  # 版本号
  version = '0.0.15',
  # 许可证
  license='MIT',
  # 描述
  description = 'gMLP - GPT',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/g-mlp-gpt',
  # 关键词
  keywords = [
    'artificial intelligence',
    'deep learning',
    'multi-layered-preceptrons'
  ],
  # 安装依赖
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
  ],
  # 分类
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```