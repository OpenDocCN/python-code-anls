# `.\lucidrains\learning-to-expire-pytorch\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的信息
setup(
  # 包名
  name = 'learning-to-expire-pytorch',
  # 查找包，排除 examples 文件夹
  packages = find_packages(exclude=['examples']),
  # 版本号
  version = '0.0.2',
  # 许可证
  license='MIT',
  # 描述
  description = 'Learning to Expire - Pytorch',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/learning-to-expire-pytorch',
  # 关键词
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'memory'
  ],
  # 安装依赖
  install_requires=[
    'torch>=1.6',
    'einops>=0.3'
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