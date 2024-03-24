# `.\lucidrains\lumiere-pytorch\setup.py`

```py
# 导入设置安装和查找包的函数
from setuptools import setup, find_packages

# 设置包的信息
setup(
  # 包名
  name = 'lumiere-pytorch',
  # 查找所有包，不排除任何包
  packages = find_packages(exclude=[]),
  # 版本号
  version = '0.0.20',
  # 许可证
  license='MIT',
  # 描述
  description = 'Lumiere',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 项目链接
  url = 'https://github.com/lucidrains/lumiere-pytorch',
  # 关键词
  keywords = [
    'artificial intelligence',
    'deep learning',
    'text-to-video'
  ],
  # 安装依赖
  install_requires=[
    'beartype',
    'einops>=0.7.0',
    'optree',
    'torch>=2.0',
    'x-transformers'
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