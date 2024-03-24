# `.\lucidrains\rotary-embedding-torch\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'rotary-embedding-torch',
  # 查找并包含所有包
  packages = find_packages(),
  # 版本号
  version = '0.5.3',
  # 许可证
  license='MIT',
  # 描述
  description = 'Rotary Embedding - Pytorch',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/rotary-embedding-torch',
  # 关键词
  keywords = [
    'artificial intelligence',
    'deep learning',
    'positional embedding'    
  ],
  # 安装依赖
  install_requires=[
    'beartype',
    'einops>=0.7',
    'torch>=2.0'
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