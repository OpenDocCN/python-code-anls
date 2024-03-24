# `.\lucidrains\BS-RoFormer\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的信息
setup(
  # 包名
  name = 'BS-RoFormer',
  # 查找所有包，不排除任何包
  packages = find_packages(exclude=[]),
  # 版本号
  version = '0.4.0',
  # 许可证
  license='MIT',
  # 描述
  description = 'BS-RoFormer - Band-Split Rotary Transformer for SOTA Music Source Separation',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 项目链接
  url = 'https://github.com/lucidrains/BS-RoFormer',
  # 关键词
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'music source separation'
  ],
  # 安装依赖
  install_requires=[
    'beartype',
    'einops>=0.6.1',
    'librosa',
    'rotary-embedding-torch>=0.3.6',
    'torch>=2.0',
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