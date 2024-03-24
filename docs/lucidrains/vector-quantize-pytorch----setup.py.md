# `.\lucidrains\vector-quantize-pytorch\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'vector_quantize_pytorch',
  # 查找并包含所有包
  packages = find_packages(),
  # 版本号
  version = '1.14.5',
  # 许可证
  license='MIT',
  # 描述
  description = 'Vector Quantization - Pytorch',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/vector-quantizer-pytorch',
  # 关键词
  keywords = [
    'artificial intelligence',
    'deep learning',
    'pytorch',
    'quantization'
  ],
  # 安装依赖
  install_requires=[
    'einops>=0.7.0',
    'einx[torch]>=0.1.3',
    'torch'
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