# `.\lucidrains\discrete-key-value-bottleneck-pytorch\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'discrete-key-value-bottleneck-pytorch',
  # 查找所有包，不排除任何包
  packages = find_packages(exclude=[]),
  # 版本号
  version = '0.1.1',
  # 许可证类型
  license='MIT',
  # 描述信息
  description = 'Discrete Key / Value Bottleneck - Pytorch',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 项目的URL
  url = 'https://github.com/lucidrains/discrete-key-value-bottleneck-pytorch',
  # 关键词列表
  keywords = [
    'artificial intelligence',
    'deep learning',
    'quantization',
    'memory',
    'transfer learning'
  ],
  # 安装依赖项
  install_requires=[
    'einops>=0.6',
    'vector-quantize-pytorch>=1.6.28',
    'torch>=1.6',
  ],
  # 分类标签
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```