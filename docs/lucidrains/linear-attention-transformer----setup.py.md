# `.\lucidrains\linear-attention-transformer\setup.py`

```py
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'linear_attention_transformer',  # 包的名称
  packages = find_packages(exclude=['examples']),  # 查找并包含除了 examples 之外的所有包
  version = '0.19.1',  # 版本号
  license='MIT',  # 许可证
  description = 'Linear Attention Transformer',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/linear-attention-transformer',  # 项目链接
  keywords = ['transformers', 'attention', 'artificial intelligence'],  # 关键词
  install_requires=[
      'axial-positional-embedding',  # 安装所需的依赖包
      'einops',
      'linformer>=0.1.0',
      'local-attention',
      'product-key-memory>=0.1.5',
      'torch',
  ],
  classifiers=[
      'Development Status :: 4 - Beta',  # 分类器
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)
```