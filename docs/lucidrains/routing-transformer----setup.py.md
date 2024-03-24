# `.\lucidrains\routing-transformer\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'routing_transformer',  # 包的名称
  packages = find_packages(exclude=['examples']),  # 查找并包含除了 examples 之外的所有包
  version = '1.6.1',  # 版本号
  license='MIT',  # 许可证
  description = 'Routing Transformer (Pytorch)',  # 描述
  author = 'Phil Wang, Aran Komatsuzaki',  # 作者
  author_email = 'lucidrains@gmail.com, aran1234321@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/routing-transformer',  # 项目链接
  keywords = ['transformers', 'attention', 'artificial intelligence'],  # 关键词
  install_requires=[
      'einops',  # 安装所需的依赖包
      'local-attention>=1.4.0',
      'mixture-of-experts>=0.2.0',
      'product-key-memory',
      'torch'
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