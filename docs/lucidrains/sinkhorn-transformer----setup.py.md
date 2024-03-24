# `.\lucidrains\sinkhorn-transformer\setup.py`

```py
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'sinkhorn_transformer',
  # 查找并排除示例文件夹以外的所有包
  packages = find_packages(exclude=['examples']),
  # 版本号
  version = '0.11.4',
  # 许可证类型
  license='MIT',
  # 描述信息
  description = 'Sinkhorn Transformer - Sparse Sinkhorn Attention',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/sinkhorn-transformer',
  # 关键词
  keywords = ['transformers', 'attention', 'artificial intelligence'],
  # 安装依赖项
  install_requires=[
      'axial-positional-embedding>=0.1.0',
      'local-attention',
      'product-key-memory',  
      'torch'
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