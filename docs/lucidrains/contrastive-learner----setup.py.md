# `.\lucidrains\contrastive-learner\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'contrastive_learner',
  # 查找并包含所有包
  packages = find_packages(),
  # 版本号
  version = '0.1.1',
  # 许可证
  license='MIT',
  # 描述
  description = 'Self-supervised contrastive learning made simple',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/contrastive-learner',
  # 关键词
  keywords = ['self-supervised learning', 'artificial intelligence'],
  # 安装依赖
  install_requires=[
      'torch',
      'kornia'
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