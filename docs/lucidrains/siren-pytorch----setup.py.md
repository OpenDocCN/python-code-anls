# `.\lucidrains\siren-pytorch\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的信息
setup(
  # 包名
  name = 'siren-pytorch',
  # 查找所有包
  packages = find_packages(),
  # 版本号
  version = '0.1.7',
  # 许可证
  license='MIT',
  # 描述
  description = 'Implicit Neural Representations with Periodic Activation Functions',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/siren-pytorch',
  # 关键词
  keywords = ['artificial intelligence', 'deep learning'],
  # 安装依赖
  install_requires=[
      'einops',
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