# `.\lucidrains\axial-positional-embedding\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'axial_positional_embedding',  # 包的名称
  packages = find_packages(),  # 查找并包含所有包
  version = '0.2.1',  # 版本号
  license='MIT',  # 许可证
  description = 'Axial Positional Embedding',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/axial-positional-embedding',  # 项目链接
  keywords = ['transformers', 'artificial intelligence'],  # 关键词
  install_requires=[
      'torch'  # 安装所需的依赖
  ],
  classifiers=[
      'Development Status :: 4 - Beta',  # 开发状态
      'Intended Audience :: Developers',  # 预期受众
      'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 主题
      'License :: OSI Approved :: MIT License',  # 许可证类型
      'Programming Language :: Python :: 3.6',  # 使用的编程语言版本
  ],
)
```