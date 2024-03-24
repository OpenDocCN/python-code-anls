# `.\lucidrains\lion-pytorch\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'lion-pytorch',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.1.2',  # 版本号
  license='MIT',  # 许可证
  description = 'Lion Optimizer - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/lion-pytorch',  # URL
  keywords = [
    'artificial intelligence',  # 关键词：人工智能
    'deep learning',  # 关键词：深度学习
    'optimizers'  # 关键词：优化器
  ],
  install_requires=[
    'torch>=1.6'  # 安装所需的依赖项
  ],
  classifiers=[
    'Development Status :: 4 - Beta',  # 分类器：开发状态为 Beta
    'Intended Audience :: Developers',  # 分类器：目标受众为开发者
    'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 分类器：主题为科学/工程 - 人工智能
    'License :: OSI Approved :: MIT License',  # 分类器：许可证为 MIT
    'Programming Language :: Python :: 3.6',  # 分类器：编程语言为 Python 3.6
  ],
)
```