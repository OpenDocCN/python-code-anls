# `.\lucidrains\kronecker-attention-pytorch\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'kronecker-attention-pytorch',  # 包的名称
  packages = find_packages(),  # 查找所有包
  version = '0.0.6',  # 版本号
  license='MIT',  # 许可证
  description = 'Kronecker Attention - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/kronecker-attention-pytorch',  # 项目链接
  keywords = [
    'artificial intelligence',  # 关键词：人工智能
    'attention mechanism'  # 关键词：注意力机制
  ],
  install_requires=[
    'torch',  # 安装依赖：torch
    'einops>=0.3'  # 安装依赖：einops 版本大于等于0.3
  ],
  classifiers=[
    'Development Status :: 4 - Beta',  # 分类：开发状态为Beta
    'Intended Audience :: Developers',  # 分类：面向的受众为开发者
    'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 分类：主题为科学/工程 - 人工智能
    'License :: OSI Approved :: MIT License',  # 分类：许可证为MIT
    'Programming Language :: Python :: 3.6',  # 分类：编程语言为Python 3.6
  ],
)
```