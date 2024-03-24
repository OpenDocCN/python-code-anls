# `.\lucidrains\global-self-attention-network\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'gsa-pytorch', # 包的名称
  packages = find_packages(), # 查找所有包
  version = '0.2.2', # 版本号
  license='MIT', # 许可证
  description = 'Global Self-attention Network (GSA) - Pytorch', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  url = 'https://github.com/lucidrains/global-self-attention-network', # 项目链接
  keywords = [
    'artificial intelligence', # 关键词：人工智能
    'attention mechanism', # 关键词：注意力机制
    'image recognition' # 关键词：图像识别
  ],
  install_requires=[
    'torch>=1.6', # 安装所需的依赖项：torch 版本大于等于 1.6
    'einops>=0.3' # 安装所需的依赖项：einops 版本大于等于 0.3
  ],
  classifiers=[
    'Development Status :: 4 - Beta', # 分类器：开发状态为 Beta
    'Intended Audience :: Developers', # 分类器：面向的受众为开发者
    'Topic :: Scientific/Engineering :: Artificial Intelligence', # 分类器：主题为科学/工程 - 人工智能
    'License :: OSI Approved :: MIT License', # 分类器：许可证为 MIT
    'Programming Language :: Python :: 3.6', # 分类器：编程语言为 Python 3.6
  ],
)
```