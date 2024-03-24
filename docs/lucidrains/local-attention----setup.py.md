# `.\lucidrains\local-attention\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'local-attention',  # 包的名称
  packages = find_packages(),  # 查找并包含所有包
  version = '1.9.0',  # 版本号
  license='MIT',  # 许可证
  description = 'Local attention, window with lookback, for language modeling',  # 描述
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/local-attention',  # 项目链接
  keywords = [
    'transformers',  # 关键词：transformers
    'attention',  # 关键词：attention
    'artificial intelligence'  # 关键词：artificial intelligence
  ],
  install_requires=[
    'einops>=0.6.0',  # 安装所需的依赖项：einops>=0.6.0
    'torch'  # 安装所需的依赖项：torch
  ],
  classifiers=[
    'Development Status :: 4 - Beta',  # 分类器：开发状态为Beta
    'Intended Audience :: Developers',  # 分类器：面向的受众为开发者
    'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 分类器：主题为科学/工程和人工智能
    'License :: OSI Approved :: MIT License',  # 分类器：许可证为MIT
    'Programming Language :: Python :: 3.6',  # 分类器：编程语言为Python 3.6
  ],
)
```