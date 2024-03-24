# `.\lucidrains\token-shift-gpt\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'token-shift-gpt',  # 包的名称
  packages = find_packages(),  # 查找所有包
  version = '0.0.3',  # 版本号
  license='MIT',  # 许可证
  description = 'Token Shift GPT - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/token-shift-gpt',  # 项目链接
  keywords = [
    'artificial intelligence',  # 关键词
    'deep learning',  # 关键词
    'autoregressive language modeling'  # 关键词
  ],
  install_requires=[
    'einops>=0.3',  # 安装所需的依赖
    'torch>=1.6'  # 安装所需的依赖
  ],
  classifiers=[
    'Development Status :: 4 - Beta',  # 分类器
    'Intended Audience :: Developers',  # 分类器
    'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 分类器
    'License :: OSI Approved :: MIT License',  # 分类器
    'Programming Language :: Python :: 3.6',  # 分类器
  ],
)
```