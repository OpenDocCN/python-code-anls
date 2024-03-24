# `.\lucidrains\llama-qrlhf\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'llama-qrlhf', # 包的名称
  packages = find_packages(exclude=[]), # 查找所有包
  version = '0.0.1', # 版本号
  license='MIT', # 许可证
  description = 'Experimental Q-RLHF applied to Language Modeling. Made compatible with Llama of course', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  long_description_content_type = 'text/markdown', # 长描述内容类型
  url = 'https://github.com/lucidrains/llama-qrlhf', # 项目链接
  keywords = [
    'artificial intelligence',
    'deep learning',
    'reinforcement learning with human feedback',
    'q learning',
  ], # 关键词
  install_requires = [
    'accelerate',
    'beartype',
    'ema-pytorch',
    'einops>=0.7.0',
    'torch>=2.0'
  ], # 安装依赖
  classifiers=[
    'Development Status :: 4 - Beta', # 开发状态
    'Intended Audience :: Developers', # 目标受众
    'Topic :: Scientific/Engineering :: Artificial Intelligence', # 主题
    'License :: OSI Approved :: MIT License', # 许可证
    'Programming Language :: Python :: 3.6', # 编程语言
  ],
)
```