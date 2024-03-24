# `.\lucidrains\agent-attention-pytorch\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'agent-attention-pytorch',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.1.7',  # 版本号
  license='MIT',  # 许可证
  description = 'Agent Attention - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/agent-attention-pytorch',  # URL
  keywords = [
    'artificial intelligence',  # 关键词
    'deep learning',  # 关键词
    'attention',  # 关键词
    'linear attention'  # 关键词
  ],
  install_requires=[
    'einops>=0.7.0',  # 安装所需的依赖
    'torch>=2.0'  # 安装所需的依赖
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