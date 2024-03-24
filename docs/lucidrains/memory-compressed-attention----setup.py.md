# `.\lucidrains\memory-compressed-attention\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'memory_compressed_attention',  # 包的名称
  packages = find_packages(),  # 查找所有包
  version = '0.0.7',  # 版本号
  license='MIT',  # 许可证
  description = 'Memory-Compressed Self Attention',  # 描述
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/memory-compressed-attention',  # 项目链接
  keywords = ['transformers', 'artificial intelligence', 'attention mechanism'],  # 关键词
  install_requires=[
    'torch'  # 安装所需的依赖
  ],
  classifiers=[
    'Development Status :: 4 - Beta',  # 开发状态
    'Intended Audience :: Developers',  # 预期受众
    'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 主题
    'License :: OSI Approved :: MIT License',  # 许可证
    'Programming Language :: Python :: 3.6',  # 编程语言
  ],
)
```