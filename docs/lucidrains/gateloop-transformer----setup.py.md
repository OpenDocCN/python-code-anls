# `.\lucidrains\gateloop-transformer\setup.py`

```py
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'gateloop-transformer',  # 包名
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.2.4',  # 版本号
  license='MIT',  # 许可证
  description = 'GateLoop Transformer',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/gateloop-transformer',  # 项目链接
  keywords = [
    'artificial intelligence',  # 关键词
    'deep learning',  # 关键词
    'gated linear attention'  # 关键词
  ],
  install_requires=[
    'einops>=0.7.0',  # 安装所需的依赖包
    'rotary-embedding-torch',  # 安装所需的依赖包
    'torch>=2.1',  # 安装所需的依赖包
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