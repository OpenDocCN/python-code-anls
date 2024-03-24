# `.\lucidrains\CALM-pytorch\setup.py`

```py
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的信息
setup(
  name = 'CALM-Pytorch',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.2.1',  # 版本号
  license='MIT',  # 许可证
  description = 'CALM - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/CALM-pytorch',  # URL
  keywords = [
    'artificial intelligence',  # 关键词
    'deep learning',  # 关键词
    'composing LLMs'  # 关键词
  ],
  install_requires = [  # 安装依赖
    'accelerate',  # 加速库
    'beartype',  # 类型检查库
    'einops>=0.7.0',  # 数据重塑库
    'pytorch-custom-utils>=0.0.11',  # PyTorch自定义工具库
    'torch>=2.0',  # PyTorch库
    'tqdm',  # 进度条库
    'x-transformers>=1.27.3'  # 自定义Transformer库
  ],
  classifiers=[  # 分类器
    'Development Status :: 4 - Beta',  # 开发状态
    'Intended Audience :: Developers',  # 目标受众
    'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 主题
    'License :: OSI Approved :: MIT License',  # 许可证
    'Programming Language :: Python :: 3.6',  # 编程语言
  ],
)
```