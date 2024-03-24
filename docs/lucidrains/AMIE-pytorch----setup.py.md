# `.\lucidrains\AMIE-pytorch\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'AMIE-pytorch',  # 包名
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.0.1',  # 版本号
  license='MIT',  # 许可证
  description = 'AMIE',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/AMIE-pytorch',  # URL
  keywords = [
    'artificial intelligence',  # 关键词
    'deep learning',  # 关键词
    'differential diagnosis'  # 关键词
  ],
  install_requires=[  # 安装依赖
    'accelerate',  # 加速库
    'beartype',  # 类型检查库
    'einops>=0.7.0',  # 数据操作库
    'einx>=0.1.2',  # 扩展库
    'torch>=2.0',  # PyTorch
    'tqdm'  # 进度条库
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