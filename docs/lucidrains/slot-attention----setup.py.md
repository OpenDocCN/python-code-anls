# `.\lucidrains\slot-attention\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'slot_attention',  # 包名
  packages = find_packages(),  # 查找所有包
  version = '1.1.2',  # 版本号
  license='MIT',  # 许可证
  description = 'Implementation of Slot Attention in Pytorch',  # 描述
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/slot-attention',  # 项目链接
  keywords = ['attention', 'artificial intelligence'],  # 关键词
  install_requires=[
      'torch'  # 安装依赖
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