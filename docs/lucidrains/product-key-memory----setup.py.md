# `.\lucidrains\product-key-memory\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
    name = 'product_key_memory',  # 包的名称
    packages = find_packages(),  # 查找所有包
    version = '0.2.10',  # 版本号
    license = 'MIT',  # 许可证
    description = 'Product Key Memory',  # 描述
    long_description_content_type = 'text/markdown',  # 长描述内容类型
    author = 'Aran Komatsuzaki, Phil Wang',  # 作者
    author_email = 'aran1234321@gmail.com, lucidrains@gmail.com',  # 作者邮箱
    url = 'https://github.com/lucidrains/product-key-memory',  # 项目链接
    keywords = [
        'transformers',  # 关键词：transformers
        'artificial intelligence'  # 关键词：artificial intelligence
    ],
    install_requires=[
        'colt5-attention>=0.10.14',  # 安装所需的依赖包
        'einops>=0.6',  # 安装所需的依赖包
        'torch'  # 安装所需的依赖包
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