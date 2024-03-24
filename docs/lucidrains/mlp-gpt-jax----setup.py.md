# `.\lucidrains\mlp-gpt-jax\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
    name="mlp-gpt-jax",  # 包的名称
    packages=find_packages(),  # 查找所有包
    version="0.0.20",  # 版本号
    license="MIT",  # 许可证
    description="MLP GPT - Jax",  # 描述
    author="Phil Wang",  # 作者
    author_email="",  # 作者邮箱
    url="https://github.com/lucidrains/mlp-gpt-jax",  # 项目链接
    keywords=[  # 关键词列表
        "artificial intelligence",
        "deep learning",
        "language model",
        "multilayered-perceptron",
        "jax"
    ],
    install_requires=[  # 安装依赖列表
        "click",
        "click-option-group",
        "einops>=0.3",
        "dm-haiku",
        "jax",
        "jaxlib",
        "optax",
        "torch",
        "tqdm"
    ],
    classifiers=[  # 分类器列表
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
```