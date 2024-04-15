# `.\pandas-ta\setup.py`

```
# -*- coding: utf-8 -*-
# 导入 setup 函数，用于设置 Python 包的元数据和安装信息
from distutils.core import setup

# 定义长描述信息
long_description = "An easy to use Python 3 Pandas Extension with 130+ Technical Analysis Indicators. Can be called from a Pandas DataFrame or standalone like TA-Lib. Correlation tested with TA-Lib."

# 设置函数调用，用于设置 Python 包的元数据和安装信息
setup(
    # 包的名称
    name="pandas_ta",
    # 包含的子包列表
    packages=[
        "pandas_ta",
        "pandas_ta.candles",
        "pandas_ta.cycles",
        "pandas_ta.momentum",
        "pandas_ta.overlap",
        "pandas_ta.performance",
        "pandas_ta.statistics",
        "pandas_ta.trend",
        "pandas_ta.utils",
        "pandas_ta.utils.data",
        "pandas_ta.volatility",
        "pandas_ta.volume"
    ],
    # 版本号
    version=".".join(("0", "3", "14b")),
    # 简要描述
    description=long_description,
    # 长描述
    long_description=long_description,
    # 作者
    author="Kevin Johnson",
    # 作者邮箱
    author_email="appliedmathkj@gmail.com",
    # 项目 URL
    url="https://github.com/twopirllc/pandas-ta",
    # 维护者
    maintainer="Kevin Johnson",
    # 维护者邮箱
    maintainer_email="appliedmathkj@gmail.com",
    # 下载 URL
    download_url="https://github.com/twopirllc/pandas-ta.git",
    # 关键字列表
    keywords=["technical analysis", "trading", "python3", "pandas"],
    # 许可证
    license="The MIT License (MIT)",
    # 分类信息列表
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    # 包数据配置
    package_data={
        "data": ["data/*.csv"],
    },
    # 安装依赖项
    install_requires=["pandas"],
    # 列出额外的依赖组（例如开发依赖）
    extras_require={
        "dev": [
            "alphaVantage-api", "matplotlib", "mplfinance", "scipy",
            "sklearn", "statsmodels", "stochastic",
            "talib", "tqdm", "vectorbt", "yfinance",
        ],
        "test": ["ta-lib"],
    },
)
```