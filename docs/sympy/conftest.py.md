# `D:\src\scipysrc\sympy\conftest.py`

```
# -*- coding: utf-8 -*-  # 指定编码格式为 UTF-8

from __future__ import print_function, division, absolute_import  # 导入未来版本兼容模块

import os  # 导入操作系统相关功能
from itertools import chain  # 导入迭代工具链
import json  # 导入 JSON 序列化和反序列化模块
import sys  # 导入系统相关功能
import warnings  # 导入警告模块
import pytest  # 导入 pytest 测试框架
from sympy.testing.runtests import setup_pprint, _get_doctest_blacklist  # 导入 Sympy 测试相关功能

durations_path = os.path.join(os.path.dirname(__file__), '.ci', 'durations.json')  # 定义持续时间数据文件路径
blacklist_path = os.path.join(os.path.dirname(__file__), '.ci', 'blacklisted.json')  # 定义黑名单数据文件路径

collect_ignore = _get_doctest_blacklist()  # 获取忽略列表

# 为 doctest 设置打印输出
setup_pprint(disable_line_wrap=False)
sys.__displayhook__ = sys.displayhook

# 下面的代码被注释掉了，不会被执行
# from sympy import pprint_use_unicode
# pprint_use_unicode(False)

# 根据组字典创建分组
def _mk_group(group_dict):
    return list(chain(*[[k+'::'+v for v in files] for k, files in group_dict.items()]))

# 如果持续时间文件存在，则读取其中的数据
if os.path.exists(durations_path):
    with open(durations_path, 'rt') as fin:
        text = fin.read()
    veryslow_group, slow_group = [_mk_group(group_dict) for group_dict in json.loads(text)]
else:
    # 如果不存在持续时间文件，发出警告
    warnings.warn("conftest.py:22: Could not find %s, --quickcheck and --veryquickcheck will have no effect.\n" % durations_path)
    veryslow_group, slow_group = [], []

# 如果黑名单文件存在，则读取其中的数据
if os.path.exists(blacklist_path):
    with open(blacklist_path, 'rt') as stream:
        blacklist_group = _mk_group(json.load(stream))
else:
    # 如果不存在黑名单文件，发出警告
    warnings.warn("conftest.py:28: Could not find %s, no tests will be skipped due to blacklisting\n" % blacklist_path)
    blacklist_group = []

# 添加 pytest 的选项：跳过非常慢的测试
def pytest_addoption(parser):
    parser.addoption("--quickcheck", dest="runquick", action="store_true",
                     help="Skip very slow tests (see ./ci/parse_durations_log.py)")
    parser.addoption("--veryquickcheck", dest="runveryquick", action="store_true",
                     help="Skip slow & very slow (see ./ci/parse_durations_log.py)")

# 配置 pytest
def pytest_configure(config):
    # 注册额外的标记
    config.addinivalue_line("markers", "slow: manually marked test as slow (use .ci/durations.json instead)")
    config.addinivalue_line("markers", "quickcheck: skip very slow tests")
    config.addinivalue_line("markers", "veryquickcheck: skip slow & very slow tests")

# 运行测试前的设置
def pytest_runtest_setup(item):
    if isinstance(item, pytest.Function):
        # 如果测试在非常慢的组中，并且指定了 --quickcheck 或 --veryquickcheck，则跳过测试
        if item.nodeid in veryslow_group and (item.config.getvalue("runquick") or
                                              item.config.getvalue("runveryquick")):
            pytest.skip("very slow test, skipping since --quickcheck or --veryquickcheck was passed.")
            return
        # 如果测试在慢速组中，并且指定了 --veryquickcheck，则跳过测试
        if item.nodeid in slow_group and item.config.getvalue("runveryquick"):
            pytest.skip("slow test, skipping since --veryquickcheck was passed.")
            return

        # 如果测试在黑名单组中，则跳过测试
        if item.nodeid in blacklist_group:
            pytest.skip("blacklisted test, see %s" % blacklist_path)
            return
```