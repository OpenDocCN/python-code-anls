# `MetaGPT\tests\metagpt\utils\test_config.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/1 11:19
@Author  : alexanderwu
@File    : test_config.py
@Modified By: mashenquan, 2013/8/20, Add `test_options`; remove global configuration `CONFIG`, enable configuration support for business isolation.
"""
# 导入模块
from pathlib import Path
# 导入 pytest 模块
import pytest
# 导入自定义的 Config 类
from metagpt.config import Config

# 测试获取配置项不存在的情况
def test_config_class_get_key_exception():
    # 使用 pytest 断言异常
    with pytest.raises(Exception) as exc_info:
        # 创建 Config 对象
        config = Config()
        # 获取不存在的配置项
        config.get("wtf")
    # 断言异常信息
    assert str(exc_info.value) == "Key 'wtf' not found in environment variables or in the YAML file"

# 测试配置文件不存在的情况
def test_config_yaml_file_not_exists():
    # FIXME: 由于这里是单例，所以会导致Config重新创建失效。后续要将Config改为非单例模式。
    # 创建 Config 对象，传入不存在的配置文件名
    _ = Config("wtf.yaml")
    # 测试配置项不存在的情况
    # with pytest.raises(Exception) as exc_info:
    #     config.get("OPENAI_BASE_URL")
    # assert str(exc_info.value) == "Set OPENAI_API_KEY or Anthropic_API_KEY first"

# 测试获取配置项的选项
def test_options():
    # 获取配置文件的路径
    filename = Path(__file__).resolve().parent.parent.parent.parent / "config/config.yaml"
    # 创建 Config 对象，传入配置文件路径
    config = Config(filename)
    # 断言配置项不为空
    assert config.options

# 如果是主程序，则执行 test_options 函数
if __name__ == "__main__":
    test_options()

```