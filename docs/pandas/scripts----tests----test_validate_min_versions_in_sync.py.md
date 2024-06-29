# `D:\src\scipysrc\pandas\scripts\tests\test_validate_min_versions_in_sync.py`

```
# 导入必要的模块
import pathlib  # 导入处理路径的模块
import sys  # 导入系统相关的模块

import pytest  # 导入用于编写测试用例的 pytest 模块
import yaml  # 导入处理 YAML 格式的模块

# 根据 Python 版本导入不同的模块
if sys.version_info >= (3, 11):
    import tomllib  # Python 3.11 及以上版本使用 tomllib 模块
else:
    import tomli as tomllib  # Python 3.10 及以下版本使用 tomli 模块

# 从指定的脚本中导入特定函数
from scripts.validate_min_versions_in_sync import (
    get_toml_map_from,  # 导入函数 get_toml_map_from
    get_yaml_map_from,  # 导入函数 get_yaml_map_from
    pin_min_versions_to_yaml_file,  # 导入函数 pin_min_versions_to_yaml_file
)

# 使用 pytest 的 parametrize 装饰器来定义多个测试参数
@pytest.mark.parametrize(
    "src_toml, src_yaml, expected_yaml",
    [
        (
            pathlib.Path("scripts/tests/data/deps_minimum.toml"),  # 源 TOML 文件路径
            pathlib.Path("scripts/tests/data/deps_unmodified_random.yaml"),  # 源 YAML 文件路径
            pathlib.Path("scripts/tests/data/deps_expected_random.yaml"),  # 预期输出的 YAML 文件路径
        ),
        (
            pathlib.Path("scripts/tests/data/deps_minimum.toml"),  # 源 TOML 文件路径
            pathlib.Path("scripts/tests/data/deps_unmodified_same_version.yaml"),  # 源 YAML 文件路径
            pathlib.Path("scripts/tests/data/deps_expected_same_version.yaml"),  # 预期输出的 YAML 文件路径
        ),
        (
            pathlib.Path("scripts/tests/data/deps_minimum.toml"),  # 源 TOML 文件路径
            pathlib.Path("scripts/tests/data/deps_unmodified_duplicate_package.yaml"),  # 源 YAML 文件路径
            pathlib.Path("scripts/tests/data/deps_expected_duplicate_package.yaml"),  # 预期输出的 YAML 文件路径
        ),
        (
            pathlib.Path("scripts/tests/data/deps_minimum.toml"),  # 源 TOML 文件路径
            pathlib.Path("scripts/tests/data/deps_unmodified_no_version.yaml"),  # 源 YAML 文件路径
            pathlib.Path("scripts/tests/data/deps_expected_no_version.yaml"),  # 预期输出的 YAML 文件路径
        ),
        (
            pathlib.Path("scripts/tests/data/deps_minimum.toml"),  # 源 TOML 文件路径
            pathlib.Path("scripts/tests/data/deps_unmodified_range.yaml"),  # 源 YAML 文件路径
            pathlib.Path("scripts/tests/data/deps_expected_range.yaml"),  # 预期输出的 YAML 文件路径
        ),
    ],
)
def test_pin_min_versions_to_yaml_file(src_toml, src_yaml, expected_yaml) -> None:
    # 打开并加载 TOML 文件，得到 TOML 文件的映射
    with open(src_toml, "rb") as toml_f:
        toml_map = tomllib.load(toml_f)
    
    # 打开并读取 YAML 文件内容
    with open(src_yaml, encoding="utf-8") as yaml_f:
        yaml_file_data = yaml_f.read()
    
    # 使用 yaml.safe_load 将 YAML 数据转换为 Python 字典
    yaml_file = yaml.safe_load(yaml_file_data)
    
    # 获取 YAML 文件中的 "dependencies" 键对应的字典
    yaml_dependencies = yaml_file["dependencies"]
    
    # 从 YAML 的依赖字典中获取 YAML 映射
    yaml_map = get_yaml_map_from(yaml_dependencies)
    
    # 获取 TOML 映射
    toml_map = get_toml_map_from(toml_map)
    
    # 将最小版本固定到 YAML 文件中，并返回结果 YAML 数据
    result_yaml_file = pin_min_versions_to_yaml_file(yaml_map, toml_map, yaml_file_data)
    
    # 打开预期输出的 YAML 文件，并读取其内容
    with open(expected_yaml, encoding="utf-8") as yaml_f:
        dummy_yaml_expected_file_1 = yaml_f.read()
    
    # 断言生成的 YAML 数据与预期的 YAML 数据一致
    assert result_yaml_file == dummy_yaml_expected_file_1
```