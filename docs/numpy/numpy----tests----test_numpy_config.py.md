# `.\numpy\numpy\tests\test_numpy_config.py`

```
"""
Check the numpy config is valid.
"""
# 导入所需的库和模块
import numpy as np
import pytest
from unittest.mock import Mock, patch

# 标记当前测试为需要 Meson 构建的跳过状态，若 numpy.__config__ 没有 "_built_with_meson" 属性
pytestmark = pytest.mark.skipif(
    not hasattr(np.__config__, "_built_with_meson"),
    reason="Requires Meson builds",
)

# 定义测试 NumPy 配置的类
class TestNumPyConfigs:
    # 必要的配置键列表
    REQUIRED_CONFIG_KEYS = [
        "Compilers",
        "Machine Information",
        "Python Information",
    ]

    # 测试当 pyyaml 未安装时的警告行为
    @patch("numpy.__config__._check_pyyaml")
    def test_pyyaml_not_found(self, mock_yaml_importer):
        mock_yaml_importer.side_effect = ModuleNotFoundError()
        # 在测试中捕获 UserWarning
        with pytest.warns(UserWarning):
            np.show_config()

    # 测试以字典模式显示配置信息
    def test_dict_mode(self):
        # 调用 NumPy 的 show_config 方法，以字典模式返回配置信息
        config = np.show_config(mode="dicts")

        # 断言返回的配置信息是字典类型，并且包含所有必需的配置键
        assert isinstance(config, dict)
        assert all([key in config for key in self.REQUIRED_CONFIG_KEYS]), (
            "Required key missing,"
            " see index of `False` with `REQUIRED_CONFIG_KEYS`"
        )

    # 测试使用无效模式时是否引发 AttributeError
    def test_invalid_mode(self):
        # 断言调用 show_config 方法时，使用无效的模式参数会引发 AttributeError
        with pytest.raises(AttributeError):
            np.show_config(mode="foo")

    # 测试警告添加新测试模式时的行为
    def test_warn_to_add_tests(self):
        # 断言 np.__config__.DisplayModes 的长度为 2，以确保未添加新模式
        assert len(np.__config__.DisplayModes) == 2, (
            "New mode detected,"
            " please add UT if applicable and increment this count"
        )
```