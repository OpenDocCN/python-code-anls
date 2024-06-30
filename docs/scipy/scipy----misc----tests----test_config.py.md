# `D:\src\scipysrc\scipy\scipy\misc\tests\test_config.py`

```
"""
Check the SciPy config is valid.
"""
# 导入需要的库和模块
import scipy
import pytest
from unittest.mock import patch

# 标记当前测试用例，如果 SciPy 没有使用 Meson 构建，则跳过执行
pytestmark = pytest.mark.skipif(
    not hasattr(scipy.__config__, "_built_with_meson"),
    reason="Requires Meson builds",
)


class TestSciPyConfigs:
    # 所需的配置键列表
    REQUIRED_CONFIG_KEYS = [
        "Compilers",
        "Machine Information",
        "Python Information",
    ]

    @patch("scipy.__config__._check_pyyaml")
    # 测试当 pyyaml 模块未找到时的情况
    def test_pyyaml_not_found(self, mock_yaml_importer):
        mock_yaml_importer.side_effect = ModuleNotFoundError()
        # 测试中希望产生 UserWarning
        with pytest.warns(UserWarning):
            # 调用 scipy.show_config() 方法
            scipy.show_config()

    # 测试以字典模式显示配置信息
    def test_dict_mode(self):
        # 调用 scipy.show_config() 方法以字典形式获取配置信息
        config = scipy.show_config(mode="dicts")

        # 断言 config 是一个字典类型
        assert isinstance(config, dict)
        # 断言所需的所有配置键都存在于 config 中
        assert all([key in config for key in self.REQUIRED_CONFIG_KEYS]), (
            "Required key missing,"
            " see index of `False` with `REQUIRED_CONFIG_KEYS`"
        )

    # 测试使用无效模式时的行为
    def test_invalid_mode(self):
        # 断言调用 scipy.show_config() 方法时使用无效的 mode 参数会引发 AttributeError 异常
        with pytest.raises(AttributeError):
            scipy.show_config(mode="foo")

    # 测试警告以添加测试用例
    def test_warn_to_add_tests(self):
        # 断言 scipy.__config__.DisplayModes 中有两个模式
        assert len(scipy.__config__.DisplayModes) == 2, (
            "New mode detected,"
            " please add UT if applicable and increment this count"
        )
```