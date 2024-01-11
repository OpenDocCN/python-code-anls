# `ZeroNet\src\Test\TestConfig.py`

```
# 导入 pytest 模块
import pytest

# 导入 Config 模块
import Config

# 使用 resetSettings 修饰器来重置设置
@pytest.mark.usefixtures("resetSettings")
class TestConfig:
    # 测试解析方法
    def testParse(self):
        # 默认情况下的测试
        config_test = Config.Config("zeronet.py".split(" "))
        config_test.parse(silent=True, parse_config=False)
        # 断言调试模式和调试套接字都为假
        assert not config_test.debug
        assert not config_test.debug_socket

        # 测试解析带有未知参数（ui_password）的命令行
        config_test = Config.Config("zeronet.py --debug --debug_socket --ui_password hello".split(" "))
        config_test.parse(silent=True, parse_config=False)
        # 断言调试模式和调试套接字为真，同时断言会引发 AttributeError 异常
        assert config_test.debug
        assert config_test.debug_socket
        with pytest.raises(AttributeError):
            config_test.ui_password

        # 更复杂的测试
        args = "zeronet.py --unknown_arg --debug --debug_socket --ui_restrict 127.0.0.1 1.2.3.4 "
        args += "--another_unknown argument --use_openssl False siteSign address privatekey --inner_path users/content.json"
        config_test = Config.Config(args.split(" "))
        config_test.parse(silent=True, parse_config=False)
        # 断言调试模式为真，ui_restrict 中包含 "1.2.3.4"，use_openssl 为假，inner_path 为 "users/content.json"
        assert config_test.debug
        assert "1.2.3.4" in config_test.ui_restrict
        assert not config_test.use_openssl
        assert config_test.inner_path == "users/content.json"
```