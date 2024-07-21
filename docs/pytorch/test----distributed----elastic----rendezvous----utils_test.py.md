# `.\pytorch\test\distributed\elastic\rendezvous\utils_test.py`

```
# 导入必要的模块和库
import socket  # 提供网络通信的基础功能
import threading  # 提供多线程支持
import time  # 提供时间相关的功能
from datetime import timedelta  # 从datetime模块中导入timedelta类
from typing import List  # 引入类型提示，声明List类型
from unittest import TestCase  # 导入单元测试框架的TestCase类
from unittest.mock import patch  # 导入用于模拟(patch)的函数

# 从torch.distributed.elastic.rendezvous.utils模块中导入多个工具函数
from torch.distributed.elastic.rendezvous.utils import (
    _delay,  # 延迟函数
    _matches_machine_hostname,  # 判断是否匹配机器主机名
    _parse_rendezvous_config,  # 解析会面配置的函数
    _PeriodicTimer,  # 周期性计时器类
    _try_parse_port,  # 尝试解析端口号的函数
    parse_rendezvous_endpoint,  # 解析会面终端点的函数
)


class UtilsTest(TestCase):
    # 测试_parse_rendezvous_config函数返回一个字典
    def test_parse_rendezvous_config_returns_dict(self) -> None:
        # 预期的配置字典
        expected_config = {
            "a": "dummy1",
            "b": "dummy2",
            "c": "dummy3=dummy4",
            "d": "dummy5/dummy6",
        }

        # 调用_parse_rendezvous_config函数，解析配置字符串并返回配置字典
        config = _parse_rendezvous_config(
            " b= dummy2  ,c=dummy3=dummy4,  a =dummy1,d=dummy5/dummy6"
        )

        # 断言解析后的配置字典与预期的配置字典相等
        self.assertEqual(config, expected_config)

    # 测试_parse_rendezvous_config函数处理空字符串时返回空字典
    def test_parse_rendezvous_returns_empty_dict_if_str_is_empty(self) -> None:
        # 空字符串和纯空格组成的字符串作为测试用例
        config_strs = ["", "   "]

        # 遍历测试用例
        for config_str in config_strs:
            with self.subTest(config_str=config_str):
                # 调用_parse_rendezvous_config函数，解析配置字符串并返回配置字典
                config = _parse_rendezvous_config(config_str)

                # 断言解析后的配置字典为空字典
                self.assertEqual(config, {})

    # 测试_parse_rendezvous_config函数处理无效字符串时引发异常
    def test_parse_rendezvous_raises_error_if_str_is_invalid(self) -> None:
        # 不合法的配置字符串作为测试用例
        config_strs = [
            "a=dummy1,",
            "a=dummy1,,c=dummy2",
            "a=dummy1,   ,c=dummy2",
            "a=dummy1,=  ,c=dummy2",
            "a=dummy1, = ,c=dummy2",
            "a=dummy1,  =,c=dummy2",
            " ,  ",
        ]

        # 遍历测试用例
        for config_str in config_strs:
            with self.subTest(config_str=config_str):
                # 使用assertRaisesRegex断言_parse_rendezvous_config函数对于不合法的配置字符串会引发ValueError异常
                with self.assertRaisesRegex(
                    ValueError,
                    r"^The rendezvous configuration string must be in format "
                    r"<key1>=<value1>,...,<keyN>=<valueN>.$",
                ):
                    _parse_rendezvous_config(config_str)

    # 测试_parse_rendezvous_config函数处理值为空的情况时引发异常
    def test_parse_rendezvous_raises_error_if_value_is_empty(self) -> None:
        # 值为空的配置字符串作为测试用例
        config_strs = [
            "b=dummy1,a,c=dummy2",
            "b=dummy1,c=dummy2,a",
            "b=dummy1,a=,c=dummy2",
            "  a ",
        ]

        # 遍历测试用例
        for config_str in config_strs:
            with self.subTest(config_str=config_str):
                # 使用assertRaisesRegex断言_parse_rendezvous_config函数对于值为空的配置字符串会引发ValueError异常
                with self.assertRaisesRegex(
                    ValueError,
                    r"^The rendezvous configuration option 'a' must have a value specified.$",
                ):
                    _parse_rendezvous_config(config_str)

    # 测试_try_parse_port函数解析端口号字符串并返回整数端口号
    def test_try_parse_port_returns_port(self) -> None:
        # 调用_try_parse_port函数，解析端口号字符串并返回整数端口号
        port = _try_parse_port("123")

        # 断言解析后的端口号与预期的端口号相等
        self.assertEqual(port, 123)
    # 测试 _try_parse_port 函数处理无效字符串时是否返回 None
    def test_try_parse_port_returns_none_if_str_is_invalid(self) -> None:
        # 不同的无效端口字符串列表
        port_strs = [
            "",
            "   ",
            "  1",
            "1  ",
            " 1 ",
            "abc",
        ]

        # 遍历每个无效端口字符串
        for port_str in port_strs:
            # 使用 subTest 创建子测试，传入当前端口字符串
            with self.subTest(port_str=port_str):
                # 调用 _try_parse_port 函数尝试解析端口
                port = _try_parse_port(port_str)

                # 断言端口应为 None
                self.assertIsNone(port)

    # 测试 parse_rendezvous_endpoint 函数是否正确返回 host 和 port 的元组
    def test_parse_rendezvous_endpoint_returns_tuple(self) -> None:
        # 不同的有效端点列表
        endpoints = [
            "dummy.com:0",
            "dummy.com:123",
            "dummy.com:65535",
            "dummy-1.com:0",
            "dummy-1.com:123",
            "dummy-1.com:65535",
            "123.123.123.123:0",
            "123.123.123.123:123",
            "123.123.123.123:65535",
            "[2001:db8::1]:0",
            "[2001:db8::1]:123",
            "[2001:db8::1]:65535",
        ]

        # 遍历每个端点字符串
        for endpoint in endpoints:
            # 使用 subTest 创建子测试，传入当前端点字符串
            with self.subTest(endpoint=endpoint):
                # 调用 parse_rendezvous_endpoint 函数解析端点字符串，设置默认端口为 123
                host, port = parse_rendezvous_endpoint(endpoint, default_port=123)

                # 从端点字符串中分割出期望的 host 和 port
                expected_host, expected_port = endpoint.rsplit(":", 1)

                # 如果 host 是 IPv6 地址，去掉首尾的方括号
                if expected_host[0] == "[" and expected_host[-1] == "]":
                    expected_host = expected_host[1:-1]

                # 断言解析出的 host 和 port 是否与期望一致
                self.assertEqual(host, expected_host)
                self.assertEqual(port, int(expected_port))

    # 测试 parse_rendezvous_endpoint 函数在端点没有指定端口时是否正确返回 host 和默认端口的元组
    def test_parse_rendezvous_endpoint_returns_tuple_if_endpoint_has_no_port(
        self,
    ) -> None:
        # 不同的没有指定端口的端点列表
        endpoints = ["dummy.com", "dummy-1.com", "123.123.123.123", "[2001:db8::1]"]

        # 遍历每个端点字符串
        for endpoint in endpoints:
            # 使用 subTest 创建子测试，传入当前端点字符串
            with self.subTest(endpoint=endpoint):
                # 调用 parse_rendezvous_endpoint 函数解析端点字符串，设置默认端口为 123
                host, port = parse_rendezvous_endpoint(endpoint, default_port=123)

                # 期望的 host 应与端点字符串一致
                expected_host = endpoint

                # 如果 host 是 IPv6 地址，去掉首尾的方括号
                if expected_host[0] == "[" and expected_host[-1] == "]":
                    expected_host = expected_host[1:-1]

                # 断言解析出的 host 和 port 是否与期望一致
                self.assertEqual(host, expected_host)
                self.assertEqual(port, 123)

    # 测试 parse_rendezvous_endpoint 函数在端点为空时是否正确返回默认的本地 host 和端口的元组
    def test_parse_rendezvous_endpoint_returns_tuple_if_endpoint_is_empty(self) -> None:
        # 空端点和仅包含空格的端点列表
        endpoints = ["", "  "]

        # 遍历每个端点字符串
        for endpoint in endpoints:
            # 使用 subTest 创建子测试，传入当前端点字符串
            with self.subTest(endpoint=endpoint):
                # 调用 parse_rendezvous_endpoint 函数解析空端点字符串，设置默认端口为 123
                host, port = parse_rendezvous_endpoint("", default_port=123)

                # 断言解析出的 host 和 port 是否与预期的本地 host 和默认端口一致
                self.assertEqual(host, "localhost")
                self.assertEqual(port, 123)

    # 测试 parse_rendezvous_endpoint 函数在主机名无效时是否正确引发 ValueError 异常
    def test_parse_rendezvous_endpoint_raises_error_if_hostname_is_invalid(
        self,
    ) -> None:
        # 不同的无效主机名端点列表
        endpoints = ["~", "dummy.com :123", "~:123", ":123"]

        # 遍历每个端点字符串
        for endpoint in endpoints:
            # 使用 subTest 创建子测试，传入当前端点字符串
            with self.subTest(endpoint=endpoint):
                # 使用 assertRaisesRegex 断言解析无效端点字符串时会引发指定错误消息的 ValueError 异常
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The hostname of the rendezvous endpoint '{endpoint}' must be a "
                    r"dot-separated list of labels, an IPv4 address, or an IPv6 address.$",
                ):
                    # 调用 parse_rendezvous_endpoint 函数解析无效端点字符串，设置默认端口为 123
                    parse_rendezvous_endpoint(endpoint, default_port=123)
    # 定义测试函数：测试解析会议点端点如果端口无效时是否引发错误
    def test_parse_rendezvous_endpoint_raises_error_if_port_is_invalid(self) -> None:
        # 定义多个会议点端点示例
        endpoints = ["dummy.com:", "dummy.com:abc", "dummy.com:-123", "dummy.com:-"]

        # 遍历每个会议点端点示例
        for endpoint in endpoints:
            # 使用子测试，设置当前测试的端点参数
            with self.subTest(endpoint=endpoint):
                # 断言解析会议点端点时是否引发指定的 ValueError 异常，检查端口号是否在有效范围内
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The port number of the rendezvous endpoint '{endpoint}' must be an integer "
                    r"between 0 and 65536.$",
                ):
                    # 调用解析函数，期望引发异常
                    parse_rendezvous_endpoint(endpoint, default_port=123)

    # 定义测试函数：测试解析会议点端点如果端口超出范围时是否引发错误
    def test_parse_rendezvous_endpoint_raises_error_if_port_is_too_big(self) -> None:
        # 定义多个会议点端点示例，其中端口号超出有效范围
        endpoints = ["dummy.com:65536", "dummy.com:70000"]

        # 遍历每个会议点端点示例
        for endpoint in endpoints:
            # 使用子测试，设置当前测试的端点参数
            with self.subTest(endpoint=endpoint):
                # 断言解析会议点端点时是否引发指定的 ValueError 异常，检查端口号是否在有效范围内
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The port number of the rendezvous endpoint '{endpoint}' must be an integer "
                    r"between 0 and 65536.$",
                ):
                    # 调用解析函数，期望引发异常
                    parse_rendezvous_endpoint(endpoint, default_port=123)

    # 定义测试函数：测试匹配机器主机名函数是否能正确识别回环地址
    def test_matches_machine_hostname_returns_true_if_hostname_is_loopback(
        self,
    ) -> None:
        # 定义多个主机名示例，包括回环地址 IPv4 和 IPv6
        hosts = [
            "localhost",
            "127.0.0.1",
            "::1",
            "0000:0000:0000:0000:0000:0000:0000:0001",
        ]

        # 遍历每个主机名示例
        for host in hosts:
            # 使用子测试，设置当前测试的主机参数
            with self.subTest(host=host):
                # 断言调用匹配机器主机名函数返回 True
                self.assertTrue(_matches_machine_hostname(host))

    # 定义测试函数：测试匹配机器主机名函数是否能正确识别本机主机名
    def test_matches_machine_hostname_returns_true_if_hostname_is_machine_hostname(
        self,
    ) -> None:
        # 获取当前机器的主机名
        host = socket.gethostname()

        # 断言调用匹配机器主机名函数返回 True
        self.assertTrue(_matches_machine_hostname(host))

    # 定义测试函数：测试匹配机器主机名函数是否能正确识别本机完全限定域名（FQDN）
    def test_matches_machine_hostname_returns_true_if_hostname_is_machine_fqdn(
        self,
    ) -> None:
        # 获取当前机器的完全限定域名（FQDN）
        host = socket.getfqdn()

        # 断言调用匹配机器主机名函数返回 True
        self.assertTrue(_matches_machine_hostname(host))

    # 定义测试函数：测试匹配机器主机名函数是否能正确识别本机地址
    def test_matches_machine_hostname_returns_true_if_hostname_is_machine_address(
        self,
    ) -> None:
        # 获取当前机器的所有地址信息列表
        addr_list = socket.getaddrinfo(
            socket.gethostname(), None, proto=socket.IPPROTO_TCP
        )

        # 遍历每个地址信息，取出地址字段
        for addr in (addr_info[4][0] for addr_info in addr_list):
            # 使用子测试，设置当前测试的地址参数
            with self.subTest(addr=addr):
                # 断言调用匹配机器主机名函数返回 True
                self.assertTrue(_matches_machine_hostname(addr))

    # 定义测试函数：测试匹配机器主机名函数是否能正确识别非本机地址
    def test_matches_machine_hostname_returns_false_if_hostname_does_not_match(
        self,
    ) -> None:
        # 定义多个非本机地址的主机名示例
        hosts = ["dummy", "0.0.0.0", "::2"]

        # 遍历每个主机名示例
        for host in hosts:
            # 使用子测试，设置当前测试的主机参数
            with self.subTest(host=host):
                # 断言调用匹配机器主机名函数返回 False
                self.assertFalse(_matches_machine_hostname(host))

    # 定义测试函数：测试延迟函数是否能正确挂起线程
    def test_delay_suspends_thread(self) -> None:
        # 定义多个延迟时间示例
        for seconds in 0.2, (0.2, 0.4):
            # 使用子测试，设置当前测试的延迟时间参数
            with self.subTest(seconds=seconds):
                # 获取当前时间
                time1 = time.monotonic()

                # 调用延迟函数，暂停当前线程指定的时间
                _delay(seconds)  # type: ignore[arg-type]

                # 获取延迟后的时间
                time2 = time.monotonic()

                # 断言延迟后的时间大于等于设定的延迟时间
                self.assertGreaterEqual(time2 - time1, 0.2)
    @patch(
        "socket.getaddrinfo",
        # 使用 patch 装饰器模拟 socket.getaddrinfo 的行为，返回两组不同的数据
        side_effect=[
            [(None, None, 0, "a_host", ("1.2.3.4", 0))],  # 第一次调用返回的数据，模拟匹配的情况
            [(None, None, 0, "a_different_host", ("1.2.3.4", 0))],  # 第二次调用返回的数据，模拟不匹配的情况
        ],
    )
    # 测试函数：当主机名为 "a_host" 时，验证 _matches_machine_hostname 函数返回 True
    def test_matches_machine_hostname_returns_true_if_ip_address_match_between_hosts(
        self,
        _0,  # 接收 patch 的副作用函数作为参数
    ) -> None:
        self.assertTrue(_matches_machine_hostname("a_host"))  # 断言 _matches_machine_hostname 返回 True

    @patch(
        "socket.getaddrinfo",
        # 使用 patch 装饰器再次模拟 socket.getaddrinfo 的行为，返回两组不同的数据
        side_effect=[
            [(None, None, 0, "a_host", ("1.2.3.4", 0))],  # 第一次调用返回的数据，模拟匹配的情况
            [(None, None, 0, "another_host_with_different_ip", ("1.2.3.5", 0))],  # 第二次调用返回的数据，模拟不匹配的情况
        ],
    )
    # 测试函数：当主机名为 "a_host" 时，验证 _matches_machine_hostname 函数返回 False
    def test_matches_machine_hostname_returns_false_if_ip_address_not_match_between_hosts(
        self,
        _0,  # 接收 patch 的副作用函数作为参数
    ) -> None:
        self.assertFalse(_matches_machine_hostname("a_host"))  # 断言 _matches_machine_hostname 返回 False
class PeriodicTimerTest(TestCase):
    # 测试只能调用一次 start 方法
    def test_start_can_be_called_only_once(self) -> None:
        # 创建一个周期定时器，间隔1秒执行一次空操作
        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        # 启动定时器
        timer.start()

        # 使用断言检查重复调用 start 方法时是否会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, r"^The timer has already started.$"):
            timer.start()

        # 取消定时器
        timer.cancel()

    # 测试可以多次调用 cancel 方法
    def test_cancel_can_be_called_multiple_times(self) -> None:
        # 创建一个周期定时器，间隔1秒执行一次空操作
        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        # 启动定时器
        timer.start()

        # 多次调用 cancel 方法
        timer.cancel()
        timer.cancel()

    # 测试 cancel 方法能够停止后台线程
    def test_cancel_stops_background_thread(self) -> None:
        # 线程名称
        name = "PeriodicTimer_CancelStopsBackgroundThreadTest"

        # 创建一个周期定时器，间隔1秒执行一次空操作
        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        # 设置定时器的名称
        timer.set_name(name)

        # 启动定时器
        timer.start()

        # 使用断言检查后台线程中是否存在指定名称的线程
        self.assertTrue(any(t.name == name for t in threading.enumerate()))

        # 取消定时器
        timer.cancel()

        # 使用断言检查后台线程中是否不存在指定名称的线程
        self.assertTrue(all(t.name != name for t in threading.enumerate()))

    # 测试删除定时器对象能够停止后台线程
    def test_delete_stops_background_thread(self) -> None:
        # 线程名称
        name = "PeriodicTimer_DeleteStopsBackgroundThreadTest"

        # 创建一个周期定时器，间隔1秒执行一次空操作
        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        # 设置定时器的名称
        timer.set_name(name)

        # 启动定时器
        timer.start()

        # 使用断言检查后台线程中是否存在指定名称的线程
        self.assertTrue(any(t.name == name for t in threading.enumerate()))

        # 删除定时器对象
        del timer

        # 使用断言检查后台线程中是否不存在指定名称的线程
        self.assertTrue(all(t.name != name for t in threading.enumerate()))

    # 测试启动后不能再调用 set_name 方法
    def test_set_name_cannot_be_called_after_start(self) -> None:
        # 创建一个周期定时器，间隔1秒执行一次空操作
        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        # 启动定时器
        timer.start()

        # 使用断言检查在定时器启动后调用 set_name 方法是否会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, r"^The timer has already started.$"):
            timer.set_name("dummy_name")

        # 取消定时器
        timer.cancel()
    # 定义一个测试方法，用于测试定时器是否在规定间隔内调用后台线程
    def test_timer_calls_background_thread_at_regular_intervals(self) -> None:
        # 记录定时器启动时的时间戳
        timer_begin_time: float

        # 设置函数每200毫秒被调用一次
        call_interval = 0.2

        # 记录每次调用之间的实际时间间隔
        actual_call_intervals: List[float] = []

        # 记录函数被调用的次数
        call_count = 0

        # 为了避免测试不稳定性，我们设置一个最小的必要调用次数下限，确保正确实现
        min_required_call_count = 4

        # 创建一个线程事件，用于停止定时器
        timer_stop_event = threading.Event()

        # 定义一个函数用于记录每次函数调用的时间间隔
        def log_call(self):
            nonlocal timer_begin_time, call_count

            # 计算当前调用与上次调用的时间间隔，并添加到列表中
            actual_call_intervals.append(time.monotonic() - timer_begin_time)

            # 增加调用次数，并检查是否达到最小要求的调用次数
            call_count += 1
            if call_count == min_required_call_count:
                timer_stop_event.set()

            # 更新定时器开始时间
            timer_begin_time = time.monotonic()

        # 创建一个周期性定时器对象，每隔指定时间调用一次 log_call 函数
        timer = _PeriodicTimer(timedelta(seconds=call_interval), log_call, self)

        # 记录定时器开始的时间戳
        timer_begin_time = time.monotonic()

        # 启动定时器
        timer.start()

        # 等待定时器运行或者超过60秒后停止
        timer_stop_event.wait(60)

        # 取消定时器
        timer.cancel()

        # 设置测试断言时不显示详细消息
        self.longMessage = False

        # 断言函数调用的实际次数不少于最小要求的调用次数
        self.assertGreaterEqual(
            call_count,
            min_required_call_count,
            f"The function has been called {call_count} time(s) but expected to be called at least "
            f"{min_required_call_count} time(s).",
        )

        # 对每次调用的时间间隔进行断言，确保每次调用间隔不少于指定的时间间隔
        for actual_call_interval in actual_call_intervals:
            self.assertGreaterEqual(
                actual_call_interval,
                call_interval,
                f"The interval between two function calls was {actual_call_interval} second(s) but "
                f"expected to be at least {call_interval} second(s).",
            )
```