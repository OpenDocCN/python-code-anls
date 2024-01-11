# `ZeroNet\src\Test\TestUpnpPunch.py`

```
# 导入所需的模块
import socket
from urllib.parse import urlparse
# 导入 pytest 和 mock 模块
import pytest
import mock
# 导入自定义的 util 模块中的 UpnpPunch 类
from util import UpnpPunch as upnp

# 创建一个模拟的 socket 对象
@pytest.fixture
def mock_socket():
    mock_socket = mock.MagicMock()
    mock_socket.recv = mock.MagicMock(return_value=b'Hello')
    mock_socket.bind = mock.MagicMock()
    mock_socket.send_to = mock.MagicMock()

    return mock_socket

# 创建一个模拟的 URL 对象
@pytest.fixture
def url_obj():
    return urlparse('http://192.168.1.1/ctrlPoint.xml')

# 创建一个模拟的 UPnP 设备描述文件
@pytest.fixture(params=['WANPPPConnection', 'WANIPConnection'])
def igd_profile(request):
    return """<root><serviceList><service>
  <serviceType>urn:schemas-upnp-org:service:{}:1</serviceType>
  <serviceId>urn:upnp-org:serviceId:wanpppc:pppoa</serviceId>
  <controlURL>/upnp/control/wanpppcpppoa</controlURL>
  <eventSubURL>/upnp/event/wanpppcpppoa</eventSubURL>
  <SCPDURL>/WANPPPConnection.xml</SCPDURL>
</service></serviceList></root>""".format(request.param)

# 创建一个模拟的 HTTP 响应对象
@pytest.fixture
def httplib_response():
    class FakeResponse(object):
        def __init__(self, status=200, body='OK'):
            self.status = status
            self.body = body

        def read(self):
            return self.body
    return FakeResponse

# 测试 UpnpPunch 类的 perform_m_search 方法
class TestUpnpPunch(object):
    def test_perform_m_search(self, mock_socket):
        local_ip = '127.0.0.1'

        # 使用 mock.patch 模拟 socket.socket 方法，返回模拟的 socket 对象
        with mock.patch('util.UpnpPunch.socket.socket',
                        return_value=mock_socket):
            # 调用 perform_m_search 方法
            result = upnp.perform_m_search(local_ip)
            # 断言返回结果为 'Hello'
            assert result == 'Hello'
            # 断言模拟 socket 对象的 bind 方法被调用，且参数为 local_ip
            assert local_ip == mock_socket.bind.call_args_list[0][0][0][0]
            # 断言模拟 socket 对象的 sendto 方法被调用，且参数为 ('239.255.255.250', 1900)
            assert ('239.255.255.250',
                    1900) == mock_socket.sendto.call_args_list[0][0][1]
    # 测试在执行 M 搜索时出现套接字错误的情况
    def test_perform_m_search_socket_error(self, mock_socket):
        # 模拟套接字接收数据时出现超时错误
        mock_socket.recv.side_effect = socket.error('Timeout error')

        # 使用模拟的套接字对象进行测试
        with mock.patch('util.UpnpPunch.socket.socket',
                        return_value=mock_socket):
            # 断言执行 M 搜索时会抛出 UpnpError 异常
            with pytest.raises(upnp.UpnpError):
                upnp.perform_m_search('127.0.0.1')

    # 测试从 SSDP 中检索位置信息
    def test_retrieve_location_from_ssdp(self, url_obj):
        # 获取 URL 对象的地址
        ctrl_location = url_obj.geturl()
        # 解析地址
        parsed_location = urlparse(ctrl_location)
        # 构造模拟的响应数据
        rsp = ('auth: gibberish\r\nlocation: {0}\r\n'
               'Content-Type: text/html\r\n\r\n').format(ctrl_location)
        # 调用函数，检索位置信息
        result = upnp._retrieve_location_from_ssdp(rsp)
        # 断言结果与解析的位置信息相等
        assert result == parsed_location

    # 测试从 SSDP 中检索位置信息时没有头部信息的情况
    def test_retrieve_location_from_ssdp_no_header(self):
        # 构造模拟的响应数据
        rsp = 'auth: gibberish\r\nContent-Type: application/json\r\n\r\n'
        # 断言调用函数会抛出 IGDError 异常
        with pytest.raises(upnp.IGDError):
            upnp._retrieve_location_from_ssdp(rsp)

    # 测试检索 IGD 档案
    def test_retrieve_igd_profile(self, url_obj):
        # 使用模拟的 urlopen 函数进行测试
        with mock.patch('urllib.request.urlopen') as mock_urlopen:
            upnp._retrieve_igd_profile(url_obj)
            # 断言调用 urlopen 函数时传入了指定的 URL 和超时时间
            mock_urlopen.assert_called_with(url_obj.geturl(), timeout=5)

    # 测试检索 IGD 档案时出现超时的情况
    def test_retrieve_igd_profile_timeout(self, url_obj):
        # 使用模拟的 urlopen 函数进行测试
        with mock.patch('urllib.request.urlopen') as mock_urlopen:
            # 模拟 urlopen 函数出现套接字超时错误
            mock_urlopen.side_effect = socket.error('Timeout error')
            # 断言调用函数会抛出 IGDError 异常
            with pytest.raises(upnp.IGDError):
                upnp._retrieve_igd_profile(url_obj)

    # 测试解析 IGD 档案的服务类型
    def test_parse_igd_profile_service_type(self, igd_profile):
        # 调用函数，解析 IGD 档案
        control_path, upnp_schema = upnp._parse_igd_profile(igd_profile)
        # 断言解析出的控制路径和 UPnP 模式符合预期
        assert control_path == '/upnp/control/wanpppcpppoa'
        assert upnp_schema in ('WANPPPConnection', 'WANIPConnection',)

    # 测试解析 IGD 档案时没有控制 URL 的情况
    def test_parse_igd_profile_no_ctrlurl(self, igd_profile):
        # 替换控制 URL 为无效值
        igd_profile = igd_profile.replace('controlURL', 'nope')
        # 断言调用函数会抛出 IGDError 异常
        with pytest.raises(upnp.IGDError):
            control_path, upnp_schema = upnp._parse_igd_profile(igd_profile)
    # 测试解析没有模式的 igd_profile
    def test_parse_igd_profile_no_schema(self, igd_profile):
        # 替换 igd_profile 中的 'Connection' 为 'nope'
        igd_profile = igd_profile.replace('Connection', 'nope')
        # 断言抛出 upnp.IGDError 异常
        with pytest.raises(upnp.IGDError):
            # 调用 upnp._parse_igd_profile 方法
            control_path, upnp_schema = upnp._parse_igd_profile(igd_profile)
    
    # 测试创建可解析的 open 消息
    def test_create_open_message_parsable(self):
        # 导入 ExpatError 异常
        from xml.parsers.expat import ExpatError
        # 调用 upnp._create_open_message 方法
        msg, _ = upnp._create_open_message('127.0.0.1', 8888)
        try:
            # 解析 XML 消息
            upnp.parseString(msg)
        except ExpatError as e:
            # 断言失败，输出错误信息
            pytest.fail('Incorrect XML message: {}'.format(e))
    
    # 测试创建包含正确内容的 open 消息
    def test_create_open_message_contains_right_stuff(self):
        # 设置参数字典
        settings = {'description': 'test desc',
                    'protocol': 'test proto',
                    'upnp_schema': 'test schema'}
        # 调用 upnp._create_open_message 方法
        msg, fn_name = upnp._create_open_message('127.0.0.1', 8888, **settings)
        # 断言 fn_name 等于 'AddPortMapping'
        assert fn_name == 'AddPortMapping'
        # 断言 '127.0.0.1' 在 msg 中
        assert '127.0.0.1' in msg
        # 断言 '8888' 在 msg 中
        assert '8888' in msg
        # 断言 settings['description'] 在 msg 中
        assert settings['description'] in msg
        # 断言 settings['protocol'] 在 msg 中
        assert settings['protocol'] in msg
        # 断言 settings['upnp_schema'] 在 msg 中
        assert settings['upnp_schema'] in msg
    
    # 测试解析错误的响应
    def test_parse_for_errors_bad_rsp(self, httplib_response):
        # 创建状态码为 500 的响应
        rsp = httplib_response(status=500)
        with pytest.raises(upnp.IGDError) as err:
            # 调用 upnp._parse_for_errors 方法
            upnp._parse_for_errors(rsp)
        # 断言异常信息中包含 'Unable to parse'
        assert 'Unable to parse' in str(err.value)
    
    # 测试解析错误的 SOAP 响应
    def test_parse_for_errors_error(self, httplib_response):
        # 创建包含错误信息的 SOAP 响应
        soap_error = ('<document>'
                      '<errorCode>500</errorCode>'
                      '<errorDescription>Bad request</errorDescription>'
                      '</document>')
        rsp = httplib_response(status=500, body=soap_error)
        with pytest.raises(upnp.IGDError) as err:
            # 调用 upnp._parse_for_errors 方法
            upnp._parse_for_errors(rsp)
        # 断言异常信息中包含 'SOAP request error'
        assert 'SOAP request error' in str(err.value)
    
    # 测试解析正确的响应
    def test_parse_for_errors_good_rsp(self, httplib_response):
        # 创建状态码为 200 的响应
        rsp = httplib_response(status=200)
        # 断言 rsp 等于 upnp._parse_for_errors(rsp) 的返回值
        assert rsp == upnp._parse_for_errors(rsp)
    # 测试发送请求成功的情况
    def test_send_requests_success(self):
        # 使用 mock.patch 临时替换 _send_soap_request 方法，并创建 mock_send_request 对象
        with mock.patch(
                'util.UpnpPunch._send_soap_request') as mock_send_request:
            # 设置 mock_send_request 的返回值为 status=200 的 MagicMock 对象
            mock_send_request.return_value = mock.MagicMock(status=200)
            # 调用 _send_requests 方法
            upnp._send_requests(['msg'], None, None, None)

        # 断言 mock_send_request 被调用过
        assert mock_send_request.called

    # 测试发送请求失败的情况
    def test_send_requests_failed(self):
        # 使用 mock.patch 临时替换 _send_soap_request 方法，并创建 mock_send_request 对象
        with mock.patch(
                'util.UpnpPunch._send_soap_request') as mock_send_request:
            # 设置 mock_send_request 的返回值为 status=500 的 MagicMock 对象
            mock_send_request.return_value = mock.MagicMock(status=500)
            # 使用 pytest.raises 断言 upnp.UpnpError 异常被抛出
            with pytest.raises(upnp.UpnpError):
                # 调用 _send_requests 方法
                upnp._send_requests(['msg'], None, None, None)

        # 断言 mock_send_request 被调用过
        assert mock_send_request.called

    # 测试收集 IDG 数据的情况
    def test_collect_idg_data(self):
        # 空方法，无需注释

    # 测试请求打开端口成功的情况
    @mock.patch('util.UpnpPunch._get_local_ips')
    @mock.patch('util.UpnpPunch._collect_idg_data')
    @mock.patch('util.UpnpPunch._send_requests')
    def test_ask_to_open_port_success(self, mock_send_requests,
                                      mock_collect_idg, mock_local_ips):
        # 设置 mock_collect_idg 的返回值为 {'upnp_schema': 'schema-yo'}
        mock_collect_idg.return_value = {'upnp_schema': 'schema-yo'}
        # 设置 mock_local_ips 的返回值为 ['192.168.0.12']
        mock_local_ips.return_value = ['192.168.0.12']

        # 调用 ask_to_open_port 方法
        result = upnp.ask_to_open_port(retries=5)

        # 获取调用 mock_send_requests 时传入的参数
        soap_msg = mock_send_requests.call_args[0][0][0][0]

        # 断言结果为 True
        assert result is True
        # 断言 mock_collect_idg 被调用过
        assert mock_collect_idg.called
        # 断言 soap_msg 中包含 '192.168.0.12'
        assert '192.168.0.12' in soap_msg
        # 断言 soap_msg 中包含 '15441'
        assert '15441' in soap_msg
        # 断言 soap_msg 中包含 'schema-yo'
        assert 'schema-yo' in soap_msg

    # 测试请求打开端口失败的情况
    @mock.patch('util.UpnpPunch._get_local_ips')
    @mock.patch('util.UpnpPunch._collect_idg_data')
    @mock.patch('util.UpnpPunch._send_requests')
    def test_ask_to_open_port_failure(self, mock_send_requests,
                                      mock_collect_idg, mock_local_ips):
        # 设置 mock_local_ips 的返回值为 ['192.168.0.12']
        mock_local_ips.return_value = ['192.168.0.12']
        # 设置 mock_collect_idg 的返回值为 {'upnp_schema': 'schema-yo'}
        mock_collect_idg.return_value = {'upnp_schema': 'schema-yo'}
        # 设置 mock_send_requests 抛出 upnp.UpnpError 异常
        mock_send_requests.side_effect = upnp.UpnpError()

        # 使用 pytest.raises 断言 upnp.UpnpError 异常被抛出
        with pytest.raises(upnp.UpnpError):
            # 调用 ask_to_open_port 方法
            upnp.ask_to_open_port()
    # 使用 mock.patch 装饰器模拟 _collect_idg_data 方法
    # 使用 mock.patch 装饰器模拟 _send_requests 方法
    def test_orchestrate_soap_request(self, mock_send_requests,
                                      mock_collect_idg):
        # 创建一个魔术方法的模拟对象
        soap_mock = mock.MagicMock()
        # 定义参数列表
        args = ['127.0.0.1', 31337, soap_mock, 'upnp-test', {'upnp_schema': 'schema-yo'}]
        # 设置 mock_collect_idg 方法的返回值
        mock_collect_idg.return_value = args[-1]
    
        # 调用 _orchestrate_soap_request 方法
        upnp._orchestrate_soap_request(*args[:-1])
    
        # 断言 mock_collect_idg 方法被调用
        assert mock_collect_idg.called
        # 断言 soap_mock 方法被调用
        soap_mock.assert_called_with(*args[:2] + ['upnp-test', 'UDP', 'schema-yo'])
        # 断言 mock_send_requests 方法被调用
    
        assert mock_send_requests.called
    
    # 使用 mock.patch 装饰器模拟 _collect_idg_data 方法
    # 使用 mock.patch 装饰器模拟 _send_requests 方法
    def test_orchestrate_soap_request_without_desc(self, mock_send_requests,
                                                   mock_collect_idg):
        # 创建一个魔术方法的模拟对象
        soap_mock = mock.MagicMock()
        # 定义参数列表
        args = ['127.0.0.1', 31337, soap_mock, {'upnp_schema': 'schema-yo'}]
        # 设置 mock_collect_idg 方法的返回值
        mock_collect_idg.return_value = args[-1]
    
        # 调用 _orchestrate_soap_request 方法
        upnp._orchestrate_soap_request(*args[:-1])
    
        # 断言 mock_collect_idg 方法被调用
        assert mock_collect_idg.called
        # 断言 soap_mock 方法被调用
        soap_mock.assert_called_with(*args[:2] + [None, 'UDP', 'schema-yo'])
        # 断言 mock_send_requests 方法被调用
    
        assert mock_send_requests.called
    
    # 测试 create_close_message_parsable 方法是否可解析
    def test_create_close_message_parsable(self):
        # 导入 ExpatError 异常
        from xml.parsers.expat import ExpatError
        # 调用 _create_close_message 方法，获取消息和 _
        msg, _ = upnp._create_close_message('127.0.0.1', 8888)
        try:
            # 尝试解析消息
            upnp.parseString(msg)
        except ExpatError as e:
            # 如果解析失败，则抛出异常
            pytest.fail('Incorrect XML message: {}'.format(e))
    # 测试创建和关闭消息是否包含正确的内容
    def test_create_close_message_contains_right_stuff(self):
        # 设置参数字典
        settings = {'protocol': 'test proto',
                    'upnp_schema': 'test schema'}
        # 调用 _create_close_message 方法，获取消息和函数名
        msg, fn_name = upnp._create_close_message('127.0.0.1', 8888, **settings)
        # 断言函数名为 'DeletePortMapping'
        assert fn_name == 'DeletePortMapping'
        # 断言消息中包含 '8888'
        assert '8888' in msg
        # 断言消息中包含设置的协议
        assert settings['protocol'] in msg
        # 断言消息中包含设置的 UPnP schema
        assert settings['upnp_schema'] in msg
    
    # 测试与 IGD 通信是否成功
    @mock.patch('util.UpnpPunch._get_local_ips')
    @mock.patch('util.UpnpPunch._orchestrate_soap_request')
    def test_communicate_with_igd_success(self, mock_orchestrate, mock_get_local_ips):
        # 模拟获取本地 IP 地址
        mock_get_local_ips.return_value = ['192.168.0.12']
        # 调用 _communicate_with_igd 方法
        upnp._communicate_with_igd()
        # 断言获取本地 IP 地址的方法被调用
        assert mock_get_local_ips.called
        # 断言 SOAP 请求方法被调用
        assert mock_orchestrate.called
    
    # 测试即使出现单个失败也能成功与 IGD 通信
    @mock.patch('util.UpnpPunch._get_local_ips')
    @mock.patch('util.UpnpPunch._orchestrate_soap_request')
    def test_communicate_with_igd_succeed_despite_single_failure(self, mock_orchestrate, mock_get_local_ips):
        # 模拟获取本地 IP 地址
        mock_get_local_ips.return_value = ['192.168.0.12']
        # 模拟 SOAP 请求方法出现一次 UpnpError 异常，一次正常调用
        mock_orchestrate.side_effect = [upnp.UpnpError, None]
        # 调用 _communicate_with_igd 方法，设置重试次数为2
        upnp._communicate_with_igd(retries=2)
        # 断言获取本地 IP 地址的方法被调用
        assert mock_get_local_ips.called
        # 断言 SOAP 请求方法被调用
        assert mock_orchestrate.called
    
    # 测试与 IGD 完全失败的情况
    @mock.patch('util.UpnpPunch._get_local_ips')
    @mock.patch('util.UpnpPunch._orchestrate_soap_request')
    def test_communicate_with_igd_total_failure(self, mock_orchestrate, mock_get_local_ips):
        # 模拟获取本地 IP 地址
        mock_get_local_ips.return_value = ['192.168.0.12']
        # 模拟 SOAP 请求方法连续出现 UpnpError 和 IGDError 异常
        mock_orchestrate.side_effect = [upnp.UpnpError, upnp.IGDError]
        # 使用 pytest 断言捕获 UpnpError 异常
        with pytest.raises(upnp.UpnpError):
            # 调用 _communicate_with_igd 方法，设置重试次数为2
            upnp._communicate_with_igd(retries=2)
        # 断言获取本地 IP 地址的方法被调用
        assert mock_get_local_ips.called
        # 断言 SOAP 请求方法被调用
        assert mock_orchestrate.called
```