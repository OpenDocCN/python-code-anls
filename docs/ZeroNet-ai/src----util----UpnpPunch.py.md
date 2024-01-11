# `ZeroNet\src\util\UpnpPunch.py`

```
import re  # 导入正则表达式模块
import urllib.request  # 导入用于发送 HTTP 请求的模块
import http.client  # 导入用于处理 HTTP 连接的模块
import logging  # 导入日志记录模块
from urllib.parse import urlparse  # 从 urllib.parse 模块中导入 urlparse 函数
from xml.dom.minidom import parseString  # 从 xml.dom.minidom 模块中导入 parseString 函数
from xml.parsers.expat import ExpatError  # 从 xml.parsers.expat 模块中导入 ExpatError 异常类
from gevent import socket  # 从 gevent 模块中导入 socket 函数
import gevent  # 导入协程模块

# Relevant UPnP spec:
# http://www.upnp.org/specs/gw/UPnP-gw-WANIPConnection-v1-Service.pdf
# 相关的 UPnP 规范链接

# General TODOs:
# Handle 0 or >1 IGDs
# 一般的待办事项：处理 0 个或多个 IGDs

logger = logging.getLogger("Upnp")  # 获取名为 "Upnp" 的日志记录器对象

class UpnpError(Exception):  # 定义 UpnpError 异常类，继承自 Exception 类
    pass

class IGDError(UpnpError):  # 定义 IGDError 异常类，继承自 UpnpError 类
    """
    Signifies a problem with the IGD.
    表示与 IGD 有问题。
    """
    pass

REMOVE_WHITESPACE = re.compile(r'>\s*<')  # 编译正则表达式，用于匹配空白字符

def perform_m_search(local_ip):
    """
    Broadcast a UDP SSDP M-SEARCH packet and return response.
    广播 UDP SSDP M-SEARCH 数据包并返回响应。
    """
    search_target = "urn:schemas-upnp-org:device:InternetGatewayDevice:1"  # 设置搜索目标

    ssdp_request = ''.join(
        ['M-SEARCH * HTTP/1.1\r\n',
         'HOST: 239.255.255.250:1900\r\n',
         'MAN: "ssdp:discover"\r\n',
         'MX: 2\r\n',
         'ST: {0}\r\n'.format(search_target),
         '\r\n']
    ).encode("utf8")  # 构造 SSDP 请求数据包并编码为 utf8 格式

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建 UDP 套接字

    sock.bind((local_ip, 0))  # 绑定本地 IP 地址和端口号

    sock.sendto(ssdp_request, ('239.255.255.250', 1900))  # 发送 SSDP 请求数据包到指定地址和端口
    if local_ip == "127.0.0.1":  # 如果本地 IP 地址为 "127.0.0.1"
        sock.settimeout(1)  # 设置超时时间为 1 秒
    else:
        sock.settimeout(5)  # 设置超时时间为 5 秒

    try:
        return sock.recv(2048).decode("utf8")  # 接收响应数据并解码为 utf8 格式
    except socket.error:  # 捕获 socket 错误
        raise UpnpError("No reply from IGD using {} as IP".format(local_ip))  # 抛出 UpnpError 异常
    finally:
        sock.close()  # 关闭套接字

def _retrieve_location_from_ssdp(response):
    """
    Parse raw HTTP response to retrieve the UPnP location header
    and return a ParseResult object.
    解析原始的 HTTP 响应以检索 UPnP 位置头并返回 ParseResult 对象。
    """
    parsed_headers = re.findall(r'(?P<name>.*?): (?P<value>.*?)\r\n', response)  # 使用正则表达式解析 HTTP 响应头
    header_locations = [header[1]
                        for header in parsed_headers
                        if header[0].lower() == 'location']  # 获取所有 location 头的值

    if len(header_locations) < 1:  # 如果 location 头的数量小于 1
        raise IGDError('IGD response does not contain a "location" header.')  # 抛出 IGDError 异常

    return urlparse(header_locations[0])  # 解析第一个 location 头的值并返回解析结果对象
# 从指定 URL 中检索设备的 UPnP 配置文件
def _retrieve_igd_profile(url):
    try:
        # 使用 urllib 请求指定 URL，并设置超时时间为 5 秒，读取返回的内容并以 utf-8 解码
        return urllib.request.urlopen(url.geturl(), timeout=5).read().decode('utf-8')
    except socket.error:
        # 如果发生 socket 错误，抛出 IGDError 异常，提示 IGD 配置文件查询超时
        raise IGDError('IGD profile query timed out')


# 获取节点的第一个子文本节点的文本值
def _get_first_child_data(node):
    return node.childNodes[0].data


# 解析 IGD 配置文件的 XML，查找 WANIPConnection 或 WANPPPConnection，并返回控制 URL 和服务 XML 模式
def _parse_igd_profile(profile_xml):
    try:
        # 解析 XML 字符串为 DOM 对象
        dom = parseString(profile_xml)
    except ExpatError as e:
        # 如果解析出现错误，抛出 IGDError 异常，提示无法解析 IGD 回复
        raise IGDError(
            'Unable to parse IGD reply: {0} \n\n\n {1}'.format(profile_xml, e))

    # 获取所有 serviceType 元素
    service_types = dom.getElementsByTagName('serviceType')
    for service in service_types:
        # 如果 serviceType 元素的文本值包含 'WANIPConnection' 或 'WANPPPConnection'
        if _get_first_child_data(service).find('WANIPConnection') > 0 or \
           _get_first_child_data(service).find('WANPPPConnection') > 0:
            try:
                # 获取控制 URL 和 UPnP 模式，并返回
                control_url = _get_first_child_data(
                    service.parentNode.getElementsByTagName('controlURL')[0])
                upnp_schema = _get_first_child_data(service).split(':')[-2]
                return control_url, upnp_schema
            except IndexError:
                # 如果发生索引错误，忽略并继续循环
                pass
    # 如果未找到控制 URL 或 UPNP 模式，抛出 IGDError 异常
    raise IGDError(
        'Could not find a control url or UPNP schema in IGD response.')


# 获取本地 IP 地址
def _get_local_ips():
    # 定义方法1：通过UDP和广播地址获取本地IP
    def method1():
        try:
            # 创建一个UDP套接字
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 设置套接字选项，允许广播
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            # 连接到指定的广播地址
            s.connect(('239.255.255.250', 1))
            # 返回本地IP地址列表
            return [s.getsockname()[0]]
        except:
            pass

    # 定义方法2：通过UDP和普通地址（谷歌DNS IP）获取本地IP
    def method2():
        try:
            # 创建一个UDP套接字
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 连接到指定的普通地址（谷歌DNS IP）
            s.connect(('8.8.8.8', 0))
            # 返回本地IP地址列表
            return [s.getsockname()[0]]
        except:
            pass

    # 定义方法3：通过''主机名获取本地IP（不是所有平台都支持）
    def method3():
        try:
            # 返回''主机名对应的IP地址列表
            return socket.gethostbyname_ex('')[2]
        except:
            pass

    # 创建线程列表，分别执行三种获取本地IP的方法
    threads = [
        gevent.spawn(method1),
        gevent.spawn(method2),
        gevent.spawn(method3)
    ]

    # 等待所有线程执行完成，超时时间为5秒
    gevent.joinall(threads, timeout=5)

    # 存储本地IP的列表
    local_ips = []
    # 遍历线程列表，将非空的本地IP列表合并到local_ips中
    for thread in threads:
        if thread.value:
            local_ips += thread.value

    # 删除重复的IP地址
    local_ips = list(set(local_ips))

    # 按照IP地址是否以"192"开头进行排序，降序排列
    local_ips = sorted(local_ips, key=lambda a: a.startswith("192"), reverse=True)

    # 返回本地IP地址列表
    return local_ips
def _create_open_message(local_ip,
                         port,
                         description="UPnPPunch",
                         protocol="TCP",
                         upnp_schema='WANIPConnection'):
    """
    Build a SOAP AddPortMapping message.
    """

    # 构建一个 SOAP AddPortMapping 消息
    soap_message = """<?xml version="1.0"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
    <s:Body>
        <u:AddPortMapping xmlns:u="urn:schemas-upnp-org:service:{upnp_schema}:1">
            <NewRemoteHost></NewRemoteHost>
            <NewExternalPort>{port}</NewExternalPort>
            <NewProtocol>{protocol}</NewProtocol>
            <NewInternalPort>{port}</NewInternalPort>
            <NewInternalClient>{host_ip}</NewInternalClient>
            <NewEnabled>1</NewEnabled>
            <NewPortMappingDescription>{description}</NewPortMappingDescription>
            <NewLeaseDuration>0</NewLeaseDuration>
        </u:AddPortMapping>
    </s:Body>
</s:Envelope>""".format(port=port,
                        protocol=protocol,
                        host_ip=local_ip,
                        description=description,
                        upnp_schema=upnp_schema)
    return (REMOVE_WHITESPACE.sub('><', soap_message), 'AddPortMapping')


def _create_close_message(local_ip,
                          port,
                          description=None,
                          protocol='TCP',
                          upnp_schema='WANIPConnection'):
    # 构建一个 SOAP DeletePortMapping 消息
    soap_message = """<?xml version="1.0"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
    <s:Body>
        <u:DeletePortMapping xmlns:u="urn:schemas-upnp-org:service:{upnp_schema}:1">
            <NewRemoteHost></NewRemoteHost>
            <NewExternalPort>{port}</NewExternalPort>
            <NewProtocol>{protocol}</NewProtocol>
        </u:DeletePortMapping>
    </s:Body>
</s:Envelope>""".format(port=port,
                        protocol=protocol,
                        upnp_schema=upnp_schema)
    # 返回格式化后的 SOAP 消息和操作名称
    return (REMOVE_WHITESPACE.sub('><', soap_message), 'DeletePortMapping')


def _parse_for_errors(soap_response):
    # 记录 SOAP 响应状态
    logger.debug(soap_response.status)
    # 如果响应状态码大于等于 400，则处理错误
    if soap_response.status >= 400:
        # 读取响应数据
        response_data = soap_response.read()
        logger.debug(response_data)
        try:
            # 解析响应数据中的错误信息
            err_dom = parseString(response_data)
            err_code = _get_first_child_data(err_dom.getElementsByTagName(
                'errorCode')[0])
            err_msg = _get_first_child_data(
                err_dom.getElementsByTagName('errorDescription')[0]
            )
        except Exception as err:
            # 如果解析错误，则抛出 IGDError 异常
            raise IGDError(
                'Unable to parse SOAP error: {0}. Got: "{1}"'.format(
                    err, response_data))
        # 抛出包含错误码和错误消息的 IGDError 异常
        raise IGDError(
            'SOAP request error: {0} - {1}'.format(err_code, err_msg)
        )
    # 返回 SOAP 响应
    return soap_response


def _send_soap_request(location, upnp_schema, control_path, soap_fn,
                       soap_message):
    """
    Send out SOAP request to UPnP device and return a response.
    """
    # 设置请求头部信息
    headers = {
        'SOAPAction': (
            '"urn:schemas-upnp-org:service:{schema}:'
            '1#{fn_name}"'.format(schema=upnp_schema, fn_name=soap_fn)
        ),
        'Content-Type': 'text/xml'
    }
    # 记录发送 SOAP 请求的目标地址和端口
    logger.debug("Sending UPnP request to {0}:{1}...".format(
        location.hostname, location.port))
    # 创建 HTTP 连接
    conn = http.client.HTTPConnection(location.hostname, location.port)
    # 发送 POST 请求，并获取响应
    conn.request('POST', control_path, soap_message, headers)

    response = conn.getresponse()
    conn.close()

    # 解析响应，处理可能的错误
    return _parse_for_errors(response)


def _collect_idg_data(ip_addr):
    idg_data = {}
    # 执行 M-SEARCH 操作，获取 IDG 响应
    idg_response = perform_m_search(ip_addr)
    # 从 SSDP 响应中提取设备位置信息
    idg_data['location'] = _retrieve_location_from_ssdp(idg_response)
    # 调用_retrieve_igd_profile函数获取IGD配置文件的位置，并解析IGD配置文件，将结果分别赋值给idg_data字典的'control_path'和'upnp_schema'键
    idg_data['control_path'], idg_data['upnp_schema'] = _parse_igd_profile(
        _retrieve_igd_profile(idg_data['location']))
    # 返回idg_data字典
    return idg_data
# 发送多个 SOAP 请求，并返回响应
def _send_requests(messages, location, upnp_schema, control_path):
    # 使用列表推导式发送多个 SOAP 请求，并获取响应
    responses = [_send_soap_request(location, upnp_schema, control_path, message_tup[1], message_tup[0])
                 for message_tup in messages]

    # 如果所有响应的状态码都为 200，则返回
    if all(rsp.status == 200 for rsp in responses):
        return
    # 否则抛出 UPnP 错误
    raise UpnpError('Sending requests using UPnP failed.')


# 发起 SOAP 请求的协调函数
def _orchestrate_soap_request(ip, port, msg_fn, desc=None, protos=("TCP", "UDP")):
    # 记录调试信息
    logger.debug("Trying using local ip: %s" % ip)
    # 收集 IDG 数据
    idg_data = _collect_idg_data(ip)

    # 生成 SOAP 消息列表
    soap_messages = [
        msg_fn(ip, port, desc, proto, idg_data['upnp_schema'])
        for proto in protos
    ]

    # 发送 SOAP 请求
    _send_requests(soap_messages, **idg_data)


# 与 IGD 通信的函数
def _communicate_with_igd(port=15441, desc="UpnpPunch", retries=3, fn=_create_open_message, protos=("TCP", "UDP")):
    """
    Manage sending a message generated by 'fn'.
    """
    # 获取本地 IP 地址
    local_ips = _get_local_ips()
    success = False

    # 定义任务函数
    def job(local_ip):
        # 尝试多次发送 SOAP 请求
        for retry in range(retries):
            try:
                _orchestrate_soap_request(local_ip, port, fn, desc, protos)
                return True
            except Exception as e:
                # 记录调试信息
                logger.debug('Upnp request using "{0}" failed: {1}'.format(local_ip, e))
                # 休眠 1 秒
                gevent.sleep(1)
        return False

    threads = []

    # 遍历本地 IP 地址，创建并启动协程
    for local_ip in local_ips:
        job_thread = gevent.spawn(job, local_ip)
        threads.append(job_thread)
        gevent.sleep(0.1)
        # 如果有任何一个协程返回 True，则设置 success 为 True，并跳出循环
        if any([thread.value for thread in threads]):
            success = True
            break

    # 再等待 10 秒，检查是否所有协程都执行完毕或有任何一个成功
    for _ in range(10):
        all_done = all([thread.value is not None for thread in threads])
        any_succeed = any([thread.value for thread in threads])
        if all_done or any_succeed:
            break
        gevent.sleep(1)
    # 如果任何一个线程的值为真，则将 success 设置为 True
    if any([thread.value for thread in threads]):
        success = True

    # 如果 success 为假，则抛出 UpnpError 异常，包含端口和重试次数信息
    if not success:
        raise UpnpError(
            'Failed to communicate with igd using port {0} on local machine after {1} tries.'.format(
                port, retries))

    # 返回 success 的值
    return success
# 定义函数，用于请求打开指定端口
def ask_to_open_port(port=15441, desc="UpnpPunch", retries=3, protos=("TCP", "UDP")):
    # 打印调试信息，尝试打开指定端口
    logger.debug("Trying to open port %d." % port)
    # 调用内部函数，与IGD通信，请求打开端口
    return _communicate_with_igd(port=port,
                          desc=desc,
                          retries=retries,
                          fn=_create_open_message,
                          protos=protos)

# 定义函数，用于请求关闭指定端口
def ask_to_close_port(port=15441, desc="UpnpPunch", retries=3, protos=("TCP", "UDP")):
    # 打印调试信息，尝试关闭指定端口
    logger.debug("Trying to close port %d." % port)
    # retries=1 because multiple successes cause 500 response and failure
    # 调用内部函数，与IGD通信，请求关闭端口
    return _communicate_with_igd(port=port,
                          desc=desc,
                          retries=retries,
                          fn=_create_close_message,
                          protos=protos)

# 主程序入口
if __name__ == "__main__":
    # 导入monkey模块，用于协程支持
    from gevent import monkey
    monkey.patch_all()
    # 配置日志级别为DEBUG
    logging.basicConfig(level=logging.DEBUG)
    # 导入时间模块
    import time

    # 记录当前时间
    s = time.time()
    # 打印信息，正在打开端口
    print("Opening port...")
    # 打印信息，请求打开指定端口，并打印是否成功
    print("Success:", ask_to_open_port(15443, "ZeroNet", protos=["TCP"]))
    # 打印信息，计算并打印操作耗时
    print("Done in", time.time() - s)

    # 打印信息，正在关闭端口
    print("Closing port...")
    # 打印信息，请求关闭指定端口，并打印是否成功
    print("Success:", ask_to_close_port(15443, "ZeroNet", protos=["TCP"]))
    # 打印信息，计算并打印操作耗时
    print("Done in", time.time() - s)
```