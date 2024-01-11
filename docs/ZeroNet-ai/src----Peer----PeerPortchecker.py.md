# `ZeroNet\src\Peer\PeerPortchecker.py`

```
# 导入 logging 模块
import logging
# 导入 urllib.request 模块
import urllib.request
# 导入 urllib.parse 模块
import urllib.parse
# 导入 re 模块
import re
# 导入 time 模块
import time
# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 从 util 模块中导入 UpnpPunch 类
from util import UpnpPunch

# 定义 PeerPortchecker 类
class PeerPortchecker(object):
    # 定义 checker_functions 字典
    checker_functions = {
        "ipv4": ["checkIpfingerprints", "checkCanyouseeme"],
        "ipv6": ["checkMyaddr", "checkIpv6scanner"]
    }
    # 定义初始化方法
    def __init__(self, file_server):
        # 获取名为 "PeerPortchecker" 的 logger 对象
        self.log = logging.getLogger("PeerPortchecker")
        # 初始化 upnp_port_opened 属性为 False
        self.upnp_port_opened = False
        # 将 file_server 参数赋值给 file_server 属性
        self.file_server = file_server

    # 定义 requestUrl 方法
    def requestUrl(self, url, post_data=None):
        # 如果 post_data 是字典类型，则将其转换为 URL 编码的字节流
        if type(post_data) is dict:
            post_data = urllib.parse.urlencode(post_data).encode("utf8")
        # 创建一个 urllib.request.Request 对象
        req = urllib.request.Request(url, post_data)
        # 添加 Referer 和 User-Agent 头信息
        req.add_header("Referer", url)
        req.add_header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11")
        req.add_header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        # 发起 URL 请求并设置超时时间为 20 秒
        return urllib.request.urlopen(req, timeout=20.0)

    # 定义 portOpen 方法
    def portOpen(self, port):
        # 记录日志信息
        self.log.info("Trying to open port using UpnpPunch...")
        try:
            # 调用 UpnpPunch 模块的 ask_to_open_port 方法来尝试打开端口
            UpnpPunch.ask_to_open_port(port, 'ZeroNet', retries=3, protos=["TCP"])
            # 设置 upnp_port_opened 属性为 True
            self.upnp_port_opened = True
        except Exception as err:
            # 记录警告日志信息
            self.log.warning("UpnpPunch run error: %s" % Debug.formatException(err))
            return False
        return True

    # 定义 portClose 方法
    def portClose(self, port):
        # 调用 UpnpPunch 模块的 ask_to_close_port 方法来尝试关闭端口
        return UpnpPunch.ask_to_close_port(port, protos=["TCP"])
    # 检查指定端口是否开放，可以指定 IP 类型，默认为 IPv4
    def portCheck(self, port, ip_type="ipv4"):
        # 获取对应 IP 类型的检查函数列表
        checker_functions = self.checker_functions[ip_type]

        # 遍历检查函数列表
        for func_name in checker_functions:
            # 获取对应的检查函数
            func = getattr(self, func_name)
            # 记录当前时间
            s = time.time()
            try:
                # 调用检查函数，获取结果
                res = func(port)
                # 如果结果为真
                if res:
                    # 记录检查结果日志
                    self.log.info(
                        "Checked port %s (%s) using %s result: %s in %.3fs" %
                        (port, ip_type, func_name, res, time.time() - s)
                    )
                    # 等待0.1秒
                    time.sleep(0.1)
                    # 如果端口开放且没有外部连接，则记录警告日志
                    if res["opened"] and not self.file_server.had_external_incoming:
                        res["opened"] = False
                        self.log.warning("Port %s:%s looks opened, but no incoming connection" % (res["ip"], port))
                    # 结束循环
                    break
            # 捕获异常
            except Exception as err:
                # 记录异常日志
                self.log.warning(
                    "%s check error: %s in %.3fs" %
                    (func_name, Debug.formatException(err), time.time() - s)
                )
                # 设置默认结果
                res = {"ip": None, "opened": False}

        # 返回结果
        return res

    # 使用canyouseeme.org网站检查端口是否开放
    def checkCanyouseeme(self, port):
        # 发送请求并获取响应数据
        data = urllib.request.urlopen("https://www.canyouseeme.org/", b"ip=1.1.1.1&port=%s" % str(port).encode("ascii"), timeout=20.0).read().decode("utf8")

        # 从响应数据中提取信息
        message = re.match(r'.*<p style="padding-left:15px">(.*?)</p>', data, re.DOTALL).group(1)
        message = re.sub(r"<.*?>", "", message.replace("<br>", " ").replace("&nbsp;", " "))  # 去除 HTML 标签

        # 从信息中提取 IP 地址
        match = re.match(r".*service on (.*?) on", message)
        if match:
            ip = match.group(1)
        else:
            # 抛出异常
            raise Exception("Invalid response: %s" % message)

        # 根据信息判断端口是否开放，并返回结果
        if "Success" in message:
            return {"ip": ip, "opened": True}
        elif "Error" in message:
            return {"ip": ip, "opened": False}
        else:
            # 抛出异常
            raise Exception("Invalid response: %s" % message)
    # 检查 IP 地址的指纹，通过指定端口进行扫描
    def checkIpfingerprints(self, port):
        # 通过请求 URL 获取数据，并解码成 UTF-8 格式
        data = self.requestUrl("https://www.ipfingerprints.com/portscan.php").read().decode("utf8")
        # 从数据中匹配出远程主机的 IP 地址
        ip = re.match(r'.*name="remoteHost".*?value="(.*?)"', data, re.DOTALL).group(1)

        # 构造 POST 请求的数据
        post_data = {
            "remoteHost": ip, "start_port": port, "end_port": port,
            "normalScan": "Yes", "scan_type": "connect2", "ping_type": "none"
        }
        # 通过请求 URL 发送 POST 请求，并获取返回的数据
        message = self.requestUrl("https://www.ipfingerprints.com/scripts/getPortsInfo.php", post_data).read().decode("utf8")

        # 根据返回的消息判断端口是否开放，并返回相应的结果
        if "open" in message:
            return {"ip": ip, "opened": True}
        elif "filtered" in message or "closed" in message:
            return {"ip": ip, "opened": False}
        else:
            raise Exception("Invalid response: %s" % message)

    # 检查本机 IP 地址的指纹，通过指定端口进行扫描
    def checkMyaddr(self, port):
        # 设置 URL
        url = "http://ipv6.my-addr.com/online-ipv6-port-scan.php"

        # 通过请求 URL 获取数据，并解码成 UTF-8 格式
        data = self.requestUrl(url).read().decode("utf8")

        # 从数据中匹配出本机 IP 地址
        ip = re.match(r'.*Your IP address is:[ ]*([0-9\.:a-z]+)', data.replace("&nbsp;", ""), re.DOTALL).group(1)

        # 构造 POST 请求的数据
        post_data = {"addr": ip, "ports_selected": "", "ports_list": port}
        # 通过请求 URL 发送 POST 请求，并获取返回的数据
        data = self.requestUrl(url, post_data).read().decode("utf8")

        # 从返回的数据中匹配出消息部分
        message = re.match(r".*<table class='table_font_16'>(.*?)</table>", data, re.DOTALL).group(1)

        # 根据返回的消息判断端口是否开放，并返回相应的结果
        if "ok.png" in message:
            return {"ip": ip, "opened": True}
        elif "fail.png" in message:
            return {"ip": ip, "opened": False}
        else:
            raise Exception("Invalid response: %s" % message)
    # 检查 IPv6scanner 端口是否开放
    def checkIpv6scanner(self, port):
        # 定义 IPv6scanner 网站的 URL
        url = "http://www.ipv6scanner.com/cgi-bin/main.py"

        # 发起请求并读取响应数据
        data = self.requestUrl(url).read().decode("utf8")

        # 从响应数据中提取 IP 地址
        ip = re.match(r'.*Your IP address is[ ]*([0-9\.:a-z]+)', data.replace("&nbsp;", ""), re.DOTALL).group(1)

        # 构造 POST 请求数据
        post_data = {"host": ip, "scanType": "1", "port": port, "protocol": "tcp", "authorized": "yes"}
        # 发起带有 POST 数据的请求并读取响应数据
        data = self.requestUrl(url, post_data).read().decode("utf8")

        # 从响应数据中提取扫描结果信息
        message = re.match(r".*<table id='scantable'>(.*?)</table>", data, re.DOTALL).group(1)
        # 处理扫描结果信息，去除 HTML 标签和空格
        message_text = re.sub("<.*?>", " ", message.replace("<br>", " ").replace("&nbsp;", " ").strip())  # Strip http tags

        # 根据扫描结果信息判断端口是否开放，并返回相应的结果
        if "OPEN" in message_text:
            return {"ip": ip, "opened": True}
        elif "CLOSED" in message_text or "FILTERED" in message_text:
            return {"ip": ip, "opened": False}
        else:
            raise Exception("Invalid response: %s" % message_text)

    # 检查 Portchecker 端口是否开放（目前无法使用：Forbidden）
    def checkPortchecker(self, port):
        # 发起请求并读取响应数据
        data = self.requestUrl("https://portchecker.co").read().decode("utf8")
        # 从响应数据中提取 CSRF 令牌
        csrf = re.match(r'.*name="_csrf" value="(.*?)"', data, re.DOTALL).group(1)

        # 发起带有端口和 CSRF 令牌的 POST 请求并读取响应数据
        data = self.requestUrl("https://portchecker.co", {"port": port, "_csrf": csrf}).read().decode("utf8")
        # 从响应数据中提取扫描结果信息
        message = re.match(r'.*<div id="results-wrapper">(.*?)</div>', data, re.DOTALL).group(1)
        # 处理扫描结果信息，去除 HTML 标签和空格
        message = re.sub(r"<.*?>", "", message.replace("<br>", " ").replace("&nbsp;", " ").strip())  # Strip http tags

        # 从响应数据中提取目标 IP 地址
        match = re.match(r".*targetIP.*?value=\"(.*?)\"", data, re.DOTALL)
        if match:
            ip = match.group(1)
        else:
            raise Exception("Invalid response: %s" % message)

        # 根据扫描结果信息判断端口是否开放，并返回相应的结果
        if "open" in message:
            return {"ip": ip, "opened": True}
        elif "closed" in message:
            return {"ip": ip, "opened": False}
        else:
            raise Exception("Invalid response: %s" % message)
    # 检查子网是否在线，传入端口号作为参数
    def checkSubnetonline(self, port):  # Not working: Invalid response
        # 设置要访问的 URL
        url = "https://www.subnetonline.com/pages/ipv6-network-tools/online-ipv6-port-scanner.php"

        # 发起请求并读取响应数据，解码成 UTF-8 格式
        data = self.requestUrl(url).read().decode("utf8")

        # 从响应数据中匹配出 IP 地址和 token
        ip = re.match(r'.*Your IP is.*?name="host".*?value="(.*?)"', data, re.DOTALL).group(1)
        token = re.match(r'.*name="token".*?value="(.*?)"', data, re.DOTALL).group(1)

        # 构造 POST 请求的数据
        post_data = {"host": ip, "port": port, "allow": "on", "token": token, "submit": "Scanning.."}
        # 发起带有 POST 数据的请求，并读取响应数据，解码成 UTF-8 格式
        data = self.requestUrl(url, post_data).read().decode("utf8")

        # 打印 POST 数据和响应数据
        print(post_data, data)

        # 从响应数据中匹配出消息内容
        message = re.match(r".*<div class='formfield'>(.*?)</div>", data, re.DOTALL).group(1)
        # 去除 HTML 标签，并替换特殊字符，然后去除首尾空格
        message = re.sub(r"<.*?>", "", message.replace("<br>", " ").replace("&nbsp;", " ").strip())  # Strip http tags

        # 根据消息内容判断 IP 地址是否在线，返回相应的结果
        if "online" in message:
            return {"ip": ip, "opened": True}
        elif "closed" in message:
            return {"ip": ip, "opened": False}
        else:
            # 抛出异常，表示响应消息无效
            raise Exception("Invalid response: %s" % message)
```