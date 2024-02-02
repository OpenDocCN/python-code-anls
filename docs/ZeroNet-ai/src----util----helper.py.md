# `ZeroNet\src\util\helper.py`

```py
# 导入所需的模块
import os
import stat
import socket
import struct
import re
import collections
import time
import logging
import base64
import json

# 导入 gevent 模块
import gevent

# 从 Config 模块中导入 config 变量

from Config import config

# 定义一个原子写入函数，用于原子性地写入文件内容
def atomicWrite(dest, content, mode="wb"):
    try:
        # 打开临时文件进行写入
        with open(dest + "-tmpnew", mode) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        # 如果存在之前未完成的写入文件，则重命名为带时间戳的文件
        if os.path.isfile(dest + "-tmpold"):  # Previous incomplete write
            os.rename(dest + "-tmpold", dest + "-tmpold-%s" % time.time())
        # 如果目标文件存在，则将其重命名为带有 -tmpold 后缀的文件
        if os.path.isfile(dest):  # Rename old file to -tmpold
            os.rename(dest, dest + "-tmpold")
        # 将临时文件重命名为目标文件
        os.rename(dest + "-tmpnew", dest)
        # 如果存在带有 -tmpold 后缀的文件，则删除它
        if os.path.isfile(dest + "-tmpold"):
            os.unlink(dest + "-tmpold")  # Remove old file
        return True
    except Exception as err:
        # 导入 Debug 模块中的 Debug 类
        from Debug import Debug
        # 记录写入失败的日志，并尝试回滚操作
        logging.error(
            "File %s write failed: %s, (%s) reverting..." %
            (dest, Debug.formatException(err), Debug.formatStack())
        )
        # 如果存在带有 -tmpold 后缀的文件，并且目标文件不存在，则将其重命名为目标文件
        if os.path.isfile(dest + "-tmpold") and not os.path.isfile(dest):
            os.rename(dest + "-tmpold", dest)
        return False

# 定义一个将数据转换为 JSON 格式的函数
def jsonDumps(data):
    # 将数据转换为 JSON 格式的字符串，并进行格式化缩进和排序
    content = json.dumps(data, indent=1, sort_keys=True)

    # 通过正则表达式去除 JSON 字符串中的不必要空白字符，使其更加紧凑
    def compact_dict(match):
        if "\n" in match.group(0):
            return match.group(0).replace(match.group(1), match.group(1).strip())
        else:
            return match.group(0)

    content = re.sub(r"\{(\n[^,\[\{]{10,100000}?)\}[, ]{0,2}\n", compact_dict, content, flags=re.DOTALL)

    def compact_list(match):
        if "\n" in match.group(0):
            stripped_lines = re.sub("\n[ ]*", "", match.group(1))
            return match.group(0).replace(match.group(1), stripped_lines)
        else:
            return match.group(0)

    content = re.sub(r"\[([^\[\{]{2,100000}?)\][, ]{0,2}\n", compact_list, content, flags=re.DOTALL)

    # 移除 JSON 字符串末尾的空白字符
    # 使用正则表达式替换多行内容中的末尾空格，替换为空字符串
    content = re.sub(r"(?m)[ ]+$", "", content)
    # 返回处理后的内容
    return content
# 打开文件并加锁，返回文件对象
def openLocked(path, mode="wb"):
    try:
        # 如果操作系统是 POSIX
        if os.name == "posix":
            # 导入 fcntl 模块
            import fcntl
            # 以指定模式打开文件
            f = open(path, mode)
            # 对文件加锁
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # 如果操作系统是 Windows
        elif os.name == "nt":
            # 导入 msvcrt 模块
            import msvcrt
            # 以指定模式打开文件
            f = open(path, mode)
            # 对文件加锁
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            # 以指定模式打开文件
            f = open(path, mode)
    # 捕获可能的异常
    except (IOError, PermissionError, BlockingIOError) as err:
        # 抛出异常
        raise BlockingIOError("Unable to lock file: %s" % err)
    # 返回文件对象
    return f


# 获取可用空间大小
def getFreeSpace():
    free_space = -1
    # 如果操作系统是 Unix
    if "statvfs" in dir(os):
        # 获取文件系统状态信息
        statvfs = os.statvfs(config.data_dir.encode("utf8"))
        # 计算可用空间大小
        free_space = statvfs.f_frsize * statvfs.f_bavail
    else:  # 如果操作系统是 Windows
        try:
            # 导入 ctypes 模块
            import ctypes
            # 创建一个无符号长长整型指针
            free_space_pointer = ctypes.c_ulonglong(0)
            # 获取磁盘可用空间大小
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(config.data_dir), None, None, ctypes.pointer(free_space_pointer)
            )
            # 获取可用空间大小
            free_space = free_space_pointer.value
        except Exception as err:
            # 记录错误日志
            logging.error("GetFreeSpace error: %s" % err)
    # 返回可用空间大小
    return free_space


# 对 SQL 查询值进行转义处理
def sqlquote(value):
    # 如果值的类型是整型
    if type(value) is int:
        # 转换为字符串并返回
        return str(value)
    else:
        # 对字符串值进行转义处理并返回
        return "'%s'" % value.replace("'", "''")


# 对命令行参数进行转义处理
def shellquote(*args):
    # 如果参数个数为1
    if len(args) == 1:
        # 对参数进行转义处理并返回
        return '"%s"' % args[0].replace('"', "")
    else:
        # 对多个参数进行转义处理并返回
        return tuple(['"%s"' % arg.replace('"', "") for arg in args])


# 将对等节点信息打包
def packPeers(peers):
    # 初始化打包后的对等节点信息
    packed_peers = {"ipv4": [], "ipv6": [], "onion": []}
    # 遍历对等节点列表
    for peer in peers:
        try:
            # 获取 IP 类型
            ip_type = getIpType(peer.ip)
            # 如果 IP 类型在打包后的对等节点信息中
            if ip_type in packed_peers:
                # 将对等节点信息打包并添加到对应类型的列表中
                packed_peers[ip_type].append(peer.packMyAddress())
        except Exception:
            # 记录调试日志
            logging.debug("Error packing peer address: %s" % peer)
    # 返回打包后的对等节点信息
    return packed_peers


# 将 IP 和端口打包成6字节或18字节格式
def packAddress(ip, port):
    # 如果 IP 地址中包含冒号，说明是 IPv6 地址
    if ":" in ip:
        # 将 IPv6 地址转换为网络字节序的二进制表示，并添加端口号的二进制表示
        return socket.inet_pton(socket.AF_INET6, ip) + struct.pack("H", port)
    # 如果 IP 地址中不包含冒号，说明是 IPv4 地址
    else:
        # 将 IPv4 地址转换为网络字节序的二进制表示，并添加端口号的二进制表示
        return socket.inet_aton(ip) + struct.pack("H", port)
# 从6字节或18字节格式转换为IP和端口
def unpackAddress(packed):
    # 如果长度为18，则使用IPv6解析前16个字节为IP地址，后2个字节为端口
    if len(packed) == 18:
        return socket.inet_ntop(socket.AF_INET6, packed[0:16]), struct.unpack_from("H", packed, 16)[0]
    else:
        # 如果长度不为6或18，则抛出异常
        if len(packed) != 6:
            raise Exception("Invalid length ip4 packed address: %s" % len(packed))
        # 使用IPv4解析前4个字节为IP地址，后2个字节为端口
        return socket.inet_ntoa(packed[0:4]), struct.unpack_from("H", packed, 4)[0]


# onion和端口转换为12字节格式
def packOnionAddress(onion, port):
    # 去除.onion后缀，将大写的onion进行base32解码，然后与端口进行打包
    onion = onion.replace(".onion", "")
    return base64.b32decode(onion.upper()) + struct.pack("H", port)


# 从12字节格式转换为IP和端口
def unpackOnionAddress(packed):
    # 对12字节格式进行解码，然后转换为小写并添加.onion后缀，最后解析出端口
    return base64.b32encode(packed[0:-2]).lower().decode() + ".onion", struct.unpack("H", packed[-2:])[0]


# 从文件获取目录
# 返回：data/site/content.json -> data/site/
def getDirname(path):
    # 如果路径中包含"/"，则返回最后一个"/"之前的内容，并去除左侧的"/"
    if "/" in path:
        return path[:path.rfind("/") + 1].lstrip("/")
    else:
        return ""


# 从文件获取文件名
# 返回：data/site/content.json -> content.json
def getFilename(path):
    # 返回路径中最后一个"/"之后的内容
    return path[path.rfind("/") + 1:]


# 获取文件大小
def getFilesize(path):
    try:
        s = os.stat(path)
    except Exception:
        return None
    # 如果是文件，则返回文件大小
    if stat.S_ISREG(s.st_mode):  # Test if it's file
        return s.st_size
    else:
        return None


# 将哈希转换为哈希ID以用于哈希字段
def toHashId(hash):
    # 将哈希的前4个字符转换为16进制数
    return int(hash[0:4], 16)


# 合并字典值
def mergeDicts(dicts):
    back = collections.defaultdict(set)
    for d in dicts:
        for key, val in d.items():
            back[key].update(val)
    return dict(back)


# 使用gevent SSL错误解决方法请求https URL
def httpRequest(url, as_file=False):
    # 如果URL以"http://"开头，则使用urllib.request发送请求
    if url.startswith("http://"):
        import urllib.request
        response = urllib.request.urlopen(url)
    # 用于避免 Python gevent SSL 错误的临时解决方案
    else:  
        # 导入需要的模块
        import socket
        import http.client
        import ssl

        # 从 URL 中提取主机名和请求路径
        host, request = re.match("https://(.*?)(/.*?)$", url).groups()

        # 创建 HTTPS 连接
        conn = http.client.HTTPSConnection(host)
        # 创建套接字连接
        sock = socket.create_connection((conn.host, conn.port), conn.timeout, conn.source_address)
        # 将套接字包装成 SSL 套接字
        conn.sock = ssl.wrap_socket(sock, conn.key_file, conn.cert_file)
        # 发送 GET 请求
        conn.request("GET", request)
        # 获取响应
        response = conn.getresponse()
        # 如果响应状态码是重定向类型，则记录重定向信息并重新发送请求
        if response.status in [301, 302, 303, 307, 308]:
            logging.info("Redirect to: %s" % response.getheader('Location'))
            response = httpRequest(response.getheader('Location'))

    # 如果需要将响应内容保存为文件
    if as_file:
        # 导入需要的模块
        import io
        # 创建一个字节流对象
        data = io.BytesIO()
        # 循环读取响应内容并写入字节流对象
        while True:
            buff = response.read(1024 * 16)
            if not buff:
                break
            data.write(buff)
        # 返回字节流对象
        return data
    # 如果不需要保存为文件，则直接返回响应对象
    else:
        return response
# 定义一个函数，用于在指定时间后调用指定的函数
def timerCaller(secs, func, *args, **kwargs):
    # 使用协程在指定时间后调用 timerCaller 函数
    gevent.spawn_later(secs, timerCaller, secs, func, *args, **kwargs)
    # 调用指定的函数
    func(*args, **kwargs)

# 定义一个函数，用于在指定时间后调用指定的函数，并返回协程对象
def timer(secs, func, *args, **kwargs):
    # 返回在指定时间后调用 timerCaller 函数的协程对象
    return gevent.spawn_later(secs, timerCaller, secs, func, *args, **kwargs)

# 创建一个网络连接
def create_connection(address, timeout=None, source_address=None):
    # 如果地址在本地 IP 列表中
    if address in config.ip_local:
        # 创建连接并返回套接字对象
        sock = socket.create_connection_original(address, timeout, source_address)
    else:
        # 创建连接并返回套接字对象
        sock = socket.create_connection_original(address, timeout, socket.bind_addr)
    return sock

# 对 socket 进行绑定的 Monkey Patch
def socketBindMonkeyPatch(bind_ip, bind_port):
    import socket
    # 记录绑定的 IP 和端口
    logging.info("Monkey patching socket to bind to: %s:%s" % (bind_ip, bind_port))
    # 设置绑定的 IP 和端口
    socket.bind_addr = (bind_ip, int(bind_port))
    # 保存原始的创建连接方法
    socket.create_connection_original = socket.create_connection
    # 替换创建连接方法为自定义的 create_connection 函数
    socket.create_connection = create_connection

# 限制 Gzip 文件大小的函数
def limitedGzipFile(*args, **kwargs):
    import gzip
    # 定义一个继承自 GzipFile 的类，重写 read 方法限制文件大小
    class LimitedGzipFile(gzip.GzipFile):
        def read(self, size=-1):
            return super(LimitedGzipFile, self).read(1024 * 1024 * 25)
    return LimitedGzipFile(*args, **kwargs)

# 计算列表中元素的平均值
def avg(items):
    if len(items) > 0:
        return sum(items) / len(items)
    else:
        return 0

# 判断是否为 IP 地址
def isIp(ip):
    if ":" in ip:  # IPv6
        try:
            # 尝试将字符串转换为 IPv6 地址
            socket.inet_pton(socket.AF_INET6, ip)
            return True
        except Exception:
            return False
    else:  # IPv4
        try:
            # 尝试将字符串转换为 IPv4 地址
            socket.inet_aton(ip)
            return True
        except Exception:
            return False

# 匹配私有 IP 地址的正则表达式
local_ip_pattern = re.compile(r"^127\.|192\.168\.|10\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[0-1]\.|169\.254\.|::1$|fe80")
# 判断是否为私有 IP 地址
def isPrivateIp(ip):
    return local_ip_pattern.match(ip)

# 获取 IP 地址类型
def getIpType(ip):
    if ip.endswith(".onion"):
        return "onion"
    elif ":" in ip:
        return "ipv6"
    elif re.match(r"[0-9\.]+$", ip):
        return "ipv4"
    else:
        return "unknown"

# 创建套接字
def createSocket(ip, sock_type=socket.SOCK_STREAM):
    # 获取 IP 类型（IPv4 或 IPv6）
    ip_type = getIpType(ip)
    # 如果 IP 类型为 IPv6，则创建 IPv6 地址族的套接字
    if ip_type == "ipv6":
        return socket.socket(socket.AF_INET6, sock_type)
    # 如果 IP 类型为 IPv4 或其他类型，则创建 IPv4 地址族的套接字
    else:
        return socket.socket(socket.AF_INET, sock_type)
# 获取指定类型的接口IP地址列表，默认为ipv4
def getInterfaceIps(ip_type="ipv4"):
    res = []  # 存储结果的列表
    if ip_type == "ipv6":
        test_ips = ["ff0e::c", "2606:4700:4700::1111"]  # IPv6测试地址列表
    else:
        test_ips = ['239.255.255.250', "8.8.8.8"]  # IPv4测试地址列表

    for test_ip in test_ips:
        try:
            s = createSocket(test_ip, sock_type=socket.SOCK_DGRAM)  # 创建套接字
            s.connect((test_ip, 1))  # 连接到指定IP地址和端口
            res.append(s.getsockname()[0])  # 将套接字的本地IP地址添加到结果列表
        except Exception:
            pass

    try:
        res += [ip[4][0] for ip in socket.getaddrinfo(socket.gethostname(), 1)]  # 获取主机名对应的IP地址列表，并添加到结果列表
    except Exception:
        pass

    # 对结果列表中的IP地址进行处理，保留指定类型的IP地址并去除无效的部分
    res = [re.sub("%.*", "", ip) for ip in res if getIpType(ip) == ip_type and isIp(ip)]
    return list(set(res))  # 返回去重后的结果列表


# 比较函数，返回a和b的大小关系
def cmp(a, b):
    return (a > b) - (a < b)


# 装饰器函数，将返回的数据从utf8编码转换为字节流
def encodeResponse(func):  # Encode returned data from utf8 to bytes
    def wrapper(*args, **kwargs):
        back = func(*args, **kwargs)  # 调用被装饰函数
        if "__next__" in dir(back):  # 如果返回的是迭代器
            for part in back:
                if type(part) == bytes:  # 如果是字节流类型
                    yield part
                else:
                    yield part.encode()  # 将其他类型转换为字节流
        else:
            if type(back) == bytes:  # 如果返回的是字节流类型
                yield back
            else:
                yield back.encode()  # 将其他类型转换为字节流

    return wrapper  # 返回包装后的函数
```