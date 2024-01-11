# `ZeroNet\src\lib\subtl\subtl.py`

```
'''
Based on the specification at http://bittorrent.org/beps/bep_0015.html
'''
# 导入所需的模块
import binascii
import random
import struct
import time
import socket
from collections import defaultdict

# 定义版本号
__version__ = '0.0.1'

# 定义常量
CONNECT = 0
ANNOUNCE = 1
SCRAPE = 2
ERROR = 3

# 定义自定义异常类
class UdpTrackerClientException(Exception):
    pass

# 定义 UdpTrackerClient 类
class UdpTrackerClient:

    # 初始化方法
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.peer_port = 6881
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.conn_id = 0x41727101980
        self.transactions = {}
        self.peer_id = self._generate_peer_id()
        self.timeout = 9

    # 连接方法
    def connect(self):
        return self._send(CONNECT)

    # 发送请求方法
    def announce(self, **kwargs):
        if not kwargs:
            raise UdpTrackerClientException('arguments missing')
        args = {
            'peer_id': self.peer_id,
            'downloaded': 0,
            'left': 0,
            'uploaded': 0,
            'event': 0,
            'key': 0,
            'num_want': 10,
            'ip_address': 0,
            'port': self.peer_port,
        }
        args.update(kwargs)

        fields = 'info_hash peer_id downloaded left uploaded event ' \
            'ip_address key num_want port'

        # 检查并抛出异常，如果缺少字段
        self._check_fields(args, fields)

        # 人类倾向于使用哈希的十六进制表示。浪费资源的人类。
        args['info_hash'] = args['info_hash']

        values = [args[a] for a in fields.split()]
        values[1] = values[1].encode("utf8")
        payload = struct.pack('!20s20sQQQLLLLH', *values)
        return self._send(ANNOUNCE, payload)
    # 根据给定的 info_hash_list 进行数据抓取
    def scrape(self, info_hash_list):
        # 如果 info_hash_list 的长度大于 74，则抛出异常
        if len(info_hash_list) > 74:
            raise UdpTrackerClientException('Max info_hashes is 74')

        # 初始化 payload 为空字符串
        payload = ''
        # 遍历 info_hash_list，将每个 info_hash 添加到 payload 中
        for info_hash in info_hash_list:
            payload += info_hash

        # 发送 SCRAPE 请求，并返回响应
        trans = self._send(SCRAPE, payload)
        # 将发送的 info_hash_list 添加到响应中
        trans['sent_hashes'] = info_hash_list
        # 返回响应
        return trans

    # 单次轮询
    def poll_once(self):
        # 设置套接字超时时间
        self.sock.settimeout(self.timeout)
        try:
            # 接收响应数据
            response = self.sock.recv(10240)
        except socket.timeout:
            return

        # 解析响应头和响应数据
        header = response[:8]
        payload = response[8:]
        action, trans_id = struct.unpack('!LL', header)
        try:
            # 根据 trans_id 获取对应的事务
            trans = self.transactions[trans_id]
        except KeyError:
            # 如果 trans_id 不存在，则输出错误信息
            self.error('transaction_id not found')
            return
        # 处理响应数据，并将处理后的结果添加到事务中
        trans['response'] = self._process_response(action, payload, trans)
        # 标记事务为已完成
        trans['completed'] = True
        # 删除已完成的事务
        del self.transactions[trans_id]
        # 返回事务
        return trans

    # 输出错误信息
    def error(self, message):
        raise Exception('error: {}'.format(message))

    # 发送请求
    def _send(self, action, payload=None):
        # 如果 payload 为空，则初始化为 b''
        if not payload:
            payload = b''
        # 生成事务 ID 和请求头
        trans_id, header = self._request_header(action)
        # 将事务添加到 transactions 字典中
        self.transactions[trans_id] = trans = {
            'action': action,
            'time': time.time(),
            'payload': payload,
            'completed': False,
        }
        # 连接到指定的主机和端口，并发送请求头和 payload
        self.sock.connect((self.host, self.port))
        self.sock.send(header + payload)
        # 返回事务
        return trans

    # 生成请求头
    def _request_header(self, action):
        # 生成随机的事务 ID
        trans_id = random.randint(0, (1 << 32) - 1)
        # 封装连接 ID、action 和事务 ID 成请求头
        return trans_id, struct.pack('!QLL', self.conn_id, action, trans_id)
    # 处理响应动作，根据不同的动作调用相应的处理函数
    def _process_response(self, action, payload, trans):
        # 如果动作是连接，则调用处理连接的函数
        if action == CONNECT:
            return self._process_connect(payload, trans)
        # 如果动作是通告，则调用处理通告的函数
        elif action == ANNOUNCE:
            return self._process_announce(payload, trans)
        # 如果动作是爬取，则调用处理爬取的函数
        elif action == SCRAPE:
            return self._process_scrape(payload, trans)
        # 如果动作是错误，则调用处理错误的函数
        elif action == ERROR:
            return self._process_error(payload, trans)
        # 如果动作未知，则抛出异常
        else:
            raise UdpTrackerClientException(
                'Unknown action response: {}'.format(action))

    # 处理连接动作的函数
    def _process_connect(self, payload, trans):
        # 从 payload 中解析出连接 ID，并赋值给 self.conn_id
        self.conn_id = struct.unpack('!Q', payload)[0]
        return self.conn_id

    # 处理通告动作的函数
    def _process_announce(self, payload, trans):
        # 初始化响应字典
        response = {}

        # 解析出 interval, leechers, seeders 的信息
        info_struct = '!LLL'
        info_size = struct.calcsize(info_struct)
        info = payload[:info_size]
        interval, leechers, seeders = struct.unpack(info_struct, info)

        # 解析出 peer 数据
        peer_data = payload[info_size:]
        peer_struct = '!LH'
        peer_size = struct.calcsize(peer_struct)
        peer_count = int(len(peer_data) / peer_size)
        peers = []

        # 遍历 peer 数据，解析出每个 peer 的地址和端口，并添加到 peers 列表中
        for peer_offset in range(peer_count):
            off = peer_size * peer_offset
            peer = peer_data[off:off + peer_size]
            addr, port = struct.unpack(peer_struct, peer)
            peers.append({
                'addr': socket.inet_ntoa(struct.pack('!L', addr)),
                'port': port,
            })

        # 返回包含 interval, leechers, seeders 和 peers 的响应字典
        return {
            'interval': interval,
            'leechers': leechers,
            'seeders': seeders,
            'peers': peers,
        }
    # 处理从 tracker 收到的爬取信息，将其转换为字典形式返回
    def _process_scrape(self, payload, trans):
        # 定义信息结构的格式
        info_struct = '!LLL'
        # 计算信息结构的大小
        info_size = struct.calcsize(info_struct)
        # 计算 payload 中包含的信息数量
        info_count = len(payload) / info_size
        # 获取已发送哈希值列表
        hashes = trans['sent_hashes']
        # 初始化响应字典
        response = {}
        # 遍历信息数量，解析每个信息并添加到响应字典中
        for info_offset in range(info_count):
            off = info_size * info_offset
            info = payload[off:off + info_size]
            seeders, completed, leechers = struct.unpack(info_struct, info)
            response[hashes[info_offset]] = {
                'seeders': seeders,
                'completed': completed,
                'leechers': leechers,
            }
        # 返回响应字典
        return response

    # 处理错误信息，目前未见过该类型的动作从 tracker 发送过来，但为了可能性，保留该方法
    def _process_error(self, payload, trans):
        '''
        I haven't seen this action type be sent from a tracker, but I've left
        it here for the possibility.
        '''
        # 调用 error 方法处理错误信息
        self.error(payload)
        # 返回 False
        return False

    # 生成 peer id
    def _generate_peer_id(self):
        '''http://www.bittorrent.org/beps/bep_0020.html'''
        # 根据规范生成 peer id
        peer_id = '-PU' + __version__.replace('.', '-') + '-'
        remaining = 20 - len(peer_id)
        numbers = [str(random.randint(0, 9)) for _ in range(remaining)]
        peer_id += ''.join(numbers)
        # 确保 peer id 的长度为 20
        assert(len(peer_id) == 20)
        # 返回生成的 peer id
        return peer_id

    # 检查参数中是否包含指定的字段
    def _check_fields(self, args, fields):
        for f in fields:
            try:
                args.get(f)
            except KeyError:
                # 抛出异常，指出缺少的字段
                raise UdpTrackerClientException('field missing: {}'.format(f))
```