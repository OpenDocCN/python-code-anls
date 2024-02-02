# `ZeroNet\plugins\Benchmark\BenchmarkPlugin.py`

```py
# 导入所需的模块
import os
import time
import io
import math
import hashlib
import re
import sys

# 从Config模块中导入config变量
from Config import config
# 从Crypt模块中导入CryptHash类
from Crypt import CryptHash
# 从Plugin模块中导入PluginManager类
from Plugin import PluginManager
# 从Debug模块中导入Debug类
from Debug import Debug
# 从util模块中导入helper函数
from util import helper

# 获取当前文件所在目录
plugin_dir = os.path.dirname(__file__)

# 初始化benchmark_key变量
benchmark_key = None

# 将UiRequestPlugin类注册到PluginManager的"UiRequest"插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 对actionBenchmark方法进行编码响应处理
    @helper.encodeResponse
    def actionBenchmark(self):
        # 声明benchmark_key为全局变量
        global benchmark_key
        # 获取script_nonce
        script_nonce = self.getScriptNonce()
        # 如果benchmark_key为空，则生成一个随机值
        if not benchmark_key:
            benchmark_key = CryptHash.random(encoding="base64")
        # 发送响应头部信息，包括script_nonce
        self.sendHeader(script_nonce=script_nonce)

        # 如果插件管理器中包含"Multiuser"插件，并且config.multiuser_local为False，则返回禁用信息
        if "Multiuser" in PluginManager.plugin_manager.plugin_names and not config.multiuser_local:
            yield "This function is disabled on this proxy"
            return

        # 渲染benchmark.html页面，并传入相应的参数
        data = self.render(
            plugin_dir + "/media/benchmark.html",
            script_nonce=script_nonce,
            benchmark_key=benchmark_key,
            filter=re.sub("[^A-Za-z0-9]", "", self.get.get("filter", ""))
        )
        yield data

    # 对actionBenchmarkResult方法进行编码响应处理
    @helper.encodeResponse
    def actionBenchmarkResult(self):
        # 声明benchmark_key为全局变量
        global benchmark_key
        # 如果请求中的benchmark_key与benchmark_key不匹配，则返回403错误
        if self.get.get("benchmark_key", "") != benchmark_key:
            return self.error403("Invalid benchmark key")

        # 发送响应头部信息，设置content_type为"text/plain"，noscript为True
        self.sendHeader(content_type="text/plain", noscript=True)

        # 如果插件管理器中包含"Multiuser"插件，并且config.multiuser_local为False，则返回禁用信息
        if "Multiuser" in PluginManager.plugin_manager.plugin_names and not config.multiuser_local:
            yield "This function is disabled on this proxy"
            return

        # 输出1024个空格，用于流式传输的头部信息
        yield " " * 1024  # Head (required for streaming)

        # 导入main模块
        import main
        # 记录当前时间
        s = time.time()

        # 调用main模块中的testBenchmark方法，并根据请求中的filter参数进行测试
        for part in main.actions.testBenchmark(filter=self.get.get("filter", "")):
            yield part

        # 输出总耗时
        yield "\n - Total time: %.3fs" % (time.time() - s)

# 将ActionsPlugin类注册到PluginManager的"Actions"插件中
@PluginManager.registerTo("Actions")
class ActionsPlugin:
    # 根据乘数返回对应的标题
    def getMultiplerTitle(self, multipler):
        # 如果乘数小于0.3，标题为"Sloooow"
        if multipler < 0.3:
            multipler_title = "Sloooow"
        # 如果乘数小于0.6，标题为"Ehh"
        elif multipler < 0.6:
            multipler_title = "Ehh"
        # 如果乘数小于0.8，标题为"Goodish"
        elif multipler < 0.8:
            multipler_title = "Goodish"
        # 如果乘数小于1.2，标题为"OK"
        elif multipler < 1.2:
            multipler_title = "OK"
        # 如果乘数小于1.7，标题为"Fine"
        elif multipler < 1.7:
            multipler_title = "Fine"
        # 如果乘数小于2.5，标题为"Fast"
        elif multipler < 2.5:
            multipler_title = "Fast"
        # 如果乘数小于3.5，标题为"WOW"
        elif multipler < 3.5:
            multipler_title = "WOW"
        # 如果乘数大于等于3.5，标题为"Insane!!"
        else:
            multipler_title = "Insane!!"
        return multipler_title

    # 格式化测试结果
    def formatResult(self, taken, standard):
        # 如果没有标准时间，返回格式化后的时间
        if not standard:
            return " Done in %.3fs" % taken

        # 如果花费时间大于0，计算乘数
        if taken > 0:
            multipler = standard / taken
        else:
            multipler = 99
        # 获取乘数对应的标题
        multipler_title = self.getMultiplerTitle(multipler)

        # 返回格式化后的结果
        return " Done in %.3fs = %s (%.2fx)" % (taken, multipler_title, multipler)

    # 测试 HTTPS 连接
    def testHttps(self, num_run=1):
        """
        Test https connection with valid and invalid certs
        """
        # 导入 urllib.request 和 urllib.error 模块
        import urllib.request
        import urllib.error

        # 获取 google.com 的响应内容
        body = urllib.request.urlopen("https://google.com").read()
        # 断言响应内容长度大于100
        assert len(body) > 100
        yield "."

        # 定义包含不同类型证书的 URL 列表
        badssl_urls = [
            "https://expired.badssl.com/",
            "https://wrong.host.badssl.com/",
            "https://self-signed.badssl.com/",
            "https://untrusted-root.badssl.com/"
        ]
        # 遍历不同类型证书的 URL
        for badssl_url in badssl_urls:
            try:
                # 获取不同类型证书的响应内容
                body = urllib.request.urlopen(badssl_url).read()
                https_err = None
            except urllib.error.URLError as err:
                # 如果出现 URLError，将其赋值给 https_err
                https_err = err
            # 断言 https_err 存在
            assert https_err
            yield "."
    # 定义一个测试哈希函数的方法，可以指定运行次数和哈希类型
    def testCryptHash(self, num_run=1, hash_type="sha256"):
        """
        Test hashing functions
        """
        # 生成一个包含 5MB 数据的字节流
        yield "(5MB) "

        # 导入 Crypt 模块中的 CryptHash 函数
        from Crypt import CryptHash

        # 定义不同哈希类型对应的哈希函数和有效哈希值
        hash_types = {
            "sha256": {"func": CryptHash.sha256sum, "hash_valid": "8cd629d9d6aff6590da8b80782a5046d2673d5917b99d5603c3dcb4005c45ffa"},
            "sha512": {"func": CryptHash.sha512sum, "hash_valid": "9ca7e855d430964d5b55b114e95c6bbb114a6d478f6485df93044d87b108904d"}
        }
        # 根据指定的哈希类型获取对应的哈希函数和有效哈希值
        hash_func = hash_types[hash_type]["func"]
        hash_valid = hash_types[hash_type]["hash_valid"]

        # 生成一个包含 5MB 数据的字节流
        data = io.BytesIO(b"Hello" * 1024 * 1024)  # 5MB
        # 循环运行指定次数
        for i in range(num_run):
            # 重置数据流的位置到开头
            data.seek(0)
            # 对数据流进行哈希计算
            hash = hash_func(data)
            yield "."
        # 断言计算得到的哈希值与有效哈希值相等
        assert hash == hash_valid, "%s != %s" % (hash, hash_valid)

    # 定义一个测试哈希函数的方法，可以指定运行次数和哈希类型
    def testCryptHashlib(self, num_run=1, hash_type="sha3_256"):
        """
        Test SHA3 hashing functions
        """
        # 生成一个包含 5MB 数据的字节流
        yield "x 5MB "

        # 定义不同哈希类型对应的哈希函数和有效哈希值
        hash_types = {
            "sha3_256": {"func": hashlib.sha3_256, "hash_valid": "c8aeb3ef9fe5d6404871c0d2a4410a4d4e23268e06735648c9596f436c495f7e"},
            "sha3_512": {"func": hashlib.sha3_512, "hash_valid": "b75dba9472d8af3cc945ce49073f3f8214d7ac12086c0453fb08944823dee1ae83b3ffbc87a53a57cc454521d6a26fe73ff0f3be38dddf3f7de5d7692ebc7f95"},
        }

        # 根据指定的哈希类型获取对应的哈希函数和有效哈希值
        hash_func = hash_types[hash_type]["func"]
        hash_valid = hash_types[hash_type]["hash_valid"]

        # 生成一个包含 5MB 数据的字节流
        data = io.BytesIO(b"Hello" * 1024 * 1024)  # 5MB
        # 循环运行指定次数
        for i in range(num_run):
            # 重置数据流的位置到开头
            data.seek(0)
            # 创建哈希对象
            h = hash_func()
            # 循环读取数据流并更新哈希对象
            while 1:
                buff = data.read(1024 * 64)
                if not buff:
                    break
                h.update(buff)
            # 获取哈希值的十六进制表示
            hash = h.hexdigest()
            yield "."
        # 断言计算得到的哈希值与有效哈希值相等
        assert hash == hash_valid, "%s != %s" % (hash, hash_valid)
    # 定义一个测试生成随机数据的函数，可以指定运行次数，默认为1次
    def testRandom(self, num_run=1):
        """
        Test generating random data
        """
        # 生成包含"x 1000 x 256 bytes "的生成器
        yield "x 1000 x 256 bytes "
        # 循环指定次数
        for i in range(num_run):
            # 初始化上一次的数据为None
            data_last = None
            # 循环1000次
            for y in range(1000):
                # 生成256字节的随机数据
                data = os.urandom(256)
                # 断言本次生成的数据与上一次不相同
                assert data != data_last
                # 断言数据长度为256
                assert len(data) == 256
                # 更新上一次的数据
                data_last = data
            # 生成一个"."
            yield "."

    # 定义一个测试从主种子生成确定性私钥的函数，可以指定运行次数，默认为2次
    def testHdPrivatekey(self, num_run=2):
        """
        Test generating deterministic private keys from a master seed
        """
        # 导入CryptBitcoin模块
        from Crypt import CryptBitcoin
        # 主种子
        seed = "e180efa477c63b0f2757eac7b1cce781877177fe0966be62754ffd4c8592ce38"
        # 存储私钥的列表
        privatekeys = []
        # 循环指定次数
        for i in range(num_run):
            # 生成私钥并添加到列表中
            privatekeys.append(CryptBitcoin.hdPrivatekey(seed, i * 10))
            # 生成一个"."
            yield "."
        # 预期的有效私钥
        valid = "5JSbeF5PevdrsYjunqpg7kAGbnCVYa1T4APSL3QRu8EoAmXRc7Y"
        # 断言第一个私钥与预期的有效私钥相等
        assert privatekeys[0] == valid, "%s != %s" % (privatekeys[0], valid)
        # 如果私钥数量大于1，断言第一个私钥与最后一个私钥不相等
        if len(privatekeys) > 1:
            assert privatekeys[0] != privatekeys[-1]

    # 定义一个测试使用私钥对数据进行签名的函数，可以指定运行次数，默认为1次
    def testSign(self, num_run=1):
        """
        Test signing data using a private key
        """
        # 导入CryptBitcoin模块
        from Crypt import CryptBitcoin
        # 待签名的数据
        data = "Hello" * 1024
        # 私钥
        privatekey = "5JsunC55XGVqFQj5kPGK4MWgTL26jKbnPhjnmchSNPo75XXCwtk"
        # 循环指定次数
        for i in range(num_run):
            # 生成一个"."
            yield "."
            # 对数据进行签名
            sign = CryptBitcoin.sign(data, privatekey)
            # 预期的有效签名
            valid = "G1GXaDauZ8vX/N9Jn+MRiGm9h+I94zUhDnNYFaqMGuOiBHB+kp4cRPZOL7l1yqK5BHa6J+W97bMjvTXtxzljp6w="
            # 断言生成的签名与预期的有效签名相等
            assert sign == valid, "%s != %s" % (sign, valid)
    # 定义一个测试验证生成签名的方法，可以指定运行次数和验证库
    def testVerify(self, num_run=1, lib_verify="sslcrypto"):
        """
        Test verification of generated signatures
        """
        # 导入 CryptBitcoin 模块
        from Crypt import CryptBitcoin
        # 加载指定的验证库
        CryptBitcoin.loadLib(lib_verify, silent=True)

        # 生成测试数据
        data = "Hello" * 1024
        # 设置私钥
        privatekey = "5JsunC55XGVqFQj5kPGK4MWgTL26jKbnPhjnmchSNPo75XXCwtk"
        # 通过私钥生成地址
        address = CryptBitcoin.privatekeyToAddress(privatekey)
        # 设置签名
        sign = "G1GXaDauZ8vX/N9Jn+MRiGm9h+I94zUhDnNYFaqMGuOiBHB+kp4cRPZOL7l1yqK5BHa6J+W97bMjvTXtxzljp6w="

        # 循环运行验证方法
        for i in range(num_run):
            # 调用 CryptBitcoin 模块的验证方法
            ok = CryptBitcoin.verify(data, address, sign, lib_verify=lib_verify)
            # 生成进度标记
            yield "."
            # 断言验证结果
            assert ok, "does not verify from %s" % address

        # 如果使用的是 sslcrypto 验证库，则输出其后端信息
        if lib_verify == "sslcrypto":
            yield("(%s)" % CryptBitcoin.sslcrypto.ecc.get_backend())

    # 测试所有活动的开放端口检查器
    def testPortCheckers(self):
        """
        Test all active open port checker
        """
        # 导入 PeerPortchecker 模块
        from Peer import PeerPortchecker
        # 遍历不同类型的 IP 地址和对应的检查器函数
        for ip_type, func_names in PeerPortchecker.PeerPortchecker.checker_functions.items():
            yield "\n- %s:" % ip_type
            for func_name in func_names:
                yield "\n - Tracker %s: " % func_name
                try:
                    # 调用单个开放端口检查器的测试方法
                    for res in self.testPortChecker(func_name):
                        yield res
                except Exception as err:
                    yield Debug.formatException(err)

    # 测试单个开放端口检查器
    def testPortChecker(self, func_name):
        """
        Test single open port checker
        """
        # 导入 PeerPortchecker 模块
        from Peer import PeerPortchecker
        # 创建 PeerPortchecker 对象
        peer_portchecker = PeerPortchecker.PeerPortchecker(None)
        # 获取指定名称的检查器函数
        announce_func = getattr(peer_portchecker, func_name)
        # 调用检查器函数
        res = announce_func(3894)
        yield res

    # 运行所有测试以检查系统与 ZeroNet 功能的兼容性
    def testAll(self):
        """
        Run all tests to check system compatibility with ZeroNet functions
        """
        # 遍历运行基准测试的进度
        for progress in self.testBenchmark(online=not config.offline, num_run=1):
            yield progress
# 将 ConfigPlugin 类注册到 PluginManager 的 ConfigPlugin 插件中
@PluginManager.registerTo("ConfigPlugin")
class ConfigPlugin(object):
    # 创建参数
    def createArguments(self):
        # 调用父类的 createArguments 方法，并将返回值保存在 back 变量中
        back = super(ConfigPlugin, self).createArguments()
        # 如果命令行参数中的 test 值为 benchmark
        if self.getCmdlineValue("test") == "benchmark":
            # 向测试解析器添加 num_multipler 参数，用于设置基准运行时间的倍数
            self.test_parser.add_argument(
                '--num_multipler', help='Benchmark run time multipler',
                default=1.0, type=float, metavar='num'
            )
            # 向测试解析器添加 filter 参数，用于过滤运行基准测试
            self.test_parser.add_argument(
                '--filter', help='Filter running benchmark',
                default=None, metavar='test name'
            )
        # 如果命令行参数中的 test 值为 portChecker
        elif self.getCmdlineValue("test") == "portChecker":
            # 向测试解析器添加 func_name 参数，用于设置打开端口检查函数的名称
            self.test_parser.add_argument(
                '--func_name', help='Name of open port checker function',
                default=None, metavar='func_name'
            )
        # 返回之前保存的 back 变量
        return back
```