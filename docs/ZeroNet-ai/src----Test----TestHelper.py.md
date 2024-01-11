# `ZeroNet\src\Test\TestHelper.py`

```
# 导入 socket 模块
import socket
# 导入 struct 模块
import struct
# 导入 os 模块
import os

# 导入 pytest 模块
import pytest
# 从 util 模块中导入 helper 函数
from util import helper
# 从 Config 模块中导入 config 变量
from Config import config

# 使用 pytest 的 usefixtures 装饰器，重置设置
@pytest.mark.usefixtures("resetSettings")
class TestHelper:
    # 测试 shellquote 函数
    def testShellquote(self):
        # 断言 shellquote 函数对字符串 "hel'lo" 的处理结果
        assert helper.shellquote("hel'lo") == "\"hel'lo\""  # Allow '
        # 断言 shellquote 函数对字符串 'hel"lo' 的处理结果
        assert helper.shellquote('hel"lo') == '"hello"'  # Remove "
        # 断言 shellquote 函数对字符串 "hel'lo" 和 'hel"lo' 的处理结果
        assert helper.shellquote("hel'lo", 'hel"lo') == ('"hel\'lo"', '"hello"')

    # 测试 packAddress 函数
    def testPackAddress(self):
        # 遍历端口号和 IP 地址的组合
        for port in [1, 1000, 65535]:
            for ip in ["1.1.1.1", "127.0.0.1", "0.0.0.0", "255.255.255.255", "192.168.1.1"]:
                # 断言 packAddress 函数对 IP 地址和端口号的处理结果
                assert len(helper.packAddress(ip, port)) == 6
                assert helper.unpackAddress(helper.packAddress(ip, port)) == (ip, port)

            for ip in ["1:2:3:4:5:6:7:8", "::1", "2001:19f0:6c01:e76:5400:1ff:fed6:3eca", "2001:4860:4860::8888"]:
                # 断言 packAddress 函数对 IPv6 地址和端口号的处理结果
                assert len(helper.packAddress(ip, port)) == 18
                assert helper.unpackAddress(helper.packAddress(ip, port)) == (ip, port)

            # 断言 packOnionAddress 函数对 .onion 地址和端口号的处理结果
            assert len(helper.packOnionAddress("boot3rdez4rzn36x.onion", port)) == 12
            assert helper.unpackOnionAddress(helper.packOnionAddress("boot3rdez4rzn36x.onion", port)) == ("boot3rdez4rzn36x.onion", port)

        # 使用 pytest 的 raises 断言捕获 struct.error 异常
        with pytest.raises(struct.error):
            helper.packAddress("1.1.1.1", 100000)

        # 使用 pytest 的 raises 断言捕获 socket.error 异常
        with pytest.raises(socket.error):
            helper.packAddress("999.1.1.1", 1)

        # 使用 pytest 的 raises 断言捕获所有异常
        with pytest.raises(Exception):
            helper.unpackAddress("X")

    # 测试 getDirname 函数
    def testGetDirname(self):
        # 断言 getDirname 函数对文件路径的处理结果
        assert helper.getDirname("data/users/content.json") == "data/users/"
        assert helper.getDirname("data/users") == "data/"
        assert helper.getDirname("") == ""
        assert helper.getDirname("content.json") == ""
        assert helper.getDirname("data/users/") == "data/users/"
        assert helper.getDirname("/data/users/content.json") == "data/users/"
    # 测试获取文件名的函数
    def testGetFilename(self):
        # 断言获取文件名函数对指定路径的文件名提取正确
        assert helper.getFilename("data/users/content.json") == "content.json"
        # 断言获取文件名函数对指定路径的文件夹名提取正确
        assert helper.getFilename("data/users") == "users"
        # 断言获取文件名函数对空路径返回空字符串
        assert helper.getFilename("") == ""
        # 断言获取文件名函数对单个文件名提取正确
        assert helper.getFilename("content.json") == "content.json"
        # 断言获取文件名函数对以斜杠结尾的路径返回空字符串
        assert helper.getFilename("data/users/") == ""
        # 断言获取文件名函数对以斜杠开头的路径提取正确
        assert helper.getFilename("/data/users/content.json") == "content.json"
    
    # 测试判断是否为 IP 地址的函数
    def testIsIp(self):
        # 断言判断 IP 地址函数对合法 IP 地址返回 True
        assert helper.isIp("1.2.3.4")
        # 断言判断 IP 地址函数对合法 IP 地址返回 True
        assert helper.isIp("255.255.255.255")
        # 断言判断 IP 地址函数对非 IP 地址返回 False
        assert not helper.isIp("any.host")
        # 断言判断 IP 地址函数对非 IP 地址返回 False
        assert not helper.isIp("1.2.3.4.com")
        # 断言判断 IP 地址函数对非 IP 地址返回 False
        assert not helper.isIp("1.2.3.4.any.host")
    
    # 测试判断是否为私有 IP 地址的函数
    def testIsPrivateIp(self):
        # 断言判断私有 IP 地址函数对私有 IP 地址返回 True
        assert helper.isPrivateIp("192.168.1.1")
        # 断言判断私有 IP 地址函数对非私有 IP 地址返回 False
        assert not helper.isPrivateIp("1.1.1.1")
        # 断言判断私有 IP 地址函数对 IPv6 地址返回 True
        assert helper.isPrivateIp("fe80::44f0:3d0:4e6:637c")
        # 断言判断私有 IP 地址函数对非私有 IPv6 地址返回 False
        assert not helper.isPrivateIp("fca5:95d6:bfde:d902:8951:276e:1111:a22c")  # cjdns
    
    # 测试以锁定方式打开文件的函数
    def testOpenLocked(self):
        # 以锁定方式打开文件，并断言成功
        locked_f = helper.openLocked(config.data_dir + "/locked.file")
        assert locked_f
        # 使用 pytest 断言捕获到 BlockingIOError 异常
        with pytest.raises(BlockingIOError):
            locked_f_again = helper.openLocked(config.data_dir + "/locked.file")
        # 以锁定方式打开另一个文件，并断言成功
        locked_f_different = helper.openLocked(config.data_dir + "/locked_different.file")
        # 关闭文件
        locked_f.close()
        locked_f_different.close()
        # 删除临时文件
        os.unlink(locked_f.name)
        os.unlink(locked_f_different.name)
```