# `.\pytorch\torch\package\_stdlib.py`

```
# mypy: allow-untyped-defs
# 导入 sys 模块，用于获取 Python 运行时信息
import sys

# 判断给定的模块是否是 Python 标准库的一部分
def is_stdlib_module(module: str) -> bool:
    # 获取模块的基础部分（不包括子模块），例如 "os.path" 的基础部分是 "os"
    base_module = module.partition(".")[0]
    # 判断基础模块是否在标准库模块集合中
    return base_module in _get_stdlib_modules()

# 获取当前 Python 版本下的标准库模块集合
def _get_stdlib_modules():
    if sys.version_info.major == 3:
        if sys.version_info.minor == 8:
            return stdlib3_8
        if sys.version_info.minor == 9:
            return stdlib3_9
        if sys.version_info.minor >= 10:
            return sys.stdlib_module_names  # type: ignore[attr-defined]
    elif sys.version_info.major > 3:
        return sys.stdlib_module_names  # type: ignore[attr-defined]

    # 如果 Python 版本不在支持范围内，抛出运行时错误
    raise RuntimeError(f"Unsupported Python version: {sys.version_info}")

# Python 3.8 版本下的标准库模块集合
stdlib3_8 = {
    "_dummy_thread",
    "_thread",
    "abc",
    "aifc",
    "argparse",
    "array",
    "ast",
    "asynchat",
    "asyncio",
    "asyncore",
    "atexit",
    "audioop",
    "base64",
    "bdb",
    "binascii",
    "binhex",
    "bisect",
    "builtins",
    "bz2",
    "cProfile",
    "calendar",
    "cgi",
    "cgitb",
    "chunk",
    "cmath",
    "cmd",
    "code",
    "codecs",
    "codeop",
    "collections",
    "colorsys",
    "compileall",
    "concurrent",
    "configparser",
    "contextlib",
    "contextvars",
    "copy",
    "copyreg",
    "crypt",
    "csv",
    "ctypes",
    "curses",
    "dataclasses",
    "datetime",
    "dbm",
    "decimal",
    "difflib",
    "dis",
    "distutils",
    "doctest",
    "dummy_threading",
    "email",
    "encodings",
    "ensurepip",
    "enum",
    "errno",
    "faulthandler",
    "fcntl",
    "filecmp",
    "fileinput",
    "fnmatch",
    "formatter",
    "fractions",
    "ftplib",
    "functools",
    "gc",
    "getopt",
    "getpass",
    "gettext",
    "glob",
    "grp",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "imaplib",
    "imghdr",
    "imp",
    "importlib",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "keyword",
    "lib2to3",
    "linecache",
    "locale",
    "logging",
    "lzma",
    "mailbox",
    "mailcap",
    "marshal",
    "math",
    "mimetypes",
    "mmap",
    "modulefinder",
    "msilib",
    "msvcrt",
    "multiprocessing",
    "netrc",
    "nis",
    "nntplib",
    "ntpath",
    "numbers",
    "operator",
    "optparse",
    "os",
    "ossaudiodev",
    "parser",
    "pathlib",
    "pdb",
    "pickle",
    "pickletools",
    "pipes",
    "pkgutil",
    "platform",
    "plistlib",
    "poplib",
    "posix",
    "posixpath",
    "pprint",
    "profile",
    "pstats",
    "pty",
    "pwd",
    "py_compile",
    "pyclbr",
    "pydoc",
    "queue",
    "quopri",
    "random",
    "re",
    "readline",
    "reprlib",
    # 导入标准库中的模块和包
    import resource          # 提供对系统资源的访问
    import rlcompleter      # 实现了交互式解释器的自动补全
    import runpy            # 用于运行 Python 脚本的库
    import sched            # 提供调度（定时执行）功能的模块
    import secrets          # 生成安全随机数的模块
    import select           # 提供高级 I/O 多路复用机制的模块
    import selectors        # 提供对高级 I/O 多路复用的封装
    import shelve           # 简单的持久化字典模块
    import shlex            # 用于解析 shell 命令行的模块
    import shutil           # 文件操作相关的高级模块
    import signal           # 提供与信号处理相关的函数
    import site             # Site-specific 文件夹配置模块
    import smtpd            # 实现了 SMTP 服务器的基本框架
    import smtplib          # 发送邮件的 SMTP 协议客户端模块
    import sndhdr           # 识别音频文件格式的模块
    import socket           # 提供了网络通信功能的模块
    import socketserver     # 简化了网络服务器的实现
    import spwd             # 提供获取加密密码模块
    import sqlite3          # Python 内置的轻量级数据库模块
    import sre              # 正则表达式模块
    import sre_compile      # 编译正则表达式模块
    import sre_constants    # 正则表达式常量模块
    import sre_parse        # 解析正则表达式模块
    import ssl              # 提供安全套接字（SSL）功能的模块
    import stat             # 提供解释 stat() 系统调用的功能
    import statistics       # 提供统计学方法的模块
    import string           # 字符串操作模块
    import stringprep       # 实现 Stringprep 框架的模块
    import struct           # 解析和打包原始数据的模块
    import subprocess       # 用于创建和控制子进程的模块
    import sunau            # 音频文件处理模块
    import symbol           # Python 语法分析树中的语法符号（符号表）
    import symtable         # 解析 Python 源代码的符号表
    import sys              # 提供 Python 解释器的运行时环境
    import sysconfig        # 提供 Python 解释器配置信息的模块
    import syslog           # 提供与系统日志相关的功能
    import tabnanny         # 检查 Python 缩进风格的工具
    import tarfile          # 提供处理 tar 文件功能的模块
    import telnetlib        # 提供 Telnet 客户端功能的模块
    import tempfile         # 提供创建临时文件和目录的模块
    import termios          # 提供 POSIX 风格的终端 I/O 控制
    import test             # Python 自带的测试框架
    import textwrap         # 提供文本包装和填充的模块
    import threading        # 提供多线程编程的模块
    import time             # 提供时间处理功能的模块
    import timeit           # 提供测量小段代码执行时间的模块
    import tkinter          # Python 的标准 GUI 库
    import token            # Python 词法分析的标记常量
    import tokenize         # 用于词法分析 Python 源代码的模块
    import trace            # 用于跟踪 Python 程序的执行
    import traceback        # 提供提取和格式化异常回溯信息的功能
    import tracemalloc      # 跟踪内存分配的模块
    import tty              # 提供操作终端设备的模块
    import turtle           # 绘制图形的 Turtle 图形库
    import turtledemo       # Turtle 图形库的演示程序
    import types            # 动态创建和控制 Python 类和对象的模块
    import typing           # Python 的类型提示模块
    import unicodedata      # 提供对 Unicode 字符数据库的访问
    import unittest         # Python 自带的单元测试框架
    import urllib           # 提供 URL 处理功能的模块
    import uu               # 提供 uu 编码和解码的模块
    import uuid             # 提供 UUID 对象和功能的模块
    import venv             # 提供创建虚拟环境的功能
    import warnings         # 控制警告消息输出的模块
    import wave             # WAV 文件处理模块
    import weakref          # 提供弱引用对象的模块
    import webbrowser       # 启动默认的 Web 浏览器模块
    import winreg           # 访问 Windows 注册表的模块
    import winsound         # 提供声音播放和控制的模块
    import wsgiref          # WSGI 规范的参考实现
    import xdrlib           # XDR 数据序列化和解析的模块
    import xml              # 提供 XML 处理的模块
    import xmlrpc           # 实现 XML-RPC 协议的模块
    import zipapp           # 创建和执行 Zip 应用的模块
    import zipfile          # 提供读写 ZIP 文件的模块
    import zipimport        # 从 ZIP 归档中导入 Python 模块的模块
    import zlib             # 提供数据压缩功能的模块
# 定义一个名为 stdlib3_9 的集合，包含了 Python 标准库中从 "_thread" 到 "winreg" 的模块名
stdlib3_9 = {
    "_thread",
    "abc",
    "aifc",
    "argparse",
    "array",
    "ast",
    "asynchat",
    "asyncio",
    "asyncore",
    "atexit",
    "audioop",
    "base64",
    "bdb",
    "binascii",
    "binhex",
    "bisect",
    "builtins",
    "bz2",
    "cProfile",
    "calendar",
    "cgi",
    "cgitb",
    "chunk",
    "cmath",
    "cmd",
    "code",
    "codecs",
    "codeop",
    "collections",
    "colorsys",
    "compileall",
    "concurrent",
    "configparser",
    "contextlib",
    "contextvars",
    "copy",
    "copyreg",
    "crypt",
    "csv",
    "ctypes",
    "curses",
    "dataclasses",
    "datetime",
    "dbm",
    "decimal",
    "difflib",
    "dis",
    "distutils",
    "doctest",
    "email",
    "encodings",
    "ensurepip",
    "enum",
    "errno",
    "faulthandler",
    "fcntl",
    "filecmp",
    "fileinput",
    "fnmatch",
    "formatter",
    "fractions",
    "ftplib",
    "functools",
    "gc",
    "getopt",
    "getpass",
    "gettext",
    "glob",
    "graphlib",
    "grp",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "imaplib",
    "imghdr",
    "imp",
    "importlib",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "keyword",
    "lib2to3",
    "linecache",
    "locale",
    "logging",
    "lzma",
    "mailbox",
    "mailcap",
    "marshal",
    "math",
    "mimetypes",
    "mmap",
    "modulefinder",
    "msilib",
    "msvcrt",
    "multiprocessing",
    "netrc",
    "nis",
    "nntplib",
    "ntpath",
    "numbers",
    "operator",
    "optparse",
    "os",
    "ossaudiodev",
    "parser",
    "pathlib",
    "pdb",
    "pickle",
    "pickletools",
    "pipes",
    "pkgutil",
    "platform",
    "plistlib",
    "poplib",
    "posix",
    "posixpath",
    "pprint",
    "profile",
    "pstats",
    "pty",
    "pwd",
    "py_compile",
    "pyclbr",
    "pydoc",
    "queue",
    "quopri",
    "random",
    "re",
    "readline",
    "reprlib",
    "resource",
    "rlcompleter",
    "runpy",
    "sched",
    "secrets",
    "select",
    "selectors",
    "shelve",
    "shlex",
    "shutil",
    "signal",
    "site",
    "smtpd",
    "smtplib",
    "sndhdr",
    "socket",
    "socketserver",
    "spwd",
    "sqlite3",
    "sre",
    "sre_compile",
    "sre_constants",
    "sre_parse",
    "ssl",
    "stat",
    "statistics",
    "string",
    "stringprep",
    "struct",
    "subprocess",
    "sunau",
    "symbol",
    "symtable",
    "sys",
    "sysconfig",
    "syslog",
    "tabnanny",
    "tarfile",
    "telnetlib",
    "tempfile",
    "termios",
    "test",
    "textwrap",
    "threading",
    "time",
    "timeit",
    "tkinter",
    "token",
    "tokenize",
    "trace",
    "traceback",
    "tracemalloc",
    "tty",
    "turtle",
    "turtledemo",
    "types",
    "typing",
    "unicodedata",
    "unittest",
    "urllib",
    "uu",
    "uuid",
    "venv",
    "warnings",
    "wave",
    "weakref",
    "webbrowser",
    "winreg",
}
    # 这些是 Python 标准库中的模块名，每个模块提供了不同的功能和特性。
        "winsound",     # 提供对 Windows 系统声音播放功能的访问
        "wsgiref",      # 提供 WSGI（Web 服务器网关接口）的参考实现和工具
        "xdrlib",       # 支持 XDR（External Data Representation）数据编码和解码
        "xml",          # 提供 XML 处理相关的模块集合
        "xmlrpc",       # 提供 XML-RPC 协议的客户端和服务器实现
        "zipapp",       # 支持创建和运行 zip 格式的可执行 Python 应用
        "zipfile",      # 提供对 ZIP 归档文件的读写支持
        "zipimport",    # 允许直接导入位于 ZIP 文件中的 Python 模块
        "zlib",         # 提供对数据压缩和解压缩的支持
        "zoneinfo",     # 提供对时区数据库和时区信息的访问
}


注释：


# 结束一个代码块，这里匹配了一个开放的大括号 '{'
```