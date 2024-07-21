# `.\pytorch\torch\utils\show_pickle.py`

```py
#!/usr/bin/env python3
# mypy: allow-untyped-defs
import sys
import pickle
import struct
import pprint
import zipfile
import fnmatch
from typing import Any, IO, BinaryIO, Union

__all__ = ["FakeObject", "FakeClass", "DumpUnpickler", "main"]

class FakeObject:
    def __init__(self, module, name, args):
        self.module = module
        self.name = name
        self.args = args
        # NOTE: We don't distinguish between state never set and state set to None.
        self.state = None

    def __repr__(self):
        state_str = "" if self.state is None else f"(state={self.state!r})"
        return f"{self.module}.{self.name}{self.args!r}{state_str}"

    def __setstate__(self, state):
        self.state = state

    @staticmethod
    def pp_format(printer, obj, stream, indent, allowance, context, level):
        if not obj.args and obj.state is None:
            stream.write(repr(obj))
            return
        if obj.state is None:
            stream.write(f"{obj.module}.{obj.name}")
            printer._format(obj.args, stream, indent + 1, allowance + 1, context, level)
            return
        if not obj.args:
            stream.write(f"{obj.module}.{obj.name}()(state=\n")
            indent += printer._indent_per_level
            stream.write(" " * indent)
            printer._format(obj.state, stream, indent, allowance + 1, context, level + 1)
            stream.write(")")
            return
        raise Exception("Need to implement")  # noqa: TRY002


class FakeClass:
    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.__new__ = self.fake_new  # type: ignore[assignment]

    def __repr__(self):
        return f"{self.module}.{self.name}"

    def __call__(self, *args):
        return FakeObject(self.module, self.name, args)

    def fake_new(self, *args):
        return FakeObject(self.module, self.name, args[1:])


class DumpUnpickler(pickle._Unpickler):  # type: ignore[name-defined]
    def __init__(
            self,
            file,
            *,
            catch_invalid_utf8=False,
            **kwargs):
        super().__init__(file, **kwargs)
        self.catch_invalid_utf8 = catch_invalid_utf8

    def find_class(self, module, name):
        # 创建并返回 FakeClass 对象，用于反序列化时查找类
        return FakeClass(module, name)

    def persistent_load(self, pid):
        # 返回一个带有持久化加载的 FakeObject 实例
        return FakeObject("pers", "obj", (pid,))

    dispatch = dict(pickle._Unpickler.dispatch)  # type: ignore[attr-defined]

    # Custom objects in TorchScript are able to return invalid UTF-8 strings
    # from their pickle (__getstate__) functions.  Install a custom loader
    # for strings that catches the decode exception and replaces it with
    # a sentinel object.
    # 从数据流中读取一个无符号整数（4字节），表示后续要读取的字符串的长度
    strlen, = struct.unpack("<I", self.read(4))  # type: ignore[attr-defined]
    
    # 如果字符串长度超过系统最大限制，则抛出异常
    if strlen > sys.maxsize:
        raise Exception("String too long.")  # noqa: TRY002
    
    # 根据读取的长度从数据流中读取相应字节数的数据，这些数据代表一个UTF-8编码的字符串
    str_bytes = self.read(strlen)  # type: ignore[attr-defined]
    obj: Any
    
    # 尝试使用UTF-8编解码器解码字节数据，使用"surrogatepass"处理不合法的码位
    try:
        obj = str(str_bytes, "utf-8", "surrogatepass")
    except UnicodeDecodeError as exn:
        # 如果解码失败且允许捕获不合法的UTF-8编码，则创建一个FakeObject对象表示Unicode解码错误
        if not self.catch_invalid_utf8:
            raise
        obj = FakeObject("builtin", "UnicodeDecodeError", (str(exn),))
    
    # 将解码后的对象添加到实例的列表中
    self.append(obj)  # type: ignore[attr-defined]
    
# 将load_binunicode函数注册为处理pickle模块中BINUNICODE类型数据的方法
dispatch[pickle.BINUNICODE[0]] = load_binunicode  # type: ignore[assignment]

# 类方法，从输入流中加载数据并解析，使用pprint将结果格式化输出到指定的输出流，然后返回加载的值
@classmethod
def dump(cls, in_stream, out_stream):
    value = cls(in_stream).load()
    pprint.pprint(value, stream=out_stream)
    return value
# 定义程序的主函数，接受命令行参数和输出流作为参数
def main(argv, output_stream=None):
    # 检查命令行参数是否为两个
    if len(argv) != 2:
        # 如果没有使用标准输出流，避免在标准错误输出重复信息
        if output_stream is not None:
            # 抛出异常，要求传入长度为2的argv
            raise Exception("Pass argv of length 2.")  # noqa: TRY002
        # 向标准错误输出写入使用说明
        sys.stderr.write("usage: show_pickle PICKLE_FILE\n")
        sys.stderr.write("  PICKLE_FILE can be any of:\n")
        sys.stderr.write("    path to a pickle file\n")
        sys.stderr.write("    file.zip@member.pkl\n")
        sys.stderr.write("    file.zip@*/pattern.*\n")
        sys.stderr.write("      (shell glob pattern for members)\n")
        sys.stderr.write("      (only first match will be shown)\n")
        # 返回退出码2，表示参数错误
        return 2

    # 获取文件名参数
    fname = argv[1]
    # 声明一个句柄，可以是字节流IO或者二进制IO
    handle: Union[IO[bytes], BinaryIO]
    # 如果文件名中不包含"@"符号
    if "@" not in fname:
        # 使用只读二进制模式打开文件，并作为句柄传递给DumpUnpickler.dump方法处理
        with open(fname, "rb") as handle:
            DumpUnpickler.dump(handle, output_stream)
    else:
        # 如果文件名中包含"@"符号，则按照ZIP文件处理
        zfname, mname = fname.split("@", 1)
        # 使用zipfile.ZipFile打开ZIP文件对象
        with zipfile.ZipFile(zfname) as zf:
            # 如果通配符"*"不在成员名中
            if "*" not in mname:
                # 直接打开ZIP文件中的指定成员，并作为句柄传递给DumpUnpickler.dump方法处理
                with zf.open(mname) as handle:
                    DumpUnpickler.dump(handle, output_stream)
            else:
                found = False
                # 遍历ZIP文件中的信息列表
                for info in zf.infolist():
                    # 如果文件名匹配通配符模式
                    if fnmatch.fnmatch(info.filename, mname):
                        # 打开匹配的成员，并作为句柄传递给DumpUnpickler.dump方法处理
                        with zf.open(info) as handle:
                            DumpUnpickler.dump(handle, output_stream)
                        found = True
                        break
                # 如果未找到匹配的成员，则抛出异常
                if not found:
                    raise Exception(f"Could not find member matching {mname} in {zfname}")  # noqa: TRY002

if __name__ == "__main__":
    # 这个Hack适用于我测试过的所有Python版本。
    # 我测试过以下版本：
    #   3.7.4
    if True:
        # 将FakeObject.__repr__映射到FakeObject.pp_format，绕过类型检查
        pprint.PrettyPrinter._dispatch[FakeObject.__repr__] = FakeObject.pp_format  # type: ignore[attr-defined]

    # 调用main函数处理命令行参数，并以其返回值作为退出码退出程序
    sys.exit(main(sys.argv))
```