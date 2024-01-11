# `ZeroNet\src\lib\bencode_open\__init__.py`

```
def loads(data):
    # 检查输入的数据类型是否为字节流，如果不是则抛出类型错误异常
    if not isinstance(data, bytes):
        raise TypeError("Expected 'bytes' object, got {}".format(type(data)))

    # 初始化偏移量
    offset = 0


    def parseInteger():
        nonlocal offset

        # 偏移量加1
        offset += 1
        # 标记是否已经遇到数字
        had_digit = False
        # 绝对值
        abs_value = 0

        # 符号，默认为正
        sign = 1
        # 如果下一个字节是负号，则将符号置为负，并将偏移量加1
        if data[offset] == ord("-"):
            sign = -1
            offset += 1
        # 遍历数据
        while offset < len(data):
            # 如果遇到'e'表示整数结束
            if data[offset] == ord("e"):
                # 偏移量加1
                offset += 1
                # 如果没有遇到数字，则抛出数值错误异常
                if not had_digit:
                    raise ValueError("Integer without value")
                break
            # 如果是数字，则计算绝对值
            if ord("0") <= data[offset] <= ord("9"):
                abs_value = abs_value * 10 + int(chr(data[offset]))
                had_digit = True
                offset += 1
            else:
                # 如果不是数字，则抛出无效整数异常
                raise ValueError("Invalid integer")
        else:
            # 如果遍历结束仍未遇到'e'，则抛出意外的文件结束异常
            raise ValueError("Unexpected EOF, expected integer")

        # 如果没有遇到数字，则抛出空整数异常
        if not had_digit:
            raise ValueError("Empty integer")

        # 返回带符号的绝对值
        return sign * abs_value


    def parseString():
        nonlocal offset

        # 读取字符串长度
        length = int(chr(data[offset]))
        offset += 1

        # 遍历数据
        while offset < len(data):
            # 如果遇到':'表示字符串长度结束
            if data[offset] == ord(":"):
                offset += 1
                break
            # 如果是数字，则计算字符串长度
            if ord("0") <= data[offset] <= ord("9"):
                length = length * 10 + int(chr(data[offset]))
                offset += 1
            else:
                # 如果不是数字，则抛出无效字符串长度异常
                raise ValueError("Invalid string length")
        else:
            # 如果遍历结束仍未遇到':'，则抛出意外的文件结束异常
            raise ValueError("Unexpected EOF, expected string contents")

        # 如果偏移量加上字符串长度超出数据长度，则抛出意外的文件结束异常
        if offset + length > len(data):
            raise ValueError("Unexpected EOF, expected string contents")
        # 更新偏移量
        offset += length

        # 返回字符串内容
        return data[offset - length:offset]
    # 解析列表类型数据
    def parseList():
        nonlocal offset
        # 偏移量加1
        offset += 1
        # 初始化空列表
        values = []

        # 循环直到偏移量小于数据长度
        while offset < len(data):
            # 如果当前字符是'e'，表示列表结束
            if data[offset] == ord("e"):
                # 偏移量加1
                offset += 1
                # 返回列表内容
                return values
            else:
                # 否则解析列表中的元素并添加到values列表中
                values.append(parse())

        # 如果循环结束仍未遇到列表结束标志'e'，则抛出异常
        raise ValueError("Unexpected EOF, expected list contents")


    # 解析字典类型数据
    def parseDict():
        nonlocal offset
        # 偏移量加1
        offset += 1
        # 初始化空字典
        items = {}

        # 循环直到偏移量小于数据长度
        while offset < len(data):
            # 如果当前字符是'e'，表示字典结束
            if data[offset] == ord("e"):
                # 偏移量加1
                offset += 1
                # 返回字典内容
                return items
            else:
                # 否则解析字典中的键值对并添加到items字典中
                key, value = parse(), parse()
                # 检查键是否为字节字符串
                if not isinstance(key, bytes):
                    raise ValueError("A dict key must be a byte string")
                # 检查是否有重复的键
                if key in items:
                    raise ValueError("Duplicate dict key: {}".format(key))
                items[key] = value

        # 如果循环结束仍未遇到字典结束标志'e'，则抛出异常
        raise ValueError("Unexpected EOF, expected dict contents")


    # 解析数据
    def parse():
        nonlocal offset
        # 如果当前字符是'i'，表示整数
        if data[offset] == ord("i"):
            return parseInteger()
        # 如果当前字符是'l'，表示列表
        elif data[offset] == ord("l"):
            return parseList()
        # 如果当前字符是'd'，表示字典
        elif data[offset] == ord("d"):
            return parseDict()
        # 如果当前字符是数字0-9，表示字符串
        elif ord("0") <= data[offset] <= ord("9"):
            return parseString()

        # 如果遇到未知类型的数据，则抛出异常
        raise ValueError("Unknown type specifier: '{}'".format(chr(data[offset]))

    # 解析结果
    result = parse()

    # 如果偏移量不等于数据长度，表示数据解析未完成，抛出异常
    if offset != len(data):
        raise ValueError("Expected EOF, got {} bytes left".format(len(data) - offset))

    # 返回解析结果
    return result
# 将数据转换为 Bencode 编码的字节流
def dumps(data):
    result = bytearray()

    # 定义内部函数，用于递归转换数据
    def convert(data):
        nonlocal result

        # 如果数据是字符串，则抛出数值错误
        if isinstance(data, str):
            raise ValueError("bencode only supports bytes, not str. Use encode")

        # 如果数据是字节流，则将其长度和数据本身编码后添加到结果中
        if isinstance(data, bytes):
            result += str(len(data)).encode() + b":" + data
        # 如果数据是整数，则将其编码后添加到结果中
        elif isinstance(data, int):
            result += b"i" + str(data).encode() + b"e"
        # 如果数据是列表，则将其编码后添加到结果中
        elif isinstance(data, list):
            result += b"l"
            for val in data:
                convert(val)
            result += b"e"
        # 如果数据是字典，则将其编码后添加到结果中
        elif isinstance(data, dict):
            result += b"d"
            for key in sorted(data.keys()):
                # 如果字典的键不是字节流，则抛出数值错误
                if not isinstance(key, bytes):
                    raise ValueError("Dict key can only be bytes, not {}".format(type(key)))
                convert(key)
                convert(data[key])
            result += b"e"
        # 如果数据类型不是支持的类型，则抛出数值错误
        else:
            raise ValueError("bencode only supports bytes, int, list and dict")

    # 调用内部函数进行数据转换
    convert(data)

    # 返回结果字节流
    return bytes(result)
```