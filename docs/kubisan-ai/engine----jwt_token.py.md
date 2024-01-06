# `KubiScan\engine\jwt_token.py`

```
# 导入 base64 模块

# 解码 base64 编码的 JWT token
def decode_base64_jwt_token(base64_token):
    # 解码 base64 编码的 JWT token，并将结果传递给 decode_jwt_token_data 函数
    return decode_jwt_token_data(decode_base64_bytes_to_string(decode_base64(base64_token)))

# 解码 JWT token 数据
def decode_jwt_token_data(jwt_token):
    try:
        # 将 JWT token 按照 "." 分割成三部分，取第二部分进行 base64 解码
        splitted_string = jwt_token.split(".")
        decoded_data_base64 = decode_base64(splitted_string[1])
        # 将解码后的数据转换为字符串
        return decode_base64_bytes_to_string(decoded_data_base64)
    except Exception:
        # 如果出现异常，返回 None
        return None

# 将 base64 编码的字节数据解码为字符串
def decode_base64_bytes_to_string(decoded_data_base64):
    decoded_data = ''
    try:
        # 尝试使用 utf-8 编码解码 base64 数据
        decoded_data = decoded_data_base64.decode("utf-8")
    except Exception as e:
        # 如果出现异常，打印错误信息
        print('[*] An error occured while trying to deocde the JWT token:')
        # 打印异常信息
        print(str(e))
        # 打印提示信息
        print("[*] Decoding the token with latin-1 instead of UTF-8...")
        # 使用 latin-1 解码 base64 编码的数据
        decoded_data = decoded_data_base64.decode("latin-1")
    # 返回解码后的数据
    return decoded_data


def decode_base64(data):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    # 计算缺少的填充字符数量
    missing_padding = len(data) % 4
    # 如果缺少填充字符，则添加相应数量的 '='
    if missing_padding != 0:
        data += '='* (4 - missing_padding)
    # 返回 base64 解码后的数据
    return base64.b64decode(data)
```