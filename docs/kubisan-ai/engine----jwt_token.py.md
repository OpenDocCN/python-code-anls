# `KubiScan\engine\jwt_token.py`

```

# 导入 base64 模块

# 解码 base64 编码的 JWT 令牌
def decode_base64_jwt_token(base64_token):
    # 解码 base64 编码的令牌，然后解码 JWT 令牌数据
    return decode_jwt_token_data(decode_base64_bytes_to_string(decode_base64(base64_token)))

# 解码 JWT 令牌数据
def decode_jwt_token_data(jwt_token):
    try:
        # 将 JWT 令牌按照 "." 分割，取第二部分进行 base64 解码
        splitted_string = jwt_token.split(".")
        decoded_data_base64 = decode_base64(splitted_string[1])
        return decode_base64_bytes_to_string(decoded_data_base64)
    except Exception:
        return None

# 将解码后的 base64 字节数据转换为字符串
def decode_base64_bytes_to_string(decoded_data_base64):
    decoded_data = ''
    try:
        decoded_data = decoded_data_base64.decode("utf-8")
    except Exception as e:
        print('[*] An error occured while trying to deocde the JWT token:')
        print(str(e))
        print("[*] Decoding the token with latin-1 instead of UTF-8...")
        decoded_data = decoded_data_base64.decode("latin-1")
    return decoded_data

# 解码 base64 编码的数据
def decode_base64(data):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    # 计算缺少的填充位数，补充 "="
    missing_padding = len(data) % 4
    if missing_padding != 0:
        data += '='* (4 - missing_padding)
    return base64.b64decode(data)

```