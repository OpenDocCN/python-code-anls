# `.\MinerU\magic_pdf\libs\path_utils.py`

```
# 根据 S3 路径移除非官方的参数
def remove_non_official_s3_args(s3path):
    # 示例: s3://abc/xxxx.json?bytes=0,81350 变为 s3://abc/xxxx.json
    arr = s3path.split("?")  # 通过 "?" 分割 S3 路径
    return arr[0]  # 返回分割后的第一个部分，即无参数的路径

# 解析 S3 路径，返回桶名和键
def parse_s3path(s3path: str):
    # from s3pathlib import S3Path  # 导入 S3Path 类（注释掉的代码）
    # p = S3Path(remove_non_official_s3_args(s3path))  # 创建 S3Path 对象（注释掉的代码）
    # return p.bucket, p.key  # 返回桶名和键（注释掉的代码）
    s3path = remove_non_official_s3_args(s3path).strip()  # 移除参数并去除空格
    if s3path.startswith(('s3://', 's3a://')):  # 检查路径是否以有效的前缀开始
        prefix, path = s3path.split('://', 1)  # 分割前缀和路径
        bucket_name, key = path.split('/', 1)  # 分割桶名和键
        return bucket_name, key  # 返回桶名和键
    elif s3path.startswith('/'):  # 检查路径是否以斜杠开头
        raise ValueError("The provided path starts with '/'. This does not conform to a valid S3 path format.")  # 抛出无效路径错误
    else:  # 处理其他无效情况
        raise ValueError("Invalid S3 path format. Expected 's3://bucket-name/key' or 's3a://bucket-name/key'.")  # 抛出格式错误

# 解析 S3 路径中的字节范围参数
def parse_s3_range_params(s3path: str):
    # 示例: s3://abc/xxxx.json?bytes=0,81350 变为 [0, 81350]
    arr = s3path.split("?bytes=")  # 通过 "?bytes=" 分割 S3 路径
    if len(arr) == 1:  # 如果没有字节参数
        return None  # 返回 None
    return arr[1].split(",")  # 返回字节范围的列表
```