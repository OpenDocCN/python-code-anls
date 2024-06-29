# `.\numpy\numpy\_core\code_generators\verify_c_api_version.py`

```py
#!/usr/bin/env python3
# 导入必要的库
import os
import sys
import argparse

# 自定义异常类，用于处理 C API 版本不匹配的情况
class MismatchCAPIError(ValueError):
    pass

# 获取当前 C API 的哈希值和记录的哈希值
def get_api_versions(apiversion):
    """
    Return current C API checksum and the recorded checksum.

    Return current C API checksum and the recorded checksum for the given
    version of the C API version.

    """
    # 将当前文件的路径添加到系统路径中
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    try:
        # 导入 genapi 模块
        m = __import__('genapi')
        # 导入 numpy_api 模块
        numpy_api = __import__('numpy_api')
        # 计算当前完整 API 的哈希值
        curapi_hash = m.fullapi_hash(numpy_api.full_api)
        # 获取已记录的各个 API 版本的哈希值
        apis_hash = m.get_versions_hash()
    finally:
        # 删除添加的路径，恢复原始状态
        del sys.path[0]

    return curapi_hash, apis_hash[apiversion]

# 检查指定的 C API 版本是否需要更新
def check_api_version(apiversion):
    """Emits a MismatchCAPIWarning if the C API version needs updating."""
    # 获取当前 C API 的哈希值和记录的哈希值
    curapi_hash, api_hash = get_api_versions(apiversion)

    # 如果当前哈希值和记录的哈希值不同，则抛出异常
    if not curapi_hash == api_hash:
        msg = ("API mismatch detected, the C API version "
               "numbers have to be updated. Current C api version is "
               f"{apiversion}, with checksum {curapi_hash}, but recorded "
               f"checksum in _core/codegen_dir/cversions.txt is {api_hash}. "
               "If functions were added in the C API, you have to update "
               f"C_API_VERSION in {__file__}."
               )
        raise MismatchCAPIError(msg)

# 主函数，解析命令行参数并调用检查 API 版本的函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-version",
        type=str,
        help="C API version to verify (as a hex string)"
    )
    args = parser.parse_args()

    check_api_version(int(args.api_version, base=16))

# 如果作为脚本直接执行，则调用主函数
if __name__ == "__main__":
    main()
```