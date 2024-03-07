# `.\YOLO-World\yolo_world\version.py`

```py
# 版权声明
# 版权所有 © 腾讯公司

# 定义版本号
__version__ = '0.1.0'

# 解析版本信息的函数
def parse_version_info(version_str):
    """Parse a version string into a tuple.

    Args:
        version_str (str): The version string.
    Returns:
        tuple[int | str]: The version info, e.g., "1.3.0" is parsed into
            (1, 3, 0), and "2.0.0rc1" is parsed into (2, 0, 0, 'rc1').
    """
    # 初始化版本信息列表
    version_info = []
    # 根据 '.' 分割版本号字符串
    for x in version_str.split('.'):
        # 如果是数字，则转换为整数
        if x.isdigit():
            version_info.append(int(x))
        # 如果包含 'rc'，则分割出补丁版本号
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info.append(int(patch_version[0]))
            version_info.append(f'rc{patch_version[1]}')
    # 返回版本信息元组
    return tuple(version_info)

# 调用解析版本信息函数，得到版本信息元组
version_info = parse_version_info(__version__)

# 导出的变量列表
__all__ = ['__version__', 'version_info', 'parse_version_info']
```