# `D:\src\scipysrc\seaborn\seaborn\external\appdirs.py`

```
# 导入系统相关的模块
import sys
# 导入操作系统相关的功能
import os

# 将字符串类型指定为unicode，以便向后兼容
unicode = str

# 如果运行环境是Java
if sys.platform.startswith('java'):
    # 导入平台相关的模块
    import platform
    # 获取Java虚拟机版本信息
    os_name = platform.java_ver()[3][0]
    # 根据操作系统名称判断系统类型
    if os_name.startswith('Windows'):  # 如果是Windows系统
        system = 'win32'
    elif os_name.startswith('Mac'):  # 如果是Mac系统
        system = 'darwin'
    else:  # 如果是其他Unix系统
        # 当前代码只检查Windows和Mac系统，对于其他系统，将其视为Linux系统
        system = 'linux2'
else:
    # 使用当前系统的平台标识
    system = sys.platform


def user_cache_dir(appname=None, appauthor=None, version=None, opinion=True):
    # 返回此应用程序的用户特定缓存目录的完整路径。
    
    # "appname" 是应用程序的名称。
    # 如果为 None，则仅返回系统目录。
    # "appauthor"（仅在 Windows 上使用）是此应用程序的 appauthor 或分发实体的名称。
    # 通常是所属公司的名称。如果未提供，则默认为 appname。您可以传递 False 禁用它。
    
    # "version" 是要附加到路径的可选版本路径元素。
    # 如果您希望您的应用程序的多个版本能够独立运行，则可能会使用此选项。
    # 如果使用，则通常为 "<major>.<minor>"。仅当存在 appname 时应用。
    
    # "opinion"（布尔值）可以设置为 False，以禁用在 Windows 的基本应用数据目录后附加 "Cache"。
    # 有关详细讨论，请参阅下面的说明。
    
    # 典型的用户缓存目录包括：
    #   Mac OS X：   ~/Library/Caches/<AppName>
    #   Unix：       ~/.cache/<AppName>（XDG 默认）
    #   Win XP：     C:\Documents and Settings\<username>\Local Settings\Application Data\<AppAuthor>\<AppName>\Cache
    #   Vista：      C:\Users\<username>\AppData\Local\<AppAuthor>\<AppName>\Cache
    
    # 在 Windows 上，MSDN 文档中唯一建议的是本地设置应放在 `CSIDL_LOCAL_APPDATA` 目录中。
    # 这与非漫游应用数据目录相同（由上述 `user_data_dir` 默认返回）。
    # 应用程序通常会将缓存数据放在此处指定目录的 *下面*。一些示例：
    #   ...\Mozilla\Firefox\Profiles\<ProfileName>\Cache
    #   ...\Acme\SuperApp\Cache\1.0
    
    # OPINION: 此函数将 "Cache" 附加到 `CSIDL_LOCAL_APPDATA` 值中。
    # 可以通过选项 `opinion=False` 禁用此行为。
    
    if system == "win32":
        if appauthor is None:
            appauthor = appname
        path = os.path.normpath(_get_win_folder("CSIDL_LOCAL_APPDATA"))
        if appname:
            if appauthor is not False:
                path = os.path.join(path, appauthor, appname)
            else:
                path = os.path.join(path, appname)
            if opinion:
                path = os.path.join(path, "Cache")
    elif system == 'darwin':
        path = os.path.expanduser('~/Library/Caches')
        if appname:
            path = os.path.join(path, appname)
    else:
        path = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
        if appname:
            path = os.path.join(path, appname)
    if appname and version:
        path = os.path.join(path, version)
    # 返回构建的缓存路径
    return path
#---- internal support stuff

# 从注册表获取 Windows 文件夹路径。这种方法仅作为备用，不能保证对所有 CSIDL_* 名称都能返回正确的结果。
def _get_win_folder_from_registry(csidl_name):
    import winreg as _winreg

    # 根据 csidl_name 获取相应的 Shell 文件夹名称
    shell_folder_name = {
        "CSIDL_APPDATA": "AppData",
        "CSIDL_COMMON_APPDATA": "Common AppData",
        "CSIDL_LOCAL_APPDATA": "Local AppData",
    }[csidl_name]

    # 打开注册表中的特定路径
    key = _winreg.OpenKey(
        _winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
    )
    # 查询注册表键值，获取文件夹路径和类型信息
    dir, type = _winreg.QueryValueEx(key, shell_folder_name)
    return dir


# 使用 pywin32 获取 Windows 文件夹路径
def _get_win_folder_with_pywin32(csidl_name):
    from win32com.shell import shellcon, shell

    # 使用 shellcon 和 shell 模块获取文件夹路径
    dir = shell.SHGetFolderPath(0, getattr(shellcon, csidl_name), 0, 0)
    
    # 尝试将路径转换为 Unicode 格式，因为 SHGetFolderPath 在路径中含有 Unicode 数据时不会返回 Unicode 字符串。
    try:
        dir = unicode(dir)

        # 如果路径中包含高位字符，则将其转换为短路径名
        has_high_char = False
        for c in dir:
            if ord(c) > 255:
                has_high_char = True
                break
        if has_high_char:
            try:
                import win32api
                dir = win32api.GetShortPathName(dir)
            except ImportError:
                pass
    except UnicodeError:
        pass
    return dir


# 使用 ctypes 获取 Windows 文件夹路径
def _get_win_folder_with_ctypes(csidl_name):
    import ctypes

    # 定义 CSIDL 常量对应的整数值
    csidl_const = {
        "CSIDL_APPDATA": 26,
        "CSIDL_COMMON_APPDATA": 35,
        "CSIDL_LOCAL_APPDATA": 28,
    }[csidl_name]

    # 创建一个 Unicode 缓冲区来存储文件夹路径
    buf = ctypes.create_unicode_buffer(1024)
    # 使用 ctypes 调用 WinAPI 函数 SHGetFolderPathW 获取文件夹路径
    ctypes.windll.shell32.SHGetFolderPathW(None, csidl_const, None, 0, buf)

    # 如果路径中包含高位字符，则将其转换为短路径名
    has_high_char = False
    for c in buf:
        if ord(c) > 255:
            has_high_char = True
            break
    if has_high_char:
        buf2 = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(buf.value, buf2, 1024):
            buf = buf2

    return buf.value


# 使用 JNA 获取 Windows 文件夹路径
def _get_win_folder_with_jna(csidl_name):
    import array
    from com.sun import jna
    from com.sun.jna.platform import win32

    # 定义缓冲区大小
    buf_size = win32.WinDef.MAX_PATH * 2
    # 创建一个字节数组缓冲区
    buf = array.zeros('c', buf_size)
    shell = win32.Shell32.INSTANCE
    # 使用 JNA 调用 WinAPI 函数 SHGetFolderPath 获取文件夹路径
    shell.SHGetFolderPath(None, getattr(win32.ShlObj, csidl_name), None, win32.ShlObj.SHGFP_TYPE_CURRENT, buf)
    # 将字节数组转换为字符串，并移除末尾的空字符
    dir = jna.Native.toString(buf.tostring()).rstrip("\0")

    # 如果路径中包含高位字符，则将其转换为短路径名
    has_high_char = False
    for c in dir:
        if ord(c) > 255:
            has_high_char = True
            break
    # 如果存在高字符（Unicode字符），则执行以下代码块
    if has_high_char:
        # 使用 array.zeros() 创建一个空的字符数组，大小为 buf_size
        buf = array.zeros('c', buf_size)
        # 获取 Windows 平台上的 Kernel32 实例
        kernel = win32.Kernel32.INSTANCE
        # 如果成功获取到短路径名，则执行以下代码块
        if kernel.GetShortPathName(dir, buf, buf_size):
            # 将 buf 转换为字符串，移除末尾的空字符（\0）
            dir = jna.Native.toString(buf.tostring()).rstrip("\0")
    
    # 返回处理后的目录路径
    return dir
# 如果操作系统是 Windows (标识为 "win32")，则执行以下代码块
if system == "win32":
    # 尝试导入 win32com.shell 模块，用于获取 Windows 文件夹路径
    try:
        import win32com.shell
        # 如果导入成功，则选择使用带有 pywin32 的方法获取 Windows 文件夹路径
        _get_win_folder = _get_win_folder_with_pywin32
    # 如果导入失败，则处理 ImportError 异常
    except ImportError:
        # 尝试导入 ctypes.windll 模块，用于备选的 Windows 文件夹路径获取方法
        try:
            from ctypes import windll
            # 如果导入成功，则选择使用带有 ctypes 的方法获取 Windows 文件夹路径
            _get_win_folder = _get_win_folder_with_ctypes
        # 如果导入失败，则处理 ImportError 异常
        except ImportError:
            # 尝试导入 com.sun.jna 模块，用于另一个备选的 Windows 文件夹路径获取方法
            try:
                import com.sun.jna
                # 如果导入成功，则选择使用带有 JNA 的方法获取 Windows 文件夹路径
                _get_win_folder = _get_win_folder_with_jna
            # 如果导入失败，则处理 ImportError 异常
            except ImportError:
                # 如果所有方法均导入失败，则使用注册表中的方法获取 Windows 文件夹路径
                _get_win_folder = _get_win_folder_from_registry
```