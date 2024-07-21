# `.\pytorch\torch\_appdirs.py`

```
# 如果当前平台是Java虚拟机，则需要额外处理平台信息
if sys.platform.startswith("java"):
    # 导入平台模块以获取Java虚拟机的操作系统信息
    import platform
    # 获取Java虚拟机的操作系统版本信息，其中版本信息数组的第四个元素是操作系统名称
    os_name = platform.java_ver()[3][0]
    # 根据操作系统名称判断当前系统是否为Windows
    if os_name.startswith("Windows"):  # "Windows XP", "Windows 7", etc.
        system = "win32"
    # 根据操作系统名称判断当前系统是否为Mac OS X
    elif os_name.startswith("Mac"):  # "Mac OS X", etc.
        system = "darwin"
    else:  # 对于其他操作系统，如Linux、SunOS、FreeBSD等，默认为Linux
        # 尽管设置为"linux2"不是理想的做法，但仅检查了Windows和Mac OS X，其余操作系统都使用sys.platform风格的字符串
        system = "linux2"
else:
    # 对于非Java虚拟机的情况，直接使用sys.platform获取系统信息
    system = sys.platform
    # 返回特定应用程序的用户数据目录的完整路径
    
    # 如果系统为 Windows
    if system == "win32":
        # 如果未指定 appauthor，则使用 appname
        if appauthor is None:
            appauthor = appname
        # 根据 roaming 参数选择 CSIDL_APPDATA 或 CSIDL_LOCAL_APPDATA
        const = roaming and "CSIDL_APPDATA" or "CSIDL_LOCAL_APPDATA"
        # 获取标准化路径
        path = os.path.normpath(_get_win_folder(const))
        # 如果存在 appname
        if appname:
            # 如果 appauthor 不为 False，则路径为 CSIDL_APPDATA/appauthor/appname
            # 否则路径为 CSIDL_APPDATA/appname
            if appauthor is not False:
                path = os.path.join(path, appauthor, appname)
            else:
                path = os.path.join(path, appname)
    
    # 如果系统为 macOS
    elif system == "darwin":
        # 获取用户的 Application Support 目录
        path = os.path.expanduser("~/Library/Application Support/")
        # 如果存在 appname，则在路径后加上 appname
        if appname:
            path = os.path.join(path, appname)
    
    # 对于其他系统（Unix-like）
    else:
        # 获取环境变量 XDG_DATA_HOME 的值，如果未设置则默认为 ~/.local/share
        path = os.getenv("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
        # 如果存在 appname，则在路径后加上 appname
        if appname:
            path = os.path.join(path, appname)
    
    # 如果存在 appname 和 version，则在路径后加上 version
    if appname and version:
        path = os.path.join(path, version)
    
    # 返回构建好的最终路径
    return path
# 返回应用程序的用户共享数据目录的完整路径
def site_data_dir(appname=None, appauthor=None, version=None, multipath=False):
    r"""Return full path to the user-shared data dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "multipath" is an optional parameter only applicable to *nix
            which indicates that the entire list of data dirs should be
            returned. By default, the first item from XDG_DATA_DIRS is
            returned, or '/usr/local/share/<AppName>',
            if XDG_DATA_DIRS is not set

    Typical site data directories are:
        Mac OS X:   /Library/Application Support/<AppName>
        Unix:       /usr/local/share/<AppName> or /usr/share/<AppName>
        Win XP:     C:\Documents and Settings\All Users\Application Data\<AppAuthor>\<AppName>
        Vista:      (Fail! "C:\ProgramData" is a hidden *system* directory on Vista.)
        Win 7:      C:\ProgramData\<AppAuthor>\<AppName>   # Hidden, but writeable on Win 7.

    For Unix, this is using the $XDG_DATA_DIRS[0] default.

    WARNING: Do not use this on Windows. See the Vista-Fail note above for why.
    """
    # 如果系统是 Windows
    if system == "win32":
        # 如果未指定 appauthor，则使用 appname
        if appauthor is None:
            appauthor = appname
        # 获取公共应用程序数据路径
        path = os.path.normpath(_get_win_folder("CSIDL_COMMON_APPDATA"))
        # 如果有 appname
        if appname:
            # 如果 appauthor 不为 False，则连接 appauthor 和 appname
            if appauthor is not False:
                path = os.path.join(path, appauthor, appname)
            else:
                # 否则，只连接 appname
                path = os.path.join(path, appname)
    # 如果系统是 macOS
    elif system == "darwin":
        # 扩展用户目录下的应用程序支持目录
        path = os.path.expanduser("/Library/Application Support")
        # 如果有 appname，则连接 appname
        if appname:
            path = os.path.join(path, appname)
    else:
        # 对于 *nix 系统，默认使用 $XDG_DATA_DIRS 的第一个路径
        path = os.getenv(
            "XDG_DATA_DIRS", os.pathsep.join(["/usr/local/share", "/usr/share"])
        )
        # 去除路径末尾的分隔符并拆分为列表
        pathlist = [
            os.path.expanduser(x.rstrip(os.sep)) for x in path.split(os.pathsep)
        ]
        # 如果有 appname
        if appname:
            # 如果有 version，则将 appname 和 version 连接作为新的 appname
            if version:
                appname = os.path.join(appname, version)
            # 对于每个路径在 pathlist 中，连接 appname
            pathlist = [os.sep.join([x, appname]) for x in pathlist]

        # 如果 multipath 为 True，则返回所有路径的分隔符连接
        if multipath:
            path = os.pathsep.join(pathlist)
        else:
            # 否则，返回第一个路径
            path = pathlist[0]
        # 返回最终的路径
        return path

    # 如果同时有 appname 和 version，则连接它们作为新的路径
    if appname and version:
        path = os.path.join(path, version)
    # 返回最终的路径
    return path
# 返回特定应用程序的用户配置目录的完整路径
def user_config_dir(appname=None, appauthor=None, version=None, roaming=False):
    r"""Return full path to the user-specific config dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "roaming" (boolean, default False) can be set True to use the Windows
            roaming appdata directory. That means that for users on a Windows
            network setup for roaming profiles, this user data will be
            sync'd on login. See
            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>
            for a discussion of issues.

    Typical user config directories are:
        Mac OS X:               ~/Library/Preferences/<AppName>
        Unix:                   ~/.config/<AppName>     # or in $XDG_CONFIG_HOME, if defined
        Win *:                  same as user_data_dir

    For Unix, we follow the XDG spec and support $XDG_CONFIG_HOME.
    That means, by default "~/.config/<AppName>".
    """
    # 根据系统类型确定用户配置目录的路径
    if system == "win32":
        # 如果是 Windows 系统，调用 user_data_dir 函数获取用户数据目录
        path = user_data_dir(appname, appauthor, None, roaming)
    elif system == "darwin":
        # 如果是 macOS 系统，设置路径为用户偏好设置目录
        path = os.path.expanduser("~/Library/Preferences/")
        # 如果有应用程序名称，则在路径中添加应用程序名称
        if appname:
            path = os.path.join(path, appname)
    else:
        # 如果是 Unix 系统，根据 XDG 规范设置配置目录路径
        path = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
        # 如果有应用程序名称，则在路径中添加应用程序名称
        if appname:
            path = os.path.join(path, appname)
    # 如果有应用程序名称和版本号，则在路径中添加版本号
    if appname and version:
        path = os.path.join(path, version)
    # 返回最终的配置目录路径
    return path


def site_config_dir(appname=None, appauthor=None, version=None, multipath=False):
    # 根据系统和应用信息返回用户共享数据目录的完整路径
    r"""Return full path to the user-shared data dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "multipath" is an optional parameter only applicable to *nix
            which indicates that the entire list of config dirs should be
            returned. By default, the first item from XDG_CONFIG_DIRS is
            returned, or '/etc/xdg/<AppName>', if XDG_CONFIG_DIRS is not set

    Typical site config directories are:
        Mac OS X:   same as site_data_dir
        Unix:       /etc/xdg/<AppName> or $XDG_CONFIG_DIRS[i]/<AppName> for each value in
                    $XDG_CONFIG_DIRS
        Win *:      same as site_data_dir
        Vista:      (Fail! "C:\ProgramData" is a hidden *system* directory on Vista.)

    For Unix, this is using the $XDG_CONFIG_DIRS[0] default, if multipath=False

    WARNING: Do not use this on Windows. See the Vista-Fail note above for why.
    """
    # 根据不同的操作系统设置路径变量
    if system == "win32":
        # 对于 Windows，调用 site_data_dir 函数获取路径
        path = site_data_dir(appname, appauthor)
        # 如果有应用名和版本信息，则添加版本路径元素
        if appname and version:
            path = os.path.join(path, version)
    elif system == "darwin":
        # 对于 macOS，使用 /Library/Preferences 扩展用户目录
        path = os.path.expanduser("/Library/Preferences")
        # 如果有应用名，则在路径中添加应用名
        if appname:
            path = os.path.join(path, appname)
    else:
        # 对于 Unix-like 系统，获取 $XDG_CONFIG_DIRS 的默认值，如果 multipath=False 则只返回第一个路径
        path = os.getenv("XDG_CONFIG_DIRS", "/etc/xdg")
        # 将路径分割成列表，扩展用户目录并去除尾部分隔符
        pathlist = [
            os.path.expanduser(x.rstrip(os.sep)) for x in path.split(os.pathsep)
        ]
        # 如果有应用名，则根据版本信息修改应用名并在路径列表中添加路径
        if appname:
            if version:
                appname = os.path.join(appname, version)
            pathlist = [os.sep.join([x, appname]) for x in pathlist]

        # 如果 multipath=True，则返回所有路径的字符串表示，否则返回第一个路径
        if multipath:
            path = os.pathsep.join(pathlist)
        else:
            path = pathlist[0]
    # 返回最终的路径
    return path
# 返回特定应用程序的用户专用缓存目录的完整路径
def user_cache_dir(appname=None, appauthor=None, version=None, opinion=True):
    # 当系统为 Windows 时
    if system == "win32":
        # 如果未提供 appauthor，则默认与 appname 相同
        if appauthor is None:
            appauthor = appname
        # 获取 CSIDL_LOCAL_APPDATA 目录的标准化路径
        path = os.path.normpath(_get_win_folder("CSIDL_LOCAL_APPDATA"))
        # 如果提供了 appname
        if appname:
            # 如果 appauthor 不为 False，则将其作为路径的一部分
            if appauthor is not False:
                path = os.path.join(path, appauthor, appname)
            else:
                path = os.path.join(path, appname)
            # 如果 opinion 为 True，则在路径后附加 "Cache"
            if opinion:
                path = os.path.join(path, "Cache")
    # 当系统为 macOS 时
    elif system == "darwin":
        # 获取当前用户的 Library/Caches 目录路径
        path = os.path.expanduser("~/Library/Caches")
        # 如果提供了 appname，则将其作为路径的一部分
        if appname:
            path = os.path.join(path, appname)
    # 对于其他 Unix 类系统
    else:
        # 获取 XDG_CACHE_HOME 环境变量的值，或者默认为 ~/.cache
        path = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        # 如果提供了 appname，则将其作为路径的一部分
        if appname:
            path = os.path.join(path, appname)
    # 如果提供了 appname 和 version，则将 version 作为路径的一部分
    if appname and version:
        path = os.path.join(path, version)
    # 返回最终的路径
    return path


def user_state_dir(appname=None, appauthor=None, version=None, roaming=False):
    r"""Return full path to the user-specific state dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "roaming" (boolean, default False) can be set True to use the Windows
            roaming appdata directory. That means that for users on a Windows
            network setup for roaming profiles, this user data will be
            sync'd on login. See
            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>
            for a discussion of issues.

    Typical user state directories are:
        Mac OS X:  same as user_data_dir
        Unix:      ~/.local/state/<AppName>   # or in $XDG_STATE_HOME, if defined
        Win *:     same as user_data_dir

    For Unix, we follow this Debian proposal <https://wiki.debian.org/XDGBaseDirectorySpecification#state>
    to extend the XDG spec and support $XDG_STATE_HOME.

    That means, by default "~/.local/state/<AppName>".
    """
    # 根据操作系统类型选择合适的用户状态目录路径
    if system in ["win32", "darwin"]:
        # 如果是 Windows 或者 macOS，使用特定函数获取用户数据目录路径
        path = user_data_dir(appname, appauthor, None, roaming)
    else:
        # 对于 Unix 系统，首先尝试获取环境变量 XDG_STATE_HOME 的值，如果不存在则使用默认路径 "~/.local/state"
        path = os.getenv("XDG_STATE_HOME", os.path.expanduser("~/.local/state"))
        # 如果指定了应用名 appname，则将其加入到路径中
        if appname:
            path = os.path.join(path, appname)
    # 如果指定了应用名 appname 和版本 version，则将版本号添加到路径末尾
    if appname and version:
        path = os.path.join(path, version)
    # 返回最终确定的用户状态目录路径
    return path
# 返回特定应用程序的用户日志目录的完整路径
def user_log_dir(appname=None, appauthor=None, version=None, opinion=True):
    r"""Return full path to the user-specific log dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "opinion" (boolean) can be False to disable the appending of
            "Logs" to the base app data dir for Windows, and "log" to the
            base cache dir for Unix. See discussion below.

    Typical user log directories are:
        Mac OS X:   ~/Library/Logs/<AppName>
        Unix:       ~/.cache/<AppName>/log  # or under $XDG_CACHE_HOME if defined
        Win XP:     C:\Documents and Settings\<username>\Local Settings\Application Data\<AppAuthor>\<AppName>\Logs
        Vista:      C:\Users\<username>\AppData\Local\<AppAuthor>\<AppName>\Logs

    On Windows the only suggestion in the MSDN docs is that local settings
    go in the `CSIDL_LOCAL_APPDATA` directory. (Note: I'm interested in
    examples of what some windows apps use for a logs dir.)

    OPINION: This function appends "Logs" to the `CSIDL_LOCAL_APPDATA`
    value for Windows and appends "log" to the user cache dir for Unix.
    This can be disabled with the `opinion=False` option.
    """
    # 根据操作系统类型设置日志路径
    if system == "darwin":
        path = os.path.join(os.path.expanduser("~/Library/Logs"), appname)
    elif system == "win32":
        # 获取用户数据目录，并将 version 设置为 False
        path = user_data_dir(appname, appauthor, version)
        version = False
        # 如果 opinion 为 True，在 Windows 下追加 "Logs" 到路径中
        if opinion:
            path = os.path.join(path, "Logs")
    else:
        # 获取用户缓存目录，并将 version 设置为 False
        path = user_cache_dir(appname, appauthor, version)
        version = False
        # 如果 opinion 为 True，在 Unix 下追加 "log" 到路径中
        if opinion:
            path = os.path.join(path, "log")
    # 如果 appname 和 version 都存在，则将 version 追加到路径中
    if appname and version:
        path = os.path.join(path, version)
    # 返回最终确定的路径
    return path
    # 返回应用程序数据目录路径，根据应用名称、作者、版本号和是否使用多路径来确定
    def site_data_dir(self):
        return site_data_dir(
            self.appname, self.appauthor, version=self.version, multipath=self.multipath
        )

    # 返回用户配置文件目录路径，根据应用名称、作者、版本号和是否漫游来确定
    @property
    def user_config_dir(self):
        return user_config_dir(
            self.appname, self.appauthor, version=self.version, roaming=self.roaming
        )

    # 返回站点配置文件目录路径，根据应用名称、作者、版本号和是否使用多路径来确定
    @property
    def site_config_dir(self):
        return site_config_dir(
            self.appname, self.appauthor, version=self.version, multipath=self.multipath
        )

    # 返回用户缓存目录路径，根据应用名称、作者和版本号来确定
    @property
    def user_cache_dir(self):
        return user_cache_dir(self.appname, self.appauthor, version=self.version)

    # 返回用户状态目录路径，根据应用名称、作者和版本号来确定
    @property
    def user_state_dir(self):
        return user_state_dir(self.appname, self.appauthor, version=self.version)

    # 返回用户日志目录路径，根据应用名称、作者和版本号来确定
    @property
    def user_log_dir(self):
        return user_log_dir(self.appname, self.appauthor, version=self.version)
# ---- internal support stuff

# 从注册表中获取 Windows 文件夹路径的函数，用于 CSIDL_* 命名空间
def _get_win_folder_from_registry(csidl_name):
    """This is a fallback technique at best. I'm not sure if using the
    registry for this guarantees us the correct answer for all CSIDL_*
    names.
    """
    import winreg as _winreg

    # 根据 CSIDL 名称映射对应的 Shell 文件夹名称
    shell_folder_name = {
        "CSIDL_APPDATA": "AppData",
        "CSIDL_COMMON_APPDATA": "Common AppData",
        "CSIDL_LOCAL_APPDATA": "Local AppData",
    }[csidl_name]

    # 打开注册表键，查询相应的 Shell 文件夹路径和类型
    key = _winreg.OpenKey(
        _winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
    )
    dir, type = _winreg.QueryValueEx(key, shell_folder_name)
    return dir


# 使用 pywin32 获取 Windows 文件夹路径的函数
def _get_win_folder_with_pywin32(csidl_name):
    from win32com.shell import shell, shellcon

    # 使用 pywin32 的 SHGetFolderPath 获取指定 CSIDL 文件夹路径
    dir = shell.SHGetFolderPath(0, getattr(shellcon, csidl_name), 0, 0)
    # 尝试将路径转换为 Unicode 格式，因为 SHGetFolderPath 在路径中有 Unicode 数据时不会返回 Unicode 字符串
    try:
        dir = unicode(dir)

        # 如果路径中包含高位字符，则降级为短路径名。参见 Bug 记录：<http://bugs.activestate.com/show_bug.cgi?id=85099>
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


# 使用 ctypes 获取 Windows 文件夹路径的函数
def _get_win_folder_with_ctypes(csidl_name):
    import ctypes

    # 根据 CSIDL 名称映射对应的常量值
    csidl_const = {
        "CSIDL_APPDATA": 26,
        "CSIDL_COMMON_APPDATA": 35,
        "CSIDL_LOCAL_APPDATA": 28,
    }[csidl_name]

    # 创建 Unicode 缓冲区，用于存储路径
    buf = ctypes.create_unicode_buffer(1024)
    # 使用 ctypes 调用 shell32 库的 SHGetFolderPathW 函数获取指定 CSIDL 文件夹路径
    ctypes.windll.shell32.SHGetFolderPathW(None, csidl_const, None, 0, buf)

    # 如果路径中包含高位字符，则降级为短路径名。参见 Bug 记录：<http://bugs.activestate.com/show_bug.cgi?id=85099>
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


# 使用 JNA 获取 Windows 文件夹路径的函数
def _get_win_folder_with_jna(csidl_name):
    import array

    from com.sun import jna
    from com.sun.jna.platform import win32

    # 定义缓冲区大小
    buf_size = win32.WinDef.MAX_PATH * 2
    # 创建数组缓冲区
    buf = array.zeros("c", buf_size)
    shell = win32.Shell32.INSTANCE
    # 使用 JNA 的 Shell32.INSTANCE 的 SHGetFolderPath 函数获取指定 CSIDL 文件夹路径
    shell.SHGetFolderPath(
        None,
        getattr(win32.ShlObj, csidl_name),
        None,
        win32.ShlObj.SHGFP_TYPE_CURRENT,
        buf,
    )
    # 将字符数组转换为字符串，并移除末尾的空字符
    dir = jna.Native.toString(buf.tostring()).rstrip("\0")

    # 如果路径中包含高位字符，则降级为短路径名。参见 Bug 记录：<http://bugs.activestate.com/show_bug.cgi?id=85099>
    has_high_char = False
    for c in dir:
        if ord(c) > 255:
            has_high_char = True
            break
    # 如果存在高字符（Unicode字符），则执行以下操作
    if has_high_char:
        # 使用array模块创建一个零填充的字节数组buf，大小为buf_size
        buf = array.zeros("c", buf_size)
        # 获取win32 Kernel32的实例
        kernel = win32.Kernel32.INSTANCE
        # 如果成功获取到指定目录的短路径名，则执行以下操作
        if kernel.GetShortPathName(dir, buf, buf_size):
            # 将字节数组buf转换为字符串，并移除末尾的空字符'\0'
            dir = jna.Native.toString(buf.tostring()).rstrip("\0")
    
    # 返回处理后的目录路径
    return dir
# 如果操作系统是 Windows ("win32")，则执行以下操作
if system == "win32":
    try:
        # 尝试导入 win32com.shell 模块
        import win32com.shell

        # 如果成功导入，则使用 _get_win_folder_with_pywin32 函数
        _get_win_folder = _get_win_folder_with_pywin32
    except ImportError:
        # 如果导入失败，则执行以下操作
        try:
            # 尝试导入 ctypes 模块的 windll 子模块
            from ctypes import windll

            # 如果成功导入，则使用 _get_win_folder_with_ctypes 函数
            _get_win_folder = _get_win_folder_with_ctypes
        except ImportError:
            # 如果导入失败，则执行以下操作
            try:
                # 尝试导入 com.sun.jna 模块
                import com.sun.jna

                # 如果成功导入，则使用 _get_win_folder_with_jna 函数
                _get_win_folder = _get_win_folder_with_jna
            except ImportError:
                # 如果导入失败，则使用 _get_win_folder_from_registry 函数
                _get_win_folder = _get_win_folder_from_registry

# ---- self test code

# 如果当前脚本被当作主程序运行，则执行以下自测代码
if __name__ == "__main__":
    # 定义应用程序名称和作者
    appname = "MyApp"
    appauthor = "MyCompany"

    # 定义需要获取的属性列表
    props = (
        "user_data_dir",
        "user_config_dir",
        "user_cache_dir",
        "user_state_dir",
        "user_log_dir",
        "site_data_dir",
        "site_config_dir",
    )

    # 输出应用程序的版本信息
    print(f"-- app dirs {__version__} --")

    # 输出带版本号的应用程序目录
    print("-- app dirs (with optional 'version')")
    # 创建 AppDirs 对象，并遍历属性列表获取相应属性的值并打印
    dirs = AppDirs(appname, appauthor, version="1.0")
    for prop in props:
        print(f"{prop}: {getattr(dirs, prop)}")

    # 输出不带版本号的应用程序目录
    print("\n-- app dirs (without optional 'version')")
    # 创建 AppDirs 对象，并遍历属性列表获取相应属性的值并打印
    dirs = AppDirs(appname, appauthor)
    for prop in props:
        print(f"{prop}: {getattr(dirs, prop)}")

    # 输出没有指定 'appauthor' 的应用程序目录
    print("\n-- app dirs (without optional 'appauthor')")
    # 创建 AppDirs 对象，并遍历属性列表获取相应属性的值并打印
    dirs = AppDirs(appname)
    for prop in props:
        print(f"{prop}: {getattr(dirs, prop)}")

    # 输出禁用 'appauthor' 的应用程序目录
    print("\n-- app dirs (with disabled 'appauthor')")
    # 创建 AppDirs 对象，并遍历属性列表获取相应属性的值并打印
    dirs = AppDirs(appname, appauthor=False)
    for prop in props:
        print(f"{prop}: {getattr(dirs, prop)}")
```