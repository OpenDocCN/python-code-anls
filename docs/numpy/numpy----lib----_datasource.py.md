# `.\numpy\numpy\lib\_datasource.py`

```
"""
A file interface for handling local and remote data files.

The goal of datasource is to abstract some of the file system operations
when dealing with data files so the researcher doesn't have to know all the
low-level details.  Through datasource, a researcher can obtain and use a
file with one function call, regardless of location of the file.

DataSource is meant to augment standard python libraries, not replace them.
It should work seamlessly with standard file IO operations and the os
module.

DataSource files can originate locally or remotely:

- local files : '/home/guido/src/local/data.txt'
- URLs (http, ftp, ...) : 'http://www.scipy.org/not/real/data.txt'

DataSource files can also be compressed or uncompressed.  Currently only
gzip, bz2 and xz are supported.

Example::

    >>> # Create a DataSource, use os.curdir (default) for local storage.
    >>> from numpy import DataSource
    >>> ds = DataSource()
    >>>
    >>> # Open a remote file.
    >>> # DataSource downloads the file, stores it locally in:
    >>> #     './www.google.com/index.html'
    >>> # opens the file and returns a file object.
    >>> fp = ds.open('http://www.google.com/') # doctest: +SKIP
    >>>
    >>> # Use the file as you normally would
    >>> fp.read() # doctest: +SKIP
    >>> fp.close() # doctest: +SKIP

"""
import os

from .._utils import set_module

# Alias for the built-in 'open' function
_open = open


def _check_mode(mode, encoding, newline):
    """Check mode and that encoding and newline are compatible.

    Parameters
    ----------
    mode : str
        File open mode.
    encoding : str
        File encoding.
    newline : str
        Newline for text files.

    """
    if "t" in mode:
        if "b" in mode:
            raise ValueError("Invalid mode: %r" % (mode,))
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")


# Using a class instead of a module-level dictionary
# to reduce the initial 'import numpy' overhead by
# deferring the import of lzma, bz2 and gzip until needed

# TODO: .zip support, .tar support?
class _FileOpeners:
    """
    Container for different methods to open (un-)compressed files.

    `_FileOpeners` contains a dictionary that holds one method for each
    supported file format. Attribute lookup is implemented in such a way
    that an instance of `_FileOpeners` itself can be indexed with the keys
    of that dictionary. Currently uncompressed files as well as files
    compressed with ``gzip``, ``bz2`` or ``xz`` compression are supported.

    Notes
    -----
    `_file_openers`, an instance of `_FileOpeners`, is made available for
    use in the `_datasource` module.

    Examples
    --------
    >>> import gzip
    >>> np.lib._datasource._file_openers.keys()
    [None, '.bz2', '.gz', '.xz', '.lzma']
    >>> np.lib._datasource._file_openers['.gz'] is gzip.open
    True

"""
    """
    初始化函数，设置初始状态和文件打开器字典。
    """
    def __init__(self):
        # 标记对象是否已加载
        self._loaded = False
        # 文件打开器字典，初始包含了一个默认的普通文件打开器
        self._file_openers = {None: open}

    """
    加载支持的压缩文件类型的文件打开器。
    """
    def _load(self):
        # 如果已经加载过，直接返回
        if self._loaded:
            return

        # 尝试导入 bz2 模块并添加对应的打开器
        try:
            import bz2
            self._file_openers[".bz2"] = bz2.open
        except ImportError:
            pass

        # 尝试导入 gzip 模块并添加对应的打开器
        try:
            import gzip
            self._file_openers[".gz"] = gzip.open
        except ImportError:
            pass

        # 尝试导入 lzma 模块并添加对应的打开器
        try:
            import lzma
            self._file_openers[".xz"] = lzma.open
            self._file_openers[".lzma"] = lzma.open
        except (ImportError, AttributeError):
            # 捕获可能的 ImportError 或 AttributeError
            # 兼容性较差的 lzma 后端可能没有 lzma.open 属性
            pass

        # 标记加载完成
        self._loaded = True

    """
    返回当前支持的文件打开器的键列表。
    """
    def keys(self):
        # 确保加载了所有支持的文件打开器
        self._load()
        # 返回文件打开器字典的键列表
        return list(self._file_openers.keys())

    """
    获取指定键对应的文件打开器。
    """
    def __getitem__(self, key):
        # 确保加载了所有支持的文件打开器
        self._load()
        # 返回指定键对应的文件打开器
        return self._file_openers[key]
# 创建一个 _FileOpeners 的实例对象，用于管理文件的打开操作
_file_openers = _FileOpeners()

# 定义一个 open 函数，用于打开文件或 URL，并返回文件对象
def open(path, mode='r', destpath=os.curdir, encoding=None, newline=None):
    """
    Open `path` with `mode` and return the file object.

    If ``path`` is an URL, it will be downloaded, stored in the
    `DataSource` `destpath` directory and opened from there.

    Parameters
    ----------
    path : str or pathlib.Path
        Local file path or URL to open.
    mode : str, optional
        Mode to open `path`. Mode 'r' for reading, 'w' for writing, 'a' to
        append. Available modes depend on the type of object specified by
        path.  Default is 'r'.
    destpath : str, optional
        Path to the directory where the source file gets downloaded to for
        use.  If `destpath` is None, a temporary directory will be created.
        The default path is the current directory.
    encoding : {None, str}, optional
        Open text file with given encoding. The default encoding will be
        what `open` uses.
    newline : {None, str}, optional
        Newline to use when reading text file.

    Returns
    -------
    out : file object
        The opened file.

    Notes
    -----
    This is a convenience function that instantiates a `DataSource` and
    returns the file object from ``DataSource.open(path)``.

    """
    # 创建一个 DataSource 的实例对象
    ds = DataSource(destpath)
    # 调用 DataSource 的 open 方法打开指定的 path，并返回文件对象
    return ds.open(path, mode, encoding=encoding, newline=newline)


# 设置当前模块为 'numpy.lib.npyio'
@set_module('numpy.lib.npyio')
class DataSource:
    """
    DataSource(destpath='.')

    A generic data source file (file, http, ftp, ...).

    DataSources can be local files or remote files/URLs.  The files may
    also be compressed or uncompressed. DataSource hides some of the
    low-level details of downloading the file, allowing you to simply pass
    in a valid file path (or URL) and obtain a file object.

    Parameters
    ----------
    destpath : str or None, optional
        Path to the directory where the source file gets downloaded to for
        use.  If `destpath` is None, a temporary directory will be created.
        The default path is the current directory.

    Notes
    -----
    URLs require a scheme string (``http://``) to be used, without it they
    will fail::

        >>> repos = np.lib.npyio.DataSource()
        >>> repos.exists('www.google.com/index.html')
        False
        >>> repos.exists('http://www.google.com/index.html')
        True

    Temporary directories are deleted when the DataSource is deleted.

    Examples
    --------

    """
    # 初始化方法，创建一个 DataSource 实例，指定下载目录 destpath，默认为当前目录
    def __init__(self, destpath='.'):
        self._destpath = destpath

    # 打开指定 path 的文件或 URL，并返回文件对象
    def open(self, path, mode='r', encoding=None, newline=None):
        """
        Open `path` with `mode` and return the file object.

        Parameters
        ----------
        path : str or pathlib.Path
            Local file path or URL to open.
        mode : str, optional
            Mode to open `path`. Mode 'r' for reading, 'w' for writing, 'a' to
            append. Available modes depend on the type of object specified by
            path.  Default is 'r'.
        encoding : {None, str}, optional
            Open text file with given encoding. The default encoding will be
            what `open` uses.
        newline : {None, str}, optional
            Newline to use when reading text file.

        Returns
        -------
        out : file object
            The opened file.

        Notes
        -----
        This method opens `path` using specified `mode`, handles downloading
        from URL if necessary, and returns the file object.

        """
        pass  # 此处省略了具体的打开文件操作，待实现

    # 检查指定 path 的文件或 URL 是否存在
    def exists(self, path):
        """
        Check if the `path` exists in the data source.

        Parameters
        ----------
        path : str or pathlib.Path
            Local file path or URL to check existence.

        Returns
        -------
        out : bool
            True if `path` exists, False otherwise.

        """
        pass  # 此处省略了具体的检查文件存在性的操作，待实现
    def __init__(self, destpath=os.curdir):
        """Create a DataSource with a local path at destpath."""
        # 如果提供了目标路径，则使用其绝对路径；否则创建临时目录作为目标路径
        if destpath:
            self._destpath = os.path.abspath(destpath)
            self._istmpdest = False
        else:
            import tempfile  # 延迟导入以提升启动速度
            # 创建临时目录作为目标路径
            self._destpath = tempfile.mkdtemp()
            self._istmpdest = True

    def __del__(self):
        # 清理临时目录
        # 如果目标路径是临时目录，则在实例被删除时删除该临时目录
        if hasattr(self, '_istmpdest') and self._istmpdest:
            import shutil
            shutil.rmtree(self._destpath)

    def _iszip(self, filename):
        """Test if the filename is a zip file by looking at the file extension.

        """
        # 检查文件名的扩展名是否在已知的压缩文件扩展名中
        fname, ext = os.path.splitext(filename)
        return ext in _file_openers.keys()

    def _iswritemode(self, mode):
        """Test if the given mode will open a file for writing."""

        # 检查给定的文件打开模式是否允许写入操作
        _writemodes = ("w", "+")
        for c in mode:
            if c in _writemodes:
                return True
        return False

    def _splitzipext(self, filename):
        """Split zip extension from filename and return filename.

        Returns
        -------
        base, zip_ext : {tuple}

        """
        # 分离文件名中的压缩扩展名，并返回文件名及其扩展名
        if self._iszip(filename):
            return os.path.splitext(filename)
        else:
            return filename, None

    def _possible_names(self, filename):
        """Return a tuple containing compressed filename variations."""
        # 返回包含可能的压缩文件名变体的元组
        names = [filename]
        if not self._iszip(filename):
            # 如果文件名不是已知的压缩文件，则为其添加已知的压缩扩展名变体
            for zipext in _file_openers.keys():
                if zipext:
                    names.append(filename + zipext)
        return names

    def _isurl(self, path):
        """Test if path is a net location.  Tests the scheme and netloc."""

        # 减少初始导入时对 'import numpy' 的依赖
        from urllib.parse import urlparse

        # 检查路径是否为网络位置，通过检查其 scheme 和 netloc 来判断
        scheme, netloc, upath, uparams, uquery, ufrag = urlparse(path)
        return bool(scheme and netloc)
    def _cache(self, path):
        """缓存指定路径的文件。

        在数据源缓存中创建文件的副本。

        """
        # 在这里导入是因为导入它们很慢，且占据了 numpy 总导入时间的显著部分。
        import shutil  # 导入 shutil 库，用于文件操作
        from urllib.request import urlopen  # 导入 urlopen 函数，用于打开 URL

        upath = self.abspath(path)  # 获取路径的绝对路径

        # 确保目录存在
        if not os.path.exists(os.path.dirname(upath)):
            os.makedirs(os.path.dirname(upath))  # 创建目录结构

        # TODO: Doesn't handle compressed files!
        if self._isurl(path):  # 如果是 URL
            with urlopen(path) as openedurl:  # 打开 URL
                with _open(upath, 'wb') as f:  # 打开目标文件进行写入操作
                    shutil.copyfileobj(openedurl, f)  # 复制 URL 内容到目标文件
        else:  # 如果是本地文件
            shutil.copyfile(path, upath)  # 复制本地文件到目标路径
        return upath  # 返回目标路径的绝对路径

    def _findfile(self, path):
        """搜索指定路径的文件并返回完整路径（如果找到）。

        如果路径是一个 URL，_findfile 将缓存一个本地副本并返回缓存文件的路径。如果路径是本地文件，_findfile 将返回该本地文件的路径。

        搜索将包括可能的压缩版本文件，并返回找到的第一个文件路径。

        """

        # 构建可能的本地文件路径列表
        if not self._isurl(path):  # 如果不是 URL
            # 有效的本地路径
            filelist = self._possible_names(path)
            # self._destpath 中的路径
            filelist += self._possible_names(self.abspath(path))
        else:  # 如果是 URL
            # 缓存的 URL 在 self._destpath 中
            filelist = self._possible_names(self.abspath(path))
            # 远程 URL
            filelist = filelist + self._possible_names(path)

        for name in filelist:  # 遍历文件列表
            if self.exists(name):  # 如果文件存在
                if self._isurl(name):  # 如果是 URL
                    name = self._cache(name)  # 缓存 URL 文件到本地并更新 name
                return name  # 返回找到的文件路径
        return None  # 如果未找到文件，则返回 None
    def abspath(self, path):
        """
        Return absolute path of file in the DataSource directory.

        If `path` is an URL, then `abspath` will return either the location
        the file exists locally or the location it would exist when opened
        using the `open` method.

        Parameters
        ----------
        path : str or pathlib.Path
            Can be a local file or a remote URL.

        Returns
        -------
        out : str
            Complete path, including the `DataSource` destination directory.

        Notes
        -----
        The functionality is based on `os.path.abspath`.

        """
        # We do this here to reduce the 'import numpy' initial import time.
        from urllib.parse import urlparse

        # TODO:  This should be more robust.  Handles case where path includes
        #        the destpath, but not other sub-paths. Failing case:
        #        path = /home/guido/datafile.txt
        #        destpath = /home/alex/
        #        upath = self.abspath(path)
        #        upath == '/home/alex/home/guido/datafile.txt'

        # handle case where path includes self._destpath
        splitpath = path.split(self._destpath, 2)
        if len(splitpath) > 1:
            path = splitpath[1]
        
        # Parse the URL or path to extract components
        scheme, netloc, upath, uparams, uquery, ufrag = urlparse(path)
        
        # Sanitize netloc and upath to ensure valid file path components
        netloc = self._sanitize_relative_path(netloc)
        upath = self._sanitize_relative_path(upath)
        
        # Construct and return the absolute path using the destination directory
        return os.path.join(self._destpath, netloc, upath)

    def _sanitize_relative_path(self, path):
        """Return a sanitised relative path for which
        os.path.abspath(os.path.join(base, path)).startswith(base)
        """
        last = None
        path = os.path.normpath(path)
        
        # Normalize and sanitize the path iteratively
        while path != last:
            last = path
            # Note: os.path.join treats '/' as os.sep on Windows
            path = path.lstrip(os.sep).lstrip('/')
            path = path.lstrip(os.pardir).lstrip('..')
            drive, path = os.path.splitdrive(path)  # for Windows
        
        # Return the sanitized relative path
        return path
    # 检查路径是否存在的方法
    def exists(self, path):
        """
        Test if path exists.

        Test if `path` exists as (and in this order):

        - a local file.
        - a remote URL that has been downloaded and stored locally in the
          `DataSource` directory.
        - a remote URL that has not been downloaded, but is valid and
          accessible.

        Parameters
        ----------
        path : str or pathlib.Path
            Can be a local file or a remote URL.

        Returns
        -------
        out : bool
            True if `path` exists.

        Notes
        -----
        When `path` is an URL, `exists` will return True if it's either
        stored locally in the `DataSource` directory, or is a valid remote
        URL.  `DataSource` does not discriminate between the two, the file
        is accessible if it exists in either location.

        """

        # First test for local path
        # 首先检查是否为本地文件
        if os.path.exists(path):
            return True

        # We import this here because importing urllib is slow and
        # a significant fraction of numpy's total import time.
        # 在此处导入 urllib，因为导入 urllib 很慢且占用了 numpy 总导入时间的显著部分。
        from urllib.request import urlopen
        from urllib.error import URLError

        # Test cached url
        # 检查缓存的 URL 是否存在
        upath = self.abspath(path)
        if os.path.exists(upath):
            return True

        # Test remote url
        # 检查远程 URL 是否存在
        if self._isurl(path):
            try:
                netfile = urlopen(path)
                netfile.close()
                del(netfile)
                return True
            except URLError:
                return False
        return False
    # 定义一个方法用于打开文件，返回文件对象

    """
    Open and return file-like object.

    If `path` is an URL, it will be downloaded, stored in the
    `DataSource` directory and opened from there.

    Parameters
    ----------
    path : str or pathlib.Path
        Local file path or URL to open.
    mode : {'r', 'w', 'a'}, optional
        Mode to open `path`.  Mode 'r' for reading, 'w' for writing,
        'a' to append. Available modes depend on the type of object
        specified by `path`. Default is 'r'.
    encoding : {None, str}, optional
        Open text file with given encoding. The default encoding will be
        what `open` uses.
    newline : {None, str}, optional
        Newline to use when reading text file.

    Returns
    -------
    out : file object
        File object.
    """

    # TODO: There is no support for opening a file for writing which
    #       doesn't exist yet (creating a file).  Should there be?
    # 提示：没有支持创建尚不存在的文件以进行写入的功能。应该增加这个功能吗？

    # TODO: Add a ``subdir`` parameter for specifying the subdirectory
    #       used to store URLs in self._destpath.
    # 提示：添加一个“subdir”参数，用于指定存储URL在self._destpath中的子目录。

    # 检查路径是否为URL，并且模式是否为写入模式，如果是则抛出异常
    if self._isurl(path) and self._iswritemode(mode):
        raise ValueError("URLs are not writeable")

    # NOTE: _findfile will fail on a new file opened for writing.
    # 注意：_findfile 在尝试打开一个新文件进行写入时会失败。
    found = self._findfile(path)
    if found:
        _fname, ext = self._splitzipext(found)
        # 如果文件有扩展名为 'bz2'，则调整模式（去掉可能存在的 '+' 符号）
        if ext == 'bz2':
            mode.replace("+", "")
        # 使用适当的文件打开器打开文件并返回文件对象
        return _file_openers[ext](found, mode=mode,
                                  encoding=encoding, newline=newline)
    else:
        # 如果找不到文件，则抛出 FileNotFoundError 异常
        raise FileNotFoundError(f"{path} not found.")
class Repository (DataSource):
    """
    Repository(baseurl, destpath='.')

    A data repository where multiple DataSource's share a base
    URL/directory.

    `Repository` extends `DataSource` by prepending a base URL (or
    directory) to all the files it handles. Use `Repository` when you will
    be working with multiple files from one base URL.  Initialize
    `Repository` with the base URL, then refer to each file by its filename
    only.

    Parameters
    ----------
    baseurl : str
        Path to the local directory or remote location that contains the
        data files.
    destpath : str or None, optional
        Path to the directory where the source file gets downloaded to for
        use.  If `destpath` is None, a temporary directory will be created.
        The default path is the current directory.

    Examples
    --------
    To analyze all files in the repository, do something like this
    (note: this is not self-contained code)::

        >>> repos = np.lib._datasource.Repository('/home/user/data/dir/')
        >>> for filename in filelist:
        ...     fp = repos.open(filename)
        ...     fp.analyze()
        ...     fp.close()

    Similarly you could use a URL for a repository::

        >>> repos = np.lib._datasource.Repository('http://www.xyz.edu/data')

    """

    def __init__(self, baseurl, destpath=os.curdir):
        """
        Create a Repository with a shared url or directory of baseurl.

        Parameters
        ----------
        baseurl : str
            The base URL or directory containing the data files.
        destpath : str, optional
            Path to the local directory where files will be stored,
            default is the current directory.
        """
        # 调用父类 DataSource 的初始化方法
        DataSource.__init__(self, destpath=destpath)
        # 设置 Repository 的基础 URL 或目录
        self._baseurl = baseurl

    def __del__(self):
        # 调用父类 DataSource 的析构方法
        DataSource.__del__(self)

    def _fullpath(self, path):
        """
        Return complete path for path.  Prepends baseurl if necessary.

        Parameters
        ----------
        path : str
            Path to the file.

        Returns
        -------
        str
            Complete path, including the base URL or directory.
        """
        # 将给定的路径 path 加上 baseurl 形成完整路径
        splitpath = path.split(self._baseurl, 2)
        if len(splitpath) == 1:
            result = os.path.join(self._baseurl, path)
        else:
            result = path    # path contains baseurl already
        return result

    def _findfile(self, path):
        """
        Extend DataSource method to prepend baseurl to ``path``.

        Parameters
        ----------
        path : str
            Path to the file.

        Returns
        -------
        str
            Complete path with base URL prepended.
        """
        # 调用父类 DataSource 的 _findfile 方法，并将 baseurl 加到 path 前面
        return DataSource._findfile(self, self._fullpath(path))

    def abspath(self, path):
        """
        Return absolute path of file in the Repository directory.

        If `path` is an URL, then `abspath` will return either the location
        the file exists locally or the location it would exist when opened
        using the `open` method.

        Parameters
        ----------
        path : str or pathlib.Path
            Can be a local file or a remote URL. This may, but does not
            have to, include the `baseurl` with which the `Repository` was
            initialized.

        Returns
        -------
        str
            Complete path, including the `DataSource` destination directory.
        """
        # 调用父类 DataSource 的 abspath 方法，传入完整路径
        return DataSource.abspath(self, self._fullpath(path))
    # 检查指定路径是否存在，会根据 Repository 的基本 URL 加上路径进行检查
    def exists(self, path):
        """
        Test if path exists prepending Repository base URL to path.

        Test if `path` exists as (and in this order):

        - a local file.
        - a remote URL that has been downloaded and stored locally in the
          `DataSource` directory.
        - a remote URL that has not been downloaded, but is valid and
          accessible.

        Parameters
        ----------
        path : str or pathlib.Path
            Can be a local file or a remote URL. This may, but does not
            have to, include the `baseurl` with which the `Repository` was
            initialized.

        Returns
        -------
        out : bool
            True if `path` exists.

        Notes
        -----
        When `path` is an URL, `exists` will return True if it's either
        stored locally in the `DataSource` directory, or is a valid remote
        URL.  `DataSource` does not discriminate between the two, the file
        is accessible if it exists in either location.

        """
        # 调用 DataSource 的 exists 方法，传入完整路径来检查路径是否存在
        return DataSource.exists(self, self._fullpath(path))

    # 打开并返回文件对象，会根据 Repository 的基本 URL 处理路径
    def open(self, path, mode='r', encoding=None, newline=None):
        """
        Open and return file-like object prepending Repository base URL.

        If `path` is an URL, it will be downloaded, stored in the
        DataSource directory and opened from there.

        Parameters
        ----------
        path : str or pathlib.Path
            Local file path or URL to open. This may, but does not have to,
            include the `baseurl` with which the `Repository` was
            initialized.
        mode : {'r', 'w', 'a'}, optional
            Mode to open `path`.  Mode 'r' for reading, 'w' for writing,
            'a' to append. Available modes depend on the type of object
            specified by `path`. Default is 'r'.
        encoding : {None, str}, optional
            Open text file with given encoding. The default encoding will be
            what `open` uses.
        newline : {None, str}, optional
            Newline to use when reading text file.

        Returns
        -------
        out : file object
            File object.

        """
        # 调用 DataSource 的 open 方法，传入完整路径来打开文件
        return DataSource.open(self, self._fullpath(path), mode,
                               encoding=encoding, newline=newline)

    # 列出源 Repository 中的文件列表
    def listdir(self):
        """
        List files in the source Repository.

        Returns
        -------
        files : list of str or pathlib.Path
            List of file names (not containing a directory part).

        Notes
        -----
        Does not currently work for remote repositories.

        """
        # 如果基本 URL 是一个 URL，则抛出未实现错误，不支持对 URL 的目录列表操作
        if self._isurl(self._baseurl):
            raise NotImplementedError(
                  "Directory listing of URLs, not supported yet.")
        else:
            # 否则，使用 os.listdir 返回基本 URL 对应目录下的文件列表
            return os.listdir(self._baseurl)
```