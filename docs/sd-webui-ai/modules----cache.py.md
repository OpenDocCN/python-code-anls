# `stable-diffusion-webui\modules\cache.py`

```py
# 导入所需的模块
import json
import os
import os.path
import threading
import time

# 从模块 paths 中导入 data_path 和 script_path
from modules.paths import data_path, script_path

# 获取缓存文件名，如果环境变量中没有指定，则使用默认路径
cache_filename = os.environ.get('SD_WEBUI_CACHE_FILE', os.path.join(data_path, "cache.json"))
# 初始化缓存数据为空
cache_data = None
# 创建一个线程锁对象
cache_lock = threading.Lock()

# 初始化将缓存写入磁盘的时间和线程对象
dump_cache_after = None
dump_cache_thread = None

# 定义将缓存写入磁盘的函数
def dump_cache():
    """
    Marks cache for writing to disk. 5 seconds after no one else flags the cache for writing, it is written.
    """

    global dump_cache_after
    global dump_cache_thread

    # 定义线程函数
    def thread_func():
        global dump_cache_after
        global dump_cache_thread

        # 当标记的时间未到时，线程等待
        while dump_cache_after is not None and time.time() < dump_cache_after:
            time.sleep(1)

        # 获取缓存锁，写入缓存数据到临时文件
        with cache_lock:
            cache_filename_tmp = cache_filename + "-"
            with open(cache_filename_tmp, "w", encoding="utf8") as file:
                json.dump(cache_data, file, indent=4, ensure_ascii=False)

            # 将临时文件替换为正式的缓存文件
            os.replace(cache_filename_tmp, cache_filename)

            dump_cache_after = None
            dump_cache_thread = None

    # 获取缓存锁，设置标记时间，并启动线程
    with cache_lock:
        dump_cache_after = time.time() + 5
        if dump_cache_thread is None:
            dump_cache_thread = threading.Thread(name='cache-writer', target=thread_func)
            dump_cache_thread.start()


# 定义用于获取或初始化特定子部分缓存的函数
def cache(subsection):
    """
    Retrieves or initializes a cache for a specific subsection.

    Parameters:
        subsection (str): The subsection identifier for the cache.

    Returns:
        dict: The cache data for the specified subsection.
    """

    global cache_data
    # 如果缓存数据为空
    if cache_data is None:
        # 使用缓存锁进行同步
        with cache_lock:
            # 再次检查缓存数据是否为空，避免多线程竞争条件
            if cache_data is None:
                # 如果缓存文件不存在
                if not os.path.isfile(cache_filename):
                    # 初始化缓存数据为空字典
                    cache_data = {}
                else:
                    try:
                        # 尝试以 UTF-8 编码打开缓存文件，加载数据到缓存
                        with open(cache_filename, "r", encoding="utf8") as file:
                            cache_data = json.load(file)
                    except Exception:
                        # 如果出现异常，将当前缓存文件移动到临时目录，并创建新的缓存
                        os.replace(cache_filename, os.path.join(script_path, "tmp", "cache.json"))
                        print('[ERROR] issue occurred while trying to read cache.json, move current cache to tmp/cache.json and create new cache')
                        cache_data = {}

    # 获取指定子部分的缓存数据，如果不存在则返回空字典
    s = cache_data.get(subsection, {})
    # 更新缓存数据中指定子部分的数据
    cache_data[subsection] = s

    # 返回指定子部分的缓存数据
    return s
# 为特定文件检索或生成数据，并使用缓存机制
def cached_data_for_file(subsection, title, filename, func):
    """
    Retrieves or generates data for a specific file, using a caching mechanism.

    Parameters:
        subsection (str): The subsection of the cache to use.
        title (str): The title of the data entry in the subsection of the cache.
        filename (str): The path to the file to be checked for modifications.
        func (callable): A function that generates the data if it is not available in the cache.

    Returns:
        dict or None: The cached or generated data, or None if data generation fails.

    The `cached_data_for_file` function implements a caching mechanism for data stored in files.
    It checks if the data associated with the given `title` is present in the cache and compares the
    modification time of the file with the cached modification time. If the file has been modified,
    the cache is considered invalid and the data is regenerated using the provided `func`.
    Otherwise, the cached data is returned.

    If the data generation fails, None is returned to indicate the failure. Otherwise, the generated
    or cached data is returned as a dictionary.
    """

    # 获取现有缓存
    existing_cache = cache(subsection)
    # 获取文件的最后修改时间
    ondisk_mtime = os.path.getmtime(filename)

    # 检查缓存中是否存在与给定标题相关的数据
    entry = existing_cache.get(title)
    if entry:
        # 获取缓存中的最后修改时间
        cached_mtime = entry.get("mtime", 0)
        # 如果文件的修改时间大于缓存中的修改时间，则缓存无效
        if ondisk_mtime > cached_mtime:
            entry = None

    # 如果缓存中不存在数据或数据中不包含'value'键
    if not entry or 'value' not in entry:
        # 生成数据
        value = func()
        # 如果数据生成失败，则返回None
        if value is None:
            return None

        # 更新缓存中的数据和最后修改时间
        entry = {'mtime': ondisk_mtime, 'value': value}
        existing_cache[title] = entry

        # 将缓存数据写入文件
        dump_cache()

    # 返回生成或缓存的数据
    return entry['value']
```