# `.\cogvideo-finetune\sat\sgm\webds.py`

```py
# 导入所需的标准库
import sys  # 系统相关的功能
import io  # 输入输出操作
import os  # 操作系统功能
import re  # 正则表达式操作
import json  # JSON 数据处理
import tarfile  # TAR 文件处理
from functools import partial  # 偏函数应用

# 导入 webdataset 库的相关模块
import webdataset as wds  # webdataset 的主模块
from webdataset import ResampledShards, DataPipeline, tarfile_to_samples  # 导入特定功能
from webdataset.filters import pipelinefilter  # 导入过滤功能
from webdataset.tariterators import url_opener, group_by_keys  # 导入 TAR 迭代器相关功能
from webdataset.handlers import reraise_exception  # 导入异常处理功能
from webdataset.gopen import gopen_schemes, gopen  # 导入打开函数和方案

def pytorch_worker_info(group=None):  # sourcery skip: use-contextlib-suppress
    """返回 PyTorch 和一些分布式环境的节点和工作者信息。"""
    rank = 0  # 初始化节点秩
    world_size = 1  # 初始化世界大小
    worker = 0  # 初始化工作者 ID
    num_workers = 1  # 初始化工作者数量
    try:
        import torch.distributed  # 导入分布式 PyTorch 模块

        # 检查分布式模块是否可用和已初始化
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            group = group or torch.distributed.group.WORLD  # 设置组为 WORLD
            rank = torch.distributed.get_rank(group=group)  # 获取节点秩
            world_size = torch.distributed.get_world_size(group=group)  # 获取世界大小
    except ModuleNotFoundError:
        pass  # 如果未找到模块，则跳过
    try:
        import torch.utils.data  # 导入数据工具模块

        worker_info = torch.utils.data.get_worker_info()  # 获取工作者信息
        if worker_info is not None:  # 如果工作者信息存在
            worker = worker_info.id  # 获取工作者 ID
            num_workers = worker_info.num_workers  # 获取工作者总数
    except ModuleNotFoundError:
        pass  # 如果未找到模块，则跳过

    return rank, world_size, worker, num_workers  # 返回节点信息

def pytorch_worker_seed(group=None):
    """为每个工作者和节点计算唯一且确定性的随机种子。"""
    rank, world_size, worker, num_workers = pytorch_worker_info(group=group)  # 获取工作者信息
    return rank * 1000 + worker  # 计算并返回随机种子

def worker_seed_sat(group=None, seed=0):
    return pytorch_worker_seed(group=group) + seed * 23  # 计算工作者的随机种子并增加偏移

class ConfiguredResampledShards(ResampledShards):
    def __init__(self, urls, seed, nshards=sys.maxsize, deterministic=True):
        from sat.helpers import print_rank0  # 导入打印功能

        try:
            from megatron.core.parallel_state import get_data_parallel_group  # 尝试导入 Megatron 数据并行组

            group = get_data_parallel_group()  # 获取数据并行组
            print_rank0("Using megatron data parallel group.")  # 打印使用的组信息
        except:
            from sat.mpu import get_data_parallel_group  # 导入备用的数据并行组

            try:
                group = get_data_parallel_group()  # 获取数据并行组
                print_rank0("Using sat data parallel group.")  # 打印使用的组信息
            except AssertionError:
                group = None  # 如果没有指定组，则设置为 None
                print_rank0("No data parallel group is specified!")  # 打印警告信息
        worker_seed_sat_this = partial(worker_seed_sat, group=group, seed=seed)  # 创建偏函数
        super().__init__(urls, nshards, worker_seed_sat_this, deterministic)  # 调用父类构造函数

class SimpleDistributedWebDataset(DataPipeline):  # 定义简单的分布式 Web 数据集类
    # 初始化方法，接收路径、处理函数、种子以及可选的洗牌缓冲区大小
    def __init__(self, path, process_fn, seed, *, shuffle_buffer=1000):
        # 如果将 shuffle_buffer 设置为 1，则禁用洗牌，模型并行将会有所不同
        try:
            # 从 sat.mpu 模块导入获取模型并行世界大小的函数
            from sat.mpu import get_model_parallel_world_size
    
            # 检查模型并行世界大小，如果大于 1，则将洗牌缓冲区设置为 1
            if get_model_parallel_world_size() > 1:
                shuffle_buffer = 1
        except Exception:
            # 捕获异常并忽略
            pass
        # 调用父类构造函数，初始化配置的重采样分片及相关参数
        super().__init__(
            ConfiguredResampledShards(path, seed),  # 使用指定路径和种子初始化重采样分片，推荐使用多个分片以避免不均匀
            tarfile_to_samples(),  # 将 tar 文件转换为样本
            wds.shuffle(shuffle_buffer),  # 使用洗牌函数，并传入洗牌缓冲区大小
            process_fn,  # 传入处理函数
        )
# 定义一个迭代器函数，用于遍历 tar 文件，生成文件名和内容的对
def tar_file_iterator_with_meta(
    # 输入的字节流对象，适用于 tarfile
    fileobj, 
    # 元数据文件中不同项的键
    meta_names, 
    # 用于跳过某些键的正则表达式（默认值为 r"__[^/]*__($|/)"）
    skip_meta=r"__[^/]*__($|/)", 
    # 文件后缀名（可选）
    suffix=None, 
    # 异常处理的处理程序（默认为 reraise_exception）
    handler=reraise_exception, 
    # 元数据流（可选）
    meta_stream=None
):
    """遍历 tar 文件，返回给定 tar 流的文件名和内容对。

    :param fileobj: 适用于 tarfile 的字节流
    :param meta_names: 元数据文件中不同项的键
    :param skip_meta: 完全跳过的键的正则表达式（默认值 = r"__[^/]*__($|/)"）
    """
    # 打开 tar 文件流，以读取模式
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    # 从文件对象中提取数据目录和文件名
    data_dir, filename = fileobj.name.rsplit("/", 1)
    # 初始化元数据字典，用于存储元数据
    meta_data = {}  # {id: {meta_name: meta_value, meta_name2: meta_value2, ...}}

    # 如果没有提供元数据流
    if meta_stream is None:
        # 生成元数据文件名，使用文件名的前缀加上 ".meta.jsonl"
        meta_file_name = filename.split(".")[0] + ".meta.jsonl"
        # 构建元数据文件的完整路径
        meta_path = os.path.join(data_dir, meta_file_name)
        # 如果元数据文件存在，则打开文件
        if os.path.exists(meta_path):
            meta_stream = open(meta_path, "r")
    else:
        # 如果提供了元数据流，则使用其名称
        meta_file_name = meta_stream.name

    # 如果元数据流存在
    if meta_stream is not None:
        # 遍历元数据流的每一行，记录行号和内容
        for lineno, line in enumerate(meta_stream):
            meta_list = []
            try:
                # 尝试将行内容解析为 JSON 对象
                meta_list.append(json.loads(line))
            except Exception as exn:
                # 导入帮助函数以打印错误
                from sat.helpers import print_rank0

                # 打印解析 JSONL 时的错误信息
                print_rank0(f"Error in loading jsonl {meta_file_name}, lineno {lineno}: {line}", level="DEBUG")
                # 继续下一行
                continue
            # 遍历解析出的每个元数据项
            for item in meta_list:
                # 如果元数据项的键不存在于元数据字典中，则初始化它
                if not item["key"] in meta_data:
                    meta_data[item["key"]] = {}
                # 遍历所有指定的元数据名称
                for meta_name in meta_names:
                    # 如果元数据项中包含该元数据名称，则将其值存储在元数据字典中
                    if meta_name in item:
                        meta_data[item["key"]][meta_name] = item[meta_name]
        # 关闭元数据流
        meta_stream.close()
    # 尝试处理流中的每个项目
        try:
            # 遍历流中的每个 tarinfo 对象
            for tarinfo in stream:
                # 获取当前项目的文件名
                fname = tarinfo.name
                try:
                    # 如果不是常规文件，则跳过
                    if not tarinfo.isreg():
                        continue
                    # 如果文件名为空，则跳过
                    if fname is None:
                        continue
                    # 跳过以双下划线开头和结尾的元数据文件
                    if "/" not in fname and fname.startswith("__") and fname.endswith("__"):
                        # 目前跳过元数据
                        continue
                    # 如果指定了 skip_meta，且文件名匹配，则跳过
                    if skip_meta is not None and re.match(skip_meta, fname):
                        continue
                    # 如果文件是 txt 类型且有后缀，则读取内容并附加后缀
                    if fname.endswith(".txt") and suffix is not None:
                        data = (stream.extractfile(tarinfo).read().decode() + suffix).encode()
                    else:
                        # 否则仅读取文件内容
                        data = stream.extractfile(tarinfo).read()
                    # 创建包含文件名和数据的字典
                    result = dict(fname=fname, data=data)
                    # 生成结果字典
                    yield result
    
                    # 如果文件名以 .id 结尾
                    if fname.endswith(".id"):
                        # 获取文件 ID，去掉扩展名
                        fid = fname.split(".")[0]
                        # 检查文件 ID 是否包含特定字符串并处理
                        if "-$#%@&" in fid:
                            sfid = fid.split("-$#%@&")[0]
                        else:
                            sfid = fid
                        # 从元数据中获取相关数据
                        meta_data_fid = meta_data.get(sfid, {})
                        # 遍历元数据名称
                        for meta_name in meta_names:
                            # 构建元数据文件名
                            meta_fname = fid + "." + meta_name
                            # 获取元数据内容
                            meta = meta_data_fid.get(meta_name, None)
                            # 生成包含元数据的字典
                            yield dict(fname=meta_fname, data=meta)
                    # 清空流的成员列表
                    stream.members = []
                except Exception as exn:
                    # 如果异常有参数，则附加当前文件对象信息
                    if hasattr(exn, "args") and len(exn.args) > 0:
                        exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
                    # 处理异常，如果处理成功则继续
                    if handler(exn):
                        continue
                    else:
                        # 否则跳出循环
                        break
        except Exception as exn:
            # 打印外层异常信息
            print(exn)
        # 删除流对象以释放资源
        del stream
# 扩展一个打开的 tar 文件流，并返回包含文件内容的迭代器
def tar_file_expander_with_meta(data, meta_names, handler=reraise_exception):
    """Expand a stream of open tar files into a stream of tar file contents.

    This returns an iterator over (filename, file_contents).
    """
    # 遍历输入数据中的每个源
    for source in data:
        # 从源字典中获取 URL
        url = source["url"]
        try:
            # 确保源是字典类型
            assert isinstance(source, dict)
            # 确保源字典中包含 "stream" 键
            assert "stream" in source
            # 遍历 tar 文件内容生成器
            for sample in tar_file_iterator_with_meta(source["stream"], meta_names, meta_stream=source["meta_stream"]):
                # 确保样本是字典并包含 "data" 和 "fname"
                assert isinstance(sample, dict) and "data" in sample and "fname" in sample
                # 将 URL 添加到样本字典中
                sample["__url__"] = url
                # 生成样本
                yield sample
        except Exception as exn:
            # 追加流和 URL 到异常参数中
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            # 如果处理异常的函数返回真，继续循环
            if handler(exn):
                continue
            else:
                # 否则，退出循环
                break


# 打开 URL 并返回 URL 和流的配对迭代器
def url_opener(
    data,
    handler,
    **kw,
):
    """Open URLs and yield a stream of url+stream pairs.

    Args:
        data: iterator over dict(url=...)
        handler: exception handler.
        kw: keyword arguments for gopen.gopen.

    Yields:
        a stream of url+stream pairs.
    """
    # 遍历输入数据中的每个样本
    for sample in data:
        # 确保样本是字典类型
        assert isinstance(sample, dict), sample
        # 确保样本字典中包含 "url" 键
        assert "url" in sample
        # 从样本中获取 URL
        url = sample["url"]
        try:
            # 打开 URL 并获取流
            stream = gopen(url, **kw)
            # 检查流是否有 meta_stream 属性
            if hasattr(stream, "meta_stream"):
                # 获取 meta_stream，并删除该属性
                meta_stream = stream.meta_stream
                del stream.meta_stream
            else:
                # 如果没有，则设为 None
                meta_stream = None
            # 更新样本字典，包含流和 meta_stream
            sample.update(stream=stream, meta_stream=meta_stream)
            # 生成样本
            yield sample
        except Exception as exn:
            # 追加 URL 到异常参数中
            exn.args = exn.args + (url,)
            # 如果处理异常的函数返回真，继续循环
            if handler(exn):
                continue
            else:
                # 否则，退出循环
                break


# 使用元数据扩展 tar 文件样本
def tarfile_samples_with_meta(src, meta_names, handler=reraise_exception):
    # 使用 URL 打开器获取流
    streams = url_opener(src, handler=handler)
    # 扩展 tar 文件流并获取文件样本
    files = tar_file_expander_with_meta(streams, meta_names, handler)
    # 按键对样本进行分组
    samples = group_by_keys(files, handler=handler)
    # 返回样本
    return samples


# 定义带有元信息文件的分布式 Web 数据集类
class MetaDistributedWebDataset(DataPipeline):
    """WebDataset with meta information files
    Extra Format:
        in webdataset (tar), for each sample there is a '.id';
        for each tar file, there is a '.meta.jsonl' file with the same name;
        The '.meta.jsonl' file contains lines of json objects, each with a 'key' field to match '.id'.
    """

    # 初始化方法，设置数据集参数
    def __init__(
        self, path, process_fn, seed, *, meta_names=[], nshards=sys.maxsize, shuffle_buffer=1000, include_dirs=None
    ):
        # 设置环境变量，控制是否显示种子（注释掉）
        # os.environ['WDS_SHOW_SEED'] = '1'
        # 导入 PyTorch 库
        import torch

        # 检查当前进程是否为主进程
        if torch.distributed.get_rank() == 0:
            # 如果包含的目录不为 None
            if include_dirs is not None:  # /webdatasets/A,/webdatasets/C
                # 初始化其他路径列表
                other_paths = []
                # 将包含的目录字符串按逗号分割
                include_dirs = include_dirs.split(",")
                # 遍历每个包含的目录
                for include_dir in include_dirs:
                    # 如果目录名中包含通配符 "*"
                    if "*" in include_dir:
                        # 分割目录名和数量
                        include_dir, n = include_dir.split("*")
                        n = int(n)  # 转换数量为整数
                    else:
                        n = 1  # 默认数量为 1
                    # 遍历当前目录及其子目录
                    for cur_dir, dirs, files in os.walk(include_dir):
                        # 遍历所有文件
                        for f in files:
                            # 检查文件是否以 "tar" 结尾且文件大小大于 0
                            if f.endswith("tar") and os.path.getsize(os.path.join(cur_dir, f)) > 0:
                                # 将符合条件的文件路径添加到其他路径列表中
                                # other_paths.append(os.path.join(cur_dir,f))
                                other_paths.extend([os.path.join(cur_dir, f)] * n)  # 根据数量扩展列表
                # print(f'Adding dataset paths {",".join(other_paths)}')
                # 从 braceexpand 库导入
                from braceexpand import braceexpand

                # 如果路径字符串不为空
                if len(path) > 0:  # not ""
                    # 扩展路径并与其他路径合并
                    path = list(braceexpand(path)) + other_paths
                else:
                    # 如果路径为空，仅使用其他路径
                    path = other_paths
            # 将路径包装成列表
            path = [path]
        else:
            # 如果不是主进程，将路径设置为 None
            path = [
                None,
            ]
        # 广播路径列表到所有进程
        torch.distributed.broadcast_object_list(path, src=0)
        # 选择第一个路径
        path = path[0]

        # 生成带元数据的 tar 文件样本处理函数
        tarfile_samples = partial(tarfile_samples_with_meta, meta_names=meta_names)
        # 对 tar 文件样本进行管道过滤
        tarfile_to_samples = pipelinefilter(tarfile_samples)

        # 如果模型并行，设置打乱缓冲区大小为 1 以禁用打乱
        try:
            # 从 sat.mpu 模块导入获取模型并行世界大小的函数
            from sat.mpu import get_model_parallel_world_size

            # 检查模型并行世界大小是否大于 1
            if get_model_parallel_world_size() > 1:
                shuffle_buffer = 1  # 设置打乱缓冲区大小
        except Exception:
            pass  # 忽略导入错误

        # 调用父类初始化方法，传入配置好的重采样分片、样本处理管道及其他参数
        super().__init__(
            ConfiguredResampledShards(path, seed, nshards=nshards),
            tarfile_to_samples(),
            wds.shuffle(shuffle_buffer),
            process_fn,
        )
# rclone 支持
from webdataset.gopen import Pipe  # 从 webdataset.gopen 导入 Pipe 类


def gopen_rclone(url, mode="rb", bufsize=1024 * 1024 * 32):
    """使用 `curl` 打开一个 URL。

    :param url: rclone URL，例如 data:bucket1/foo.tar，数据需要被配置。
    :param mode: 文件模式
    :param bufsize: 缓冲区大小
    """
    # 去掉 URL 前缀 "rclone://"
    url = url.replace("rclone://", "")
    # 如果模式以 "r" 开头，准备读取命令
    if mode[0] == "r":
        cmd = f"rclone cat '{url}'"  # 生成 rclone 读取命令
        return Pipe(  # 返回 Pipe 对象用于读取
            cmd,
            mode=mode,  # 设置文件模式
            shell=True,  # 通过 shell 执行命令
            bufsize=bufsize,  # 设置缓冲区大小
            ignore_status=[141, 23],  # 忽略特定的返回状态
        )  # skipcq: BAN-B604
    # 如果模式以 "w" 开头，准备写入命令
    elif mode[0] == "w":
        cmd = f"rclone cp - '{url}'"  # 生成 rclone 写入命令
        return Pipe(  # 返回 Pipe 对象用于写入
            cmd,
            mode=mode,  # 设置文件模式
            shell=True,  # 通过 shell 执行命令
            bufsize=bufsize,  # 设置缓冲区大小
            ignore_status=[141, 26],  # 忽略特定的返回状态
        )  # skipcq: BAN-B604
    else:
        # 如果模式未知，抛出错误
        raise ValueError(f"{mode}: unknown mode")


def gopen_boto3(url, mode="rb", bufsize=8192 * 2):
    """使用 boto3 API 打开一个 URL。

    :param url: boto3 URL，例如 boto3://bucket1/foo.tar，数据需要被配置。
    :param mode: 文件模式
    :param bufsize: 缓冲区大小
    """
    import boto3  # 导入 boto3 库

    # boto3.set_stream_logger('botocore', level='DEBUG')  # 设置日志记录（已注释）
    # 如果 URL 以 "boto3://" 开头，去掉前缀并标记是否需要元数据
    if url.startswith("boto3://"):
        url = url.replace("boto3://", "")
        need_meta = False
    else:
        url = url.replace("metaboto3://", "")  # 去掉 "metaboto3://" 前缀
        need_meta = True  # 需要元数据

    # 从环境变量获取 S3 配置
    endpoint_url = os.environ.get("S3_ENDPOINT_URL", None)  # S3 端点 URL
    access_key = os.environ.get("S3_ACCESS_KEY_ID", None)  # 访问密钥 ID
    secret_key = os.environ.get("S3_SECRET_ACCESS_KEY", None)  # 秘密访问密钥

    # 如果模式以 "r" 开头，准备读取 S3 对象
    if mode[0] == "r":
        # 创建 S3 客户端
        s3_client = boto3.client(
            "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )
        # 分割桶名和对象键
        bucket, key = url.split("/", 1)

        # 如果需要元数据，下载相应的元数据文件
        if need_meta:
            # 下载一个元数据 JSON 文件
            meta_file_key = key.split(".")[0] + ".meta.jsonl"  # 元数据文件名
            meta_stream = io.BytesIO()  # 创建一个字节流用于存储元数据
            s3_client.download_fileobj(bucket, meta_file_key, meta_stream)  # 从 S3 下载元数据
            meta_stream.seek(0)  # 重置字节流位置
            meta_stream.name = meta_file_key  # 设置字节流名称
        else:
            meta_stream = None  # 不需要元数据时设置为 None

        # 获取数据流对象
        response = s3_client.get_object(Bucket=bucket, Key=key)  # 获取 S3 对象
        response["Body"].name = key  # 设置对象的名称（实际未使用）
        response["Body"].meta_stream = meta_stream  # 将元数据流关联到对象
        return response["Body"]  # 返回数据流对象
    else:
        # 如果模式未知，抛出错误
        raise ValueError(f"{mode}: unknown mode")


# 注册 gopen_rclone 和 gopen_boto3 到 gopen_schemes 字典
gopen_schemes["rclone"] = gopen_rclone
gopen_schemes["boto3"] = gopen_boto3
gopen_schemes["metaboto3"] = gopen_boto3
```