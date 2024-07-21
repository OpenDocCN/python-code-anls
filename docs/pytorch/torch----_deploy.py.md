# `.\pytorch\torch\_deploy.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类
import io  # 导入io模块，用于处理字节流

import torch  # 导入PyTorch库
from torch.package import Importer, OrderedImporter, PackageImporter, sys_importer  # 导入PyTorch的包管理相关模块
from torch.package._package_pickler import create_pickler  # 导入创建pickler的函数
from torch.package._package_unpickler import PackageUnpickler  # 导入PackageUnpickler类
from torch.serialization import _maybe_decode_ascii  # 导入_maybe_decode_ascii函数


# 定义一个函数来保存存储对象
def _save_storages(importer, obj):
    # 初始化空列表来存储序列化后的存储对象和数据类型
    serialized_storages = []
    serialized_dtypes = []

    # 确保importer是PackageImporter类型的对象，如果不是则设为None
    importer = importer if isinstance(importer, torch.package.PackageImporter) else None

    # 创建importers对象，使用OrderedImporter来保证导入顺序
    importers: Importer
    if importer is not None:
        importers = OrderedImporter(importer, sys_importer)
    else:
        importers = sys_importer

    # 定义一个函数persistent_id，用于持久化标识对象
    def persistent_id(obj):
        # 如果是Tensor的存储对象或TypedStorage类型的对象
        if torch.is_storage(obj) or isinstance(obj, torch.storage.TypedStorage):
            if isinstance(obj, torch.storage.TypedStorage):
                # 对于TypedStorage对象，获取其未类型化的存储和数据类型
                storage = obj._untyped_storage
                dtype = obj.dtype
            else:
                # 对于Tensor的存储对象，默认数据类型为torch.uint8
                storage = obj
                dtype = torch.uint8

            # 将存储对象和数据类型添加到对应的序列化列表中，并返回其索引作为持久化标识
            serialized_storages.append(obj)
            serialized_dtypes.append(dtype)
            return ("storage", len(serialized_storages) - 1)

        # 如果对象具有__reduce_deploy__方法，则执行特定的序列化流程
        if hasattr(obj, "__reduce_deploy__"):
            if _serialized_reduces.get(id(obj)) is None:
                _serialized_reduces[id(obj)] = (
                    "reduce_deploy",
                    id(obj),
                    *obj.__reduce_deploy__(importers),
                )
            return _serialized_reduces[id(obj)]

        return None

    # 创建一个字节流缓冲区来存储pickler的序列化数据
    data_buf = io.BytesIO()
    # 创建pickler对象，并设置其持久化标识函数为persistent_id
    pickler = create_pickler(data_buf, importers)
    pickler.persistent_id = persistent_id
    # 对输入的obj执行序列化操作
    pickler.dump(obj)
    # 获取序列化后的数据值
    data_value = data_buf.getvalue()
    # 返回序列化后的数据值、存储对象列表、数据类型列表以及导入器的zip_reader（如果存在）
    return (
        data_value,
        serialized_storages,
        serialized_dtypes,
        importer.zip_reader if importer else None,
    )


# 定义一个函数来加载存储对象
def _load_storages(id, zip_reader, obj_bytes, serialized_storages, serialized_dtypes):
    # 定义一个函数persistent_load，用于加载持久化标识的对象
    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        # 如果typename为"storage"，则根据索引从serialized_storages和serialized_dtypes中恢复TypedStorage对象
        if typename == "storage":
            storage = serialized_storages[data[0]]
            dtype = serialized_dtypes[data[0]]
            return torch.storage.TypedStorage(
                wrap_storage=storage.untyped(), dtype=dtype
            )

        # 如果typename为"reduce_deploy"，则执行特定的反序列化流程
        if typename == "reduce_deploy":
            reduce_id, func, args = data
            if reduce_id not in _loaded_reduces:
                _loaded_reduces[reduce_id] = func(_raw_packages[zip_reader], *args)
            return _loaded_reduces[reduce_id]

        return None

    # 如果zip_reader不为None，则创建OrderedImporter对象用于导入zip_reader中的包
    importer: Importer
    if zip_reader is not None:
        importer = OrderedImporter(_get_package(zip_reader), sys_importer)
    else:
        # 如果未指定 sys_importer，则使用默认的 importer
        importer = sys_importer

    # 创建 PackageUnpickler 对象，用于反序列化 obj_bytes 数据
    unpickler = PackageUnpickler(importer, io.BytesIO(obj_bytes))
    # 设置 unpickler 对象的 persistent_load 方法为指定的 persistent_load 函数
    unpickler.persistent_load = persistent_load  # type: ignore[method-assign]
    # 载入并反序列化 obj_bytes 中的数据，结果存储在 result 变量中
    result = _deploy_objects[id] = unpickler.load()
    # 将结果存储在 _deploy_objects 字典中，并返回该结果
    return result
# 定义一个函数 _get_package，用于获取指定 ZIP 读取器对应的包导入器对象
def _get_package(zip_reader):
    # 如果指定的 ZIP 读取器不在 _raw_packages 字典中
    if zip_reader not in _raw_packages:
        # 创建一个新的 PackageImporter 对象，并将其存储在 _raw_packages 字典中
        _raw_packages[zip_reader] = PackageImporter(zip_reader)
    # 返回存储在 _raw_packages 中的指定 ZIP 读取器对应的包导入器对象
    return _raw_packages[zip_reader]

# 定义几个空字典，用于存储不同类型的对象
_raw_packages: dict = {}
_deploy_objects: dict = {}
_serialized_reduces: dict = {}
_loaded_reduces: dict = {}
```