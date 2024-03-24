# `.\lucidrains\DALLE2-pytorch\dalle2_pytorch\trackers.py`

```
# 导入所需的库
import urllib.request
import os
import json
from pathlib import Path
import shutil
from itertools import zip_longest
from typing import Any, Optional, List, Union
from pydantic import BaseModel

import torch
from dalle2_pytorch.dalle2_pytorch import Decoder, DiffusionPrior
from dalle2_pytorch.utils import import_or_print_error
from dalle2_pytorch.trainer import DecoderTrainer, DiffusionPriorTrainer
from dalle2_pytorch.version import __version__
from packaging import version

# 常量定义
DEFAULT_DATA_PATH = './.tracker-data'

# 辅助函数
def exists(val):
    return val is not None

# 定义基础日志类
class BaseLogger:
    """
    An abstract class representing an object that can log data.
    Parameters:
        data_path (str): A file path for storing temporary data.
        verbose (bool): Whether of not to always print logs to the console.
    """
    def __init__(self, data_path: str, resume: bool = False, auto_resume: bool = False, verbose: bool = False, **kwargs):
        self.data_path = Path(data_path)
        self.resume = resume
        self.auto_resume = auto_resume
        self.verbose = verbose

    def init(self, full_config: BaseModel, extra_config: dict, **kwargs) -> None:
        """
        Initializes the logger.
        Errors if the logger is invalid.
        full_config is the config file dict while extra_config is anything else from the script that is not defined the config file.
        """
        raise NotImplementedError

    def log(self, log, **kwargs) -> None:
        raise NotImplementedError

    def log_images(self, images, captions=[], image_section="images", **kwargs) -> None:
        raise NotImplementedError

    def log_file(self, file_path, **kwargs) -> None:
        raise NotImplementedError

    def log_error(self, error_string, **kwargs) -> None:
        raise NotImplementedError

    def get_resume_data(self, **kwargs) -> dict:
        """
        Sets tracker attributes that along with { "resume": True } will be used to resume training.
        It is assumed that after init is called this data will be complete.
        If the logger does not have any resume functionality, it should return an empty dict.
        """
        raise NotImplementedError

# 定义控制台日志类
class ConsoleLogger(BaseLogger):
    def init(self, full_config: BaseModel, extra_config: dict, **kwargs) -> None:
        print("Logging to console")

    def log(self, log, **kwargs) -> None:
        print(log)

    def log_images(self, images, captions=[], image_section="images", **kwargs) -> None:
        pass

    def log_file(self, file_path, **kwargs) -> None:
        pass

    def log_error(self, error_string, **kwargs) -> None:
        print(error_string)

    def get_resume_data(self, **kwargs) -> dict:
        return {}

# 定义Wandb日志类
class WandbLogger(BaseLogger):
    """
    Logs to a wandb run.
    Parameters:
        data_path (str): A file path for storing temporary data.
        wandb_entity (str): The wandb entity to log to.
        wandb_project (str): The wandb project to log to.
        wandb_run_id (str): The wandb run id to resume.
        wandb_run_name (str): The wandb run name to use.
    """
    def __init__(self,
        data_path: str,
        wandb_entity: str,
        wandb_project: str,
        wandb_run_id: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(data_path, **kwargs)
        self.entity = wandb_entity
        self.project = wandb_project
        self.run_id = wandb_run_id
        self.run_name = wandb_run_name
    # 初始化函数，接受完整配置、额外配置和其他参数，不返回任何内容
    def init(self, full_config: BaseModel, extra_config: dict, **kwargs) -> None:
        # 断言 wandb_entity 必须被指定以使用 wandb 记录器
        assert self.entity is not None, "wandb_entity must be specified for wandb logger"
        # 断言 wandb_project 必须被指定以使用 wandb 记录器
        assert self.project is not None, "wandb_project must be specified for wandb logger"
        # 导入 wandb 模块或打印错误信息
        self.wandb = import_or_print_error('wandb', '`pip install wandb` to use the wandb logger')
        # 设置环境变量 WANDB_SILENT 为 true
        os.environ["WANDB_SILENT"] = "true"
        # 初始化 wandb 运行对象
        init_object = {
            "entity": self.entity,
            "project": self.project,
            "config": {**full_config.dict(), **extra_config}
        }
        # 如果指定了运行名称，则设置到初始化对象中
        if self.run_name is not None:
            init_object['name'] = self.run_name
        # 如果要恢复运行，则设置相应参数
        if self.resume:
            assert self.run_id is not None, '`wandb_run_id` must be provided if `wandb_resume` is True'
            if self.run_name is not None:
                print("You are renaming a run. I hope that is what you intended.")
            init_object['resume'] = 'must'
            init_object['id'] = self.run_id

        # 初始化 wandb 运行
        self.wandb.init(**init_object)
        print(f"Logging to wandb run {self.wandb.run.path}-{self.wandb.run.name}")

    # 记录日志函数
    def log(self, log, **kwargs) -> None:
        # 如果设置了 verbose，则打印日志
        if self.verbose:
            print(log)
        # 记录日志到 wandb
        self.wandb.log(log, **kwargs)

    # 记录图片函数
    def log_images(self, images, captions=[], image_section="images", **kwargs) -> None:
        """
        Takes a tensor of images and a list of captions and logs them to wandb.
        """
        # 创建 wandb 图像对象列表
        wandb_images = [self.wandb.Image(image, caption=caption) for image, caption in zip_longest(images, captions)]
        # 记录图像到 wandb
        self.wandb.log({ image_section: wandb_images }, **kwargs)

    # 记录文件函数
    def log_file(self, file_path, base_path: Optional[str] = None, **kwargs) -> None:
        # 如果未指定基本路径，则将文件路径的父路径作为基本路径
        if base_path is None:
            base_path = Path(file_path).parent
        # 保存文件到 wandb
        self.wandb.save(str(file_path), base_path = str(base_path))

    # 记录错误函数
    def log_error(self, error_string, step=None, **kwargs) -> None:
        # 如果设置了 verbose，则打印错误信息
        if self.verbose:
            print(error_string)
        # 记录错误信息到 wandb
        self.wandb.log({"error": error_string, **kwargs}, step=step)

    # 获取恢复数据函数
    def get_resume_data(self, **kwargs) -> dict:
        # 为了恢复运行，需要 wandb_entity、wandb_project 和 wandb_run_id
        return {
            "entity": self.entity,
            "project": self.project,
            "run_id": self.wandb.run.id
        }
# 定义一个字典，将不同的日志类型映射到对应的日志类
logger_type_map = {
    'console': ConsoleLogger,
    'wandb': WandbLogger,
}

# 创建日志记录器的函数，根据日志类型选择对应的日志类进行实例化
def create_logger(logger_type: str, data_path: str, **kwargs) -> BaseLogger:
    # 如果日志类型为'custom'，则抛出未实现错误
    if logger_type == 'custom':
        raise NotImplementedError('Custom loggers are not supported yet. Please use a different logger type.')
    try:
        # 根据日志类型从映射字典中获取对应的日志类
        logger_class = logger_type_map[logger_type]
    except KeyError:
        # 如果日志类型未知，则抛出数值错误
        raise ValueError(f'Unknown logger type: {logger_type}. Must be one of {list(logger_type_map.keys())}')
    # 返回实例化的日志类对象
    return logger_class(data_path, **kwargs)

# 定义一个抽象基类，表示可以加载模型检查点的对象
class BaseLoader:
    """
    An abstract class representing an object that can load a model checkpoint.
    Parameters:
        data_path (str): A file path for storing temporary data.
    """
    def __init__(self, data_path: str, only_auto_resume: bool = False, **kwargs):
        self.data_path = Path(data_path)
        self.only_auto_resume = only_auto_resume

    def init(self, logger: BaseLogger, **kwargs) -> None:
        raise NotImplementedError

    def recall() -> dict:
        raise NotImplementedError

# 定义一个从 URL 下载文件并加载的加载器类
class UrlLoader(BaseLoader):
    """
    A loader that downloads the file from a url and loads it
    Parameters:
        data_path (str): A file path for storing temporary data.
        url (str): The url to download the file from.
    """
    def __init__(self, data_path: str, url: str, **kwargs):
        super().__init__(data_path, **kwargs)
        self.url = url

    def init(self, logger: BaseLogger, **kwargs) -> None:
        # 确保要下载的文件存在
        pass  # TODO: Actually implement that

    def recall(self) -> dict:
        # 下载文件
        save_path = self.data_path / 'loaded_checkpoint.pth'
        urllib.request.urlretrieve(self.url, str(save_path))
        # 加载文件
        return torch.load(str(save_path), map_location='cpu')

# 定义一个从本地路径加载文件的加载器类
class LocalLoader(BaseLoader):
    """
    A loader that loads a file from a local path
    Parameters:
        data_path (str): A file path for storing temporary data.
        file_path (str): The path to the file to load.
    """
    def __init__(self, data_path: str, file_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
        self.file_path = Path(file_path)

    def init(self, logger: BaseLogger, **kwargs) -> None:
        # 确保要加载的文件存在
        if not self.file_path.exists() and not self.only_auto_resume:
            raise FileNotFoundError(f'Model not found at {self.file_path}')

    def recall(self) -> dict:
        # 加载文件
        return torch.load(str(self.file_path), map_location='cpu')

# 定义一个从 wandb 运行中加载模型的加载器类
class WandbLoader(BaseLoader):
    """
    A loader that loads a model from an existing wandb run
    """
    def __init__(self, data_path: str, wandb_file_path: str, wandb_run_path: Optional[str] = None, **kwargs):
        super().__init__(data_path, **kwargs)
        self.run_path = wandb_run_path
        self.file_path = wandb_file_path

    def init(self, logger: BaseLogger, **kwargs) -> None:
        self.wandb = import_or_print_error('wandb', '`pip install wandb` to use the wandb recall function')
        # 确保文件可以被下载
        if self.wandb.run is not None and self.run_path is None:
            self.run_path = self.wandb.run.path
            assert self.run_path is not None, 'wandb run was not found to load from. If not using the wandb logger must specify the `wandb_run_path`.'
        assert self.run_path is not None, '`wandb_run_path` must be provided for the wandb loader'
        assert self.file_path is not None, '`wandb_file_path` must be provided for the wandb loader'
        
        os.environ["WANDB_SILENT"] = "true"
        pass  # TODO: Actually implement that

    def recall(self) -> dict:
        file_reference = self.wandb.restore(self.file_path, run_path=self.run_path)
        return torch.load(file_reference.name, map_location='cpu')

# 定义一个字典，将不同的加载器类型映射到对应的加载器类
loader_type_map = {
    'url': UrlLoader,
    'local': LocalLoader,
    # 键为'wandb'，值为WandbLoader的键值对
    'wandb': WandbLoader,
# 结束当前代码块
}

# 创建数据加载器的函数，根据给定的加载器类型和数据路径返回相应的加载器对象
def create_loader(loader_type: str, data_path: str, **kwargs) -> BaseLoader:
    # 如果加载器类型为'custom'，则抛出未实现错误
    if loader_type == 'custom':
        raise NotImplementedError('Custom loaders are not supported yet. Please use a different loader type.')
    # 尝试获取对应加载器类型的加载器类
    try:
        loader_class = loader_type_map[loader_type]
    except KeyError:
        # 如果加载器类型未知，则抛出数值错误
        raise ValueError(f'Unknown loader type: {loader_type}. Must be one of {list(loader_type_map.keys())}')
    # 返回使用给定数据路径和参数初始化的加载器对象
    return loader_class(data_path, **kwargs)

# 基础保存器类
class BaseSaver:
    # 初始化函数
    def __init__(self,
        data_path: str,
        save_latest_to: Optional[Union[str, bool]] = None,
        save_best_to: Optional[Union[str, bool]] = None,
        save_meta_to: Optional[str] = None,
        save_type: str = 'checkpoint',
        **kwargs
    ):
        # 初始化保存器属性
        self.data_path = Path(data_path)
        self.save_latest_to = save_latest_to
        self.saving_latest = save_latest_to is not None and save_latest_to is not False
        self.save_best_to = save_best_to
        self.saving_best = save_best_to is not None and save_best_to is not False
        self.save_meta_to = save_meta_to
        self.saving_meta = save_meta_to is not None
        self.save_type = save_type
        # 断言保存类型为'checkpoint'或'model'
        assert save_type in ['checkpoint', 'model'], '`save_type` must be one of `checkpoint` or `model`'
        # 断言至少有一个保存选项被指定
        assert self.saving_latest or self.saving_best or self.saving_meta, 'At least one saving option must be specified'

    # 初始化函数，抛出未实现错误
    def init(self, logger: BaseLogger, **kwargs) -> None:
        raise NotImplementedError

    # 保存文件函数，抛出未实现错误
    def save_file(self, local_path: Path, save_path: str, is_best=False, is_latest=False, **kwargs) -> None:
        """
        Save a general file under save_meta_to
        """
        raise NotImplementedError

# 本地保存器类，继承自基础保存器类
class LocalSaver(BaseSaver):
    # 初始化函数
    def __init__(self,
        data_path: str,
        **kwargs
    ):
        # 调用父类初始化函数
        super().__init__(data_path, **kwargs)

    # 初始化函数，确保要保存的目录存在
    def init(self, logger: BaseLogger, **kwargs) -> None:
        print(f"Saving {self.save_type} locally")
        # 如果数据路径不存在，则创建目录
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True)

    # 保存文件函数，复制文件到指定路径
    def save_file(self, local_path: str, save_path: str, **kwargs) -> None:
        # 获取保存路径文件名
        save_path_file_name = Path(save_path).name
        # 确保父目录存在
        save_path_parent = Path(save_path).parent
        if not save_path_parent.exists():
            save_path_parent.mkdir(parents=True)
        print(f"Saving {save_path_file_name} {self.save_type} to local path {save_path}")
        # 复制文件到保存路径
        shutil.copy(local_path, save_path)

# Wandb保存器类，继承自基础保存器类
class WandbSaver(BaseSaver):
    # 初始化函数
    def __init__(self, data_path: str, wandb_run_path: Optional[str] = None, **kwargs):
        # 调用父类初始化函数
        super().__init__(data_path, **kwargs)
        self.run_path = wandb_run_path

    # 初始化函数，初始化wandb并确保用户可以上传到此运行
    def init(self, logger: BaseLogger, **kwargs) -> None:
        self.wandb = import_or_print_error('wandb', '`pip install wandb` to use the wandb logger')
        os.environ["WANDB_SILENT"] = "true"
        # 确保用户可以上传到此运行
        if self.run_path is not None:
            entity, project, run_id = self.run_path.split("/")
            self.run = self.wandb.init(entity=entity, project=project, id=run_id)
        else:
            assert self.wandb.run is not None, 'You must be using the wandb logger if you are saving to wandb and have not set `wandb_run_path`'
            self.run = self.wandb.run
        # TODO: 现在实际检查上传是否可行
        print(f"Saving to wandb run {self.run.path}-{self.run.name}")
    # 保存文件到指定路径，并在wandb中记录相同的文件结构
    def save_file(self, local_path: Path, save_path: str, **kwargs) -> None:
        # 获取保存路径中的文件名
        save_path_file_name = Path(save_path).name
        # 打印保存文件的信息，包括文件名、保存类型和wandb运行的路径和名称
        print(f"Saving {save_path_file_name} {self.save_type} to wandb run {self.run.path}-{self.run.name}")
        # 将保存路径设置为数据路径加上保存路径
        save_path = Path(self.data_path) / save_path
        # 创建保存路径的父目录，如果不存在则创建
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # 复制本地文件到保存路径
        shutil.copy(local_path, save_path)
        # 在wandb中保存文件，设置基本路径为数据路径，保存策略为立即保存
        self.run.save(str(save_path), base_path = str(self.data_path), policy='now')
class HuggingfaceSaver(BaseSaver):
    # HuggingfaceSaver 类继承自 BaseSaver 类
    def __init__(self, data_path: str, huggingface_repo: str, token_path: Optional[str] = None, **kwargs):
        # 初始化方法，接受数据路径、Huggingface 仓库、token 路径等参数
        super().__init__(data_path, **kwargs)
        # 调用父类的初始化方法
        self.huggingface_repo = huggingface_repo
        # 设置 Huggingface 仓库
        self.token_path = token_path
        # 设置 token 路径

    def init(self, logger: BaseLogger, **kwargs):
        # 初始化方法，接受 logger 和其他参数
        # 确保用户可以上传到仓库
        self.hub = import_or_print_error('huggingface_hub', '`pip install huggingface_hub` to use the huggingface saver')
        # 导入 huggingface_hub 模块
        try:
            identity = self.hub.whoami()  # Errors if not logged in
            # 获取当前用户信息，如果未登录则报错
            # 然后表示已登录
        except:
            # 如果未登录，使用 token_path 设置 token
            if not os.path.exists(self.token_path):
                raise Exception("Not logged in to huggingface and no token_path specified. Please login with `huggingface-cli login` or if that does not work set the token_path.")
            with open(self.token_path, "r") as f:
                token = f.read().strip()
            self.hub.HfApi.set_access_token(token)
            identity = self.hub.whoami()
        print(f"Saving to huggingface repo {self.huggingface_repo}")
        # 打印保存到 Huggingface 仓库的信息

    def save_file(self, local_path: Path, save_path: str, **kwargs) -> None:
        # 保存文件到 Huggingface 很简单，只需要上传文件并指定正确的名称
        save_path_file_name = Path(save_path).name
        # 获取保存路径的文件名
        print(f"Saving {save_path_file_name} {self.save_type} to huggingface repo {self.huggingface_repo}")
        # 打印保存文件的信息
        self.hub.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=str(save_path),
            repo_id=self.huggingface_repo
        )
        # 上传文件到 Huggingface 仓库

saver_type_map = {
    'local': LocalSaver,
    'wandb': WandbSaver,
    'huggingface': HuggingfaceSaver
}
# 不同的保存类型映射到不同的 Saver 类

def create_saver(saver_type: str, data_path: str, **kwargs) -> BaseSaver:
    # 创建 Saver 对象的方法，接受保存类型、数据路径和其他参数
    if saver_type == 'custom':
        raise NotImplementedError('Custom savers are not supported yet. Please use a different saver type.')
    # 如果是自定义类型，则抛出未实现错误
    try:
        saver_class = saver_type_map[saver_type]
    except KeyError:
        raise ValueError(f'Unknown saver type: {saver_type}. Must be one of {list(saver_type_map.keys())}')
    # 获取对应保存类型的 Saver 类
    return saver_class(data_path, **kwargs)
    # 返回创建的 Saver 对象

class Tracker:
    # Tracker 类
    def __init__(self, data_path: Optional[str] = DEFAULT_DATA_PATH, overwrite_data_path: bool = False, dummy_mode: bool = False):
        # 初始化方法，接受数据路径、是否覆盖数据路径和是否为虚拟模式等参数
        self.data_path = Path(data_path)
        # 设置数据路径为给定的路径
        if not dummy_mode:
            # 如果不是虚拟模式
            if not overwrite_data_path:
                assert not self.data_path.exists(), f'Data path {self.data_path} already exists. Set overwrite_data_path to True to overwrite.'
                # 断言数据路径不存在，如果存在则报错
                if not self.data_path.exists():
                    self.data_path.mkdir(parents=True)
        # 如果数据路径不存在，则创建该路径
        self.logger: BaseLogger = None
        # 初始化 logger 为 None
        self.loader: Optional[BaseLoader] = None
        # 初始化 loader 为 None
        self.savers: List[BaseSaver]= []
        # 初始化 savers 为空列表
        self.dummy_mode = dummy_mode
        # 设置虚拟模式标志
    def _load_auto_resume(self) -> bool:
        # 加载自动恢复数据
        # 如果文件不存在，则返回 False。如果自动恢复已启用，则打印警告，以便用户知道这是第一次运行。
        if not self.auto_resume_path.exists():
            if self.logger.auto_resume:
                print("Auto_resume is enabled but no auto_resume.json file exists. Assuming this is the first run.")
            return False

        # 现在我们知道自动恢复文件存在，但如果我们不自动恢复，我们应该删除它，以免下次意外加载它
        if not self.logger.auto_resume:
            print(f'Removing auto_resume.json because auto_resume is not enabled in the config')
            self.auto_resume_path.unlink()
            return False

        # 否则，我们将将 JSON 读入字典，将覆盖 logger.__dict__ 的部分
        with open(self.auto_resume_path, 'r') as f:
            auto_resume_dict = json.load(f)
        # 检查记录器是否与自动恢复保存的类型相同
        if auto_resume_dict["logger_type"] != self.logger.__class__.__name__:
            raise Exception(f'The logger type in the auto_resume file is {auto_resume_dict["logger_type"]} but the current logger is {self.logger.__class__.__name__}. Either use the original logger type, set `auto_resume` to `False`, or delete your existing tracker-data folder.')
        # 然后我们准备用自动恢复保存覆盖记录器
        self.logger.__dict__["resume"] = True
        print(f"Updating {self.logger.__dict__} with {auto_resume_dict}")
        self.logger.__dict__.update(auto_resume_dict)
        return True

    def _save_auto_resume(self):
        # 从记录器获取自动恢复字典，并将 "logger_type" 添加到其中，然后将其保存到 auto_resume 文件
        auto_resume_dict = self.logger.get_resume_data()
        auto_resume_dict['logger_type'] = self.logger.__class__.__name__
        with open(self.auto_resume_path, 'w') as f:
            json.dump(auto_resume_dict, f)

    def init(self, full_config: BaseModel, extra_config: dict):
        self.auto_resume_path = self.data_path / 'auto_resume.json'
        # 检查是否恢复运行
        self.did_auto_resume = self._load_auto_resume()
        if self.did_auto_resume:
            print(f'\n\nWARNING: RUN HAS BEEN AUTO-RESUMED WITH THE LOGGER TYPE {self.logger.__class__.__name__}.\nIf this was not your intention, stop this run and set `auto_resume` to `False` in the config.\n\n')
            print(f"New logger config: {self.logger.__dict__}")
        
        self.save_metadata = dict(
            version = version.parse(__version__)
        )  # 将保存在检查点或模型旁��的数据
        self.blacklisted_checkpoint_metadata_keys = ['scaler', 'optimizer', 'model', 'version', 'step', 'steps']  # 如果尝试将它们保存为元数据，这些键将导致我们出错

        assert self.logger is not None, '`logger` must be set before `init` is called'
        if self.dummy_mode:
            # 我们唯一需要的是一个加载器
            if self.loader is not None:
                self.loader.init(self.logger)
            return
        assert len(self.savers) > 0, '`savers` must be set before `init` is called'

        self.logger.init(full_config, extra_config)
        if self.loader is not None:
            self.loader.init(self.logger)
        for saver in self.savers:
            saver.init(self.logger)

        if self.logger.auto_resume:
            # 然后我们需要保存自动恢复文件。假定在调用 logger.init 后，记录器已准备好保存。
            self._save_auto_resume()

    def add_logger(self, logger: BaseLogger):
        self.logger = logger

    def add_loader(self, loader: BaseLoader):
        self.loader = loader

    def add_saver(self, saver: BaseSaver):
        self.savers.append(saver)
    # 记录日志，如果处于虚拟模式，则直接返回
    def log(self, *args, **kwargs):
        if self.dummy_mode:
            return
        # 调用logger对象的log方法记录日志
        self.logger.log(*args, **kwargs)
    
    # 记录图片日志，如果处于虚拟模式，则直接返回
    def log_images(self, *args, **kwargs):
        if self.dummy_mode:
            return
        # 调用logger对象的log_images方法记录图片日志
        self.logger.log_images(*args, **kwargs)

    # 记录文件日志，如果处于虚拟模式，则直接返回
    def log_file(self, *args, **kwargs):
        if self.dummy_mode:
            return
        # 调用logger对象的log_file方法记录文件日志
        self.logger.log_file(*args, **kwargs)

    # 保存配置文件，如果处于虚拟模式，则直接返回
    def save_config(self, current_config_path: str, config_name = 'config.json'):
        if self.dummy_mode:
            return
        # 将当前配置文件复制到data_path根目录下的config_name文件中
        shutil.copy(current_config_path, self.data_path / config_name)
        # 遍历所有savers，如果saver正在保存元数据，则将当前配置文件保存到指定路径下
        for saver in self.savers:
            if saver.saving_meta:
                remote_path = Path(saver.save_meta_to) / config_name
                saver.save_file(current_config_path, str(remote_path))

    # 添加保存元数据，用于与模型或解码器一起保存
    def add_save_metadata(self, state_dict_key: str, metadata: Any):
        """
        Adds a new piece of metadata that will be saved along with the model or decoder.
        """
        # 将元数据添加到save_metadata字典中
        self.save_metadata[state_dict_key] = metadata

    # 保存状态字典，根据保存类型和文件路径保存状态字典
    def _save_state_dict(self, trainer: Union[DiffusionPriorTrainer, DecoderTrainer], save_type: str, file_path: str, **kwargs) -> Path:
        """
        Gets the state dict to be saved and writes it to file_path.
        If save_type is 'checkpoint', we save the entire trainer state dict.
        If save_type is 'model', we save only the model state dict.
        """
        assert save_type in ['checkpoint', 'model']
        if save_type == 'checkpoint':
            # 创建不包含黑名单键的元数据字典，以便在创建状态字典时不出错
            metadata = {k: v for k, v in self.save_metadata.items() if k not in self.blacklisted_checkpoint_metadata_keys}
            # 保存整个trainer状态字典
            trainer.save(file_path, overwrite=True, **kwargs, **metadata)
        elif save_type == 'model':
            if isinstance(trainer, DiffusionPriorTrainer):
                prior = trainer.ema_diffusion_prior.ema_model if trainer.use_ema else trainer.diffusion_prior
                prior: DiffusionPrior = trainer.accelerator.unwrap_model(prior)
                # 如果模型中包含CLIP，则移除CLIP
                original_clip = prior.clip
                prior.clip = None
                model_state_dict = prior.state_dict()
                prior.clip = original_clip
            elif isinstance(trainer, DecoderTrainer):
                decoder: Decoder = trainer.accelerator.unwrap_model(trainer.decoder)
                # 如果模型中包含CLIP，则移除CLIP
                original_clip = decoder.clip
                decoder.clip = None
                if trainer.use_ema:
                    trainable_unets = decoder.unets
                    decoder.unets = trainer.unets  # 交换EMA unets
                    model_state_dict = decoder.state_dict()
                    decoder.unets = trainable_unets  # 恢复原始unets
                else:
                    model_state_dict = decoder.state_dict()
                decoder.clip = original_clip
            else:
                raise NotImplementedError('Saving this type of model with EMA mode enabled is not yet implemented. Actually, how did you get here?')
            # 构建状态字典，包含save_metadata和模型的state_dict
            state_dict = {
                **self.save_metadata,
                'model': model_state_dict
            }
            # 将状态字典保存到文件路径中
            torch.save(state_dict, file_path)
        return Path(file_path)
    # 保存训练器的状态和模型到指定路径
    def save(self, trainer, is_best: bool, is_latest: bool, **kwargs):
        # 如果处于虚拟模式，则直接返回
        if self.dummy_mode:
            return
        # 如果既不是最佳模型也不是最新模型，则无需保存
        if not is_best and not is_latest:
            # 无需执行任何操作
            return
        # 保存检查点和模型到指定路径
        checkpoint_path = self.data_path / 'checkpoint.pth'
        self._save_state_dict(trainer, 'checkpoint', checkpoint_path, **kwargs)
        model_path = self.data_path / 'model.pth'
        self._save_state_dict(trainer, 'model', model_path, **kwargs)
        print("Saved cached models")
        # 调用保存器的保存方法
        for saver in self.savers:
            local_path = checkpoint_path if saver.save_type == 'checkpoint' else model_path
            # 如果需要保存最新模型且当前为最新模型，则保存最新模型
            if saver.saving_latest and is_latest:
                latest_checkpoint_path = saver.save_latest_to.format(**kwargs)
                try:
                    saver.save_file(local_path, latest_checkpoint_path, is_latest=True, **kwargs)
                except Exception as e:
                    self.logger.log_error(f'Error saving checkpoint: {e}', **kwargs)
                    print(f'Error saving checkpoint: {e}')
            # 如果需要保存最佳模型且当前为最佳模型，则保存最佳模型
            if saver.saving_best and is_best:
                best_checkpoint_path = saver.save_best_to.format(**kwargs)
                try:
                    saver.save_file(local_path, best_checkpoint_path, is_best=True, **kwargs)
                except Exception as e:
                    self.logger.log_error(f'Error saving checkpoint: {e}', **kwargs)
                    print(f'Error saving checkpoint: {e}')
    
    @property
    # 定义是否可以执行回溯操作
    def can_recall(self):
        return self.loader is not None and (not self.loader.only_auto_resume or self.did_auto_resume)
    
    # 执行回溯操作
    def recall(self):
        if self.can_recall:
            return self.loader.recall()
        else:
            raise ValueError('Tried to recall, but no loader was set or auto-resume was not performed.')
```