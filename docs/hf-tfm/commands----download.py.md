# `.\commands\download.py`

```py
# 导入模块 argparse 中的 ArgumentParser 类
from argparse import ArgumentParser

# 从当前目录下的 __init__.py 文件中导入 BaseTransformersCLICommand 类
from . import BaseTransformersCLICommand

# 定义一个函数 download_command_factory，用于创建 DownloadCommand 类的实例并返回
def download_command_factory(args):
    return DownloadCommand(args.model, args.cache_dir, args.force, args.trust_remote_code)

# 定义 DownloadCommand 类，继承自 BaseTransformersCLICommand 类
class DownloadCommand(BaseTransformersCLICommand):

    # 静态方法，用于注册命令行参数
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        # 添加一个名为 "download" 的子命令解析器
        download_parser = parser.add_parser("download")

        # 添加命令行参数 --cache-dir，用于指定存储模型的路径
        download_parser.add_argument(
            "--cache-dir", type=str, default=None, help="Path to location to store the models"
        )

        # 添加命令行参数 --force，用于强制下载模型，即使已存在于 cache-dir 中
        download_parser.add_argument(
            "--force", action="store_true", help="Force the model to be download even if already in cache-dir"
        )

        # 添加命令行参数 --trust-remote-code，用于控制是否信任远程代码
        download_parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files. Use only if you've reviewed the code as it will execute on your local machine",
        )

        # 添加一个位置参数 model，用于指定要下载的模型名称
        download_parser.add_argument("model", type=str, help="Name of the model to download")

        # 设置默认的函数处理程序为 download_command_factory
        download_parser.set_defaults(func=download_command_factory)

    # DownloadCommand 类的初始化方法，接收模型名称、缓存路径、是否强制下载、是否信任远程代码四个参数
    def __init__(self, model: str, cache: str, force: bool, trust_remote_code: bool):
        self._model = model  # 将模型名称存储在实例变量 _model 中
        self._cache = cache  # 将缓存路径存储在实例变量 _cache 中
        self._force = force  # 将是否强制下载标志存储在实例变量 _force 中
        self._trust_remote_code = trust_remote_code  # 将是否信任远程代码标志存储在实例变量 _trust_remote_code 中

    # 定义 run 方法，用于执行下载模型的操作
    def run(self):
        # 从 ..models.auto 模块中导入 AutoModel 和 AutoTokenizer 类
        from ..models.auto import AutoModel, AutoTokenizer

        # 使用 AutoModel 类从预训练模型中加载模型
        AutoModel.from_pretrained(
            self._model, cache_dir=self._cache, force_download=self._force, trust_remote_code=self._trust_remote_code
        )

        # 使用 AutoTokenizer 类从预训练模型中加载 tokenizer
        AutoTokenizer.from_pretrained(
            self._model, cache_dir=self._cache, force_download=self._force, trust_remote_code=self._trust_remote_code
        )
```