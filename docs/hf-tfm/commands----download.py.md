# `.\transformers\commands\download.py`

```
# 导入必要的模块
from argparse import ArgumentParser
# 导入基础的命令行接口类
from . import BaseTransformersCLICommand

# 下载命令的工厂函数，根据参数返回下载命令对象
def download_command_factory(args):
    return DownloadCommand(args.model, args.cache_dir, args.force, args.trust_remote_code)

# 下载命令类，继承自基础的 Transformers 命令行接口类
class DownloadCommand(BaseTransformersCLICommand):
    # 注册下载命令的子命令
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        # 添加下载命令的解析器
        download_parser = parser.add_parser("download")
        # 添加解析器参数：缓存目录
        download_parser.add_argument(
            "--cache-dir", type=str, default=None, help="Path to location to store the models"
        )
        # 添加解析器参数：是否强制下载模型
        download_parser.add_argument(
            "--force", action="store_true", help="Force the model to be download even if already in cache-dir"
        )
        # 添加解析器参数：是否信任远程代码
        download_parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files. Use only if you've reviewed the code as it will execute on your local machine",
        )
        # 添加解析器参数：模型名称
        download_parser.add_argument("model", type=str, help="Name of the model to download")
        # 设置默认函数为下载命令工厂函数
        download_parser.set_defaults(func=download_command_factory)

    # 初始化下载命令对象
    def __init__(self, model: str, cache: str, force: bool, trust_remote_code: bool):
        self._model = model
        self._cache = cache
        self._force = force
        self._trust_remote_code = trust_remote_code

    # 执行下载命令
    def run(self):
        # 导入自动模型和自动分词器
        from ..models.auto import AutoModel, AutoTokenizer

        # 从预训练模型中创建自动模型对象，可指定缓存目录、是否强制下载、是否信任远程代码
        AutoModel.from_pretrained(
            self._model, cache_dir=self._cache, force_download=self._force, trust_remote_code=self._trust_remote_code
        )
        # 从预训练模型中创建自动分词器对象，可指定缓存目录、是否强制下载、是否信任远程代码
        AutoTokenizer.from_pretrained(
            self._model, cache_dir=self._cache, force_download=self._force, trust_remote_code=self._trust_remote_code
        )
```