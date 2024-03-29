# `.\lucidrains\DALLE-pytorch\dalle_pytorch\distributed_backends\deepspeed_backend.py`

```py
import json
import os

import torch

from .distributed_backend import DistributedBackend


class DeepSpeedBackend(DistributedBackend):
    """使用 DeepSpeed 引擎的分布式后端。"""

    BACKEND_MODULE_NAME = 'deepspeed'
    BACKEND_NAME = 'DeepSpeed'

    def wrap_arg_parser(self, parser):
        if not self.has_backend():
            parser.add_argument(
                '--deepspeed',
                type=lambda _: False,
                help=(
                    '是否使用 DeepSpeed '
                    "(由于不可用，此选项被忽略)"
                ),
            )
        else:
            parser = self.backend_module.add_config_arguments(parser)

        parser.add_argument(
            '--local_rank',
            type=int,
            default=-1,
            help='从分布式启动器传递的本地排名',
        )
        return parser

    def _initialize(self):
        self.backend_module.init_distributed()
        if torch.cuda.is_available():
            torch.cuda.set_device(self._get_local_rank())

    @staticmethod
    def _require_torch_distributed_init():
        """当 `torch.distributed` 尚未初始化时引发错误。"""
        assert torch.distributed.is_initialized(), \
            ('`torch.distributed` 未初始化；请在脚本开头调用 '
             '`DeepSpeedBackend.initialize`')

    def _get_world_size(self):
        self._require_torch_distributed_init()
        return torch.distributed.get_world_size()

    def _get_rank(self):
        self._require_torch_distributed_init()
        return torch.distributed.get_rank()

    def _get_local_rank(self):
        self._require_torch_distributed_init()
        return int(os.environ['LOCAL_RANK'])

    def _local_barrier(self):
        self._require_torch_distributed_init()
        torch.distributed.barrier()

    def _check_args(self, args, optimizer, lr_scheduler, kwargs):
        """在检查传递给 `distribute` 的值后，返回适当的优化器和学习率调度器。"""
        self._check_argvs(args, optimizer, lr_scheduler, kwargs)
        (optimizer, lr_scheduler) = self._check_config(
            args, optimizer, lr_scheduler, kwargs)
        return (optimizer, lr_scheduler)

    def _check_argvs(self, args, optimizer, lr_scheduler, kwargs):
        """对给定的命令行参数应用几个合理性检查。"""
        has_json_config = (hasattr(args, 'deepspeed_config')
                           and args.deepspeed_config is not None)
        has_dict_config = 'config_params' in kwargs
        if (
                # 没有给定配置
                (not has_json_config and not has_dict_config)
                # JSON 配置文件不存在
                or (not has_dict_config
                    and not os.path.isfile(args.deepspeed_config))
        ):
            # 让 DeepSpeed 处理这些参数错误。
            return

        if not args.deepspeed:
            print(
                '警告：已选择 DeepSpeed 后端；设置 `args.deepspeed = True`'
            )
            args.deepspeed = True

        if has_json_config and has_dict_config:
            print(
                '警告：DeepSpeed 配置同时以 JSON 文件和 Python 字典形式给出。Python 字典优先。'
            )
    def _check_config(self, args, optimizer, lr_scheduler, kwargs):
        """Return an appropriate optimizer and learning rate scheduler
        for the DeepSpeed configuration.
        """
        # 检查 DeepSpeed 配置，根据情况返回优化器和学习率调度器
        if 'config_params' in kwargs:
            config = kwargs['config_params']
        else:
            with open(args.deepspeed_config, 'r') as json_config_file:
                config = json.load(json_config_file)

        if 'optimizer' in config and optimizer is not None:
            print(
                'WARNING: Optimizer encountered in both DeepSpeed config and '
                'keyword arguments. Optimizer in DeepSpeed config '
                'takes precedence.'
            )
            optimizer = None

        if 'scheduler' in config and lr_scheduler is not None:
            print(
                'WARNING: Learning rate scheduler encountered in both '
                'DeepSpeed config and keyword arguments. Learning rate '
                'scheduler in DeepSpeed config takes precedence.'
            )
            # 对于 LR 调度器，JSON 配置已经具有优先权。我们这样做是为了向前兼容。
            lr_scheduler = None

        return (optimizer, lr_scheduler)

    def _distribute(
            self,
            args=None,
            model=None,
            optimizer=None,
            model_parameters=None,
            training_data=None,
            lr_scheduler=None,
            **kwargs,
    ):
        """Return a distributed model engine, optimizer, dataloader, and
        learning rate scheduler. These are obtained by wrapping the
        given values with the backend.

        For the other or other possible arguments,
        see `deepspeed.initialize`.
        """
        (optimizer, lr_scheduler) = self._check_args(
            args, optimizer, lr_scheduler, kwargs)

        return self.backend_module.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            **kwargs,
        )

    def _average_all(self, tensor):
        self._require_torch_distributed_init()
        # We copy because modification happens in-place
        averaged = tensor.detach().clone()
        # We use `all_reduce` because it is better supported than `reduce`
        torch.distributed.all_reduce(averaged, torch.distributed.ReduceOp.SUM)
        return averaged / self.get_world_size()
```