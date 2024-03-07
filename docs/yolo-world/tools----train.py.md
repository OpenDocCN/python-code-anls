# `.\YOLO-World\tools\train.py`

```
# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import logging  # 用于记录日志
import os  # 用于操作系统相关功能
import os.path as osp  # 用于操作文件路径

from mmengine.config import Config, DictAction  # 导入Config和DictAction类
from mmengine.logging import print_log  # 导入print_log函数
from mmengine.runner import Runner  # 导入Runner类

from mmyolo.registry import RUNNERS  # 导入RUNNERS变量
from mmyolo.utils import is_metainfo_lower  # 导入is_metainfo_lower函数

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')  # 创建参数解析器
    parser.add_argument('config', help='train config file path')  # 添加必需的参数
    parser.add_argument('--work-dir', help='the dir to save logs and models')  # 添加可选参数
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')  # 添加可选参数
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')  # 添加可选参数
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')  # 添加可选参数
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')  # 添加可选参数
    parser.add_argument('--local_rank', type=int, default=0)  # 添加可选参数
    args = parser.parse_args()  # 解析命令行参数
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)  # 设置环境变量LOCAL_RANK为args.local_rank的值

    return args  # 返回解析后的参数

def main():
    args = parse_args()  # 解析命令行参数并保存到args变量中

    # 加载配置文件
    cfg = Config.fromfile(args.config)  # 从配置文件路径args.config中加载配置信息
    # 用cfg.key的值替换${key}的占位符
    # 设置配置文件中的 launcher 为命令行参数中指定的 launcher
    cfg.launcher = args.launcher
    # 如果命令行参数中指定了 cfg_options，则将其合并到配置文件中
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 确定工作目录的优先级：CLI > 文件中的段 > 文件名
    if args.work_dir is not None:
        # 如果命令行参数中指定了 work_dir，则更新配置文件中的 work_dir
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # 如果配置文件中的 work_dir 为 None，则根据配置文件名设置默认的 work_dir
        if args.config.startswith('projects/'):
            config = args.config[len('projects/'):]
            config = config.replace('/configs/', '/')
            cfg.work_dir = osp.join('./work_dirs', osp.splitext(config)[0])
        else:
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])

    # 启用自动混合精度训练
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # 确定恢复训练的优先级：resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # 确定自定义元信息字段是否全部为小写
    is_metainfo_lower(cfg)

    # 从配置文件构建 runner
    # 如果配置中没有指定 'runner_type'
    if 'runner_type' not in cfg:
        # 构建默认的运行器
        runner = Runner.from_cfg(cfg)
    else:
        # 从注册表中构建定制的运行器
        # 如果配置中设置了 'runner_type'
        runner = RUNNERS.build(cfg)

    # 开始训练
    runner.train()
# 如果当前脚本被直接执行，则调用主函数
if __name__ == '__main__':
    main()
```