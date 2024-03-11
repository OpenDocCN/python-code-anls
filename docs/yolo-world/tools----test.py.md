# `.\YOLO-World\tools\test.py`

```py
# 版权声明
# 导入必要的库
import argparse
import os
import os.path as osp

# 导入自定义模块
from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner

# 导入自定义模块
from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower

# 定义解析命令行参数的函数
def parse_args():
    # 创建 ArgumentParser 对象，设置描述信息
    parser = argparse.ArgumentParser(
        description='MMYOLO test (and eval) a model')
    # 添加命令行参数
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='output result file (must be a .pkl file) in pickle format')
    parser.add_argument(
        '--json-prefix',
        type=str,
        help='the prefix of the output json file without perform evaluation, '
        'which is useful when you want to format the result to a specific '
        'format and submit it to the test server')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to use test time augmentation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    # 添加一个命令行参数，用于覆盖配置文件中的一些设置，参数为字典类型，使用自定义的DictAction处理
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    
    # 添加一个命令行参数，用于指定作业启动器的类型，可选值为['none', 'pytorch', 'slurm', 'mpi']，默认为'none'
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    
    # 添加一个命令行参数，用于指定本地进程的排名，默认为0
    parser.add_argument('--local_rank', type=int, default=0)
    
    # 解析命令行参数并返回结果
    args = parser.parse_args()
    
    # 如果环境变量中没有'LOCAL_RANK'，则将命令行参数中的local_rank值赋给'LOCAL_RANK'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    # 返回解析后的命令行参数
    return args
def main():
    # 解析命令行参数
    args = parse_args()

    # 加载配置文件
    cfg = Config.fromfile(args.config)
    # 用 cfg.key 的值替换 ${key}
    # cfg = replace_cfg_vals(cfg)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        # 根据命令行参数更新配置
        cfg.merge_from_dict(args.cfg_options)

    # 确定工作目录的优先级：CLI > 配置文件中的段 > 文件名
    if args.work_dir is not None:
        # 如果 args.work_dir 不为 None，则根据 CLI 参数更新配置
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # 如果 cfg.work_dir 为 None，则使用配置文件名作为默认工作目录
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # 加载模型参数
    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        # 触发可视化钩子
        cfg = trigger_visualization_hook(cfg, args)

    if args.deploy:
        # 添加部署钩子
        cfg.custom_hooks.append(dict(type='SwitchToDeployHook'))

    # 将 `format_only` 和 `outfile_prefix` 添加到配置中
    if args.json_prefix is not None:
        cfg_json = {
            'test_evaluator.format_only': True,
            'test_evaluator.outfile_prefix': args.json_prefix
        }
        cfg.merge_from_dict(cfg_json)

    # 确定自定义元信息字段是否全部为小写
    is_metainfo_lower(cfg)
    # 如果启用了测试时间增强（TTA），则需要检查配置中是否包含必要的参数
    if args.tta:
        # 检查配置中是否包含 tta_model 和 tta_pipeline，否则无法使用 TTA
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.' \
                                   " Can't use tta !"
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` ' \
                                      "in config. Can't use tta !"

        # 将 tta_model 合并到 model 配置中
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        test_data_cfg = cfg.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg 会强制控制输出图像的大小，与 TTA 不兼容
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = cfg.tta_pipeline

    # 根据配置构建 Runner 对象
    if 'runner_type' not in cfg:
        # 构建默认的 Runner
        runner = Runner.from_cfg(cfg)
    else:
        # 从注册表中构建自定义的 Runner，如果配置中设置了 runner_type
        runner = RUNNERS.build(cfg)

    # 添加 `DumpResults` 虚拟指标
    if args.out is not None:
        # 确保输出文件是 pkl 或 pickle 格式
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))

    # 开始测试
    runner.test()
# 如果当前脚本被直接执行，则调用主函数
if __name__ == '__main__':
    main()
```