# `.\YOLO-World\deploy\deploy_test.py`

```
# 导入必要的库
import argparse
import os.path as osp
from copy import deepcopy

# 导入自定义模块
from mmengine import DictAction
from mmdeploy.apis import build_task_processor
from mmdeploy.utils.config_utils import load_config
from mmdeploy.utils.timer import TimeCounter

# 解析命令行参数
def parse_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='MMDeploy test (and eval) a backend.')
    
    # 添加命令行参数
    parser.add_argument('deploy_cfg', help='Deploy config path')
    parser.add_argument('model_cfg', help='Model config path')
    parser.add_argument('--model', type=str, nargs='+', help='Input model files.')
    parser.add_argument('--device', help='device used for conversion', default='cpu')
    parser.add_argument('--work-dir', default='./work_dir', help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--interval', type=int, default=1, help='visualize per interval samples.')
    parser.add_argument('--wait-time', type=float, default=2, help='display time of every window. (second)')
    parser.add_argument('--log2file', type=str, help='log evaluation results and speed to file', default=None)
    # 添加命令行参数，用于激活速度测试
    parser.add_argument(
        '--speed-test', action='store_true', help='activate speed test')
    # 添加命令行参数，用于设置预热次数，在计算推理时间之前需要设置速度测试
    parser.add_argument(
        '--warmup',
        type=int,
        help='warmup before counting inference elapse, require setting '
        'speed-test first',
        default=10)
    # 添加命令行参数，用于设置日志输出间隔，在设置速度测试之前需要设置
    parser.add_argument(
        '--log-interval',
        type=int,
        help='the interval between each log, require setting '
        'speed-test first',
        default=100)
    # 添加命令行参数，用于设置测试的批次大小，会覆盖数据配置中的 `samples_per_gpu`
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='the batch size for test, would override `samples_per_gpu`'
        'in  data config.')
    # 添加命令行参数，用于设置远程推理设备的 IP 地址和端口
    parser.add_argument(
        '--uri',
        action='store_true',
        default='192.168.1.1:60000',
        help='Remote ipv4:port or ipv6:port for inference on edge device.')

    # 解析命令行参数
    args = parser.parse_args()
    # 返回解析后的参数
    return args
def main():
    # 解析命令行参数
    args = parse_args()
    # 获取部署配置文件路径和模型配置文件路径
    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg

    # 加载部署配置和模型配置
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # 确定工作目录的优先级：命令行参数 > 文件中的段落 > 文件名
    if args.work_dir is not None:
        # 如果命令行参数中指定了工作目录，则更新配置
        work_dir = args.work_dir
    elif model_cfg.get('work_dir', None) is None:
        # 如果配置文件中未指定工作目录，则使用配置文件名作为默认工作目录
        work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # 合并模型配置的选项
    if args.cfg_options is not None:
        model_cfg.merge_from_dict(args.cfg_options)

    # 构建任务处理器
    task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)

    # 准备数据集加载器
    test_dataloader = deepcopy(model_cfg['test_dataloader'])
    if isinstance(test_dataloader, list):
        dataset = []
        for loader in test_dataloader:
            # 构建数据集
            ds = task_processor.build_dataset(loader['dataset'])
            dataset.append(ds)
            loader['dataset'] = ds
            loader['batch_size'] = args.batch_size
            loader = task_processor.build_dataloader(loader)
        dataloader = test_dataloader
    else:
        test_dataloader['batch_size'] = args.batch_size
        dataset = task_processor.build_dataset(test_dataloader['dataset'])
        test_dataloader['dataset'] = dataset
        dataloader = task_processor.build_dataloader(test_dataloader)

    # 加载后端模型
    model = task_processor.build_backend_model(
        args.model,
        data_preprocessor_updater=task_processor.update_data_preprocessor)
    destroy_model = model.destroy
    is_device_cpu = (args.device == 'cpu')
    # 使用任务处理器构建测试运行器，传入模型、工作目录、是否记录日志到文件、是否展示测试结果、展示目录、等待时间、间隔时间、数据加载器等参数
    runner = task_processor.build_test_runner(
        model,
        work_dir,
        log_file=args.log2file,
        show=args.show,
        show_dir=args.show_dir,
        wait_time=args.wait_time,
        interval=args.interval,
        dataloader=dataloader)

    # 如果需要进行速度测试
    if args.speed_test:
        # 根据设备是否为 CPU 决定是否需要同步
        with_sync = not is_device_cpu

        # 激活时间计数器，设置预热次数、日志间隔、是否同步、记录日志到文件、批处理大小等参数
        with TimeCounter.activate(
                warmup=args.warmup,
                log_interval=args.log_interval,
                with_sync=with_sync,
                file=args.log2file,
                batch_size=args.batch_size):
            # 运行测试
            runner.test()

    else:
        # 运行测试
        runner.test()
    
    # 仅在后端需要显式清理时生效（例如 Ascend）
    destroy_model()
# 如果当前脚本被直接执行，则调用主函数
if __name__ == '__main__':
    main()
```