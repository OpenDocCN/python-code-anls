# `.\YOLO-World\deploy\deploy.py`

```py
# 导入必要的库
import argparse  # 用于解析命令行参数
import logging  # 用于记录日志
import os  # 用于操作系统相关功能
import os.path as osp  # 用于操作文件路径
from functools import partial  # 用于创建偏函数

import mmengine  # 导入 mmengine 库
import torch.multiprocessing as mp  # 导入多进程库
from torch.multiprocessing import Process, set_start_method  # 导入多进程相关函数

from mmdeploy.apis import (create_calib_input_data, extract_model,  # 导入一些部署相关的函数
                           get_predefined_partition_cfg, torch2onnx,
                           torch2torchscript, visualize_model)
from mmdeploy.apis.core import PIPELINE_MANAGER  # 导入核心部署管理器
from mmdeploy.apis.utils import to_backend  # 导入转换到后端的函数
from mmdeploy.backend.sdk.export_info import export2SDK  # 导入导出到 SDK 的函数
from mmdeploy.utils import (IR, Backend, get_backend, get_calib_filename,  # 导入一些工具函数
                            get_ir_config, get_partition_config,
                            get_root_logger, load_config, target_wrapper)


def parse_args():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Export model to backends.')
    # 添加命令行参数
    parser.add_argument('deploy_cfg', help='deploy config path')  # 部署配置文件路径
    parser.add_argument('model_cfg', help='model config path')  # 模型配置文件路径
    parser.add_argument('checkpoint', help='model checkpoint path')  # 模型检查点路径
    parser.add_argument('img', help='image used to convert model model')  # 用于转换模型的图像
    parser.add_argument(
        '--test-img',
        default=None,
        type=str,
        nargs='+',
        help='image used to test model')  # 用于测试模型的图像
    parser.add_argument(
        '--work-dir',
        default=os.getcwd(),
        help='the dir to save logs and models')  # 保存日志和模型的目录
    parser.add_argument(
        '--calib-dataset-cfg',
        help='dataset config path used to calibrate in int8 mode. If not \
            specified, it will use "val" dataset in model config instead.',
        default=None)  # 用于在 int8 模式下校准的数据集配置文件路径
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')  # 用于转换的设备
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))  # 设置日志级别
    # 添加命令行参数，用于显示检测输出
    parser.add_argument(
        '--show', action='store_true', help='Show detection outputs')
    # 添加命令行参数，用于输出 SDK 的信息
    parser.add_argument(
        '--dump-info', action='store_true', help='Output information for SDK')
    # 添加命令行参数，指定用于量化模型的图像目录
    parser.add_argument(
        '--quant-image-dir',
        default=None,
        help='Image directory for quantize model.')
    # 添加命令行参数，用于将模型量化为低位
    parser.add_argument(
        '--quant', action='store_true', help='Quantize model to low bit.')
    # 添加命令行参数，指定远程设备上进行推理的 IP 地址和端口
    parser.add_argument(
        '--uri',
        default='192.168.1.1:60000',
        help='Remote ipv4:port or ipv6:port for inference on edge device.')
    # 解析命令行参数
    args = parser.parse_args()
    # 返回解析后的参数
    return args
# 创建一个进程，执行指定的目标函数，并传入参数和关键字参数，可选地返回一个值
def create_process(name, target, args, kwargs, ret_value=None):
    # 获取根日志记录器
    logger = get_root_logger()
    # 记录进程开始信息
    logger.info(f'{name} start.')
    # 获取日志级别
    log_level = logger.level

    # 创建一个包装函数，用于处理目标函数的执行结果和日志记录
    wrap_func = partial(target_wrapper, target, log_level, ret_value)

    # 创建一个进程对象，指定目标函数、参数和关键字参数
    process = Process(target=wrap_func, args=args, kwargs=kwargs)
    # 启动进程
    process.start()
    # 等待进程结束
    process.join()

    # 如果有返回值，检查返回值是否为0，记录日志并退出程序
    if ret_value is not None:
        if ret_value.value != 0:
            logger.error(f'{name} failed.')
            exit(1)
        else:
            logger.info(f'{name} success.')


# 根据中间表示类型返回对应的转换函数
def torch2ir(ir_type: IR):
    """Return the conversion function from torch to the intermediate
    representation.

    Args:
        ir_type (IR): The type of the intermediate representation.
    """
    if ir_type == IR.ONNX:
        return torch2onnx
    elif ir_type == IR.TORCHSCRIPT:
        return torch2torchscript
    else:
        raise KeyError(f'Unexpected IR type {ir_type}')


# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    # 设置进程启动方法为'spawn'
    set_start_method('spawn', force=True)
    # 获取根日志记录器
    logger = get_root_logger()
    # 获取日志级别
    log_level = logging.getLevelName(args.log_level)
    # 设置日志级别
    logger.setLevel(log_level)

    # 定义一系列处理函数
    pipeline_funcs = [
        torch2onnx, torch2torchscript, extract_model, create_calib_input_data
    ]
    # 启用多进程模式，并指定处理函数
    PIPELINE_MANAGER.enable_multiprocess(True, pipeline_funcs)
    # 设置处理函数的日志级别
    PIPELINE_MANAGER.set_log_level(log_level, pipeline_funcs)

    # 获取部署配置路径、模型配置路径、检查点路径、量化标志和量化图片目录
    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    checkpoint_path = args.checkpoint
    quant = args.quant
    quant_image_dir = args.quant_image_dir

    # 加载部署配置和模型配置
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # 如果需要，创建工作目录
    mmengine.mkdir_or_exist(osp.abspath(args.work_dir))

    # 如果需要，导出到SDK
    if args.dump_info:
        export2SDK(
            deploy_cfg,
            model_cfg,
            args.work_dir,
            pth=checkpoint_path,
            device=args.device)

    # 创建一个共享变量，用于存储返回值
    ret_value = mp.Value('d', 0, lock=False)

    # 转换为中间表示
    ir_config = get_ir_config(deploy_cfg)
    # 获取保存 IR 结果的文件名
    ir_save_file = ir_config['save_file']
    # 获取 IR 类型
    ir_type = IR.get(ir_config['type'])
    # 调用 torch2ir 函数将模型转换为 IR
    torch2ir(ir_type)(
        args.img,
        args.work_dir,
        ir_save_file,
        deploy_cfg_path,
        model_cfg_path,
        checkpoint_path,
        device=args.device)

    # 转换后的 IR 文件列表
    ir_files = [osp.join(args.work_dir, ir_save_file)]

    # 获取模型分区配置
    partition_cfgs = get_partition_config(deploy_cfg)

    if partition_cfgs is not None:

        if 'partition_cfg' in partition_cfgs:
            partition_cfgs = partition_cfgs.get('partition_cfg', None)
        else:
            assert 'type' in partition_cfgs
            partition_cfgs = get_predefined_partition_cfg(
                deploy_cfg, partition_cfgs['type'])

        # 原始 IR 文件
        origin_ir_file = ir_files[0]
        ir_files = []
        # 遍历分区配置，对模型进行分区
        for partition_cfg in partition_cfgs:
            save_file = partition_cfg['save_file']
            save_path = osp.join(args.work_dir, save_file)
            start = partition_cfg['start']
            end = partition_cfg['end']
            dynamic_axes = partition_cfg.get('dynamic_axes', None)

            # 提取模型的部分并保存
            extract_model(
                origin_ir_file,
                start,
                end,
                dynamic_axes=dynamic_axes,
                save_file=save_path)

            ir_files.append(save_path)

    # 获取校准数据文件名
    calib_filename = get_calib_filename(deploy_cfg)
    if calib_filename is not None:
        calib_path = osp.join(args.work_dir, calib_filename)
        # 创建校准输入数据
        create_calib_input_data(
            calib_path,
            deploy_cfg_path,
            model_cfg_path,
            checkpoint_path,
            dataset_cfg=args.calib_dataset_cfg,
            dataset_type='val',
            device=args.device)

    # 后端文件列表
    backend_files = ir_files
    # 获取后端类型
    backend = get_backend(deploy_cfg)

    # 预处理部署配置
    # 如果选择的后端是RKNN
    if backend == Backend.RKNN:
        # TODO: 在将来将此功能添加到任务处理器中
        # 导入临时文件模块
        import tempfile

        # 从mmdeploy.utils中导入必要的函数
        from mmdeploy.utils import (get_common_config, get_normalization,
                                    get_quantization_config,
                                    get_rknn_quantization)
        
        # 获取量化配置
        quantization_cfg = get_quantization_config(deploy_cfg)
        # 获取通用配置
        common_params = get_common_config(deploy_cfg)
        
        # 如果需要RKNN量化
        if get_rknn_quantization(deploy_cfg) is True:
            # 获取归一化转换参数
            transform = get_normalization(model_cfg)
            # 更新通用参数，包括均值和标准差
            common_params.update(
                dict(
                    mean_values=[transform['mean']],
                    std_values=[transform['std']]))

        # 创建临时文件用于存储数据集文件路径
        dataset_file = tempfile.NamedTemporaryFile(suffix='.txt').name
        # 将图像文件的绝对路径写入数据集文件
        with open(dataset_file, 'w') as f:
            f.writelines([osp.abspath(args.img)])
        
        # 如果量化配置中未指定数据集，则将数据集文件路径添加到量化配置中
        if quantization_cfg.get('dataset', None) is None:
            quantization_cfg['dataset'] = dataset_file
    
    # 如果选择的后端是ASCEND
    if backend == Backend.ASCEND:
        # TODO: 在将来将此功能添加到后端管理器中
        # 如果需要输出信息
        if args.dump_info:
            # 从mmdeploy.backend.ascend中导入更新SDK管道的函数
            from mmdeploy.backend.ascend import update_sdk_pipeline
            # 更新SDK管道
            update_sdk_pipeline(args.work_dir)
    # 如果后端是VACC
    if backend == Backend.VACC:
        # TODO: 将此部分在未来添加到任务处理器中

        # 导入获取量化数据的模块
        from onnx2vacc_quant_dataset import get_quant

        # 导入获取模型输入的工具函数
        from mmdeploy.utils import get_model_inputs

        # 加载部署配置和模型配置
        deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
        # 获取模型输入
        model_inputs = get_model_inputs(deploy_cfg)

        # 遍历ONNX文件路径和模型输入
        for onnx_path, model_input in zip(ir_files, model_inputs):

            # 获取量化模式
            quant_mode = model_input.get('qconfig', {}).get('dtype', 'fp16')
            # 确保量化模式为'int8'或'fp16'
            assert quant_mode in ['int8', 'fp16'], quant_mode + ' not support now'
            shape_dict = model_input.get('shape', {})

            # 如果量化模式为'int8'
            if quant_mode == 'int8':
                # 创建处理过程
                create_process(
                    'vacc quant dataset',
                    target=get_quant,
                    args=(deploy_cfg, model_cfg, shape_dict, checkpoint_path,
                          args.work_dir, args.device),
                    kwargs=dict(),
                    ret_value=ret_value)

    # 设置日志级别
    PIPELINE_MANAGER.set_log_level(log_level, [to_backend])
    # 如果后端是TensorRT，启用多进程
    if backend == Backend.TENSORRT:
        PIPELINE_MANAGER.enable_multiprocess(True, [to_backend])
    # 将转换后的文件传递给后端处理
    backend_files = to_backend(
        backend,
        ir_files,
        work_dir=args.work_dir,
        deploy_cfg=deploy_cfg,
        log_level=log_level,
        device=args.device,
        uri=args.uri)

    # 进行ncnn量化
    # 如果后端为NCNN且需要量化
    if backend == Backend.NCNN and quant:
        # 导入获取量化表的函数
        from onnx2ncnn_quant_table import get_table
        # 导入NCNN相关函数
        from mmdeploy.apis.ncnn import get_quant_model_file, ncnn2int8
        # 获取模型参数路径列表
        model_param_paths = backend_files[::2]
        # 获取模型二进制文件路径列表
        model_bin_paths = backend_files[1::2]
        # 清空后端文件列表
        backend_files = []
        # 遍历ONNX文件路径、模型参数路径、模型二进制文件路径
        for onnx_path, model_param_path, model_bin_path in zip(
                ir_files, model_param_paths, model_bin_paths):
            # 加载部署配置和模型配置
            deploy_cfg, model_cfg = load_config(deploy_cfg_path,
                                                model_cfg_path)
            # 获取量化后的ONNX文件、量化表、量化参数、量化二进制文件
            quant_onnx, quant_table, quant_param, quant_bin = get_quant_model_file(
                onnx_path, args.work_dir)

            # 创建进程，获取量化表
            create_process(
                'ncnn quant table',
                target=get_table,
                args=(onnx_path, deploy_cfg, model_cfg, quant_onnx,
                      quant_table, quant_image_dir, args.device),
                kwargs=dict(),
                ret_value=ret_value)

            # 创建进程，进行NCNN量化
            create_process(
                'ncnn_int8',
                target=ncnn2int8,
                args=(model_param_path, model_bin_path, quant_table,
                      quant_param, quant_bin),
                kwargs=dict(),
                ret_value=ret_value)
            # 将量化参数和量化二进制文件添加到后端文件列表中
            backend_files += [quant_param, quant_bin]

    # 如果未指定测试图片，则使用默认图片
    if args.test_img is None:
        args.test_img = args.img

    # 额外参数，包括后端类型、输出文件路径、是否显示结果
    extra = dict(
        backend=backend,
        output_file=osp.join(args.work_dir, f'output_{backend.value}.jpg'),
        show_result=args.show)
    # 如果后端为SNPE，则添加URI参数
    if backend == Backend.SNPE:
        extra['uri'] = args.uri

    # 获取后端推理结果，并尝试渲染
    create_process(
        f'visualize {backend.value} model',
        target=visualize_model,
        args=(model_cfg_path, deploy_cfg_path, backend_files, args.test_img,
              args.device),
        kwargs=extra,
        ret_value=ret_value)

    # 获取PyTorch模型推理结果，尝试可视化（如果可能）
    # 创建一个进程，用于可视化 PyTorch 模型
    create_process(
        'visualize pytorch model',  # 进程名称
        target=visualize_model,  # 目标函数为 visualize_model
        args=(model_cfg_path, deploy_cfg_path, [checkpoint_path], args.test_img, args.device),  # 参数列表
        kwargs=dict(
            backend=Backend.PYTORCH,  # 使用 PyTorch 后端
            output_file=osp.join(args.work_dir, 'output_pytorch.jpg'),  # 输出文件路径
            show_result=args.show),  # 是否显示结果
        ret_value=ret_value)  # 返回值为 ret_value
    # 记录信息到日志
    logger.info('All process success.')
# 如果当前脚本被直接执行，则调用主函数
if __name__ == '__main__':
    main()
```