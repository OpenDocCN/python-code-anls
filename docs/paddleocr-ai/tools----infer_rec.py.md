# `.\PaddleOCR\tools\infer_rec.py`

```py
# 版权声明和许可证信息
# 从未来的 Python 版本导入特性
# 导入 numpy 库
# 导入操作系统、系统和 JSON 库
# 获取当前文件所在目录路径
# 将当前文件所在目录路径添加到系统路径中
# 将当前文件所在目录路径的上一级目录路径添加到系统路径中
# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
# 导入 PaddlePaddle 库
# 从 ppocr.data 模块中导入 create_operators 和 transform 函数
# 从 ppocr.modeling.architectures 模块中导入 build_model 函数
# 从 ppocr.postprocess 模块中导入 build_post_process 函数
# 从 ppocr.utils.save_load 模块中导入 load_model 函数
# 从 ppocr.utils.utility 模块中导入 get_image_file_list 函数
# 导入 tools.program 模块
def main():
    # 获取全局配置信息
    global_config = config['Global']

    # 构建后处理对象
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # 构建模型
    model = build_model(config['Architecture'])

    # 加载模型参数
    load_model(config, model)

    # 创建数据操作列表
    transforms = []
    # 遍历配置文件中数据集变换操作列表
    for op in config['Eval']['dataset']['transforms']:
        # 获取操作名
        op_name = list(op)[0]
        # 如果操作名中包含 'Label'，则跳过当前循环
        if 'Label' in op_name:
            continue
        # 如果操作名在 ['RecResizeImg'] 中
        elif op_name in ['RecResizeImg']:
            # 设置操作的 'infer_mode' 为 True
            op[op_name]['infer_mode'] = True
        # 如果操作名为 'KeepKeys'
        elif op_name == 'KeepKeys':
            # 根据不同的算法设置不同的保留键值
            if config['Architecture']['algorithm'] == "SRN":
                op[op_name]['keep_keys'] = [
                    'image', 'encoder_word_pos', 'gsrm_word_pos',
                    'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                ]
            elif config['Architecture']['algorithm'] == "SAR":
                op[op_name]['keep_keys'] = ['image', 'valid_ratio']
            elif config['Architecture']['algorithm'] == "RobustScanner":
                op[op_name]['keep_keys'] = ['image', 'valid_ratio', 'word_positons']
            else:
                op[op_name]['keep_keys'] = ['image']
        # 将操作添加到变换操作列表中
        transforms.append(op)
    # 设置全局配置中的 'infer_mode' 为 True
    global_config['infer_mode'] = True
    # 创建操作符
    ops = create_operators(transforms, global_config)

    # 获取保存结果路径，默认为 "./output/rec/predicts_rec.txt"
    save_res_path = config['Global'].get('save_res_path', "./output/rec/predicts_rec.txt")
    # 如果保存结果路径的目录不存在，则创建目录
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    # 将模型设置为评估模式
    model.eval()

    # 输出日志信息
    logger.info("success!")
# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 调用 program 模块的 preprocess 函数，获取 config, device, logger, vdl_writer 四个变量
    config, device, logger, vdl_writer = program.preprocess()
    # 调用主函数 main()
    main()
```