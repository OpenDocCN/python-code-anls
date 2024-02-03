# `.\PaddleOCR\tools\infer_sr.py`

```
# 版权声明和许可证信息
# 从未来的 Python 版本导入特性
# 导入 numpy 库
# 导入操作系统、系统和 JSON 库，以及 PIL 和 OpenCV 库
# 获取当前文件所在目录的绝对路径
# 将当前文件所在目录和上一级目录添加到系统路径中
# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
# 导入 PaddlePaddle 库
# 导入创建操作符和转换函数的模块
# 导入构建模型的模块
# 导入构建后处理过程的模块
# 导入加载模型的函数
# 导入获取图像文件列表的函数
# 导入程序工具模块
def main():
    # 获取全局配置信息
    global_config = config['Global']

    # 构建后处理过程
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # 设置转换模式为推理模式
    config['Architecture']["Transform"]['infer_mode'] = True

    # 构建模型
    model = build_model(config['Architecture'])

    # 加载模型参数
    load_model(config, model)

    # 创建数据操作符列表
    transforms = []
    # 遍历配置文件中数据集变换的操作
    for op in config['Eval']['dataset']['transforms']:
        # 获取操作名
        op_name = list(op)[0]
        # 如果操作名中包含 'Label'，则跳过当前循环
        if 'Label' in op_name:
            continue
        # 如果操作名为 'SRResize'，则设置 'infer_mode' 为 True
        elif op_name in ['SRResize']:
            op[op_name]['infer_mode'] = True
        # 如果操作名为 'KeepKeys'，则设置 'keep_keys' 为 ['img_lr']
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['img_lr']
        # 将操作添加到 transforms 列表中
        transforms.append(op)
    # 设置全局配置中的 'infer_mode' 为 True
    global_config['infer_mode'] = True
    # 创建操作符
    ops = create_operators(transforms, global_config)

    # 获取保存可视化结果的路径
    save_visual_path = config['Global'].get('save_visual', "infer_result/")
    # 如果保存路径不存在，则创建
    if not os.path.exists(os.path.dirname(save_visual_path)):
        os.makedirs(os.path.dirname(save_visual_path))

    # 设置模型为评估模式
    model.eval()
    # 遍历推理图像文件列表
    for file in get_image_file_list(config['Global']['infer_img']):
        logger.info("infer_img: {}".format(file))
        # 打开图像文件并转换为 RGB 格式
        img = Image.open(file).convert("RGB")
        data = {'image_lr': img}
        # 对数据进行变换
        batch = transform(data, ops)
        images = np.expand_dims(batch[0], axis=0)
        images = paddle.to_tensor(images)

        # 获取模型预测结果
        preds = model(images)
        sr_img = preds["sr_img"][0]
        lr_img = preds["lr_img"][0]
        # 转换预测结果为可视化图像
        fm_sr = (sr_img.numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
        fm_lr = (lr_img.numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
        img_name_pure = os.path.split(file)[-1]
        # 保存可视化图像
        cv2.imwrite("{}/sr_{}".format(save_visual_path, img_name_pure),
                    fm_sr[:, :, ::-1])
        logger.info("The visualized image saved in infer_result/sr_{}".format(
            img_name_pure))

    logger.info("success!")
# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 调用 program 模块的 preprocess 函数，获取 config, device, logger, vdl_writer 四个变量
    config, device, logger, vdl_writer = program.preprocess()
    # 调用主函数 main()
    main()
```