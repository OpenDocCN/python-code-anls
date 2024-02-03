# `.\PaddleOCR\deploy\fastdeploy\rockchip\rknpu2_tools\export.py`

```py
# 版权声明和许可证信息
# 导入所需的库
import os
import yaml
import argparse
from rknn.api import RKNN

# 定义函数用于获取配置信息
def get_config():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument("--verbose", default=True, help="rknntoolkit verbose")
    parser.add_argument("--config_path")
    parser.add_argument("--target_platform")
    # 解析参数
    args = parser.parse_args()
    return args

# 主程序入口
if __name__ == "__main__":
    # 获取配置信息
    config = get_config()
    # 读取配置文件
    with open(config.config_path) as file:
        file_data = file.read()
        yaml_config = yaml.safe_load(file_data)
    # 打印配置信息
    print(yaml_config)
    # 创建 RKNN 模型对象
    model = RKNN(config.verbose)

    # 配置模型
    mean_values = yaml_config["mean"]
    std_values = yaml_config["std"]
    model.config(
        mean_values=mean_values,
        std_values=std_values,
        target_platform=config.target_platform)

    # 加载 ONNX 模型
    if yaml_config["outputs_nodes"] is None:
        ret = model.load_onnx(model=yaml_config["model_path"])
    else:
        ret = model.load_onnx(
            model=yaml_config["model_path"],
            outputs=yaml_config["outputs_nodes"])
    assert ret == 0, "Load model failed!"

    # 构建模型
    ret = model.build(
        do_quantization=yaml_config["do_quantization"],
        dataset=yaml_config["dataset"])
    assert ret == 0, "Build model failed!"

    # 初始化运行时环境
    ret = model.init_runtime()
    assert ret == 0, "Init runtime environment failed!"
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(yaml_config["output_folder"]):
        os.mkdir(yaml_config["output_folder"])

    # 获取模型路径的基本名称
    name_list = os.path.basename(yaml_config["model_path"]).split(".")
    model_base_name = ""
    for name in name_list[0:-1]:
        model_base_name += name

    # 获取目标平台并转换为小写
    model_device_name = config.target_platform.lower()

    # 根据是否进行量化，确定模型保存名称
    if yaml_config["do_quantization"]:
        model_save_name = model_base_name + "_" + model_device_name + "_quantized" + ".rknn"
    else:
        model_save_name = model_base_name + "_" + model_device_name + "_unquantized" + ".rknn"

    # 导出模型到指定路径
    ret = model.export_rknn(
        os.path.join(yaml_config["output_folder"], model_save_name))
    
    # 检查导出是否成功
    assert ret == 0, "Export rknn model failed!"
    
    # 打印导出成功信息
    print("Export OK!")
```