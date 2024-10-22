# `.\chatglm4-finetune\intel_device_demo\openvino\convert.py`

```py
"""
该脚本用于将原始模型转换为 OpenVINO IR 格式。
可以查看原始代码 https://github.com/OpenVINO-dev-contest/chatglm3.openvino/blob/main/convert.py
"""
# 从 transformers 库导入自动分词器和配置
from transformers import AutoTokenizer, AutoConfig
# 从 optimum.intel 导入量化配置
from optimum.intel import OVWeightQuantizationConfig
# 从 optimum.intel.openvino 导入 OpenVINO 语言模型类
from optimum.intel.openvino import OVModelForCausalLM

# 导入操作系统模块
import os
# 从 pathlib 导入 Path 类
from pathlib import Path
# 导入参数解析模块
import argparse


# 主程序入口
if __name__ == '__main__':
    # 创建参数解析器，禁用帮助信息自动添加
    parser = argparse.ArgumentParser(add_help=False)
    # 添加帮助选项
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='显示帮助信息并退出。')
    # 添加模型 ID 参数，默认值为指定的模型路径
    parser.add_argument('-m',
                        '--model_id',
                        default='THUDM/glm-4-9b-chat',
                        required=False,
                        type=str,
                        help='原始模型路径')
    # 添加精度参数，默认值为 "int4"
    parser.add_argument('-p',
                        '--precision',
                        required=False,
                        default="int4",
                        type=str,
                        choices=["fp16", "int8", "int4"],
                        help='fp16、int8 或 int4')
    # 添加输出路径参数，默认值为 './glm-4-9b-ov'
    parser.add_argument('-o',
                        '--output',
                        default='./glm-4-9b-ov',
                        required=False,
                        type=str,
                        help='必需。保存 IR 模型的路径')
    # 解析命令行参数
    args = parser.parse_args()

    # 将输出路径转换为 Path 对象
    ir_model_path = Path(args.output)
    # 如果输出路径不存在，则创建该目录
    if ir_model_path.exists() == False:
        os.mkdir(ir_model_path)

    # 创建模型参数字典，包括信任远程代码和模型配置
    model_kwargs = {
        "trust_remote_code": True,
        "config": AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    }
    # 创建压缩配置字典
    compression_configs = {
        "sym": False,
        "group_size": 128,
        "ratio": 0.8,
    }

    # 打印导出 IR 的消息
    print("====Exporting IR=====")
    # 根据指定精度加载不同的模型
    if args.precision == "int4":
        # 加载 4 位量化模型
        ov_model = OVModelForCausalLM.from_pretrained(args.model_id, export=True,
                                                      compile=False, quantization_config=OVWeightQuantizationConfig(
                                                          bits=4, **compression_configs), **model_kwargs)
    elif args.precision == "int8":
        # 加载 8 位量化模型
        ov_model = OVModelForCausalLM.from_pretrained(args.model_id, export=True,
                                                      compile=False, load_in_8bit=True, **model_kwargs)
    else:
        # 加载原始模型（未量化）
        ov_model = OVModelForCausalLM.from_pretrained(args.model_id, export=True,
                                                      compile=False, load_in_8bit=False, **model_kwargs)

    # 将模型保存到指定的路径
    ov_model.save_pretrained(ir_model_path)

    # 打印导出分词器的消息
    print("====Exporting tokenizer=====")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True)
    # 将分词器保存到指定的路径
    tokenizer.save_pretrained(ir_model_path)
```