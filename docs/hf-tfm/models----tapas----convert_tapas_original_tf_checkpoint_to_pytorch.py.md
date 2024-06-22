# `.\transformers\models\tapas\convert_tapas_original_tf_checkpoint_to_pytorch.py`

```py
# 导入需要的库和模块
import argparse

# 导入转换所需的类和函数
from transformers import (
    TapasConfig,
    TapasForMaskedLM,
    TapasForQuestionAnswering,
    TapasForSequenceClassification,
    TapasModel,
    TapasTokenizer,
    load_tf_weights_in_tapas,
)
from transformers.utils import logging

# 设置日志的详细程度为 info
logging.set_verbosity_info()

# 定义一个函数用于将 TensorFlow 的检查点转换为 PyTorch 格式
def convert_tf_checkpoint_to_pytorch(
    task, reset_position_index_per_cell, tf_checkpoint_path, tapas_config_file, pytorch_dump_path
):
    # 初始化 PyTorch 模型
    # 如果你要转换的检查点使用了绝对位置嵌入，确保将 TapasConfig 的 reset_position_index_per_cell 设置为 False
    
    # 从 JSON 文件中初始化配置
    config = TapasConfig.from_json_file(tapas_config_file)
    # 设置绝对/相对位置嵌入参数
    config.reset_position_index_per_cell = reset_position_index_per_cell

    # 设置 TapasConfig 的余下参数以及基于任务的模型
    if task == "SQA":
        model = TapasForQuestionAnswering(config=config)
    elif task == "WTQ":
        # run_task_main.py hparams
        config.num_aggregation_labels = 4
        config.use_answer_as_supervision = True
        # hparam_utils.py hparams
        config.answer_loss_cutoff = 0.664694
        config.cell_selection_preference = 0.207951
        config.huber_loss_delta = 0.121194
        config.init_cell_selection_weights_to_zero = True
        config.select_one_column = True
        config.allow_empty_column_selection = False
        config.temperature = 0.0352513

        model = TapasForQuestionAnswering(config=config)
    elif task == "WIKISQL_SUPERVISED":
        # run_task_main.py hparams
        config.num_aggregation_labels = 4
        config.use_answer_as_supervision = False
        # hparam_utils.py hparams
        config.answer_loss_cutoff = 36.4519
        config.cell_selection_preference = 0.903421
        config.huber_loss_delta = 222.088
        config.init_cell_selection_weights_to_zero = True
        config.select_one_column = True
        config.allow_empty_column_selection = True
        config.temperature = 0.763141

        model = TapasForQuestionAnswering(config=config)
    elif task == "TABFACT":
        model = TapasForSequenceClassification(config=config)
    elif task == "MLM":
        model = TapasForMaskedLM(config=config)
    # 如果任务为“INTERMEDIATE_PRETRAINING”，则创建一个 TapasModel 对象，使用给定的配置参数
    elif task == "INTERMEDIATE_PRETRAINING":
        model = TapasModel(config=config)
    # 如果任务不是“INTERMEDIATE_PRETRAINING”，则抛出 ValueError 异常，提示任务不支持
    else:
        raise ValueError(f"Task {task} not supported.")

    # 打印正在根据配置构建的 PyTorch 模型
    print(f"Building PyTorch model from configuration: {config}")

    # 从 TensorFlow checkpoint 中加载权重到 TapasModel 对象
    load_tf_weights_in_tapas(model, config, tf_checkpoint_path)

    # 将 PyTorch 模型（权重和配置）保存到指定路径
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

    # 保存分词器文件到指定路径
    print(f"Save tokenizer files to {pytorch_dump_path}")
    # 创建 TapasTokenizer 对象，使用 TensorFlow checkpoint 路径中的词汇文件和最大长度为 512
    tokenizer = TapasTokenizer(vocab_file=tf_checkpoint_path[:-10] + "vocab.txt", model_max_length=512)
    tokenizer.save_pretrained(pytorch_dump_path)

    # 打印是否使用相对位置嵌入
    print("Used relative position embeddings:", model.config.reset_position_index_per_cell)
```  
# 判断当前脚本是否在主程序中执行
if __name__ == "__main__":
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--task", default="SQA", type=str, help="Model task for which to convert a checkpoint. Defaults to SQA."
    )
    # 添加必需的参数
    parser.add_argument(
        "--reset_position_index_per_cell",
        default=False,
        action="store_true",
        help="Whether to use relative position embeddings or not. Defaults to True.",
    )
    # 添加必需的参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    # 添加必需的参数
    parser.add_argument(
        "--tapas_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained TAPAS model. \n"
            "This specifies the model architecture."
        ),
    )
    # 添加必需的参数
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数，并将其保存在 args 中
    args = parser.parse_args()
    # 调用函数，将 TensorFlow 检查点文件转换为 PyTorch 模型
    convert_tf_checkpoint_to_pytorch(
        args.task,
        args.reset_position_index_per_cell,
        args.tf_checkpoint_path,
        args.tapas_config_file,
        args.pytorch_dump_path,
    )
```