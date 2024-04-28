# `.\transformers\commands\run.py`

```
# 导入必要的模块和类
from argparse import ArgumentParser

# 导入自定义模块
from ..pipelines import Pipeline, PipelineDataFormat, get_supported_tasks, pipeline
# 导入日志工具
from ..utils import logging
# 导入基础命令类
from . import BaseTransformersCLICommand

# 获取日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 从文件扩展名推断数据格式
def try_infer_format_from_ext(path: str):
    # 如果路径为空，返回管道格式
    if not path:
        return "pipe"

    # 遍历支持的格式
    for ext in PipelineDataFormat.SUPPORTED_FORMATS:
        # 如果路径以支持的格式结尾，返回该格式
        if path.endswith(ext):
            return ext

    # 如果无法推断格式，抛出异常
    raise Exception(
        f"Unable to determine file format from file extension {path}. "
        f"Please provide the format through --format {PipelineDataFormat.SUPPORTED_FORMATS}"
    )

# 创建运行命令的工厂函数
def run_command_factory(args):
    # 构建管道
    nlp = pipeline(
        task=args.task,
        model=args.model if args.model else None,
        config=args.config,
        tokenizer=args.tokenizer,
        device=args.device,
    )
    # 推断输入文件格式或使用指定格式
    format = try_infer_format_from_ext(args.input) if args.format == "infer" else args.format
    # 创建数据格式对象
    reader = PipelineDataFormat.from_str(
        format=format,
        output_path=args.output,
        input_path=args.input,
        column=args.column if args.column else nlp.default_input_names,
        overwrite=args.overwrite,
    )
    # 返回运行命令对象
    return RunCommand(nlp, reader)

# 运行命令类
class RunCommand(BaseTransformersCLICommand):
    # 初始化函数
    def __init__(self, nlp: Pipeline, reader: PipelineDataFormat):
        self._nlp = nlp
        self._reader = reader

    # 静态方法
    @staticmethod
    # 向命令行解析器添加子命令"run"，用于通过命令行界面运行管道
    def register_subcommand(parser: ArgumentParser):
        # 添加一个名为"run"的子命令，帮助信息为"Run a pipeline through the CLI"
        run_parser = parser.add_parser("run", help="Run a pipeline through the CLI")
        # 添加"--task"参数，用于指定要运行的任务，选项从支持的任务中选择
        run_parser.add_argument("--task", choices=get_supported_tasks(), help="Task to run")
        # 添加"--input"参数，用于指定用于推理的文件路径
        run_parser.add_argument("--input", type=str, help="Path to the file to use for inference")
        # 添加"--output"参数，用于指定将用于写入结果的文件路径
        run_parser.add_argument("--output", type=str, help="Path to the file that will be used post to write results.")
        # 添加"--model"参数，用于指定要实例化的模型的名称或路径
        run_parser.add_argument("--model", type=str, help="Name or path to the model to instantiate.")
        # 添加"--config"参数，用于指定要实例化的模型配置的名称或路径
        run_parser.add_argument("--config", type=str, help="Name or path to the model's config to instantiate.")
        # 添加"--tokenizer"参数，用于指定要使用的分词器的名称，默认与模型名称相同
        run_parser.add_argument(
            "--tokenizer", type=str, help="Name of the tokenizer to use. (default: same as the model name)"
        )
        # 添加"--column"参数，用于指定用作输入的列的名称，对于多列输入，如问答系统，使用"column1,column2"格式
        run_parser.add_argument(
            "--column",
            type=str,
            help="Name of the column to use as input. (For multi columns input as QA use column1,columns2)",
        )
        # 添加"--format"参数，用于指定从中读取的输入格式，默认为"infer"，选项从PipelineDataFormat.SUPPORTED_FORMATS中选择
        run_parser.add_argument(
            "--format",
            type=str,
            default="infer",
            choices=PipelineDataFormat.SUPPORTED_FORMATS,
            help="Input format to read from",
        )
        # 添加"--device"参数，用于指定要运行到的设备，-1表示CPU，>= 0表示GPU，默认为-1
        run_parser.add_argument("--device", type=int, default=-1, help="Indicate the device to run onto, -1 indicates CPU, >= 0 indicates GPU (default: -1)")
        # 添加"--overwrite"参数，用于允许覆盖输出文件
        run_parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file.")
        # 设置默认的命令处理函数为run_command_factory
        run_parser.set_defaults(func=run_command_factory)

    # 运行方法
    def run(self):
        # 获取自然语言处理对象和输出列表
        nlp, outputs = self._nlp, []
        
        # 遍历读取器的条目
        for entry in self._reader:
            # 如果读取器是多列的，则使用多个输入调用自然语言处理对象，否则使用单个输入调用自然语言处理对象
            output = nlp(**entry) if self._reader.is_multi_columns else nlp(entry)
            # 如果输出是字典，则将其添加到输出列表中，否则将其连接到输出列表中
            if isinstance(output, dict):
                outputs.append(output)
            else:
                outputs += output
        
        # 保存数据
        # 如果自然语言处理对象的输出是二进制格式
        if self._nlp.binary_output:
            # 保存输出为二进制格式，并获取保存路径
            binary_path = self._reader.save_binary(outputs)
            # 记录警告消息，指示当前管道需要输出为二进制格式，并指定保存路径
            logger.warning(f"Current pipeline requires output to be in binary format, saving at {binary_path}")
        else:
            # 保存输出
            self._reader.save(outputs)
```