# `Bert-VITS2\oldVersion\V220\text\bert_utils.py`

```py
# 从 pathlib 模块中导入 Path 类
from pathlib import Path
# 从 huggingface_hub 模块中导入 hf_hub_download 函数
from huggingface_hub import hf_hub_download
# 从 config 模块中导入 config 变量
from config import config

# 定义 MIRROR 变量，并赋值为 config.mirror
MIRROR: str = config.mirror

# 定义 _check_bert 函数，接受 repo_id、files 和 local_path 三个参数
def _check_bert(repo_id, files, local_path):
    # 遍历 files 列表
    for file in files:
        # 如果 local_path 下的 file 文件不存在
        if not Path(local_path).joinpath(file).exists():
            # 如果 MIRROR 变量的值转换为小写后等于 "openi"
            if MIRROR.lower() == "openi":
                # 导入 openi 模块
                import openi
                # 调用 openi.model.download_model 函数下载模型
                openi.model.download_model(
                    "Stardust_minus/Bert-VITS2", repo_id.split("/")[-1], "./bert"
                )
            # 如果 MIRROR 变量的值不等于 "openi"
            else:
                # 调用 hf_hub_download 函数下载模型文件
                hf_hub_download(
                    repo_id, file, local_dir=local_path, local_dir_use_symlinks=False
                )
```