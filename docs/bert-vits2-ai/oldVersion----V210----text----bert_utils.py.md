# `d:/src/tocomm/Bert-VITS2\oldVersion\V210\text\bert_utils.py`

```
from pathlib import Path  # 导入 Path 模块，用于处理文件路径
from huggingface_hub import hf_hub_download  # 从 huggingface_hub 模块中导入 hf_hub_download 函数
from config import config  # 从 config 模块中导入 config 变量

MIRROR: str = config.mirror  # 设置 MIRROR 变量为 config 模块中的 mirror 变量值

def _check_bert(repo_id, files, local_path):
    for file in files:  # 遍历 files 列表中的文件
        if not Path(local_path).joinpath(file).exists():  # 如果本地路径下的文件不存在
            if MIRROR.lower() == "openi":  # 如果 MIRROR 变量的值为 "openi"
                import openi  # 导入 openi 模块

                openi.model.download_model(  # 调用 openi 模块中的 download_model 函数
                    "Stardust_minus/Bert-VITS2", repo_id.split("/")[-1], "./bert"
                )
            else:  # 如果 MIRROR 变量的值不为 "openi"
# 从 HF Hub 下载指定 repo_id 的文件到本地目录 local_dir 中
hf_hub_download(
    repo_id, file, local_dir=local_path, local_dir_use_symlinks=False
)
```