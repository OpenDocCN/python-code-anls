# `so-vits-svc\wav_upload.py`

```
# 导入必要的模块
import argparse  # 用于解析命令行参数
import os  # 用于操作文件路径
import shutil  # 用于移动文件

from google.colab import files  # 从谷歌云端硬盘导入文件

# 如果作为独立程序运行
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数，指定上传文件的类型
    parser.add_argument("--type", type=str, required=True, help="type of file to upload")
    # 解析命令行参数
    args = parser.parse_args()
    # 获取文件类型
    file_type = args.type

    # 获取当前工作目录
    basepath = os.getcwd()
    # 上传文件
    uploaded = files.upload()
    # 确保文件类型是 'zip' 或 'audio'
    assert(file_type in ['zip', 'audio'])
    # 如果文件类型是 'zip'
    if file_type == "zip":
        # 设置上传路径
        upload_path = "./upload/"
        # 遍历上传的文件
        for filename in uploaded.keys():
            # 将上传的文件移动到指定的位置上
            shutil.move(os.path.join(basepath, filename), os.path.join(upload_path, "userzip.zip"))
    # 如果文件类型是 'audio'
    elif file_type == "audio":
        # 设置上传路径
        upload_path = "./raw/"
        # 遍历上传的文件
        for filename in uploaded.keys():
            # 将上传的文件移动到指定的位置上
            shutil.move(os.path.join(basepath, filename), os.path.join(upload_path, filename))
```