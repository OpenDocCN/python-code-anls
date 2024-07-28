# `.\comic-translate\modules\utils\file_handler.py`

```py
import os
import shutil
import tempfile
import string
from datetime import datetime
from typing import List
from .archives import extract_archive

class FileHandler:
    def __init__(self):
        self.file_paths = []  # 初始化一个空列表，用于存储文件路径
        self.archive_info = []  # 初始化一个空列表，用于存储归档信息
    
    def prepare_files(self):
        all_image_paths = []  # 初始化一个空列表，用于存储所有图像文件路径
        
        # 遍历文件路径列表
        for path in self.file_paths:
            # 检查文件路径是否是支持的归档文件格式
            if path.lower().endswith(('.cbr', '.cbz', '.zip', '.cbt', '.cb7', '.pdf', '.epub')):
                print('Extracting archive:', path)
                archive_dir = os.path.dirname(path)  # 获取归档文件的目录路径
                temp_dir = tempfile.mkdtemp(dir=archive_dir)  # 在归档文件的目录中创建临时目录
                
                extracted_files = extract_archive(path, temp_dir)  # 提取归档文件中的所有文件到临时目录
                image_paths = [f for f in extracted_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]  # 筛选出图像文件路径
                image_paths = self.sanitize_and_copy_files(image_paths)  # 调用方法对图像文件进行处理和复制
                
                all_image_paths.extend(image_paths)  # 将处理后的图像文件路径添加到总列表中
                # 添加归档信息到归档信息列表
                self.archive_info.append({
                    'archive_path': path,
                    'extracted_images': image_paths,
                    'temp_dir': temp_dir
                })
            else:
                # 对于非归档文件，调用方法进行处理和复制
                path = self.sanitize_and_copy_files([path])[0]
                all_image_paths.append(path)  # 将处理后的文件路径添加到总列表中
        
        self.file_paths = all_image_paths  # 更新文件路径列表为处理后的所有文件路径
        return self.file_paths  # 返回处理后的所有文件路径列表

    def sanitize_and_copy_files(self, file_paths):
        sanitized_paths = []  # 初始化一个空列表，用于存储处理后的文件路径
        for index, image_path in enumerate(file_paths):
            if not image_path.isascii():
                # 如果文件路径不是 ASCII 编码，则进行字符清理
                name = ''.join(c for c in image_path if c in string.printable)
                dir_name = ''.join(c for c in os.path.dirname(image_path) if c in string.printable)
                os.makedirs(dir_name, exist_ok=True)  # 创建目录（如果不存在）
                if os.path.splitext(os.path.basename(name))[1] == '':
                    basename = ""
                    ext = os.path.splitext(os.path.basename(name))[0]
                else:
                    basename = os.path.splitext(os.path.basename(name))[0]
                    ext = os.path.splitext(os.path.basename(name))[1]
                # 构造清理后的文件路径
                sanitized_path = os.path.join(dir_name, basename + str(index) + ext)
                try:
                    shutil.copy(image_path, sanitized_path)  # 复制文件到清理后的路径
                    image_path = sanitized_path  # 更新文件路径为清理后的路径
                except IOError as e:
                    print(f"An error occurred while copying or deleting the file: {e}")
            sanitized_paths.append(image_path)  # 将处理后的文件路径添加到列表中

        return sanitized_paths  # 返回处理后的所有文件路径列表
```