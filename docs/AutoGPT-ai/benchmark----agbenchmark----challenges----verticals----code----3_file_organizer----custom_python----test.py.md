# `.\AutoGPT\benchmark\agbenchmark\challenges\verticals\code\3_file_organizer\custom_python\test.py`

```py
# 导入必要的模块
import os
import subprocess
import tempfile
import unittest

# 定义测试类 TestOrganizeFiles，继承自 unittest.TestCase
class TestOrganizeFiles(unittest.TestCase):
    # 在每个测试方法执行前执行的方法
    def setUp(self):
        # 创建临时目录
        self.test_dir = tempfile.mkdtemp()

        # 文件类型及其对应的目录
        self.file_types = {
            "test_image.png": "images",
            "test_doc.txt": "documents",
            "test_audio.mp3": "audio",
        }

        # 创建测试文件
        for file_name in self.file_types.keys():
            open(os.path.join(self.test_dir, file_name), "a").close()

    # 测试方法，用于测试文件整理功能
    def test_organize_files(self):
        # 调用 organize_files.py 脚本使用 subprocess
        subprocess.call(
            ["python", "organize_files.py", "--directory_path=" + self.test_dir]
        )

        # 检查文件是否已移动到正确的目录
        for file_name, directory in self.file_types.items():
            self.assertTrue(
                os.path.isfile(os.path.join(self.test_dir, directory, file_name))
            )

    # 在每个测试方法执行后执行的方法
    def tearDown(self):
        # 删除测试目录及其内容
        for file_name, directory in self.file_types.items():
            os.remove(os.path.join(self.test_dir, directory, file_name))
        for directory in set(self.file_types.values()):
            os.rmdir(os.path.join(self.test_dir, directory))
        os.rmdir(self.test_dir)

# 如果当前脚本被直接执行，则执行测试
if __name__ == "__main__":
    unittest.main()
```