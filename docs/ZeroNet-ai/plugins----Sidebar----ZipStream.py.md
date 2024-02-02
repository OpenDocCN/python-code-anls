# `ZeroNet\plugins\Sidebar\ZipStream.py`

```py
# 导入所需的模块
import io
import os
import zipfile

# 定义 ZipStream 类
class ZipStream(object):
    # 初始化方法
    def __init__(self, dir_path):
        # 设置初始属性
        self.dir_path = dir_path
        self.pos = 0
        self.buff_pos = 0
        # 创建一个 ZipFile 对象
        self.zf = zipfile.ZipFile(self, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        # 创建一个字节流对象
        self.buff = io.BytesIO()
        # 获取文件列表
        self.file_list = self.getFileList()

    # 获取文件列表的方法
    def getFileList(self):
        # 遍历指定目录下的文件
        for root, dirs, files in os.walk(self.dir_path):
            for file in files:
                file_path = root + "/" + file
                relative_path = os.path.join(os.path.relpath(root, self.dir_path), file)
                # 返回文件路径和相对路径
                yield file_path, relative_path
        # 关闭 ZipFile 对象
        self.zf.close()

    # 读取数据的方法
    def read(self, size=60 * 1024):
        # 遍历文件列表
        for file_path, relative_path in self.file_list:
            # 将文件写入 ZipFile 对象
            self.zf.write(file_path, relative_path)
            # 如果缓冲区大小达到指定大小，则退出循环
            if self.buff.tell() >= size:
                break
        # 将缓冲区指针移动到开头，读取数据并清空缓冲区
        self.buff.seek(0)
        back = self.buff.read()
        self.buff.truncate(0)
        self.buff.seek(0)
        self.buff_pos += len(back)
        return back

    # 写入数据的方法
    def write(self, data):
        self.pos += len(data)
        self.buff.write(data)

    # 返回当前位置的方法
    def tell(self):
        return self.pos

    # 移动文件指针的方法
    def seek(self, pos, whence=0):
        if pos >= self.buff_pos:
            self.buff.seek(pos - self.buff_pos, whence)
            self.pos = pos

    # 刷新缓冲区的方法
    def flush(self):
        pass

# 主程序入口
if __name__ == "__main__":
    # 创建 ZipStream 对象
    zs = ZipStream(".")
    # 打开输出文件
    out = open("out.zip", "wb")
    # 循环读取数据并写入输出文件
    while 1:
        data = zs.read()
        print("Write %s" % len(data))
        if not data:
            break
        out.write(data)
    out.close()
```