# `ZeroNet\plugins\Benchmark\BenchmarkPack.py`

```py
# 导入所需的模块
import os
import io
from collections import OrderedDict
# 从自定义的插件管理器中导入插件基类
from Plugin import PluginManager
# 从配置文件中导入配置信息
from Config import config
# 从自定义的工具模块中导入消息打包工具
from util import Msgpack

# 将该类注册到插件管理器的"Actions"模块中
@PluginManager.registerTo("Actions")
class ActionsPlugin:
    # 创建一个 ZIP 文件的方法
    def createZipFile(self, path):
        # 导入 zipfile 模块
        import zipfile
        # 创建一个测试数据
        test_data = b"Test" * 1024
        # 创建一个文件名，包含特殊字符
        file_name = b"\xc3\x81rv\xc3\xadzt\xc5\xb1r\xc5\x91%s.txt".decode("utf8")
        # 使用 zipfile 模块创建一个 ZIP 文件，并写入数据
        with zipfile.ZipFile(path, 'w') as archive:
            for y in range(100):
                # 创建一个 ZipInfo 对象，设置文件名和其他属性
                zip_info = zipfile.ZipInfo(file_name % y, (1980, 1, 1, 0, 0, 0))
                zip_info.compress_type = zipfile.ZIP_DEFLATED
                zip_info.create_system = 3
                zip_info.flag_bits = 0
                zip_info.external_attr = 25165824
                # 将数据写入 ZIP 文件
                archive.writestr(zip_info, test_data)

    # 测试创建 ZIP 文件的方法
    def testPackZip(self, num_run=1):
        """
        Test zip file creating
        """
        # 生成一个消息，用于测试
        yield "x 100 x 5KB "
        # 导入加密哈希模块
        from Crypt import CryptHash
        # 设置 ZIP 文件的路径
        zip_path = '%s/test.zip' % config.data_dir
        # 循环创建 ZIP 文件
        for i in range(num_run):
            self.createZipFile(zip_path)
            yield "."

        # 获取生成的 ZIP 文件大小
        archive_size = os.path.getsize(zip_path) / 1024
        yield "(Generated file size: %.2fkB)" % archive_size

        # 计算 ZIP 文件的哈希值
        hash = CryptHash.sha512sum(open(zip_path, "rb"))
        # 设置一个有效的哈希值
        valid = "cb32fb43783a1c06a2170a6bc5bb228a032b67ff7a1fd7a5efb9b467b400f553"
        # 检查哈希值是否有效
        assert hash == valid, "Invalid hash: %s != %s<br>" % (hash, valid)
        # 删除生成的 ZIP 文件
        os.unlink(zip_path)
    # 定义一个测试函数，用于解压缩文件并读取内容
    def testUnpackZip(self, num_run=1):
        """
        Test zip file reading
        """
        # 生成器函数，用于生成测试结果
        yield "x 100 x 5KB "
        # 导入 zipfile 模块
        import zipfile
        # 拼接 ZIP 文件路径
        zip_path = '%s/test.zip' % config.data_dir
        # 创建测试数据
        test_data = b"Test" * 1024
        # 创建文件名
        file_name = b"\xc3\x81rv\xc3\xadzt\xc5\xb1r\xc5\x91".decode("utf8")

        # 创建 ZIP 文件
        self.createZipFile(zip_path)
        # 循环运行测试
        for i in range(num_run):
            # 使用 zipfile 模块打开 ZIP 文件
            with zipfile.ZipFile(zip_path) as archive:
                # 遍历 ZIP 文件中的文件列表
                for f in archive.filelist:
                    # 断言文件名以指定字符开头
                    assert f.filename.startswith(file_name), "Invalid filename: %s != %s" % (f.filename, file_name)
                    # 读取文件数据
                    data = archive.open(f.filename).read()
                    # 断言文件数据正确性
                    assert archive.open(f.filename).read() == test_data, "Invalid data: %s..." % data[0:30]
            # 生成测试结果
            yield "."

        # 删除 ZIP 文件
        os.unlink(zip_path)

    # 创建压缩文件
    def createArchiveFile(self, path, archive_type="gz"):
        # 导入 tarfile 和 gzip 模块
        import tarfile
        import gzip

        # 定义一个函数，用于修复 gzip 文件头中的日期信息，以保持哈希值与日期无关
        def nodate_write_gzip_header(self):
            self._write_mtime = 0
            original_write_gzip_header(self)

        # 创建测试数据的字节流
        test_data_io = io.BytesIO(b"Test" * 1024)
        # 创建文件名
        file_name = b"\xc3\x81rv\xc3\xadzt\xc5\xb1r\xc5\x91%s.txt".decode("utf8")

        # 保存原始的写入 gzip 文件头的函数
        original_write_gzip_header = gzip.GzipFile._write_gzip_header
        # 替换写入 gzip 文件头的函数
        gzip.GzipFile._write_gzip_header = nodate_write_gzip_header
        # 使用 tarfile 模块创建压缩文件
        with tarfile.open(path, 'w:%s' % archive_type) as archive:
            # 循环添加文件到压缩文件中
            for y in range(100):
                test_data_io.seek(0)
                # 创建文件信息
                tar_info = tarfile.TarInfo(file_name % y)
                tar_info.size = 4 * 1024
                # 添加文件到压缩文件中
                archive.addfile(tar_info, test_data_io)
    # 定义一个测试函数，用于创建压缩文件
    def testPackArchive(self, num_run=1, archive_type="gz"):
        """
        Test creating tar archive files
        """
        # 生成测试数据
        yield "x 100 x 5KB "
        # 导入加密模块
        from Crypt import CryptHash

        # 哈希验证字典
        hash_valid_db = {
            "gz": "92caec5121a31709cbbc8c11b0939758e670b055bbbe84f9beb3e781dfde710f",
            "bz2": "b613f41e6ee947c8b9b589d3e8fa66f3e28f63be23f4faf015e2f01b5c0b032d",
            "xz": "ae43892581d770959c8d993daffab25fd74490b7cf9fafc7aaee746f69895bcb",
        }
        # 生成压缩文件路径
        archive_path = '%s/test.tar.%s' % (config.data_dir, archive_type)
        # 循环创建压缩文件
        for i in range(num_run):
            self.createArchiveFile(archive_path, archive_type=archive_type)
            yield "."

        # 获取生成的压缩文件大小
        archive_size = os.path.getsize(archive_path) / 1024
        yield "(Generated file size: %.2fkB)" % archive_size

        # 计算生成文件的哈希值
        hash = CryptHash.sha512sum(open("%s/test.tar.%s" % (config.data_dir, archive_type), "rb"))
        valid = hash_valid_db[archive_type]
        # 检查哈希值是否有效
        assert hash == valid, "Invalid hash: %s != %s<br>" % (hash, valid)

        # 如果生成的压缩文件存在，则删除
        if os.path.isfile(archive_path):
            os.unlink(archive_path)

    # 定义一个测试函数，用于解压缩文件
    def testUnpackArchive(self, num_run=1, archive_type="gz"):
        """
        Test reading tar archive files
        """
        # 生成测试数据
        yield "x 100 x 5KB "
        # 导入tarfile模块
        import tarfile

        # 定义测试数据
        test_data = b"Test" * 1024
        file_name = b"\xc3\x81rv\xc3\xadzt\xc5\xb1r\xc5\x91%s.txt".decode("utf8")
        # 生成压缩文件路径
        archive_path = '%s/test.tar.%s' % (config.data_dir, archive_type)
        # 创建压缩文件
        self.createArchiveFile(archive_path, archive_type=archive_type)
        # 循环解压缩文件
        for i in range(num_run):
            with tarfile.open(archive_path, 'r:%s' % archive_type) as archive:
                for y in range(100):
                    # 读取并比较解压后的文件内容
                    assert archive.extractfile(file_name % y).read() == test_data
            yield "."
        # 如果生成的压缩文件存在，则删除
        if os.path.isfile(archive_path):
            os.unlink(archive_path)
```