# `ZeroNet\plugins\OptionalManager\Test\TestOptionalManager.py`

```
# 导入 copy 模块
import copy
# 导入 pytest 模块
import pytest

# 使用 resetSettings 修饰器来重置设置
@pytest.mark.usefixtures("resetSettings")
class TestOptionalManager:
    # 测试数据库填充功能
    def testDbFill(self, site):
        # 获取站点内容管理器中的内容
        contents = site.content_manager.contents
        # 断言哈希字段的长度大于 0
        assert len(site.content_manager.hashfield) > 0
        # 断言已下载文件的数量与哈希字段的长度相等
        assert contents.db.execute("SELECT COUNT(*) FROM file_optional WHERE is_downloaded = 1").fetchone()[0] == len(site.content_manager.hashfield)

    # 测试设置内容功能
    def testSetContent(self, site):
        # 获取站点内容管理器中的内容
        contents = site.content_manager.contents

        # 添加新文件
        new_content = copy.deepcopy(contents["content.json"])
        new_content["files_optional"]["testfile"] = {
            "size": 1234,
            "sha512": "aaaabbbbcccc"
        }
        # 获取添加文件前可选文件的数量
        num_optional_files_before = contents.db.execute("SELECT COUNT(*) FROM file_optional").fetchone()[0]
        # 更新内容中的 content.json
        contents["content.json"] = new_content
        # 断言更新后可选文件的数量大于添加文件前的数量
        assert contents.db.execute("SELECT COUNT(*) FROM file_optional").fetchone()[0] > num_optional_files_before

        # 删除文件
        new_content = copy.deepcopy(contents["content.json"])
        del new_content["files_optional"]["testfile"]
        # 获取删除文件前可选文件的数量
        num_optional_files_before = contents.db.execute("SELECT COUNT(*) FROM file_optional").fetchone()[0]
        # 更新内容中的 content.json
        contents["content.json"] = new_content
        # 断言更新后可选文件的数量小于删除文件前的数量
        assert contents.db.execute("SELECT COUNT(*) FROM file_optional").fetchone()[0] < num_optional_files_before

    # 测试删除内容功能
    def testDeleteContent(self, site):
        # 获取站点内容管理器中的内容
        contents = site.content_manager.contents
        # 获取删除内容前可选文件的数量
        num_optional_files_before = contents.db.execute("SELECT COUNT(*) FROM file_optional").fetchone()[0]
        # 删除内容中的 content.json
        del contents["content.json"]
        # 断言删除内容后可选文件的数量小于删除内容前的数量
        assert contents.db.execute("SELECT COUNT(*) FROM file_optional").fetchone()[0] < num_optional_files_before
    # 测试验证文件的方法，接受站点对象作为参数
    def testVerifyFiles(self, site):
        # 获取站点内容管理器中的内容
        contents = site.content_manager.contents

        # 添加新文件
        new_content = copy.deepcopy(contents["content.json"])
        new_content["files_optional"]["testfile"] = {
            "size": 1234,
            "sha512": "aaaabbbbcccc"
        }
        contents["content.json"] = new_content
        # 从内容管理器中的数据库中查询文件信息，确保文件未下载
        file_row = contents.db.execute("SELECT * FROM file_optional WHERE inner_path = 'testfile'").fetchone()
        assert not file_row["is_downloaded"]

        # 从ZeroNet外部写入文件
        site.storage.open("testfile", "wb").write(b"A" * 1234)  # 用于快速检查哈希值无关紧要，只需文件大小
        # 记录验证前哈希字段的长度
        hashfield_len_before = len(site.content_manager.hashfield)
        # 验证文件，进行快速检查
        site.storage.verifyFiles(quick_check=True)
        # 断言验证后哈希字段的长度增加了1
        assert len(site.content_manager.hashfield) == hashfield_len_before + 1
        # 从内容管理器中的数据库中查询文件信息，确保文件已下载
        file_row = contents.db.execute("SELECT * FROM file_optional WHERE inner_path = 'testfile'").fetchone()
        assert file_row["is_downloaded"]

        # 从ZeroNet外部删除文件
        site.storage.delete("testfile")
        # 再次验证文件，进行快速检查
        site.storage.verifyFiles(quick_check=True)
        # 从内容管理器中的数据库中查询文件信息，确保文件未下载
        file_row = contents.db.execute("SELECT * FROM file_optional WHERE inner_path = 'testfile'").fetchone()
        assert not file_row["is_downloaded"]
    # 测试验证文件的相同哈希 ID
    def testVerifyFilesSameHashId(self, site):
        # 获取站点内容管理器中的内容
        contents = site.content_manager.contents

        # 深拷贝content.json文件内容
        new_content = copy.deepcopy(contents["content.json"])

        # 添加两个具有相同哈希ID（前4个字符）的文件
        new_content["files_optional"]["testfile1"] = {
            "size": 1234,
            "sha512": "aaaabbbbcccc"
        }
        new_content["files_optional"]["testfile2"] = {
            "size": 2345,
            "sha512": "aaaabbbbdddd"
        }
        contents["content.json"] = new_content

        # 断言两个哈希ID相同
        assert site.content_manager.hashfield.getHashId("aaaabbbbcccc") == site.content_manager.hashfield.getHashId("aaaabbbbdddd")

        # 从ZeroNet外部写入文件（仅用于快速检查哈希值不重要，只需文件大小）
        site.storage.open("testfile1", "wb").write(b"A" * 1234)
        site.storage.open("testfile2", "wb").write(b"B" * 2345)

        # 验证文件（快速检查）
        site.storage.verifyFiles(quick_check=True)

        # 确保两个文件都已下载
        assert site.content_manager.isDownloaded("testfile1")
        assert site.content_manager.isDownloaded("testfile2")
        assert site.content_manager.hashfield.getHashId("aaaabbbbcccc") in site.content_manager.hashfield

        # 删除其中一个文件
        site.storage.delete("testfile1")
        site.storage.verifyFiles(quick_check=True)
        assert not site.content_manager.isDownloaded("testfile1")
        assert site.content_manager.isDownloaded("testfile2")
        assert site.content_manager.hashfield.getHashId("aaaabbbbdddd") in site.content_manager.hashfield
    # 测试给定站点是否已经固定了指定文件
    def testIsPinned(self, site):
        # 断言指定文件未被固定
        assert not site.content_manager.isPinned("data/img/zerotalk-upvote.png")
        # 将指定文件固定
        site.content_manager.setPin("data/img/zerotalk-upvote.png", True)
        # 再次断言指定文件已被固定
        assert site.content_manager.isPinned("data/img/zerotalk-upvote.png")

        # 断言站点内容管理器缓存中已固定文件的数量为1
        assert len(site.content_manager.cache_is_pinned) == 1
        # 清空站点内容管理器缓存
        site.content_manager.cache_is_pinned = {}
        # 再次断言指定文件已被固定
        assert site.content_manager.isPinned("data/img/zerotalk-upvote.png")

    # 测试大文件分片重置
    def testBigfilePieceReset(self, site):
        # 设置站点的坏文件字典
        site.bad_files = {
            "data/fake_bigfile.mp4|0-1024": 10,
            "data/fake_bigfile.mp4|1024-2048": 10,
            "data/fake_bigfile.mp4|2048-3064": 10
        }
        # 调用文件完成处理函数
        site.onFileDone("data/fake_bigfile.mp4|0-1024")
        # 断言站点的坏文件字典中指定分片的值为1
        assert site.bad_files["data/fake_bigfile.mp4|1024-2048"] == 1
        assert site.bad_files["data/fake_bigfile.mp4|2048-3064"] == 1

    # 测试可选删除
    def testOptionalDelete(self, site):
        # 获取站点内容管理器的内容
        contents = site.content_manager.contents
        # 将指定文件固定或取消固定
        site.content_manager.setPin("data/img/zerotalk-upvote.png", True)
        site.content_manager.setPin("data/img/zeroid.png", False)
        # 深拷贝内容管理器中的内容
        new_content = copy.deepcopy(contents["content.json"])
        # 删除内容中的可选文件
        del new_content["files_optional"]["data/img/zerotalk-upvote.png"]
        del new_content["files_optional"]["data/img/zeroid.png"]

        # 断言指定文件存在于存储中
        assert site.storage.isFile("data/img/zerotalk-upvote.png")
        assert site.storage.isFile("data/img/zeroid.png")

        # 将新内容写入存储
        site.storage.writeJson("content.json", new_content)
        # 加载新内容到内容管理器中
        site.content_manager.loadContent("content.json", force=True)

        # 断言指定文件不存在于存储中
        assert not site.storage.isFile("data/img/zeroid.png")
        assert site.storage.isFile("data/img/zerotalk-upvote.png")
    # 测试可选重命名功能，传入站点对象
    def testOptionalRename(self, site):
        # 获取站点内容管理器的内容
        contents = site.content_manager.contents

        # 设置指定文件为固定文件
        site.content_manager.setPin("data/img/zerotalk-upvote.png", True)
        
        # 深拷贝内容管理器中的 content.json 文件
        new_content = copy.deepcopy(contents["content.json"])
        
        # 将新文件名添加到可选文件中，并复制原文件的数据
        new_content["files_optional"]["data/img/zerotalk-upvote-new.png"] = new_content["files_optional"]["data/img/zerotalk-upvote.png"]
        
        # 删除原文件名对应的可选文件
        del new_content["files_optional"]["data/img/zerotalk-upvote.png"]

        # 断言原文件存在并且为固定文件
        assert site.storage.isFile("data/img/zerotalk-upvote.png")
        assert site.content_manager.isPinned("data/img/zerotalk-upvote.png")

        # 将修改后的 content.json 文件写入存储
        site.storage.writeJson("content.json", new_content)
        
        # 强制加载修改后的 content.json 文件
        site.content_manager.loadContent("content.json", force=True)

        # 断言原文件不存在并且不是固定文件，新文件存在并且是固定文件
        assert not site.storage.isFile("data/img/zerotalk-upvote.png")
        assert not site.content_manager.isPinned("data/img/zerotalk-upvote.png")
        assert site.content_manager.isPinned("data/img/zerotalk-upvote-new.png")
        assert site.storage.isFile("data/img/zerotalk-upvote-new.png")
```