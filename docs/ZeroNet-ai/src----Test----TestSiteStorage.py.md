# `ZeroNet\src\Test\TestSiteStorage.py`

```
# 导入 pytest 模块
import pytest

# 使用 resetSettings 修饰器来重置设置
@pytest.mark.usefixtures("resetSettings")
class TestSiteStorage:
    # 测试遍历文件系统
    def testWalk(self, site):
        # Rootdir
        # 获取根目录下的文件列表
        walk_root = list(site.storage.walk(""))
        # 断言 content.json 存在于根目录下
        assert "content.json" in walk_root
        # 断言 css/all.css 存在于根目录下
        assert "css/all.css" in walk_root

        # Subdir
        # 断言 data-default 目录下的文件列表
        assert list(site.storage.walk("data-default")) == ["data.json", "users/content-default.json"]

    # 测试列出文件
    def testList(self, site):
        # Rootdir
        # 获取根目录下的文件列表
        list_root = list(site.storage.list(""))
        # 断言 content.json 存在于根目录下
        assert "content.json" in list_root
        # 断言 css/all.css 不存在于根目录下
        assert "css/all.css" not in list_root

        # Subdir
        # 断言 data-default 目录下的文件列表
        assert set(site.storage.list("data-default")) == set(["data.json", "users"])

    # 测试重建数据库
    def testDbRebuild(self, site):
        # 断言重建数据库成功
        assert site.storage.rebuildDb()
```