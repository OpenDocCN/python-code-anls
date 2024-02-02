# `ZeroNet\plugins\ContentFilter\Test\TestContentFilter.py`

```py
# 导入 pytest 模块
import pytest
# 从 ContentFilter 模块中导入 ContentFilterPlugin 类
from ContentFilter import ContentFilterPlugin
# 从 Site 模块中导入 SiteManager 类
from Site import SiteManager

# 定义一个装置函数，用于创建并返回 ContentFilterPlugin 的 filter_storage 对象
@pytest.fixture
def filter_storage():
    # 将 ContentFilterPlugin 的 filter_storage 属性设置为 ContentFilterPlugin 的 ContentFilterStorage 对象
    ContentFilterPlugin.filter_storage = ContentFilterPlugin.ContentFilterStorage(SiteManager.site_manager)
    return ContentFilterPlugin.filter_storage

# 使用 resetSettings 装置函数重置设置
@pytest.mark.usefixtures("resetSettings")
# 使用 resetTempSettings 装置函数重置临时设置
@pytest.mark.usefixtures("resetTempSettings")
# 定义 TestContentFilter 类
class TestContentFilter:
    # 定义 createInclude 方法，用于创建包含特定内容的 JSON 文件
    def createInclude(self, site):
        site.storage.writeJson("filters.json", {
            "mutes": {"1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C": {}},
            "siteblocks": {site.address: {}}
        })

    # 定义 testIncludeLoad 方法，用于测试加载包含特定内容的 JSON 文件
    def testIncludeLoad(self, site, filter_storage):
        # 调用 createInclude 方法创建包含特定内容的 JSON 文件
        self.createInclude(site)
        # 将 filter_storage 对象的 file_content 属性中的特定键值对设置为指定值
        filter_storage.file_content["includes"]["%s/%s" % (site.address, "filters.json")] = {
            "date_added": 1528295893,
        }

        # 断言特定条件是否成立
        assert not filter_storage.include_filters["mutes"]
        assert not filter_storage.isMuted("1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C")
        assert not filter_storage.isSiteblocked(site.address)
        # 调用 filter_storage 对象的 includeUpdateAll 方法，更新所有包含的过滤器
        filter_storage.includeUpdateAll(update_site_dbs=False)
        assert len(filter_storage.include_filters["mutes"]) == 1
        assert filter_storage.isMuted("1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C")
        assert filter_storage.isSiteblocked(site.address)
    # 定义测试函数，测试添加包含规则的功能
    def testIncludeAdd(self, site, filter_storage):
        # 创建包含规则
        self.createInclude(site)
        # 定义查询 JSON 文件中特定目录下的记录数量的 SQL 语句
        query_num_json = "SELECT COUNT(*) AS num FROM json WHERE directory = 'users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C'"
        # 断言网站地址未被屏蔽
        assert not filter_storage.isSiteblocked(site.address)
        # 断言用户未被静音
        assert not filter_storage.isMuted("1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C")
        # 断言查询到的 JSON 文件中特定目录下的记录数量为2
        assert site.storage.query(query_num_json).fetchone()["num"] == 2

        # 添加包含规则
        filter_storage.includeAdd(site.address, "filters.json")

        # 断言网站地址已被屏蔽
        assert filter_storage.isSiteblocked(site.address)
        # 断言用户已被静音
        assert filter_storage.isMuted("1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C")
        # 断言查询到的 JSON 文件中特定目录下的记录数量为0
        assert site.storage.query(query_num_json).fetchone()["num"] == 0

        # 移除包含规则
        filter_storage.includeRemove(site.address, "filters.json")

        # 断言网站地址未被屏蔽
        assert not filter_storage.isSiteblocked(site.address)
        # 断言用户未被静音
        assert not filter_storage.isMuted("1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C")
        # 断言查询到的 JSON 文件中特定目录下的记录数量为2
        assert site.storage.query(query_num_json).fetchone()["num"] == 2
    # 测试包含变更的情况，传入站点和过滤器存储对象
    def testIncludeChange(self, site, filter_storage):
        # 创建包含站点
        self.createInclude(site)
        # 向过滤器存储对象添加包含站点的过滤器文件
        filter_storage.includeAdd(site.address, "filters.json")
        # 断言包含站点是否被阻止
        assert filter_storage.isSiteblocked(site.address)
        # 断言特定用户是否被静音
        assert filter_storage.isMuted("1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C")

        # 添加新的被阻止站点
        assert not filter_storage.isSiteblocked("1Hello")

        # 读取过滤器文件的内容
        filter_content = site.storage.loadJson("filters.json")
        # 向过滤器文件内容中添加新的被阻止站点
        filter_content["siteblocks"]["1Hello"] = {}
        # 将更新后的过滤器文件内容写入存储
        site.storage.writeJson("filters.json", filter_content)

        # 断言新的被阻止站点是否被阻止
        assert filter_storage.isSiteblocked("1Hello")

        # 添加新的被静音用户
        query_num_json = "SELECT COUNT(*) AS num FROM json WHERE directory = 'users/1C5sgvWaSgfaTpV5kjBCnCiKtENNMYo69q'"
        assert not filter_storage.isMuted("1C5sgvWaSgfaTpV5kjBCnCiKtENNMYo69q")
        assert site.storage.query(query_num_json).fetchone()["num"] == 2

        # 向过滤器文件内容中添加新的被静音用户
        filter_content["mutes"]["1C5sgvWaSgfaTpV5kjBCnCiKtENNMYo69q"] = {}
        # 将更新后的过滤器文件内容写入存储
        site.storage.writeJson("filters.json", filter_content)

        # 断言新的被静音用户是否被静音
        assert filter_storage.isMuted("1C5sgvWaSgfaTpV5kjBCnCiKtENNMYo69q")
        # 断言查询特定 JSON 目录的结果是否为 0
        assert site.storage.query(query_num_json).fetchone()["num"] == 0
```