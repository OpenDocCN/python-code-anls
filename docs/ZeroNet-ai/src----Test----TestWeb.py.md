# `ZeroNet\src\Test\TestWeb.py`

```
# 导入 urllib.request 模块
import urllib.request

# 导入 pytest 模块
import pytest

# 尝试导入 selenium 相关模块，如果失败则忽略
try:
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support.expected_conditions import staleness_of, title_is
    from selenium.common.exceptions import NoSuchElementException
except:
    pass

# 定义一个等待页面加载的类
class WaitForPageLoad(object):
    def __init__(self, browser):
        self.browser = browser

    # 进入上下文时，获取当前页面的元素
    def __enter__(self):
        self.old_page = self.browser.find_element_by_tag_name('html')

    # 退出上下文时，等待页面元素变化
    def __exit__(self, *args):
        WebDriverWait(self.browser, 10).until(staleness_of(self.old_page))

# 获取当前页面的 URL
def getContextUrl(browser):
    return browser.execute_script("return window.location.toString()")

# 根据 URL 获取页面内容
def getUrl(url):
    content = urllib.request.urlopen(url).read()
    # 断言页面内容中不包含 "server error"，否则抛出异常
    assert "server error" not in content.lower(), "Got a server error! " + repr(url)
    return content

# 使用 resetSettings fixture 和 webtest 标记进行测试
@pytest.mark.usefixtures("resetSettings")
@pytest.mark.webtest
class TestWeb:
    # 测试文件安全性，检查是否能够通过 URL 直接访问敏感文件
    assert "Not Found" in getUrl("%s/media/sites.json" % site_url)
    assert "Forbidden" in getUrl("%s/media/./sites.json" % site_url)
    assert "Forbidden" in getUrl("%s/media/../config.py" % site_url)
    assert "Forbidden" in getUrl("%s/media/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/../sites.json" % site_url)
    assert "Forbidden" in getUrl("%s/media/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/..//sites.json" % site_url)
    assert "Forbidden" in getUrl("%s/media/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/../../zeronet.py" % site_url)

    assert "Not Found" in getUrl("%s/raw/sites.json" % site_url)
    assert "Forbidden" in getUrl("%s/raw/./sites.json" % site_url)
    assert "Forbidden" in getUrl("%s/raw/../config.py" % site_url)
    assert "Forbidden" in getUrl("%s/raw/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/../sites.json" % site_url)
    assert "Forbidden" in getUrl("%s/raw/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/..//sites.json" % site_url)
    assert "Forbidden" in getUrl("%s/raw/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/../../zeronet.py" % site_url)

    assert "Forbidden" in getUrl("%s/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/../sites.json" % site_url)
    assert "Forbidden" in getUrl("%s/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/..//sites.json" % site_url)
    assert "Forbidden" in getUrl("%s/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/../../zeronet.py" % site_url)

    assert "Forbidden" in getUrl("%s/content.db" % site_url)
    assert "Forbidden" in getUrl("%s/./users.json" % site_url)
    assert "Forbidden" in getUrl("%s/./key-rsa.pem" % site_url)
    assert "Forbidden" in getUrl("%s/././././././././././//////sites.json" % site_url)
    # 定义一个测试函数，接受浏览器对象和网站 URL 作为参数
    def testRaw(self, browser, site_url):
        # 使用浏览器对象打开指定的网站原始页面
        browser.get("%s/raw/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/test/security.html" % site_url)
        # 等待页面标题为"Security tests"，最多等待10秒
        WebDriverWait(browser, 10).until(title_is("Security tests"))
        # 断言当前页面的 URL 与指定的原始页面 URL 相同
        assert getContextUrl(browser) == "%s/raw/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/test/security.html" % site_url

        # 断言页面中 id 为"script_output"的元素的文本内容为"Result: Fail"
        assert browser.find_element_by_id("script_output").text == "Result: Fail"
```