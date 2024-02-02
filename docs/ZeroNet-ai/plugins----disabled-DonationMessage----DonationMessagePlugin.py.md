# `ZeroNet\plugins\disabled-DonationMessage\DonationMessagePlugin.py`

```py
# 导入正则表达式模块
import re
# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager

# 警告：如果修改了捐赠地址，请将插件目录重命名为 "MyDonationMessage"，以防止更新脚本覆盖

# 将 UiRequestPlugin 类注册到 PluginManager 的 UiRequest 插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 在每个页面的右上角注入捐赠信息
    def renderWrapper(self, *args, **kwargs):
        # 获取包装框架的输出
        body = super(UiRequestPlugin, self).renderWrapper(*args, **kwargs)

        # 注入的 HTML 代码
        inject_html = """
            <style>
             #donation_message { position: absolute; bottom: 0px; right: 20px; padding: 7px; font-family: Arial; font-size: 11px }
            </style>
            <a id='donation_message' href='https://blockchain.info/address/1QDhxQ6PraUZa21ET5fYUCPgdrwBomnFgX' target='_blank'>Please donate to help to keep this ZeroProxy alive</a>
            </body>
            </html>
        """

        # 使用正则表达式替换页面中的 </body> 和 </html> 标签为注入的 HTML 代码
        return re.sub(r"</body>\s*</html>\s*$", inject_html, body)
```