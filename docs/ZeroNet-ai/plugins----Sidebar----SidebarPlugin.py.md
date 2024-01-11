# `ZeroNet\plugins\Sidebar\SidebarPlugin.py`

```
# 导入正则表达式模块
import re
# 导入操作系统模块
import os
# 导入 HTML 模块
import html
# 导入系统模块
import sys
# 导入数学模块
import math
# 导入时间模块
import time
# 导入 JSON 模块
import json
# 导入 IO 模块
import io
# 导入 URL 编码模块
import urllib
# 导入 URL 解析模块
import urllib.parse

# 导入协程模块
import gevent

# 导入自定义工具模块
import util
# 从 Config 模块中导入 config 对象
from Config import config
# 从 Plugin 模块中导入 PluginManager 对象
from Plugin import PluginManager
# 从 Debug 模块中导入 Debug 对象
from Debug import Debug
# 从 Translate 模块中导入 Translate 对象
from Translate import Translate
# 从 util 模块中导入 helper 函数
from util import helper
# 从 util.Flag 模块中导入 flag 对象
from util.Flag import flag
# 从当前目录下的 ZipStream 模块中导入 ZipStream 类
from .ZipStream import ZipStream

# 获取插件目录路径
plugin_dir = os.path.dirname(__file__)
# 拼接插件媒体文件目录路径
media_dir = plugin_dir + "/media"

# 本地缓存字典
loc_cache = {}
# 如果当前作用域中不存在下划线变量
if "_" not in locals():
    # 将翻译函数赋值给下划线变量
    _ = Translate(plugin_dir + "/languages/")

# 将 UiRequestPlugin 类注册到 PluginManager 的 UiRequest 插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 在原始文件流的末尾注入我们的资源
    # 定义一个处理 UI 媒体文件的方法，接受文件路径作为参数
    def actionUiMedia(self, path):
        # 如果路径是 "/uimedia/all.js" 或 "/uimedia/all.css"
        if path == "/uimedia/all.js" or path == "/uimedia/all.css":
            # 首先生成原始文件和头部信息
            body_generator = super(UiRequestPlugin, self).actionUiMedia(path)
            # 遍历生成器，逐个返回生成的部分
            for part in body_generator:
                yield part

            # 在末尾添加我们的媒体文件
            # 通过正则表达式匹配文件扩展名
            ext = re.match(".*(js|css)$", path).group(1)
            # 根据配置文件中的调试模式，决定是否合并媒体文件
            plugin_media_file = "%s/all.%s" % (media_dir, ext)
            if config.debug:
                # 如果是调试模式，将 *.css 合并到 all.css，将 *.js 合并到 all.js
                from Debug import DebugMedia
                DebugMedia.merge(plugin_media_file)
            # 如果是 js 文件，则返回经过翻译后的数据
            if ext == "js":
                yield _.translateData(open(plugin_media_file).read()).encode("utf8")
            else:
                # 如果是 css 文件，则返回文件的生成器，不发送头部信息
                for part in self.actionFile(plugin_media_file, send_header=False):
                    yield part
        # 如果路径以 "/uimedia/globe/" 开头，用于提供 WebGL 球体文件
        elif path.startswith("/uimedia/globe/"):
            # 通过正则表达式匹配文件名
            file_name = re.match(".*/(.*)", path).group(1)
            # 构建球体文件的路径
            plugin_media_file = "%s_globe/%s" % (media_dir, file_name)
            # 如果是调试模式，并且路径以 "all.js" 结尾，则合并媒体文件
            if config.debug and path.endswith("all.js"):
                from Debug import DebugMedia
                DebugMedia.merge(plugin_media_file)
            # 返回球体文件的生成器
            for part in self.actionFile(plugin_media_file):
                yield part
        else:
            # 对于其他路径，返回原始文件的生成器
            for part in super(UiRequestPlugin, self).actionUiMedia(path):
                yield part
    # 定义一个方法用于处理 ZIP 操作
    def actionZip(self):
        # 从请求参数中获取地址
        address = self.get["address"]
        # 从服务器站点管理器中获取指定地址的站点
        site = self.server.site_manager.get(address)
        # 如果站点不存在，则返回 404 错误
        if not site:
            return self.error404("Site not found")

        # 从站点内容管理器中获取内容文件中的标题
        title = site.content_manager.contents.get("content.json", {}).get("title", "")
        # 根据标题和当前时间生成备份文件名
        filename = "%s-backup-%s.zip" % (title, time.strftime("%Y-%m-%d_%H_%M"))
        # 对文件名进行 URL 编码
        filename_quoted = urllib.parse.quote(filename)
        # 发送 ZIP 文件的响应头，指定文件名
        self.sendHeader(content_type="application/zip", extra_headers={'Content-Disposition': 'attachment; filename="%s"' % filename_quoted})

        # 调用 streamZip 方法，传入站点存储路径，返回 ZIP 流
        return self.streamZip(site.storage.getPath("."))

    # 定义一个方法用于流式传输 ZIP 文件
    def streamZip(self, dir_path):
        # 创建一个 ZipStream 对象，传入目录路径
        zs = ZipStream(dir_path)
        # 循环读取 ZIP 流中的数据
        while 1:
            data = zs.read()
            # 如果没有数据了，则退出循环
            if not data:
                break
            # 返回数据
            yield data
# 将 UiWebsocketPlugin 类注册到 PluginManager 的 UiWebsocket 插件中
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # 侧边栏渲染数据传输统计
    def sidebarRenderTransferStats(self, body, site):
        # 计算接收的数据量（MB）
        recv = float(site.settings.get("bytes_recv", 0)) / 1024 / 1024
        # 计算发送的数据量（MB）
        sent = float(site.settings.get("bytes_sent", 0)) / 1024 / 1024
        # 计算总的数据传输量（MB）
        transfer_total = recv + sent
        # 如果有数据传输
        if transfer_total:
            # 计算接收数据量占总传输量的百分比
            percent_recv = recv / transfer_total
            # 计算发送数据量占总传输量的百分比
            percent_sent = sent / transfer_total
        else:
            # 如果没有数据传输，则默认设置接收和发送数据量占比为 50%
            percent_recv = 0.5
            percent_sent = 0.5

        # 在页面 body 中添加数据传输统计的 HTML 内容
        body.append(_("""
            <li>
             <label>{_[Data transfer]}</label>
             <ul class='graph graph-stacked'>
              <li style='width: {percent_recv:.0%}' class='received back-yellow' title="{_[Received bytes]}"></li>
              <li style='width: {percent_sent:.0%}' class='sent back-green' title="{_[Sent bytes]}"></li>
             </ul>
             <ul class='graph-legend'>
              <li class='color-yellow'><span>{_[Received]}:</span><b>{recv:.2f}MB</b></li>
              <li class='color-green'<span>{_[Sent]}:</span><b>{sent:.2f}MB</b></li>
             </ul>
            </li>
        """))

    # 侧边栏渲染大小限制
    def sidebarRenderSizeLimit(self, body, site):
        # 获取剩余空间（MB）
        free_space = helper.getFreeSpace() / 1024 / 1024
        # 获取站点设置的大小（MB）
        size = float(site.settings["size"]) / 1024 / 1024
        # 获取站点的大小限制
        size_limit = site.getSizeLimit()
        # 计算站点大小占大小限制的百分比
        percent_used = size / size_limit

        # 在页面 body 中添加大小限制的 HTML 内容
        body.append(_("""
            <li>
             <label>{_[Size limit]} <small>({_[limit used]}: {percent_used:.0%}, {_[free space]}: {free_space:,.0f}MB)</small></label>
             <input type='text' class='text text-num' value="{size_limit}" id='input-sitelimit'/><span class='text-post'>MB</span>
             <a href='#Set' class='button' id='button-sitelimit'>{_[Set]}</a>
            </li>
        """))
    # 渲染可选文件的统计信息到侧边栏
    def sidebarRenderOptionalFileStats(self, body, site):
        # 获取可选文件的总大小和已下载大小
        size_total = float(site.settings["size_optional"])
        size_downloaded = float(site.settings["optional_downloaded"])

        # 如果总大小为0，则返回False
        if not size_total:
            return False

        # 计算已下载大小占总大小的百分比
        percent_downloaded = size_downloaded / size_total

        # 将总大小和已下载大小转换为MB单位
        size_formatted_total = size_total / 1024 / 1024
        size_formatted_downloaded = size_downloaded / 1024 / 1024

        # 将可选文件的统计信息添加到body中
        body.append(_("""
            <li>
             <label>{_[Optional files]}</label>
             <ul class='graph'>
              <li style='width: 100%' class='total back-black' title="{_[Total size]}"></li>
              <li style='width: {percent_downloaded:.0%}' class='connected back-green' title='{_[Downloaded files]}'></li>
             </ul>
             <ul class='graph-legend'>
              <li class='color-green'><span>{_[Downloaded]}:</span><b>{size_formatted_downloaded:.2f}MB</b></li>
              <li class='color-black'><span>{_[Total]}:</span><b>{size_formatted_total:.2f}MB</b></li>
             </ul>
            </li>
        """))

        # 返回True表示渲染成功
        return True
    # 渲染侧边栏可选文件设置，将结果添加到 body 中，site 为站点对象
    def sidebarRenderOptionalFileSettings(self, body, site):
        # 如果站点设置中包含 "autodownloadoptional"，则设置 checked 为 "checked='checked'"，否则为空字符串
        if self.site.settings.get("autodownloadoptional"):
            checked = "checked='checked'"
        else:
            checked = ""

        # 在 body 中添加 HTML 代码，包括复选框和标签
        body.append(_("""
            <li>
             <label>{_[Help distribute added optional files]}</label>
             <input type="checkbox" class="checkbox" id="checkbox-autodownloadoptional" {checked}/><div class="checkbox-skin"></div>
        """))

        # 如果配置中包含 "autodownload_bigfile_size_limit"，则获取其值，否则使用默认值
        if hasattr(config, "autodownload_bigfile_size_limit"):
            autodownload_bigfile_size_limit = int(site.settings.get("autodownload_bigfile_size_limit", config.autodownload_bigfile_size_limit))
            # 在 body 中添加 HTML 代码，包括输入框、标签和按钮
            body.append(_("""
                <div class='settings-autodownloadoptional'>
                 <label>{_[Auto download big file size limit]}</label>
                 <input type='text' class='text text-num' value="{autodownload_bigfile_size_limit}" id='input-autodownload_bigfile_size_limit'/><span class='text-post'>MB</span>
                 <a href='#Set' class='button' id='button-autodownload_bigfile_size_limit'>{_[Set]}</a>
                 <a href='#Download+previous' class='button' id='button-autodownload_previous'>{_[Download previous files]}</a>
                </div>
            """))
        # 在 body 中添加 HTML 代码，闭合 li 标签
        body.append("</li>")
    # 在侧边栏渲染出有问题的文件列表
    def sidebarRenderBadFiles(self, body, site):
        # 添加一个列表项和一个标签，用于显示需要更新的文件
        body.append(_("""
            <li>
             <label>{_[Needs to be updated]}:</label>
             <ul class='filelist'>
        """))

        # 初始化计数器
        i = 0
        # 遍历有问题的文件和尝试次数的字典
        for bad_file, tries in site.bad_files.items():
            # 计数器加一
            i += 1
            # 添加一个有问题文件的列表项，显示文件名、尝试次数和文件路径
            body.append(_("""<li class='color-red' title="{bad_file_path} ({tries})">{bad_filename}</li>""", {
                "bad_file_path": bad_file,
                "bad_filename": helper.getFilename(bad_file),
                "tries": _.pluralize(tries, "{} try", "{} tries")
            }))
            # 如果已经显示了30个文件，则退出循环
            if i > 30:
                break

        # 如果有问题的文件数量超过30个，则显示剩余文件数量
        if len(site.bad_files) > 30:
            num_bad_files = len(site.bad_files) - 30
            body.append(_("""<li class='color-red'>{_[+ {num_bad_files} more]}</li>""", nested=True))

        # 添加列表项的结束标签
        body.append("""
             </ul>
            </li>
        """)

    # 在侧边栏渲染出数据库选项
    def sidebarRenderDbOptions(self, body, site):
        # 如果存在数据库
        if site.storage.db:
            # 获取数据库的内部路径
            inner_path = site.storage.getInnerPath(site.storage.db.db_path)
            # 获取数据库文件的大小
            size = float(site.storage.getSize(inner_path)) / 1024
            # 获取数据库中的订阅源数量
            feeds = len(site.storage.db.schema.get("feeds", {}))
        else:
            # 如果不存在数据库，则显示相应提示
            inner_path = _["No database found"]
            size = 0.0
            feeds = 0

        # 添加数据库选项的标签和相关信息
        body.append(_("""
            <li>
             <label>{_[Database]} <small>({size:.2f}kB, {_[search feeds]}: {_[{feeds} query]})</small></label>
             <div class='flex'>
              <input type='text' class='text disabled' value="{inner_path}" disabled='disabled'/>
              <a href='#Reload' id="button-dbreload" class='button'>{_[Reload]}</a>
              <a href='#Rebuild' id="button-dbrebuild" class='button'>{_[Rebuild]}</a>
             </div>
            </li>
        """, nested=True))
    # 侧边栏渲染身份信息
    def sidebarRenderIdentity(self, body, site):
        # 获取用户的认证地址
        auth_address = self.user.getAuthAddress(self.site.address, create=False)
        # 获取用户的规则
        rules = self.site.content_manager.getRules("data/users/%s/content.json" % auth_address)
        # 如果规则存在并且有最大大小限制
        if rules and rules.get("max_size"):
            # 计算配额大小（单位：KB）
            quota = rules["max_size"] / 1024
            try:
                # 获取用户内容
                content = site.content_manager.contents["data/users/%s/content.json" % auth_address]
                # 计算已使用空间大小（单位：KB）
                used = len(json.dumps(content)) + sum([file["size"] for file in list(content["files"].values())])
            except:
                # 如果获取内容失败，则已使用空间为0
                used = 0
            # 将已使用空间大小转换为KB
            used = used / 1024
        else:
            # 如果规则不存在或没有最大大小限制，则配额和已使用空间都为0
            quota = used = 0

        # 将身份信息添加到页面主体
        body.append(_("""
            <li>
             <label>{_[Identity address]} <small>({_[limit used]}: {used:.2f}kB / {quota:.2f}kB)</small></label>
             <div class='flex'>
              <span class='input text disabled'>{auth_address}</span>
              <a href='#Change' class='button' id='button-identity'>{_[Change]}</a>
             </div>
            </li>
        """))
    # 渲染侧边栏控件
    def sidebarRenderControls(self, body, site):
        # 获取用户的授权地址
        auth_address = self.user.getAuthAddress(self.site.address, create=False)
        # 如果站点正在提供服务
        if self.site.settings["serving"]:
            # 设置暂停按钮为隐藏
            class_pause = ""
            # 设置恢复按钮为显示
            class_resume = "hidden"
        else:
            # 设置暂停按钮为显示
            class_pause = "hidden"
            # 设置恢复按钮为隐藏
            class_resume = ""

        # 添加站点控制按钮到页面
        body.append(_("""
            <li>
             <label>{_[Site control]}</label>
             <a href='#Update' class='button noupdate' id='button-update'>{_[Update]}</a>
             <a href='#Pause' class='button {class_pause}' id='button-pause'>{_[Pause]}</a>
             <a href='#Resume' class='button {class_resume}' id='button-resume'>{_[Resume]}</a>
             <a href='#Delete' class='button noupdate' id='button-delete'>{_[Delete]}</a>
            </li>
        """))

        # 获取站点的捐赠密钥
        donate_key = site.content_manager.contents.get("content.json", {}).get("donate", True)
        site_address = self.site.address
        # 添加站点地址到页面
        body.append(_("""
            <li>
             <label>{_[Site address]}</label><br>
             <div class='flex'>
              <span class='input text disabled'>{site_address}</span>
        """))
        # 如果没有捐赠密钥或者密钥为空，则不添加捐赠按钮
        if donate_key == False or donate_key == "":
            pass
        # 如果捐赠密钥是字符串且长度大于0，则添加捐赠按钮
        elif (type(donate_key) == str or type(donate_key) == str) and len(donate_key) > 0:
            body.append(_("""
             </div>
            </li>
            <li>
             <label>{_[Donate]}</label><br>
             <div class='flex'>
             {donate_key}
            """))
        # 否则添加默认的捐赠按钮
        else:
            body.append(_("""
              <a href='bitcoin:{site_address}' class='button' id='button-donate'>{_[Donate]}</a>
            """))
        # 添加结束标签到页面
        body.append(_("""
             </div>
            </li>
        """))
    # 渲染侧边栏拥有的复选框
    def sidebarRenderOwnedCheckbox(self, body, site):
        # 如果站点设置中包含"own"字段
        if self.site.settings["own"]:
            # 设置复选框为选中状态
            checked = "checked='checked'"
        else:
            # 设置复选框为未选中状态
            checked = ""

        # 将标题和复选框添加到侧边栏内容中
        body.append(_("""
            <h2 class='owned-title'>{_[This is my site]}</h2>
            <input type="checkbox" class="checkbox" id="checkbox-owned" {checked}/><div class="checkbox-skin"></div>
        """))

    # 渲染侧边栏拥有设置
    def sidebarRenderOwnSettings(self, body, site):
        # 获取站点标题和描述
        title = site.content_manager.contents.get("content.json", {}).get("title", "")
        description = site.content_manager.contents.get("content.json", {}).get("description", "")

        # 将标题、描述和保存按钮添加到侧边栏内容中
        body.append(_("""
            <li>
             <label for='settings-title'>{_[Site title]}</label>
             <input type='text' class='text' value="{title}" id='settings-title'/>
            </li>

            <li>
             <label for='settings-description'>{_[Site description]}</label>
             <input type='text' class='text' value="{description}" id='settings-description'/>
            </li>

            <li>
             <a href='#Save' class='button' id='button-settings'>{_[Save site settings]}</a>
            </li>
        """))
    # 侧边栏渲染内容函数，接受 body 和 site 两个参数
    def sidebarRenderContents(self, body, site):
        # 检查用户在指定站点是否有私钥，返回布尔值
        has_privatekey = bool(self.user.getSiteData(site.address, create=False).get("privatekey"))
        # 如果有私钥，则创建包含私钥保存提示和忘记私钥链接的标签
        if has_privatekey:
            tag_privatekey = _("{_[Private key saved.]} <a href='#Forget+private+key' id='privatekey-forget' class='link-right'>{_[Forget]}</a>")
        # 如果没有私钥，则创建包含添加私钥链接的标签
        else:
            tag_privatekey = _("<a href='#Add+private+key' id='privatekey-add' class='link-right'>{_[Add saved private key]}</a>")

        # 将包含标签的内容添加到 body 中
        body.append(_("""
            <li>
             <label>{_[Content publishing]} <small class='label-right'>{tag_privatekey}</small></label>
        """.replace("{tag_privatekey}", tag_privatekey)))

        # 选择要签名的内容
        body.append(_("""
             <div class='flex'>
              <input type='text' class='text' value="content.json" id='input-contents'/>
              <a href='#Sign-and-Publish' id='button-sign-publish' class='button'>{_[Sign and publish]}</a>
              <a href='#Sign-or-Publish' id='menu-sign-publish'>\u22EE</a>
             </div>
        """))

        # 获取站点内容管理器中的内容列表，并将其添加到 body 中
        contents = ["content.json"]
        contents += list(site.content_manager.contents.get("content.json", {}).get("includes", {}).keys())
        body.append(_("<div class='contents'>{_[Choose]}: "))
        for content in contents:
            body.append(_("<a href='{content}' class='contents-content'>{content}</a> "))
        body.append("</div>")
        body.append("</li>")

    # 标记为管理员权限的装饰器
    @flag.admin
    # 定义一个方法，用于生成侧边栏的 HTML 标签
    def actionSidebarGetHtmlTag(self, to):
        # 获取当前站点信息
        site = self.site

        # 创建一个空列表，用于存储生成的 HTML 内容
        body = []

        # 添加一个 div 标签
        body.append("<div>")
        # 添加一个关闭按钮
        body.append("<a href='#Close' class='close'>&times;</a>")
        # 添加一个 h1 标签，显示站点的标题
        body.append("<h1>%s</h1>" % html.escape(site.content_manager.contents.get("content.json", {}).get("title", ""), True))

        # 添加一个 div 标签，用于显示加载状态
        body.append("<div class='globe loading'></div>")

        # 添加一个无序列表标签
        body.append("<ul class='fields'>")

        # 调用其他方法，生成侧边栏的各个部分内容，并添加到 body 列表中
        self.sidebarRenderPeerStats(body, site)
        self.sidebarRenderTransferStats(body, site)
        self.sidebarRenderFileStats(body, site)
        self.sidebarRenderSizeLimit(body, site)
        # 调用方法，生成可选文件统计信息，并返回是否存在可选文件
        has_optional = self.sidebarRenderOptionalFileStats(body, site)
        # 如果存在可选文件，则调用方法，生成可选文件设置内容
        if has_optional:
            self.sidebarRenderOptionalFileSettings(body, site)
        # 调用方法，生成数据库选项内容
        self.sidebarRenderDbOptions(body, site)
        # 调用方法，生成身份信息内容
        self.sidebarRenderIdentity(body, site)
        # 调用方法，生成控制按钮内容
        self.sidebarRenderControls(body, site)
        # 如果存在错误文件，则调用方法，生成错误文件内容
        if site.bad_files:
            self.sidebarRenderBadFiles(body, site)

        # 调用方法，生成拥有文件复选框内容
        self.sidebarRenderOwnedCheckbox(body, site)
        # 添加一个 div 标签，用于显示拥有文件设置
        body.append("<div class='settings-owned'>")
        # 调用方法，生成拥有文件设置内容
        self.sidebarRenderOwnSettings(body, site)
        # 调用方法，生成内容列表内容
        self.sidebarRenderContents(body, site)
        body.append("</div>")
        body.append("</ul>")
        body.append("</div>")

        # 添加一个 div 标签，用于显示模板菜单
        body.append("<div class='menu template'>")
        body.append("<a href='#'' class='menu-item template'>Template</a>")
        body.append("</div>")

        # 调用方法，将生成的 HTML 内容返回给指定的位置
        self.response(to, "".join(body))
    # 获取 IP 地理位置信息
    def getLoc(self, geodb, ip):
        # 声明全局变量 loc_cache
        global loc_cache

        # 如果 IP 地址在 loc_cache 中，则直接返回对应的位置信息
        if ip in loc_cache:
            return loc_cache[ip]
        else:
            # 否则，尝试从 geodb 中获取 IP 地址的位置信息
            try:
                loc_data = geodb.get(ip)
            except:
                loc_data = None

            # 如果未获取到位置信息或者 loc_data 中不包含 "location" 键，则将 loc_cache 中对应的 IP 地址设为 None，并返回 None
            if not loc_data or "location" not in loc_data:
                loc_cache[ip] = None
                return None

            # 从 loc_data 中提取经纬度信息，组成 loc 字典
            loc = {
                "lat": loc_data["location"]["latitude"],
                "lon": loc_data["location"]["longitude"],
            }
            # 如果 loc_data 中包含 "city" 键，则将城市信息添加到 loc 字典中
            if "city" in loc_data:
                loc["city"] = loc_data["city"]["names"]["en"]

            # 如果 loc_data 中包含 "country" 键，则将国家信息添加到 loc 字典中
            if "country" in loc_data:
                loc["country"] = loc_data["country"]["names"]["en"]

            # 将 loc 存入 loc_cache 中，并返回 loc
            loc_cache[ip] = loc
            return loc

    # 获取地理位置数据库
    @util.Noparallel()
    def getGeoipDb(self):
        # 地理位置数据库文件名
        db_name = 'GeoLite2-City.mmdb'

        # 系统数据库路径列表
        sys_db_paths = []
        # 如果系统平台为 Linux，则将系统数据库路径添加到列表中
        if sys.platform == "linux":
            sys_db_paths += ['/usr/share/GeoIP/' + db_name]

        # 数据目录中的数据库路径
        data_dir_db_path = os.path.join(config.data_dir, db_name)

        # 数据库路径列表
        db_paths = sys_db_paths + [data_dir_db_path]

        # 遍历数据库路径列表，找到存在且大小大于 0 的数据库文件，则返回该路径
        for path in db_paths:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path

        # 如果未找到有效的数据库文件，则记录日志并尝试下载数据库文件到数据目录中
        self.log.info("GeoIP database not found at [%s]. Downloading to: %s",
                " ".join(db_paths), data_dir_db_path)
        if self.downloadGeoLiteDb(data_dir_db_path):
            return data_dir_db_path
        return None
    # 获取对等节点的地理位置信息
    def getPeerLocations(self, peers):
        # 导入 maxminddb 模块
        import maxminddb

        # 获取 GeoIP 数据库路径
        db_path = self.getGeoipDb()
        # 如果没有数据库路径，则记录日志并返回 False
        if not db_path:
            self.log.debug("Not showing peer locations: no GeoIP database")
            return False

        # 打开 GeoIP 数据库
        geodb = maxminddb.open_database(db_path)

        # 将 peers 字典的值转换为列表
        peers = list(peers.values())
        # 存储对等节点的地理位置信息
        peer_locations = []
        # 已经放置过的位置
        placed = {}  # Already placed bars here
        # 遍历对等节点
        for peer in peers:
            # 计算条形图的高度
            if peer.connection and peer.connection.last_ping_delay:
                ping = round(peer.connection.last_ping_delay * 1000)
            else:
                ping = None
            # 获取对等节点的地理位置信息
            loc = self.getLoc(geodb, peer.ip)

            # 如果没有地理位置信息，则继续下一个循环
            if not loc:
                continue
            # 创建位置数组
            lat, lon = loc["lat"], loc["lon"]
            latlon = "%s,%s" % (lat, lon)
            # 如果已经放置过条形图，并且对等节点的 IP 类型为 ipv4，则不再放置
            if latlon in placed and helper.getIpType(peer.ip) == "ipv4":  # Dont place more than 1 bar to same place, fake repos using ip address last two part
                lat += float(128 - int(peer.ip.split(".")[-2])) / 50
                lon += float(128 - int(peer.ip.split(".")[-1])) / 50
                latlon = "%s,%s" % (lat, lon)
            placed[latlon] = True
            # 存储对等节点的地理位置信息和 ping 值
            peer_location = {}
            peer_location.update(loc)
            peer_location["lat"] = lat
            peer_location["lon"] = lon
            peer_location["ping"] = ping

            peer_locations.append(peer_location)

        # 添加自己的地理位置信息
        for ip in self.site.connection_server.ip_external_list:
            my_loc = self.getLoc(geodb, ip)
            if my_loc:
                my_loc["ping"] = 0
                peer_locations.append(my_loc)

        # 返回对等节点的地理位置信息列表
        return peer_locations

    # 标记为管理员权限
    @flag.admin
    # 标记为异步运行
    @flag.async_run
    # 从给定的站点获取对等节点的位置信息
    def actionSidebarGetPeers(self, to):
        try:
            # 获取对等节点的位置信息
            peer_locations = self.getPeerLocations(self.site.peers)
            globe_data = []
            # 获取所有对等节点的ping时间
            ping_times = [
                peer_location["ping"]
                for peer_location in peer_locations
                if peer_location["ping"]
            ]
            # 如果存在ping时间，则计算平均ping时间
            if ping_times:
                ping_avg = sum(ping_times) / float(len(ping_times))
            else:
                ping_avg = 0

            # 根据对等节点的ping时间计算高度信息
            for peer_location in peer_locations:
                if peer_location["ping"] == 0:  # Me
                    height = -0.135
                elif peer_location["ping"]:
                    height = min(0.20, math.log(1 + peer_location["ping"] / ping_avg, 300))
                else:
                    height = -0.03

                # 将对等节点的位置信息添加到globe_data中
                globe_data += [peer_location["lat"], peer_location["lon"], height]

            # 返回对等节点的位置信息
            self.response(to, globe_data)
        except Exception as err:
            # 记录错误日志并返回错误信息
            self.log.debug("sidebarGetPeers error: %s" % Debug.formatException(err))
            self.response(to, {"error": str(err)})

    # 设置站点的所有权
    @flag.admin
    @flag.no_multiuser
    def actionSiteSetOwned(self, to, owned):
        # 如果站点地址为更新站点，则无法更改所有权
        if self.site.address == config.updatesite:
            return {"error": "You can't change the ownership of the updater site"}

        # 设置站点的所有权，并更新websocket
        self.site.settings["own"] = bool(owned)
        self.site.updateWebsocket(owned=owned)
        return "ok"

    # 设置站点的所有权
    @flag.admin
    @flag.no_multiuser
    # 从 Crypt 模块中导入 CryptBitcoin 类
    def actionSiteRecoverPrivatekey(self, to):
        from Crypt import CryptBitcoin

        # 获取当前用户在指定站点的数据
        site_data = self.user.sites[self.site.address]
        # 如果站点数据中已经存在私钥，则返回错误信息
        if site_data.get("privatekey"):
            return {"error": "This site already has saved privated key"}

        # 获取 content.json 文件中的 address_index
        address_index = self.site.content_manager.contents.get("content.json", {}).get("address_index")
        # 如果不存在 address_index，则返回错误信息
        if not address_index:
            return {"error": "No address_index in content.json"}

        # 根据用户的 master_seed 和 address_index 生成私钥
        privatekey = CryptBitcoin.hdPrivatekey(self.user.master_seed, address_index)
        # 将私钥转换为地址
        privatekey_address = CryptBitcoin.privatekeyToAddress(privatekey)

        # 如果生成的地址与站点地址相同，则将私钥保存到站点数据中，并更新用户数据和站点的 WebSocket
        if privatekey_address == self.site.address:
            site_data["privatekey"] = privatekey
            self.user.save()
            self.site.updateWebsocket(recover_privatekey=True)
            return "ok"
        else:
            # 如果生成的地址与站点地址不同，则返回错误信息
            return {"error": "Unable to deliver private key for this site from current user's master_seed"}

    # 标记为管理员权限，禁止多用户访问
    @flag.admin
    @flag.no_multiuser
    def actionUserSetSitePrivatekey(self, to, privatekey):
        # 将私钥保存到站点数据中，并更新用户数据和站点的 WebSocket
        site_data = self.user.sites[self.site.address]
        site_data["privatekey"] = privatekey
        self.site.updateWebsocket(set_privatekey=bool(privatekey))
        self.user.save()

        return "ok"

    # 标记为管理员权限，禁止多用户访问
    @flag.admin
    @flag.no_multiuser
    def actionSiteSetAutodownloadoptional(self, to, owned):
        # 设置站点的 autodownloadoptional 属性，并移除已解决的文件任务
        self.site.settings["autodownloadoptional"] = bool(owned)
        self.site.worker_manager.removeSolvedFileTasks()

    # 禁止多用户访问，标记为管理员权限
    @flag.no_multiuser
    @flag.admin
    def actionDbReload(self, to):
        # 关闭并重新获取站点的数据库
        self.site.storage.closeDb()
        self.site.storage.getDb()

        return self.response(to, "ok")

    # 禁止多用户访问，标记为管理员权限
    @flag.no_multiuser
    @flag.admin
    def actionDbRebuild(self, to):
        try:
            # 尝试重建站点的数据库
            self.site.storage.rebuildDb()
        except Exception as err:
            # 如果出现异常，则返回错误信息
            return self.response(to, {"error": str(err)})

        return self.response(to, "ok")
```