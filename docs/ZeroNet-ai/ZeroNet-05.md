# ZeroNet源码解析 5

# `plugins/FilePack/__init__.py`

这段代码定义了一个名为 FilePackPlugin 的类，可能是一个软件包插件，用于在文件打包时执行某些操作。

但是，没有提供上下文和完整的代码，因此无法解释该代码的作用。


```py
from . import FilePackPlugin
```

# `plugins/MergerSite/MergerSitePlugin.py`

这段代码的作用是实现了一个插件，名为“Plugin”。它允许用户通过插件的方式启用或禁用某些功能，并在运行时进行一些辅助操作。下面是具体的代码实现：

1. 导入需要用到的库：re、time、copy、os、PluginManager、Translate、RateLimit、helper、flag、Debug和OptionalManager.UiWebsocketPlugin。

2. 定义了一些变量，包括：一个Regex类型的正则表达式，用于匹配文件中的某些文本、一个时间函数用于暂停执行一段时间、一个Copy类型的副本对象用于复制数据、一个带有文件路径参数的函数，用于获取文件对象、一个带有参数的函数，用于设置某个对象的计数器、一个RateLimit类型的对象，用于限制代码在一定时间内执行的次数、一个使用Debug函数输出调试信息的函数、一个正则表达式类型的选项字符串，用于设置插件的选项。

3. 加载插件管理器、翻译服务和一些辅助函数：这些函数在插件运行时需要使用。

4. 创建一个插件管理器对象，一个翻译对象，一个RateLimit对象和一个Debug对象。这些对象将在插件运行时使用。

5. 定义了一个方法PluginManager，它实现了插件管理器的作用。这个方法包含一些辅助方法，例如：获取当前插件选项的方法，设置限制的方法，检查插件是否已加载的方法等等。

6. 定义了一个方法Translate，它实现了翻译服务的作用。这个方法包含一些辅助方法，例如：将文本翻译成目标语言的方法。

7. 定义了一个方法RateLimit，它实现了RateLimit对象的作用。这个方法包含一些辅助方法，例如：设置计数器的方法，检查限速方法等等。

8. 定义了一个方法PluginHelper，它是一个辅助类，用于执行一些通用的任务。这个方法包含一些方法，例如：获取目标语言列表的方法，检查某个单词是否是字母的方法等等。

9. 定义了一个标志类RateLimit，用于设置计数器。这个类包含一些方法，例如：设置计数器的方法，检查限速方法等等。

10. 最后，定义了一个Debug类，用于实现调试信息的记录。这个类包含一些方法，例如：记录调试信息的方法，打印调试信息的方法等等。

11. 在try语句中，尝试引入OptionalManager.UiWebsocketPlugin。如果这个插件成功加载，那么就会导入这个类，否则就不会导入。

12. 在插件的代码中，还有一些文件被导入，例如RateLimit文件、PluginManager文件、Translate文件、PluginHelper文件等等。这些文件可能包含了插件实现的具体功能。


```py
import re
import time
import copy
import os

from Plugin import PluginManager
from Translate import Translate
from util import RateLimit
from util import helper
from util.Flag import flag
from Debug import Debug
try:
    import OptionalManager.UiWebsocketPlugin  # To make optioanlFileInfo merger sites compatible
except Exception:
    pass

```

This code appears to be a Python plugin for a file merger tool. It contains a `SiteManager` class that manages the loading and unloading of different modules and their associated sites.

The `SiteManager` class has a `checkMergerPath` method that checks whether a specific site


```py
if "merger_db" not in locals().keys():  # To keep merger_sites between module reloads
    merger_db = {}  # Sites that allowed to list other sites {address: [type1, type2...]}
    merged_db = {}  # Sites that allowed to be merged to other sites {address: type, ...}
    merged_to_merger = {}  # {address: [site1, site2, ...]} cache
    site_manager = None  # Site manager for merger sites


plugin_dir = os.path.dirname(__file__)

if "_" not in locals():
    _ = Translate(plugin_dir + "/languages/")


# Check if the site has permission to this merger site
def checkMergerPath(address, inner_path):
    merged_match = re.match("^merged-(.*?)/([A-Za-z0-9]{26,35})/", inner_path)
    if merged_match:
        merger_type = merged_match.group(1)
        # Check if merged site is allowed to include other sites
        if merger_type in merger_db.get(address, []):
            # Check if included site allows to include
            merged_address = merged_match.group(2)
            if merged_db.get(merged_address) == merger_type:
                inner_path = re.sub("^merged-(.*?)/([A-Za-z0-9]{26,35})/", "", inner_path)
                return merged_address, inner_path
            else:
                raise Exception(
                    "Merger site (%s) does not have permission for merged site: %s (%s)" %
                    (merger_type, merged_address, merged_db.get(merged_address))
                )
        else:
            raise Exception("No merger (%s) permission to load: <br>%s (%s not in %s)" % (
                address, inner_path, merger_type, merger_db.get(address, []))
            )
    else:
        raise Exception("Invalid merger path: %s" % inner_path)


```



This is a class that implements the `动作网站签名`功能，用于签到动作网站的相关信息，包括签到URL、签到信息等。

该接口有两个实现，分别为`actionSiteSign`和`actionSitePublish`，其中第一个实现使用了私钥，第二个实现没有使用私钥。

对于私钥的使用，需要在调用该接口时传入`privatekey`参数，如果传入了私钥，则使用私钥进行签名，否则使用默认的签名方式。

在签名过程中，使用到了`actionSiteSign`类中的`mergerFuncWrapperWithPrivatekey`，该函数将签名信息封装到`PrivateKey`对象中，并使用传入的私钥进行签名。

对于使用私钥的情况，需要先在服务器上创建一个`PrivateKey`对象，并使用该对象进行签名，获取签名信息后，再将签名信息返回给客户端。

该接口还提供了一个`actionPermissionAdd`方法，用于添加用户对应的权限信息，包括权限类型为`Merger`的情况，需要在用户登录后进行权限检查，并在权限检查通过后，将权限信息添加到服务器上的`sites.storage`对象中，以供后续签名使用。


```py
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # Download new site
    def actionMergerSiteAdd(self, to, addresses):
        if type(addresses) != list:
            # Single site add
            addresses = [addresses]
        # Check if the site has merger permission
        merger_types = merger_db.get(self.site.address)
        if not merger_types:
            return self.response(to, {"error": "Not a merger site"})

        if RateLimit.isAllowed(self.site.address + "-MergerSiteAdd", 10) and len(addresses) == 1:
            # Without confirmation if only one site address and not called in last 10 sec
            self.cbMergerSiteAdd(to, addresses)
        else:
            self.cmd(
                "confirm",
                [_["Add <b>%s</b> new site?"] % len(addresses), "Add"],
                lambda res: self.cbMergerSiteAdd(to, addresses)
            )
        self.response(to, "ok")

    # Callback of adding new site confirmation
    def cbMergerSiteAdd(self, to, addresses):
        added = 0
        for address in addresses:
            try:
                site_manager.need(address)
                added += 1
            except Exception as err:
                self.cmd("notification", ["error", _["Adding <b>%s</b> failed: %s"] % (address, err)])
        if added:
            self.cmd("notification", ["done", _["Added <b>%s</b> new site"] % added, 5000])
        RateLimit.called(self.site.address + "-MergerSiteAdd")
        site_manager.updateMergerSites()

    # Delete a merged site
    @flag.no_multiuser
    def actionMergerSiteDelete(self, to, address):
        site = self.server.sites.get(address)
        if not site:
            return self.response(to, {"error": "No site found: %s" % address})

        merger_types = merger_db.get(self.site.address)
        if not merger_types:
            return self.response(to, {"error": "Not a merger site"})
        if merged_db.get(address) not in merger_types:
            return self.response(to, {"error": "Merged type (%s) not in %s" % (merged_db.get(address), merger_types)})

        self.cmd("notification", ["done", _["Site deleted: <b>%s</b>"] % address, 5000])
        self.response(to, "ok")

    # Lists merged sites
    def actionMergerSiteList(self, to, query_site_info=False):
        merger_types = merger_db.get(self.site.address)
        ret = {}
        if not merger_types:
            return self.response(to, {"error": "Not a merger site"})
        for address, merged_type in merged_db.items():
            if merged_type not in merger_types:
                continue  # Site not for us
            if query_site_info:
                site = self.server.sites.get(address)
                ret[address] = self.formatSiteInfo(site, create_user=False)
            else:
                ret[address] = merged_type
        self.response(to, ret)

    def hasSitePermission(self, address, *args, **kwargs):
        if super(UiWebsocketPlugin, self).hasSitePermission(address, *args, **kwargs):
            return True
        else:
            if self.site.address in [merger_site.address for merger_site in merged_to_merger.get(address, [])]:
                return True
            else:
                return False

    # Add support merger sites for file commands
    def mergerFuncWrapper(self, func_name, to, inner_path, *args, **kwargs):
        if inner_path.startswith("merged-"):
            merged_address, merged_inner_path = checkMergerPath(self.site.address, inner_path)

            # Set the same cert for merged site
            merger_cert = self.user.getSiteData(self.site.address).get("cert")
            if merger_cert and self.user.getSiteData(merged_address).get("cert") != merger_cert:
                self.user.setCert(merged_address, merger_cert)

            req_self = copy.copy(self)
            req_self.site = self.server.sites.get(merged_address)  # Change the site to the merged one

            func = getattr(super(UiWebsocketPlugin, req_self), func_name)
            return func(to, merged_inner_path, *args, **kwargs)
        else:
            func = getattr(super(UiWebsocketPlugin, self), func_name)
            return func(to, inner_path, *args, **kwargs)

    def actionFileList(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionFileList", to, inner_path, *args, **kwargs)

    def actionDirList(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionDirList", to, inner_path, *args, **kwargs)

    def actionFileGet(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionFileGet", to, inner_path, *args, **kwargs)

    def actionFileWrite(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionFileWrite", to, inner_path, *args, **kwargs)

    def actionFileDelete(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionFileDelete", to, inner_path, *args, **kwargs)

    def actionFileRules(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionFileRules", to, inner_path, *args, **kwargs)

    def actionFileNeed(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionFileNeed", to, inner_path, *args, **kwargs)

    def actionOptionalFileInfo(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionOptionalFileInfo", to, inner_path, *args, **kwargs)

    def actionOptionalFileDelete(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionOptionalFileDelete", to, inner_path, *args, **kwargs)

    def actionBigfileUploadInit(self, to, inner_path, *args, **kwargs):
        back = self.mergerFuncWrapper("actionBigfileUploadInit", to, inner_path, *args, **kwargs)
        if inner_path.startswith("merged-"):
            merged_address, merged_inner_path = checkMergerPath(self.site.address, inner_path)
            back["inner_path"] = "merged-%s/%s/%s" % (merged_db[merged_address], merged_address, back["inner_path"])
        return back

    # Add support merger sites for file commands with privatekey parameter
    def mergerFuncWrapperWithPrivatekey(self, func_name, to, privatekey, inner_path, *args, **kwargs):
        func = getattr(super(UiWebsocketPlugin, self), func_name)
        if inner_path.startswith("merged-"):
            merged_address, merged_inner_path = checkMergerPath(self.site.address, inner_path)
            merged_site = self.server.sites.get(merged_address)

            # Set the same cert for merged site
            merger_cert = self.user.getSiteData(self.site.address).get("cert")
            if merger_cert:
                self.user.setCert(merged_address, merger_cert)

            site_before = self.site  # Save to be able to change it back after we ran the command
            self.site = merged_site  # Change the site to the merged one
            try:
                back = func(to, privatekey, merged_inner_path, *args, **kwargs)
            finally:
                self.site = site_before  # Change back to original site
            return back
        else:
            return func(to, privatekey, inner_path, *args, **kwargs)

    def actionSiteSign(self, to, privatekey=None, inner_path="content.json", *args, **kwargs):
        return self.mergerFuncWrapperWithPrivatekey("actionSiteSign", to, privatekey, inner_path, *args, **kwargs)

    def actionSitePublish(self, to, privatekey=None, inner_path="content.json", *args, **kwargs):
        return self.mergerFuncWrapperWithPrivatekey("actionSitePublish", to, privatekey, inner_path, *args, **kwargs)

    def actionPermissionAdd(self, to, permission):
        super(UiWebsocketPlugin, self).actionPermissionAdd(to, permission)
        if permission.startswith("Merger"):
            self.site.storage.rebuildDb()

    def actionPermissionDetails(self, to, permission):
        if not permission.startswith("Merger"):
            return super(UiWebsocketPlugin, self).actionPermissionDetails(to, permission)

        merger_type = permission.replace("Merger:", "")
        if not re.match("^[A-Za-z0-9-]+$", merger_type):
            raise Exception("Invalid merger_type: %s" % merger_type)
        merged_sites = []
        for address, merged_type in merged_db.items():
            if merged_type != merger_type:
                continue
            site = self.server.sites.get(address)
            try:
                merged_sites.append(site.content_manager.contents.get("content.json").get("title", address))
            except Exception:
                merged_sites.append(address)

        details = _["Read and write permissions to sites with merged type of <b>%s</b> "] % merger_type
        details += _["(%s sites)"] % len(merged_sites)
        details += "<div style='white-space: normal; max-width: 400px'>%s</div>" % ", ".join(merged_sites)
        self.response(to, details)


```

This is a Python class that implements a SiteStoragePlugin for a FUSE-based virtual machine. It stores virtual machine data in a merged site file, which is a backup of the entire virtual machine.

The SiteStoragePlugin class has the following methods:

onAttached: This method is called when the virtual machine is attached to the site. It does nothing.

onDetached: This method is called when the virtual machine is detached from the site. It does nothing.

onUpdated: This method is called when the virtual machine data is updated. It takes the following arguments:

- inner_path: The virtual machine file path, including the .json extension if it exists.
- file: The file object, if any.

It works as follows:

1. If the virtual machine file is json, it checks if it has a matching json file in the merged site and if it does, it yields the merged site path and the file.
2. If the virtual machine file is not json, it starts to process the file by adding it to the merged site.
3. It adds the virtual machine to the merged site.
4. It informs the merged site plugin if it has updated.
5. It also notices if the merged site plugin has updated and if it does, it is also updated.

It also has a method to handle the update of the merged site, which is called when the site is updated.


```py
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # Allow to load merged site files using /merged-ZeroMe/address/file.jpg
    def parsePath(self, path):
        path_parts = super(UiRequestPlugin, self).parsePath(path)
        if "merged-" not in path:  # Optimization
            return path_parts
        path_parts["address"], path_parts["inner_path"] = checkMergerPath(path_parts["address"], path_parts["inner_path"])
        return path_parts


@PluginManager.registerTo("SiteStorage")
class SiteStoragePlugin(object):
    # Also rebuild from merged sites
    def getDbFiles(self):
        merger_types = merger_db.get(self.site.address)

        # First return the site's own db files
        for item in super(SiteStoragePlugin, self).getDbFiles():
            yield item

        # Not a merger site, that's all
        if not merger_types:
            return

        merged_sites = [
            site_manager.sites[address]
            for address, merged_type in merged_db.items()
            if merged_type in merger_types
        ]
        found = 0
        for merged_site in merged_sites:
            self.log.debug("Loading merged site: %s" % merged_site)
            merged_type = merged_db[merged_site.address]
            for content_inner_path, content in merged_site.content_manager.contents.items():
                # content.json file itself
                if merged_site.storage.isFile(content_inner_path):  # Missing content.json file
                    merged_inner_path = "merged-%s/%s/%s" % (merged_type, merged_site.address, content_inner_path)
                    yield merged_inner_path, merged_site.storage.getPath(content_inner_path)
                else:
                    merged_site.log.error("[MISSING] %s" % content_inner_path)
                # Data files in content.json
                content_inner_path_dir = helper.getDirname(content_inner_path)  # Content.json dir relative to site
                for file_relative_path in list(content.get("files", {}).keys()) + list(content.get("files_optional", {}).keys()):
                    if not file_relative_path.endswith(".json"):
                        continue  # We only interesed in json files
                    file_inner_path = content_inner_path_dir + file_relative_path  # File Relative to site dir
                    file_inner_path = file_inner_path.strip("/")  # Strip leading /
                    if merged_site.storage.isFile(file_inner_path):
                        merged_inner_path = "merged-%s/%s/%s" % (merged_type, merged_site.address, file_inner_path)
                        yield merged_inner_path, merged_site.storage.getPath(file_inner_path)
                    else:
                        merged_site.log.error("[MISSING] %s" % file_inner_path)
                    found += 1
                    if found % 100 == 0:
                        time.sleep(0.001)  # Context switch to avoid UI block

    # Also notice merger sites on a merged site file change
    def onUpdated(self, inner_path, file=None):
        super(SiteStoragePlugin, self).onUpdated(inner_path, file)

        merged_type = merged_db.get(self.site.address)

        for merger_site in merged_to_merger.get(self.site.address, []):
            if merger_site.address == self.site.address:  # Avoid infinite loop
                continue
            virtual_path = "merged-%s/%s/%s" % (merged_type, self.site.address, inner_path)
            if inner_path.endswith(".json"):
                if file is not None:
                    merger_site.storage.onUpdated(virtual_path, file=file)
                else:
                    merger_site.storage.onUpdated(virtual_path, file=self.open(inner_path))
            else:
                merger_site.storage.onUpdated(virtual_path)


```

这段代码定义了一个名为 SitePlugin 的类，用于处理网站的数据传输。

SitePlugin 类有两个方法，分别名为 fileDone 和 fileFailed。这两个方法都是针对网站中不同成员的调用，用于在文件传输过程中或传输失败时执行特定的操作。

fileDone 方法先调用父类的 fileDone 方法，如果文件传输成功，就直接返回。然后，它遍历 merged_to_merger 对象中的合并器站点，检查当前站点地址是否与当前站点地址相同。如果是，就跳过当前站点，否则，就遍历合并器站点中的所有 WebSocket，并将事件类型设置为 "siteChanged"，参数设置为当前 SitePlugin 对象以及文件路径和当前文件名。

fileFailed 方法与 fileDone 方法类似，但循环中包括所有合并器站点，而不是只与当前站点地址相同的站点。循环遍历所有合并器站点，并检查当前站点地址是否与当前站点地址相同。如果是，就跳过当前站点，否则，就遍历合并器站点中的所有 WebSocket，并将事件类型设置为 "siteChanged"，参数设置为当前 SitePlugin 对象以及文件路径和当前文件名。


```py
@PluginManager.registerTo("Site")
class SitePlugin(object):
    def fileDone(self, inner_path):
        super(SitePlugin, self).fileDone(inner_path)

        for merger_site in merged_to_merger.get(self.address, []):
            if merger_site.address == self.address:
                continue
            for ws in merger_site.websockets:
                ws.event("siteChanged", self, {"event": ["file_done", inner_path]})

    def fileFailed(self, inner_path):
        super(SitePlugin, self).fileFailed(inner_path)

        for merger_site in merged_to_merger.get(self.address, []):
            if merger_site.address == self.address:
                continue
            for ws in merger_site.websockets:
                ws.event("siteChanged", self, {"event": ["file_failed", inner_path]})


```

This is a Python class that inherits from the SiteManagerPlugin class and implements the site.save() and site.load() methods.

The site.save() method is called when the site is saved, and the `site.save()` method is overwritten with the implementation of the method. It first updates the merged type and the merged database new, and then updates the globals. It also updates the merged to


```py
@PluginManager.registerTo("SiteManager")
class SiteManagerPlugin(object):
    # Update merger site for site types
    def updateMergerSites(self):
        global merger_db, merged_db, merged_to_merger, site_manager
        s = time.time()
        merger_db_new = {}
        merged_db_new = {}
        merged_to_merger_new = {}
        site_manager = self
        if not self.sites:
            return
        for site in self.sites.values():
            # Update merged sites
            try:
                merged_type = site.content_manager.contents.get("content.json", {}).get("merged_type")
            except Exception as err:
                self.log.error("Error loading site %s: %s" % (site.address, Debug.formatException(err)))
                continue
            if merged_type:
                merged_db_new[site.address] = merged_type

            # Update merger sites
            for permission in site.settings["permissions"]:
                if not permission.startswith("Merger:"):
                    continue
                if merged_type:
                    self.log.error(
                        "Removing permission %s from %s: Merger and merged at the same time." %
                        (permission, site.address)
                    )
                    site.settings["permissions"].remove(permission)
                    continue
                merger_type = permission.replace("Merger:", "")
                if site.address not in merger_db_new:
                    merger_db_new[site.address] = []
                merger_db_new[site.address].append(merger_type)
                site_manager.sites[site.address] = site

            # Update merged to merger
            if merged_type:
                for merger_site in self.sites.values():
                    if "Merger:" + merged_type in merger_site.settings["permissions"]:
                        if site.address not in merged_to_merger_new:
                            merged_to_merger_new[site.address] = []
                        merged_to_merger_new[site.address].append(merger_site)

        # Update globals
        merger_db = merger_db_new
        merged_db = merged_db_new
        merged_to_merger = merged_to_merger_new

        self.log.debug("Updated merger sites in %.3fs" % (time.time() - s))

    def load(self, *args, **kwags):
        super(SiteManagerPlugin, self).load(*args, **kwags)
        self.updateMergerSites()

    def saveDelayed(self, *args, **kwags):
        super(SiteManagerPlugin, self).saveDelayed(*args, **kwags)
        self.updateMergerSites()

```

# `plugins/MergerSite/__init__.py`

这段代码的作用是导入名为 "MergerSitePlugin" 的自定义插件类，可能用于在特定的 MergerSite 项目中进行一些自定义的操作。

在 Python 中，导入其他模块或类通常需要使用 "from ... import ..." 的语法。这里的 "." 表示从当前模块开始导入，如果没有从模块中导出过该插件类，则需要使用 "import ..." 的语句来指定。

具体而言，这个插件类可能用于在 MergerSite 项目中自动下载、安装、配置或运行某些自定义脚本或操作，以帮助用户更高效地完成数据合并工作。


```py
from . import MergerSitePlugin
```

# `plugins/Newsfeed/NewsfeedPlugin.py`

This is a code snippet for a database query that retrieves a feed based on the input parameters. It seems to be using the search\_like parameter to retrieve a search term and then appending it to the WHERE clause of the query, along with the current date limit.

It also appears to be ordering the results by date\_added and limiting the number of rows to be retrieved.

It also logs any errors that occur.

It appears to be returning the results to the client.

Please note that this code snippet is cut off and may not work as intended.



```py
import time
import re

from Plugin import PluginManager
from Db.DbQuery import DbQuery
from Debug import Debug
from util import helper
from util.Flag import flag


@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    def formatSiteInfo(self, site, create_user=True):
        site_info = super(UiWebsocketPlugin, self).formatSiteInfo(site, create_user=create_user)
        feed_following = self.user.sites.get(site.address, {}).get("follow", None)
        if feed_following == None:
            site_info["feed_follow_num"] = None
        else:
            site_info["feed_follow_num"] = len(feed_following)
        return site_info

    def actionFeedFollow(self, to, feeds):
        self.user.setFeedFollow(self.site.address, feeds)
        self.user.save()
        self.response(to, "ok")

    def actionFeedListFollow(self, to):
        feeds = self.user.sites.get(self.site.address, {}).get("follow", {})
        self.response(to, feeds)

    @flag.admin
    def actionFeedQuery(self, to, limit=10, day_limit=3):
        from Site import SiteManager
        rows = []
        stats = []

        total_s = time.time()
        num_sites = 0

        for address, site_data in list(self.user.sites.items()):
            feeds = site_data.get("follow")
            if not feeds:
                continue
            if type(feeds) is not dict:
                self.log.debug("Invalid feed for site %s" % address)
                continue
            num_sites += 1
            for name, query_set in feeds.items():
                site = SiteManager.site_manager.get(address)
                if not site or not site.storage.has_db:
                    continue

                s = time.time()
                try:
                    query_raw, params = query_set
                    query_parts = re.split(r"UNION(?:\s+ALL|)", query_raw)
                    for i, query_part in enumerate(query_parts):
                        db_query = DbQuery(query_part)
                        if day_limit:
                            where = " WHERE %s > strftime('%%s', 'now', '-%s day')" % (db_query.fields.get("date_added", "date_added"), day_limit)
                            if "WHERE" in query_part:
                                query_part = re.sub("WHERE (.*?)(?=$| GROUP BY)", where+" AND (\\1)", query_part)
                            else:
                                query_part += where
                        query_parts[i] = query_part
                    query = " UNION ".join(query_parts)

                    if ":params" in query:
                        query_params = map(helper.sqlquote, params)
                        query = query.replace(":params", ",".join(query_params))

                    res = site.storage.query(query + " ORDER BY date_added DESC LIMIT %s" % limit)

                except Exception as err:  # Log error
                    self.log.error("%s feed query %s error: %s" % (address, name, Debug.formatException(err)))
                    stats.append({"site": site.address, "feed_name": name, "error": str(err)})
                    continue

                for row in res:
                    row = dict(row)
                    if not isinstance(row["date_added"], (int, float, complex)):
                        self.log.debug("Invalid date_added from site %s: %r" % (address, row["date_added"]))
                        continue
                    if row["date_added"] > 1000000000000:  # Formatted as millseconds
                        row["date_added"] = row["date_added"] / 1000
                    if "date_added" not in row or row["date_added"] > time.time() + 120:
                        self.log.debug("Newsfeed item from the future from from site %s" % address)
                        continue  # Feed item is in the future, skip it
                    row["site"] = address
                    row["feed_name"] = name
                    rows.append(row)
                stats.append({"site": site.address, "feed_name": name, "taken": round(time.time() - s, 3)})
                time.sleep(0.001)
        return self.response(to, {"rows": rows, "stats": stats, "num": len(rows), "sites": num_sites, "taken": round(time.time() - total_s, 3)})

    def parseSearch(self, search):
        parts = re.split("(site|type):", search)
        if len(parts) > 1:  # Found filter
            search_text = parts[0]
            parts = [part.strip() for part in parts]
            filters = dict(zip(parts[1::2], parts[2::2]))
        else:
            search_text = search
            filters = {}
        return [search_text, filters]

    def actionFeedSearch(self, to, search, limit=30, day_limit=30):
        if "ADMIN" not in self.site.settings["permissions"]:
            return self.response(to, "FeedSearch not allowed")

        from Site import SiteManager
        rows = []
        stats = []
        num_sites = 0
        total_s = time.time()

        search_text, filters = self.parseSearch(search)

        for address, site in SiteManager.site_manager.list().items():
            if not site.storage.has_db:
                continue

            if "site" in filters:
                if filters["site"].lower() not in [site.address, site.content_manager.contents["content.json"].get("title").lower()]:
                    continue

            if site.storage.db:  # Database loaded
                feeds = site.storage.db.schema.get("feeds")
            else:
                try:
                    feeds = site.storage.loadJson("dbschema.json").get("feeds")
                except:
                    continue

            if not feeds:
                continue

            num_sites += 1

            for name, query in feeds.items():
                s = time.time()
                try:
                    db_query = DbQuery(query)

                    params = []
                    # Filters
                    if search_text:
                        db_query.wheres.append("(%s LIKE ? OR %s LIKE ?)" % (db_query.fields["body"], db_query.fields["title"]))
                        search_like = "%" + search_text.replace(" ", "%") + "%"
                        params.append(search_like)
                        params.append(search_like)
                    if filters.get("type") and filters["type"] not in query:
                        continue

                    if day_limit:
                        db_query.wheres.append(
                            "%s > strftime('%%s', 'now', '-%s day')" % (db_query.fields.get("date_added", "date_added"), day_limit)
                        )

                    # Order
                    db_query.parts["ORDER BY"] = "date_added DESC"
                    db_query.parts["LIMIT"] = str(limit)

                    res = site.storage.query(str(db_query), params)
                except Exception as err:
                    self.log.error("%s feed query %s error: %s" % (address, name, Debug.formatException(err)))
                    stats.append({"site": site.address, "feed_name": name, "error": str(err), "query": query})
                    continue
                for row in res:
                    row = dict(row)
                    if not row["date_added"] or row["date_added"] > time.time() + 120:
                        continue  # Feed item is in the future, skip it
                    row["site"] = address
                    row["feed_name"] = name
                    rows.append(row)
                stats.append({"site": site.address, "feed_name": name, "taken": round(time.time() - s, 3)})
        return self.response(to, {"rows": rows, "num": len(rows), "sites": num_sites, "taken": round(time.time() - total_s, 3), "stats": stats})


```

这段代码定义了一个名为 `UserPlugin` 的类，用于在插件系统中注册用户，并将其分为两种类型： "plugin" 和 "user"。

在这个类中，使用 `@PluginManager.registerTo("User")` 注解注册了一个名为 "user" 的插件。当这个插件被加载到插件系统中时，会使用 `registerTo("User")` 注册用户的接口来获取所有的用户信息。

在 `UserPlugin` 类中，有一个名为 `setFeedFollow` 的方法，用于设置用户指定的 feed 的 follow 属性，该属性表示用户正在跟随的 feed 列表。

该方法接收两个参数： `address` 表示用户所在的地址，`feeds` 表示用户正在跟随的 feed 列表。首先，使用 `self.getSiteData(address)` 方法获取用户所在站点的基本数据，包括该站点支持的 feed 列表。然后，在基本数据的基础上，通过 `self.save()` 方法将用户的 follow 属性保存到该站点中。最后，返回修改后的站点数据。

如果用户没有指定任何站点，或者站点不支持 feed，该方法将直接返回基本数据，包含 follow 属性为空的列表。


```py
@PluginManager.registerTo("User")
class UserPlugin(object):
    # Set queries that user follows
    def setFeedFollow(self, address, feeds):
        site_data = self.getSiteData(address)
        site_data["follow"] = feeds
        self.save()
        return site_data

```

# `plugins/Newsfeed/__init__.py`

这段代码是一个Python代码片段，从标准库中导入了一个名为 "NewsfeedPlugin" 的类，然后没有做任何其他事情，只是声明了一个名为 "plugin" 的变量。

根据上下文，可以猜测 "NewsfeedPlugin" 可能是一个用于从 Newsfeed 网站获取新闻信息的插件，或者是将用户输入的信息存储到 Newsfeed 网站的功能。但是，由于没有进一步的上下文，无法确定具体的作用。


```py
from . import NewsfeedPlugin
```

# `plugins/OptionalManager/ContentDbPlugin.py`

这段代码的作用是定义了一个函数 `evaluate_regex`，它接受一个字符串参数，使用正则表达式匹配并在匹配到第一个匹配项时，使用指定的 `gevent` 库的 `time` 函数来暂停当前时间，防止时间片轮转造成计算过长。

具体来说，这段代码定义了一个名为 `evaluate_regex` 的函数，它接受一个字符串参数 `pattern`。函数内部使用 `re` 模块的 `findall` 函数来查找 `pattern` 中的所有正则表达式表达式，并将它们返回。然后，函数内部使用 `itertools` 模块的 `cycle` 函数来遍历所有匹配项，并为每个匹配项设置一个暂停的时间，时间长度为 `config.RELATIVE_PAUSE_TIME`，可以根据当前时间片轮转的时间长度进行调整。

最后，函数内部使用 `gevent` 库的 `time` 函数来暂停当前时间，防止时间片轮转造成计算过长。整个函数的作用就是快速查找一个字符串中所有匹配的正则表达式表达式，并暂停计算时间，以防止计算过长导致程序卡顿。


```py
import time
import collections
import itertools
import re

import gevent

from util import helper
from Plugin import PluginManager
from Config import config
from Debug import Debug

if "content_db" not in locals().keys():  # To keep between module reloads
    content_db = None


```



This code appears to be a file size limit tool for a file storage system. It performs the following operations:

1. Retrieves the optional file size limit from the configuration settings.
2. Checks if there is a need to delete any files. If there is, it does so by removing the specified files from the file storage system.
3. Updates the peer numbers for the file storage system.
4. Logs the activity of the file storage system.

It does not seem to perform any actual file removal, but instead appears to mark the specified files as deleted and update the numbers accordingly.


```py
@PluginManager.registerTo("ContentDb")
class ContentDbPlugin(object):
    def __init__(self, *args, **kwargs):
        global content_db
        content_db = self
        self.filled = {}  # Site addresses that already filled from content.json
        self.need_filling = False  # file_optional table just created, fill data from content.json files
        self.time_peer_numbers_updated = 0
        self.my_optional_files = {}  # Last 50 site_address/inner_path called by fileWrite (auto-pinning these files)
        self.optional_files = collections.defaultdict(dict)
        self.optional_files_loaded = False
        self.timer_check_optional = helper.timer(60 * 5, self.checkOptionalLimit)
        super(ContentDbPlugin, self).__init__(*args, **kwargs)

    def getSchema(self):
        schema = super(ContentDbPlugin, self).getSchema()

        # Need file_optional table
        schema["tables"]["file_optional"] = {
            "cols": [
                ["file_id", "INTEGER PRIMARY KEY UNIQUE NOT NULL"],
                ["site_id", "INTEGER REFERENCES site (site_id) ON DELETE CASCADE"],
                ["inner_path", "TEXT"],
                ["hash_id", "INTEGER"],
                ["size", "INTEGER"],
                ["peer", "INTEGER DEFAULT 0"],
                ["uploaded", "INTEGER DEFAULT 0"],
                ["is_downloaded", "INTEGER DEFAULT 0"],
                ["is_pinned", "INTEGER DEFAULT 0"],
                ["time_added", "INTEGER DEFAULT 0"],
                ["time_downloaded", "INTEGER DEFAULT 0"],
                ["time_accessed", "INTEGER DEFAULT 0"]
            ],
            "indexes": [
                "CREATE UNIQUE INDEX file_optional_key ON file_optional (site_id, inner_path)",
                "CREATE INDEX is_downloaded ON file_optional (is_downloaded)"
            ],
            "schema_changed": 11
        }

        return schema

    def initSite(self, site):
        super(ContentDbPlugin, self).initSite(site)
        if self.need_filling:
            self.fillTableFileOptional(site)

    def checkTables(self):
        changed_tables = super(ContentDbPlugin, self).checkTables()
        if "file_optional" in changed_tables:
            self.need_filling = True
        return changed_tables

    # Load optional files ending
    def loadFilesOptional(self):
        s = time.time()
        num = 0
        total = 0
        total_downloaded = 0
        res = content_db.execute("SELECT site_id, inner_path, size, is_downloaded FROM file_optional")
        site_sizes = collections.defaultdict(lambda: collections.defaultdict(int))
        for row in res:
            self.optional_files[row["site_id"]][row["inner_path"][-8:]] = 1
            num += 1

            # Update site size stats
            site_sizes[row["site_id"]]["size_optional"] += row["size"]
            if row["is_downloaded"]:
                site_sizes[row["site_id"]]["optional_downloaded"] += row["size"]

        # Site site size stats to sites.json settings
        site_ids_reverse = {val: key for key, val in self.site_ids.items()}
        for site_id, stats in site_sizes.items():
            site_address = site_ids_reverse.get(site_id)
            if not site_address or site_address not in self.sites:
                self.log.error("Not found site_id: %s" % site_id)
                continue
            site = self.sites[site_address]
            site.settings["size_optional"] = stats["size_optional"]
            site.settings["optional_downloaded"] = stats["optional_downloaded"]
            total += stats["size_optional"]
            total_downloaded += stats["optional_downloaded"]

        self.log.info(
            "Loaded %s optional files: %.2fMB, downloaded: %.2fMB in %.3fs" %
            (num, float(total) / 1024 / 1024, float(total_downloaded) / 1024 / 1024, time.time() - s)
        )

        if self.need_filling and self.getOptionalLimitBytes() >= 0 and self.getOptionalLimitBytes() < total_downloaded:
            limit_bytes = self.getOptionalLimitBytes()
            limit_new = round((float(total_downloaded) / 1024 / 1024 / 1024) * 1.1, 2)  # Current limit + 10%
            self.log.info(
                "First startup after update and limit is smaller than downloaded files size (%.2fGB), increasing it from %.2fGB to %.2fGB" %
                (float(total_downloaded) / 1024 / 1024 / 1024, float(limit_bytes) / 1024 / 1024 / 1024, limit_new)
            )
            config.saveValue("optional_limit", limit_new)
            config.optional_limit = str(limit_new)

    # Predicts if the file is optional
    def isOptionalFile(self, site_id, inner_path):
        return self.optional_files[site_id].get(inner_path[-8:])

    # Fill file_optional table with optional files found in sites
    def fillTableFileOptional(self, site):
        s = time.time()
        site_id = self.site_ids.get(site.address)
        if not site_id:
            return False
        cur = self.getCursor()
        res = cur.execute("SELECT * FROM content WHERE size_files_optional > 0 AND site_id = %s" % site_id)
        num = 0
        for row in res.fetchall():
            content = site.content_manager.contents[row["inner_path"]]
            try:
                num += self.setContentFilesOptional(site, row["inner_path"], content, cur=cur)
            except Exception as err:
                self.log.error("Error loading %s into file_optional: %s" % (row["inner_path"], err))
        cur.close()

        # Set my files to pinned
        from User import UserManager
        user = UserManager.user_manager.get()
        if not user:
            user = UserManager.user_manager.create()
        auth_address = user.getAuthAddress(site.address)
        res = self.execute(
            "UPDATE file_optional SET is_pinned = 1 WHERE site_id = :site_id AND inner_path LIKE :inner_path",
            {"site_id": site_id, "inner_path": "%%/%s/%%" % auth_address}
        )

        self.log.debug(
            "Filled file_optional table for %s in %.3fs (loaded: %s, is_pinned: %s)" %
            (site.address, time.time() - s, num, res.rowcount)
        )
        self.filled[site.address] = True

    def setContentFilesOptional(self, site, content_inner_path, content, cur=None):
        if not cur:
            cur = self

        num = 0
        site_id = self.site_ids[site.address]
        content_inner_dir = helper.getDirname(content_inner_path)
        for relative_inner_path, file in content.get("files_optional", {}).items():
            file_inner_path = content_inner_dir + relative_inner_path
            hash_id = int(file["sha512"][0:4], 16)
            if hash_id in site.content_manager.hashfield:
                is_downloaded = 1
            else:
                is_downloaded = 0
            if site.address + "/" + content_inner_dir in self.my_optional_files:
                is_pinned = 1
            else:
                is_pinned = 0
            cur.insertOrUpdate("file_optional", {
                "hash_id": hash_id,
                "size": int(file["size"])
            }, {
                "site_id": site_id,
                "inner_path": file_inner_path
            }, oninsert={
                "time_added": int(time.time()),
                "time_downloaded": int(time.time()) if is_downloaded else 0,
                "is_downloaded": is_downloaded,
                "peer": is_downloaded,
                "is_pinned": is_pinned
            })
            self.optional_files[site_id][file_inner_path[-8:]] = 1
            num += 1

        return num

    def setContent(self, site, inner_path, content, size=0):
        super(ContentDbPlugin, self).setContent(site, inner_path, content, size=size)
        old_content = site.content_manager.contents.get(inner_path, {})
        if (not self.need_filling or self.filled.get(site.address)) and ("files_optional" in content or "files_optional" in old_content):
            self.setContentFilesOptional(site, inner_path, content)
            # Check deleted files
            if old_content:
                old_files = old_content.get("files_optional", {}).keys()
                new_files = content.get("files_optional", {}).keys()
                content_inner_dir = helper.getDirname(inner_path)
                deleted = [content_inner_dir + key for key in old_files if key not in new_files]
                if deleted:
                    site_id = self.site_ids[site.address]
                    self.execute("DELETE FROM file_optional WHERE ?", {"site_id": site_id, "inner_path": deleted})

    def deleteContent(self, site, inner_path):
        content = site.content_manager.contents.get(inner_path)
        if content and "files_optional" in content:
            site_id = self.site_ids[site.address]
            content_inner_dir = helper.getDirname(inner_path)
            optional_inner_paths = [
                content_inner_dir + relative_inner_path
                for relative_inner_path in content.get("files_optional", {}).keys()
            ]
            self.execute("DELETE FROM file_optional WHERE ?", {"site_id": site_id, "inner_path": optional_inner_paths})
        super(ContentDbPlugin, self).deleteContent(site, inner_path)

    def updatePeerNumbers(self):
        s = time.time()
        num_file = 0
        num_updated = 0
        num_site = 0
        for site in list(self.sites.values()):
            if not site.content_manager.has_optional_files:
                continue
            if not site.isServing():
                continue
            has_updated_hashfield = next((
                peer
                for peer in site.peers.values()
                if peer.has_hashfield and peer.hashfield.time_changed > self.time_peer_numbers_updated
            ), None)

            if not has_updated_hashfield and site.content_manager.hashfield.time_changed < self.time_peer_numbers_updated:
                continue

            hashfield_peers = itertools.chain.from_iterable(
                peer.hashfield.storage
                for peer in site.peers.values()
                if peer.has_hashfield
            )
            peer_nums = collections.Counter(
                itertools.chain(
                    hashfield_peers,
                    site.content_manager.hashfield
                )
            )

            site_id = self.site_ids[site.address]
            if not site_id:
                continue

            res = self.execute("SELECT file_id, hash_id, peer FROM file_optional WHERE ?", {"site_id": site_id})
            updates = {}
            for row in res:
                peer_num = peer_nums.get(row["hash_id"], 0)
                if peer_num != row["peer"]:
                    updates[row["file_id"]] = peer_num

            for file_id, peer_num in updates.items():
                self.execute("UPDATE file_optional SET peer = ? WHERE file_id = ?", (peer_num, file_id))

            num_updated += len(updates)
            num_file += len(peer_nums)
            num_site += 1

        self.time_peer_numbers_updated = time.time()
        self.log.debug("%s/%s peer number for %s site updated in %.3fs" % (num_updated, num_file, num_site, time.time() - s))

    def queryDeletableFiles(self):
        # First return the files with atleast 10 seeder and not accessed in last week
        query = """
            SELECT * FROM file_optional
            WHERE peer > 10 AND %s
            ORDER BY time_accessed < %s DESC, uploaded / size
        """ % (self.getOptionalUsedWhere(), int(time.time() - 60 * 60 * 7))
        limit_start = 0
        while 1:
            num = 0
            res = self.execute("%s LIMIT %s, 50" % (query, limit_start))
            for row in res:
                yield row
                num += 1
            if num < 50:
                break
            limit_start += 50

        self.log.debug("queryDeletableFiles returning less-seeded files")

        # Then return files less seeder but still not accessed in last week
        query = """
            SELECT * FROM file_optional
            WHERE peer <= 10 AND %s
            ORDER BY peer DESC, time_accessed < %s DESC, uploaded / size
        """ % (self.getOptionalUsedWhere(), int(time.time() - 60 * 60 * 7))
        limit_start = 0
        while 1:
            num = 0
            res = self.execute("%s LIMIT %s, 50" % (query, limit_start))
            for row in res:
                yield row
                num += 1
            if num < 50:
                break
            limit_start += 50

        self.log.debug("queryDeletableFiles returning everyting")

        # At the end return all files
        query = """
            SELECT * FROM file_optional
            WHERE peer <= 10 AND %s
            ORDER BY peer DESC, time_accessed, uploaded / size
        """ % self.getOptionalUsedWhere()
        limit_start = 0
        while 1:
            num = 0
            res = self.execute("%s LIMIT %s, 50" % (query, limit_start))
            for row in res:
                yield row
                num += 1
            if num < 50:
                break
            limit_start += 50

    def getOptionalLimitBytes(self):
        if config.optional_limit.endswith("%"):
            limit_percent = float(re.sub("[^0-9.]", "", config.optional_limit))
            limit_bytes = helper.getFreeSpace() * (limit_percent / 100)
        else:
            limit_bytes = float(re.sub("[^0-9.]", "", config.optional_limit)) * 1024 * 1024 * 1024
        return limit_bytes

    def getOptionalUsedWhere(self):
        maxsize = config.optional_limit_exclude_minsize * 1024 * 1024
        query = "is_downloaded = 1 AND is_pinned = 0 AND size < %s" % maxsize

        # Don't delete optional files from owned sites
        my_site_ids = []
        for address, site in self.sites.items():
            if site.settings["own"]:
                my_site_ids.append(str(self.site_ids[address]))

        if my_site_ids:
            query += " AND site_id NOT IN (%s)" % ", ".join(my_site_ids)
        return query

    def getOptionalUsedBytes(self):
        size = self.execute("SELECT SUM(size) FROM file_optional WHERE %s" % self.getOptionalUsedWhere()).fetchone()[0]
        if not size:
            size = 0
        return size

    def getOptionalNeedDelete(self, size):
        if config.optional_limit.endswith("%"):
            limit_percent = float(re.sub("[^0-9.]", "", config.optional_limit))
            need_delete = size - ((helper.getFreeSpace() + size) * (limit_percent / 100))
        else:
            need_delete = size - self.getOptionalLimitBytes()
        return need_delete

    def checkOptionalLimit(self, limit=None):
        if not limit:
            limit = self.getOptionalLimitBytes()

        if limit < 0:
            self.log.debug("Invalid limit for optional files: %s" % limit)
            return False

        size = self.getOptionalUsedBytes()

        need_delete = self.getOptionalNeedDelete(size)

        self.log.debug(
            "Optional size: %.1fMB/%.1fMB, Need delete: %.1fMB" %
            (float(size) / 1024 / 1024, float(limit) / 1024 / 1024, float(need_delete) / 1024 / 1024)
        )
        if need_delete <= 0:
            return False

        self.updatePeerNumbers()

        site_ids_reverse = {val: key for key, val in self.site_ids.items()}
        deleted_file_ids = []
        for row in self.queryDeletableFiles():
            site_address = site_ids_reverse.get(row["site_id"])
            site = self.sites.get(site_address)
            if not site:
                self.log.error("No site found for id: %s" % row["site_id"])
                continue
            site.log.debug("Deleting %s %.3f MB left" % (row["inner_path"], float(need_delete) / 1024 / 1024))
            deleted_file_ids.append(row["file_id"])
            try:
                site.content_manager.optionalRemoved(row["inner_path"], row["hash_id"], row["size"])
                site.storage.delete(row["inner_path"])
                need_delete -= row["size"]
            except Exception as err:
                site.log.error("Error deleting %s: %s" % (row["inner_path"], err))

            if need_delete <= 0:
                break

        cur = self.getCursor()
        for file_id in deleted_file_ids:
            cur.execute("UPDATE file_optional SET is_downloaded = 0, is_pinned = 0, peer = peer - 1 WHERE ?", {"file_id": file_id})
        cur.close()


```

这段代码定义了一个名为 SiteManagerPlugin 的类，用于在插件系统中注册到名为 "SiteManager" 的插件中。

在 SiteManagerPlugin 的 load 方法中，定义了一个名为 "self" 的属性，该属性指向自己的对象，以便在方法内部调用父类的 load 方法。

在 load 方法的实现中，首先调用父类的 load 方法，并将传入的参数传递给它。然后，检查对象中是否包含 sites 属性，如果不包含，则执行 content_db.optional_files_loaded = True，并将 content_db.conn 设置为 True，以便加载可选文件。最后，返回调用结果。

如果 SiteManagerPlugin 插件被注册到 SiteManager 插件系统中，它将会在插件启动时加载可选文件，从而使 content_db 数据库中的内容更加丰富。


```py
@PluginManager.registerTo("SiteManager")
class SiteManagerPlugin(object):
    def load(self, *args, **kwargs):
        back = super(SiteManagerPlugin, self).load(*args, **kwargs)
        if self.sites and not content_db.optional_files_loaded and content_db.conn:
            content_db.optional_files_loaded = True
            content_db.loadFilesOptional()
        return back
```

# `plugins/OptionalManager/OptionalManagerPlugin.py`

这段代码使用了Python的gevent库，re库和collections库，以及一个名为ContentDbPlugin的插件类。

首先，它导入了time库以用于获取当前时间戳，re库以用于解析正则表达式，collections库以用于收集各种类型的对象，以及gevent库的俱乐部（gevent.俱乐部）以简化事件处理。

然后，它导入了插件管理器类PluginManager，和一个名为ContentDbPlugin的插件类。

接下来，它定义了一个名为importPluginnedClasses的方法，该方法在插件装载后执行。

在该方法中，它通过调用Config.config.get('plugins')[0]来获取配置文件中定义的插件列表。然后，它使用pluggins列表中的所有类来实例化插件，并将它们注册为插件管理器中的PluginsMap对象的一部分。

最后，它注册了一个名为ContentDbPlugin的插件类，该插件类可能需要连接到数据库，并使用ContentDbPlugin.content_db函数将内容数据库中的内容同步到该插件的堆栈中。


```py
import time
import re
import collections

import gevent

from util import helper
from Plugin import PluginManager
from . import ContentDbPlugin


# We can only import plugin host clases after the plugins are loaded
@PluginManager.afterLoad
def importPluginnedClasses():
    global config
    from Config import config


```

这段代码是一个 Python 函数 `processAccessLog`，其目的是处理 Web 访问日志并更新数据库中的 `access_log` 字段。

具体来说，代码首先定义了一个全局变量 `access_log`，如果 `access_log` 存在，则执行以下操作：

1. 从 `ContentDbPlugin` 类中获取 `content_db` 对象，并检查 `content_db` 对象中的 `conn` 属性是否为真。

2. 如果 `content_db` 对象存在并且 `conn` 属性为真，则执行以下操作：

  a. 从 `access_log` 字典中读取前一个条目，并将其存储在 `access_log_prev` 字典中。

  b. 从 `access_log_prev` 字典中读取当前条目，并将其存储在 `access_log` 字典中。

  c. 对当前条目中的每个键，从 `access_log_prev` 字典中读取相应的值，并将其添加到 `access_log` 字典中。

  d. 调用 `content_db.log` 方法记录成功更新 `access_log` 字段的日志信息。

  如果 `access_log` 不存在，则输出一条错误消息。


```py
def processAccessLog():
    global access_log
    if access_log:
        content_db = ContentDbPlugin.content_db
        if not content_db.conn:
            return False

        s = time.time()
        access_log_prev = access_log
        access_log = collections.defaultdict(dict)
        now = int(time.time())
        num = 0
        for site_id in access_log_prev:
            content_db.execute(
                "UPDATE file_optional SET time_accessed = %s WHERE ?" % now,
                {"site_id": site_id, "inner_path": list(access_log_prev[site_id].keys())}
            )
            num += len(access_log_prev[site_id])

        content_db.log.debug("Inserted %s web request stat in %.3fs" % (num, time.time() - s))


```

这段代码是一个 Python 函数 `processRequestLog`，其作用是处理请求日志并执行一些数据库操作。

具体来说，该函数包含以下步骤：

1. 定义了一个全局变量 `request_log`，表示一个字典，用于存储请求日志的信息。

2. 判断 `request_log` 是否为空，如果是，则直接返回 `False`。

3. 如果 `request_log` 不是空，则执行以下操作：

   - 创建一个名为 `content_db` 的数据库连接对象。

   - 如果 `content_db.conn` 不是有效的连接，则创建一个新的连接并返回。

   - 创建一个空字典 `request_log`，用于存储当前的请求日志信息。

   - 遍历当前的请求日志信息，并将其存储到 `request_log` 中。

   - 使用 `content_db.execute` 方法更新数据库中的文件选项，其中包含当前请求日志信息。

   - 使用 `time.time` 方法获取当前时间，并将其作为函数执行时间的标头。

   - 在函数内部使用 `content_db.log.debug` 方法输出一条日志信息，其中包含请求日志的数量和执行时间。

   - 返回 `True`以表示函数成功执行。


```py
def processRequestLog():
    global request_log
    if request_log:
        content_db = ContentDbPlugin.content_db
        if not content_db.conn:
            return False

        s = time.time()
        request_log_prev = request_log
        request_log = collections.defaultdict(lambda: collections.defaultdict(int))  # {site_id: {inner_path1: 1, inner_path2: 1...}}
        num = 0
        for site_id in request_log_prev:
            for inner_path, uploaded in request_log_prev[site_id].items():
                content_db.execute(
                    "UPDATE file_optional SET uploaded = uploaded + %s WHERE ?" % uploaded,
                    {"site_id": site_id, "inner_path": inner_path}
                )
                num += 1
        content_db.log.debug("Inserted %s file request stat in %.3fs" % (num, time.time() - s))


```

This is a Python class that appears to be a plugin for a file storage system. It has methods for pinning, ispinning, and unpinning files, as well as setting and deleting pinned optional files.

The pinning and ispinning methods accept an inner_path argument and return a boolean value indicating whether the file is pinned or ispinned. The `setPin` method takes an inner_path and an is_pinned argument and updates the is_pinned attribute of the corresponding row in the contents database. The `optionalDelete` method deletes the file from the contents database, only if it is not pinned.

The `isPinned` method checks whether a file is pinned by checking if it has been explicitly set with the `setPin` method. If the file is pinned, it hashes the contents of the file and stores the hash in the `cache_is_pinned` dictionary. This is used to quickly identify pinned files when `is_pinned` returns `True`.

The `hash_id` property is a unique identifier for the pinned file, derived from its hash value and the site ID of the file. This property is used to quickly identify the pinned file when `is_pinned` returns `True`.

The `pinned` property is a boolean that indicates whether a file has been pinned or ispinned. This property is determined by a combination of the `isPinned` and `hash_id` properties.


```py
if "access_log" not in locals().keys():  # To keep between module reloads
    access_log = collections.defaultdict(dict)  # {site_id: {inner_path1: 1, inner_path2: 1...}}
    request_log = collections.defaultdict(lambda: collections.defaultdict(int))  # {site_id: {inner_path1: 1, inner_path2: 1...}}
    helper.timer(61, processAccessLog)
    helper.timer(60, processRequestLog)


@PluginManager.registerTo("ContentManager")
class ContentManagerPlugin(object):
    def __init__(self, *args, **kwargs):
        self.cache_is_pinned = {}
        super(ContentManagerPlugin, self).__init__(*args, **kwargs)

    def optionalDownloaded(self, inner_path, hash_id, size=None, own=False):
        if "|" in inner_path:  # Big file piece
            file_inner_path, file_range = inner_path.split("|")
        else:
            file_inner_path = inner_path

        self.contents.db.executeDelayed(
            "UPDATE file_optional SET time_downloaded = :now, is_downloaded = 1, peer = peer + 1 WHERE site_id = :site_id AND inner_path = :inner_path AND is_downloaded = 0",
            {"now": int(time.time()), "site_id": self.contents.db.site_ids[self.site.address], "inner_path": file_inner_path}
        )

        return super(ContentManagerPlugin, self).optionalDownloaded(inner_path, hash_id, size, own)

    def optionalRemoved(self, inner_path, hash_id, size=None):
        res = self.contents.db.execute(
            "UPDATE file_optional SET is_downloaded = 0, is_pinned = 0, peer = peer - 1 WHERE site_id = :site_id AND inner_path = :inner_path AND is_downloaded = 1",
            {"site_id": self.contents.db.site_ids[self.site.address], "inner_path": inner_path}
        )

        if res.rowcount > 0:
            back = super(ContentManagerPlugin, self).optionalRemoved(inner_path, hash_id, size)
            # Re-add to hashfield if we have other file with the same hash_id
            if self.isDownloaded(hash_id=hash_id, force_check_db=True):
                self.hashfield.appendHashId(hash_id)
        else:
            back = False
        self.cache_is_pinned = {}
        return back

    def optionalRenamed(self, inner_path_old, inner_path_new):
        back = super(ContentManagerPlugin, self).optionalRenamed(inner_path_old, inner_path_new)
        self.cache_is_pinned = {}
        self.contents.db.execute(
            "UPDATE file_optional SET inner_path = :inner_path_new WHERE site_id = :site_id AND inner_path = :inner_path_old",
            {"site_id": self.contents.db.site_ids[self.site.address], "inner_path_old": inner_path_old, "inner_path_new": inner_path_new}
        )
        return back

    def isDownloaded(self, inner_path=None, hash_id=None, force_check_db=False):
        if hash_id and not force_check_db and hash_id not in self.hashfield:
            return False

        if inner_path:
            res = self.contents.db.execute(
                "SELECT is_downloaded FROM file_optional WHERE site_id = :site_id AND inner_path = :inner_path LIMIT 1",
                {"site_id": self.contents.db.site_ids[self.site.address], "inner_path": inner_path}
            )
        else:
            res = self.contents.db.execute(
                "SELECT is_downloaded FROM file_optional WHERE site_id = :site_id AND hash_id = :hash_id AND is_downloaded = 1 LIMIT 1",
                {"site_id": self.contents.db.site_ids[self.site.address], "hash_id": hash_id}
            )
        row = res.fetchone()
        if row and row["is_downloaded"]:
            return True
        else:
            return False

    def isPinned(self, inner_path):
        if inner_path in self.cache_is_pinned:
            self.site.log.debug("Cached is pinned: %s" % inner_path)
            return self.cache_is_pinned[inner_path]

        res = self.contents.db.execute(
            "SELECT is_pinned FROM file_optional WHERE site_id = :site_id AND inner_path = :inner_path LIMIT 1",
            {"site_id": self.contents.db.site_ids[self.site.address], "inner_path": inner_path}
        )
        row = res.fetchone()

        if row and row[0]:
            is_pinned = True
        else:
            is_pinned = False

        self.cache_is_pinned[inner_path] = is_pinned
        self.site.log.debug("Cache set is pinned: %s %s" % (inner_path, is_pinned))

        return is_pinned

    def setPin(self, inner_path, is_pinned):
        content_db = self.contents.db
        site_id = content_db.site_ids[self.site.address]
        content_db.execute("UPDATE file_optional SET is_pinned = %d WHERE ?" % is_pinned, {"site_id": site_id, "inner_path": inner_path})
        self.cache_is_pinned = {}

    def optionalDelete(self, inner_path):
        if self.isPinned(inner_path):
            self.site.log.debug("Skip deleting pinned optional file: %s" % inner_path)
            return False
        else:
            return super(ContentManagerPlugin, self).optionalDelete(inner_path)


```

这段代码定义了两个类，一个是名为 `WorkerManagerPlugin` 的类，另一个是名为 `UiRequestPlugin` 的类。这两个类都属于一个名为 `WorkerManager` 的插件。

`WorkerManagerPlugin` 类包含一个名为 `doneTask` 的方法。这个方法执行以下操作：

1. 调用父类（`super(WorkerManagerPlugin, self).doneTask(task)`）中的 `doneTask` 方法，传递给定的 `task` 参数。
2. 如果 `task` 对象中包含一个名为 `optional_hash_id` 的选项，并且在 `self.tasks` 列表中为空，那么执行以下操作：

a. 调用 `ContentDbPlugin.content_db.processDelayed()` 方法。
b. 调用 `super(WorkerManagerPlugin, self).doneTask(task)` 中的 `doneTask` 方法，传递给定的 `task` 参数。

`UiRequestPlugin` 类包含一个名为 `parsePath` 的方法。这个方法用于解析 `path` 参数，返回 `path` 部件。

这两个类都属于 `WorkerManager` 插件，用于在 `WorkerManager` 插件的请求处理过程中执行一些额外的操作。


```py
@PluginManager.registerTo("WorkerManager")
class WorkerManagerPlugin(object):
    def doneTask(self, task):
        super(WorkerManagerPlugin, self).doneTask(task)

        if task["optional_hash_id"] and not self.tasks:  # Execute delayed queries immedietly after tasks finished
            ContentDbPlugin.content_db.processDelayed()


@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    def parsePath(self, path):
        global access_log
        path_parts = super(UiRequestPlugin, self).parsePath(path)
        if path_parts:
            site_id = ContentDbPlugin.content_db.site_ids.get(path_parts["request_address"])
            if site_id:
                if ContentDbPlugin.content_db.isOptionalFile(site_id, path_parts["inner_path"]):
                    access_log[site_id][path_parts["inner_path"]] = 1
        return path_parts


```

这段代码定义了一个名为 FileRequestPlugin 的类，用于在 FileRequest 插件中执行操作。

在 FileRequestPlugin 的 actionGetFile 和 actionStreamFile 方法中，分别调用了父类 FileRequestPlugin 和 ContentDbPlugin 的同名方法，实现了文件请求的获取和发送。

在 RecordFileRequest 方法中，记录了每个文件请求的site_id 和 inner_path，以及请求发送的字节数。

当需要记录文件请求时，会检查该请求是否可选，并且如果是，将该请求的字节数加1到 request_log 字典中，其中 site_id 是请求的site_id，inner_path 是请求的inner_path。

这段代码的作用是，在 FileRequest 插件中执行文件请求的获取和发送操作，并记录每个文件请求的信息，以便后续的统计和分析。


```py
@PluginManager.registerTo("FileRequest")
class FileRequestPlugin(object):
    def actionGetFile(self, params):
        stats = super(FileRequestPlugin, self).actionGetFile(params)
        self.recordFileRequest(params["site"], params["inner_path"], stats)
        return stats

    def actionStreamFile(self, params):
        stats = super(FileRequestPlugin, self).actionStreamFile(params)
        self.recordFileRequest(params["site"], params["inner_path"], stats)
        return stats

    def recordFileRequest(self, site_address, inner_path, stats):
        if not stats:
            # Only track the last request of files
            return False
        site_id = ContentDbPlugin.content_db.site_ids[site_address]
        if site_id and ContentDbPlugin.content_db.isOptionalFile(site_id, inner_path):
            request_log[site_id][inner_path] += stats["bytes_sent"]


```

这段代码定义了一个名为 SitePlugin 的类，用于插件系统。这个插件的作用是在插件安装、卸载以及文件上传下载过程中发挥作用。下面是这个插件的详细解释：

1. `isDownloadable(inner_path)` 方法：判断是否可以下载这个文件。这个方法首先调用父类的 `isDownloadable` 方法，如果已经是，就直接返回。否则，遍历设置中 "optional_help" 的设置，如果请求下载的文件路径与设置中的路径相同，则返回 True。否则，返回 False。

2. `fileForgot(inner_path)` 方法：尝试从系统中恢复已上传但用户忘记下载的文件。这个方法首先检查文件是否已经被挂起，如果是，就返回 False。否则，调用父类的 `fileForgot` 方法来下载这个文件。

3. `fileDone(inner_path)` 方法：下载文件完成后的处理。这个方法首先检查文件是否已经被挂起，如果是，就执行一系列操作，包括：更新 "bad_files" 键值对，将 "|" 转义字符串中的所有文件设置为只读，然后检查文件是否已经下载完成。如果下载完成，就返回 True。否则，等待一段时间并再次尝试下载，最多允许 5 次尝试。


```py
@PluginManager.registerTo("Site")
class SitePlugin(object):
    def isDownloadable(self, inner_path):
        is_downloadable = super(SitePlugin, self).isDownloadable(inner_path)
        if is_downloadable:
            return is_downloadable

        for path in self.settings.get("optional_help", {}).keys():
            if inner_path.startswith(path):
                return True

        return False

    def fileForgot(self, inner_path):
        if "|" in inner_path and self.content_manager.isPinned(re.sub(r"\|.*", "", inner_path)):
            self.log.debug("File %s is pinned, no fileForgot" % inner_path)
            return False
        else:
            return super(SitePlugin, self).fileForgot(inner_path)

    def fileDone(self, inner_path):
        if "|" in inner_path and self.bad_files.get(inner_path, 0) > 5:  # Idle optional file done
            inner_path_file = re.sub(r"\|.*", "", inner_path)
            num_changed = 0
            for key, val in self.bad_files.items():
                if key.startswith(inner_path_file) and val > 1:
                    self.bad_files[key] = 1
                    num_changed += 1
            self.log.debug("Idle optional file piece done, changed retry number of %s pieces." % num_changed)
            if num_changed:
                gevent.spawn(self.retryBadFiles)

        return super(SitePlugin, self).fileDone(inner_path)


```

这段代码是一个Python类，名为`ConfigPlugin`，属于`ConfigPlugin`插件，作用是在插件定义中定义了参数。

具体来说，该类实现了`PluginManager.registerTo("ConfigPlugin")`，即注册到`ConfigPlugin`插件中。该类中包含了一个名为`createArguments`的静态方法，用于生成命令行参数。

在该方法中，首先定义了`optional_limit`和`optional_limit_exclude_minsize`这两个参数，它们用于定义可选文件的大小限制。其中`optional_limit`指定了可选文件的最大大小，`optional_limit_exclude_minsize`指定了大于该限制的大于最小可运行文件的大小的不属于该文件大小的文件将被排除在计算之内的最小大小。这两个参数都使用了`metavar`参数来描述它们的含义，即"free space %"和"MB"。

然后，该方法调用了父类中`createArguments`方法，用于生成完整的命令行参数列表，以便将参数传递给`registerTo`方法。


```py
@PluginManager.registerTo("ConfigPlugin")
class ConfigPlugin(object):
    def createArguments(self):
        group = self.parser.add_argument_group("OptionalManager plugin")
        group.add_argument('--optional_limit', help='Limit total size of optional files', default="10%", metavar="GB or free space %")
        group.add_argument('--optional_limit_exclude_minsize', help='Exclude files larger than this limit from optional size limit calculation', default=20, metavar="MB", type=int)

        return super(ConfigPlugin, self).createArguments()

```