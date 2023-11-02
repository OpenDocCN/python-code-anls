# ZeroNet源码解析 6

# `plugins/OptionalManager/UiWebsocketPlugin.py`

这段代码使用了多种Python库，包括re库、time库、html库、os库和gevent库。它的一些主要作用是：

1. 导入re库和time库，用于处理字符串和时间相关的操作。
2. 导入html库，用于处理HTML代码。
3. 导入os库，用于处理操作系统相关的操作。
4. 导入gevent库，用于实现异步编程。
5. 导入PluginManager库，用于管理插件。
6. 导入Config库，用于设置全局配置。
7. 导入helper库，用于一些辅助操作。
8. 导入Translate库，用于将内容从一种语言翻译成另一种语言。

具体来说，这段代码的作用是用于处理一个插件的日志信息。它首先定义了一些变量，包括plugin_dir、PluginManager、config、helper、Translate以及flag等。然后，它导入了一些必要的库，并定义了一些函数，如handle_command、parse_config、translate等。这些函数的具体实现可能因插件而异，因此这段代码无法提供有关插件实际操作的更多信息。


```py
import re
import time
import html
import os

import gevent

from Plugin import PluginManager
from Config import config
from util import helper
from util.Flag import flag
from Translate import Translate


plugin_dir = os.path.dirname(__file__)

```

This is a Flask command-line interface (CLI) application that appears to provide a set of options and functionality for setting or removing optional help files on a server.

The application has three main methods:

* `actionOptionalHelpCreate`: This method accepts a Flask request with a JSON payload containing the following fields: `directory`, `title`
This field is a required field and must be a string.
* `actionOptionalHelpRemove`: This method accepts a Flask request with a JSON payload containing the following fields: `directory`, `address`
This field is a required field and must be a string.
* `actionOptionalHelpAll`: This method accepts a Flask request with a JSON payload containing the following fields: `value`
This field is a required field and must be a boolean.

The application also has a `help` dictionary that appears to provide information about the available options and their usage.

It is important to note that this application is written in Python and uses the Flask framework for web development.


```py
if "_" not in locals():
    _ = Translate(plugin_dir + "/languages/")

bigfile_sha512_cache = {}


@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    def __init__(self, *args, **kwargs):
        self.time_peer_numbers_updated = 0
        super(UiWebsocketPlugin, self).__init__(*args, **kwargs)

    def actionSiteSign(self, to, privatekey=None, inner_path="content.json", *args, **kwargs):
        # Add file to content.db and set it as pinned
        content_db = self.site.content_manager.contents.db
        content_inner_dir = helper.getDirname(inner_path)
        content_db.my_optional_files[self.site.address + "/" + content_inner_dir] = time.time()
        if len(content_db.my_optional_files) > 50:  # Keep only last 50
            oldest_key = min(
                iter(content_db.my_optional_files.keys()),
                key=(lambda key: content_db.my_optional_files[key])
            )
            del content_db.my_optional_files[oldest_key]

        return super(UiWebsocketPlugin, self).actionSiteSign(to, privatekey, inner_path, *args, **kwargs)

    def updatePeerNumbers(self):
        self.site.updateHashfield()
        content_db = self.site.content_manager.contents.db
        content_db.updatePeerNumbers()
        self.site.updateWebsocket(peernumber_updated=True)

    def addBigfileInfo(self, row):
        global bigfile_sha512_cache

        content_db = self.site.content_manager.contents.db
        site = content_db.sites[row["address"]]
        if not site.settings.get("has_bigfile"):
            return False

        file_key = row["address"] + "/" + row["inner_path"]
        sha512 = bigfile_sha512_cache.get(file_key)
        file_info = None
        if not sha512:
            file_info = site.content_manager.getFileInfo(row["inner_path"])
            if not file_info or not file_info.get("piece_size"):
                return False
            sha512 = file_info["sha512"]
            bigfile_sha512_cache[file_key] = sha512

        if sha512 in site.storage.piecefields:
            piecefield = site.storage.piecefields[sha512].tobytes()
        else:
            piecefield = None

        if piecefield:
            row["pieces"] = len(piecefield)
            row["pieces_downloaded"] = piecefield.count(b"\x01")
            row["downloaded_percent"] = 100 * row["pieces_downloaded"] / row["pieces"]
            if row["pieces_downloaded"]:
                if row["pieces"] == row["pieces_downloaded"]:
                    row["bytes_downloaded"] = row["size"]
                else:
                    if not file_info:
                        file_info = site.content_manager.getFileInfo(row["inner_path"])
                    row["bytes_downloaded"] = row["pieces_downloaded"] * file_info.get("piece_size", 0)
            else:
                row["bytes_downloaded"] = 0

            row["is_downloading"] = bool(next((inner_path for inner_path in site.bad_files if inner_path.startswith(row["inner_path"])), False))

        # Add leech / seed stats
        row["peer_seed"] = 0
        row["peer_leech"] = 0
        for peer in site.peers.values():
            if not peer.time_piecefields_updated or sha512 not in peer.piecefields:
                continue
            peer_piecefield = peer.piecefields[sha512].tobytes()
            if not peer_piecefield:
                continue
            if peer_piecefield == b"\x01" * len(peer_piecefield):
                row["peer_seed"] += 1
            else:
                row["peer_leech"] += 1

        # Add myself
        if piecefield:
            if row["pieces_downloaded"] == row["pieces"]:
                row["peer_seed"] += 1
            else:
                row["peer_leech"] += 1

        return True

    # Optional file functions

    def actionOptionalFileList(self, to, address=None, orderby="time_downloaded DESC", limit=10, filter="downloaded", filter_inner_path=None):
        if not address:
            address = self.site.address

        # Update peer numbers if necessary
        content_db = self.site.content_manager.contents.db
        if time.time() - content_db.time_peer_numbers_updated > 60 * 1 and time.time() - self.time_peer_numbers_updated > 60 * 5:
            # Start in new thread to avoid blocking
            self.time_peer_numbers_updated = time.time()
            gevent.spawn(self.updatePeerNumbers)

        if address == "all" and "ADMIN" not in self.permissions:
            return self.response(to, {"error": "Forbidden"})

        if not self.hasSitePermission(address):
            return self.response(to, {"error": "Forbidden"})

        if not all([re.match("^[a-z_*/+-]+( DESC| ASC|)$", part.strip()) for part in orderby.split(",")]):
            return self.response(to, "Invalid order_by")

        if type(limit) != int:
            return self.response(to, "Invalid limit")

        back = []
        content_db = self.site.content_manager.contents.db

        wheres = {}
        wheres_raw = []
        if "bigfile" in filter:
            wheres["size >"] = 1024 * 1024 * 1
        if "downloaded" in filter:
            wheres_raw.append("(is_downloaded = 1 OR is_pinned = 1)")
        if "pinned" in filter:
            wheres["is_pinned"] = 1
        if filter_inner_path:
            wheres["inner_path__like"] = filter_inner_path

        if address == "all":
            join = "LEFT JOIN site USING (site_id)"
        else:
            wheres["site_id"] = content_db.site_ids[address]
            join = ""

        if wheres_raw:
            query_wheres_raw = "AND" + " AND ".join(wheres_raw)
        else:
            query_wheres_raw = ""

        query = "SELECT * FROM file_optional %s WHERE ? %s ORDER BY %s LIMIT %s" % (join, query_wheres_raw, orderby, limit)

        for row in content_db.execute(query, wheres):
            row = dict(row)
            if address != "all":
                row["address"] = address

            if row["size"] > 1024 * 1024:
                has_bigfile_info = self.addBigfileInfo(row)
            else:
                has_bigfile_info = False

            if not has_bigfile_info and "bigfile" in filter:
                continue

            if not has_bigfile_info:
                if row["is_downloaded"]:
                    row["bytes_downloaded"] = row["size"]
                    row["downloaded_percent"] = 100
                else:
                    row["bytes_downloaded"] = 0
                    row["downloaded_percent"] = 0

            back.append(row)
        self.response(to, back)

    def actionOptionalFileInfo(self, to, inner_path):
        content_db = self.site.content_manager.contents.db
        site_id = content_db.site_ids[self.site.address]

        # Update peer numbers if necessary
        if time.time() - content_db.time_peer_numbers_updated > 60 * 1 and time.time() - self.time_peer_numbers_updated > 60 * 5:
            # Start in new thread to avoid blocking
            self.time_peer_numbers_updated = time.time()
            gevent.spawn(self.updatePeerNumbers)

        query = "SELECT * FROM file_optional WHERE site_id = :site_id AND inner_path = :inner_path LIMIT 1"
        res = content_db.execute(query, {"site_id": site_id, "inner_path": inner_path})
        row = next(res, None)
        if row:
            row = dict(row)
            if row["size"] > 1024 * 1024:
                row["address"] = self.site.address
                self.addBigfileInfo(row)
            self.response(to, row)
        else:
            self.response(to, None)

    def setPin(self, inner_path, is_pinned, address=None):
        if not address:
            address = self.site.address

        if not self.hasSitePermission(address):
            return {"error": "Forbidden"}

        site = self.server.sites[address]
        site.content_manager.setPin(inner_path, is_pinned)

        return "ok"

    @flag.no_multiuser
    def actionOptionalFilePin(self, to, inner_path, address=None):
        if type(inner_path) is not list:
            inner_path = [inner_path]
        back = self.setPin(inner_path, 1, address)
        num_file = len(inner_path)
        if back == "ok":
            if num_file == 1:
                self.cmd("notification", ["done", _["Pinned %s"] % html.escape(helper.getFilename(inner_path[0])), 5000])
            else:
                self.cmd("notification", ["done", _["Pinned %s files"] % num_file, 5000])
        self.response(to, back)

    @flag.no_multiuser
    def actionOptionalFileUnpin(self, to, inner_path, address=None):
        if type(inner_path) is not list:
            inner_path = [inner_path]
        back = self.setPin(inner_path, 0, address)
        num_file = len(inner_path)
        if back == "ok":
            if num_file == 1:
                self.cmd("notification", ["done", _["Removed pin from %s"] % html.escape(helper.getFilename(inner_path[0])), 5000])
            else:
                self.cmd("notification", ["done", _["Removed pin from %s files"] % num_file, 5000])
        self.response(to, back)

    @flag.no_multiuser
    def actionOptionalFileDelete(self, to, inner_path, address=None):
        if not address:
            address = self.site.address

        if not self.hasSitePermission(address):
            return self.response(to, {"error": "Forbidden"})

        site = self.server.sites[address]

        content_db = site.content_manager.contents.db
        site_id = content_db.site_ids[site.address]

        res = content_db.execute("SELECT * FROM file_optional WHERE ? LIMIT 1", {"site_id": site_id, "inner_path": inner_path, "is_downloaded": 1})
        row = next(res, None)

        if not row:
            return self.response(to, {"error": "Not found in content.db"})

        removed = site.content_manager.optionalRemoved(inner_path, row["hash_id"], row["size"])
        # if not removed:
        #    return self.response(to, {"error": "Not found in hash_id: %s" % row["hash_id"]})

        content_db.execute("UPDATE file_optional SET is_downloaded = 0, is_pinned = 0, peer = peer - 1 WHERE ?", {"site_id": site_id, "inner_path": inner_path})

        try:
            site.storage.delete(inner_path)
        except Exception as err:
            return self.response(to, {"error": "File delete error: %s" % err})
        site.updateWebsocket(file_delete=inner_path)

        if inner_path in site.content_manager.cache_is_pinned:
            site.content_manager.cache_is_pinned = {}

        self.response(to, "ok")

    # Limit functions

    @flag.admin
    def actionOptionalLimitStats(self, to):
        back = {}
        back["limit"] = config.optional_limit
        back["used"] = self.site.content_manager.contents.db.getOptionalUsedBytes()
        back["free"] = helper.getFreeSpace()

        self.response(to, back)

    @flag.no_multiuser
    @flag.admin
    def actionOptionalLimitSet(self, to, limit):
        config.optional_limit = re.sub(r"\.0+$", "", limit)  # Remove unnecessary digits from end
        config.saveValue("optional_limit", limit)
        self.response(to, "ok")

    # Distribute help functions

    def actionOptionalHelpList(self, to, address=None):
        if not address:
            address = self.site.address

        if not self.hasSitePermission(address):
            return self.response(to, {"error": "Forbidden"})

        site = self.server.sites[address]

        self.response(to, site.settings.get("optional_help", {}))

    @flag.no_multiuser
    def actionOptionalHelp(self, to, directory, title, address=None):
        if not address:
            address = self.site.address

        if not self.hasSitePermission(address):
            return self.response(to, {"error": "Forbidden"})

        site = self.server.sites[address]
        content_db = site.content_manager.contents.db
        site_id = content_db.site_ids[address]

        if "optional_help" not in site.settings:
            site.settings["optional_help"] = {}

        stats = content_db.execute(
            "SELECT COUNT(*) AS num, SUM(size) AS size FROM file_optional WHERE site_id = :site_id AND inner_path LIKE :inner_path",
            {"site_id": site_id, "inner_path": directory + "%"}
        ).fetchone()
        stats = dict(stats)

        if not stats["size"]:
            stats["size"] = 0
        if not stats["num"]:
            stats["num"] = 0

        self.cmd("notification", [
            "done",
            _["You started to help distribute <b>%s</b>.<br><small>Directory: %s</small>"] %
            (html.escape(title), html.escape(directory)),
            10000
        ])

        site.settings["optional_help"][directory] = title

        self.response(to, dict(stats))

    @flag.no_multiuser
    def actionOptionalHelpRemove(self, to, directory, address=None):
        if not address:
            address = self.site.address

        if not self.hasSitePermission(address):
            return self.response(to, {"error": "Forbidden"})

        site = self.server.sites[address]

        try:
            del site.settings["optional_help"][directory]
            self.response(to, "ok")
        except Exception:
            self.response(to, {"error": "Not found"})

    def cbOptionalHelpAll(self, to, site, value):
        site.settings["autodownloadoptional"] = value
        self.response(to, value)

    @flag.no_multiuser
    def actionOptionalHelpAll(self, to, value, address=None):
        if not address:
            address = self.site.address

        if not self.hasSitePermission(address):
            return self.response(to, {"error": "Forbidden"})

        site = self.server.sites[address]

        if value:
            if "ADMIN" in self.site.settings["permissions"]:
                self.cbOptionalHelpAll(to, site, True)
            else:
                site_title = site.content_manager.contents["content.json"].get("title", address)
                self.cmd(
                    "confirm",
                    [
                        _["Help distribute all new optional files on site <b>%s</b>"] % html.escape(site_title),
                        _["Yes, I want to help!"]
                    ],
                    lambda res: self.cbOptionalHelpAll(to, site, True)
                )
        else:
            site.settings["autodownloadoptional"] = False
            self.response(to, False)

```

# `plugins/OptionalManager/__init__.py`

这两行代码定义了两个名为 "OptionalManagerPlugin" 和 "UiWebsocketPlugin" 的类，它们都属于一个名为 "Plugin" 的类。

"OptionalManagerPlugin" 可能是一个管理可选数据的类，用于在程序中选择或取消选择某些选项或数据。"UiWebsocketPlugin" 可能是一个用于在用户界面中接收实时 WebSocket 连接的类。

作为一个程序，你需要创建一个 "Plugin" 类，然后从 "OptionalManagerPlugin" 和 "UiWebsocketPlugin" 两个类中选择一个来实例化 "Plugin" 类。


```py
from . import OptionalManagerPlugin
from . import UiWebsocketPlugin

```

# `plugins/OptionalManager/Test/conftest.py`

这段代码是一个Python定义，从名为"src.Test.conftest"的包中导入了Test类。这个定义可能是一个用于定义测试函数的通用函数，也可能是某个测试框架的类或函数。

但请注意，由于缺乏上下文，无法确切地了解这段代码在测试中具体会被如何使用。


```py
from src.Test.conftest import *
```

# `plugins/OptionalManager/Test/TestOptionalManager.py`

It looks like you are testing two things:

1. The renaming of the optional image file "data/img/zerotalk-upvote.png" to "data/img/zerotalk-upvote-new.png" is working as expected.
2. The renaming of the image file "data/img/zerotalk-upvote.png" itself is not having any effect on the contents of


```py
import copy

import pytest


@pytest.mark.usefixtures("resetSettings")
class TestOptionalManager:
    def testDbFill(self, site):
        contents = site.content_manager.contents
        assert len(site.content_manager.hashfield) > 0
        assert contents.db.execute("SELECT COUNT(*) FROM file_optional WHERE is_downloaded = 1").fetchone()[0] == len(site.content_manager.hashfield)

    def testSetContent(self, site):
        contents = site.content_manager.contents

        # Add new file
        new_content = copy.deepcopy(contents["content.json"])
        new_content["files_optional"]["testfile"] = {
            "size": 1234,
            "sha512": "aaaabbbbcccc"
        }
        num_optional_files_before = contents.db.execute("SELECT COUNT(*) FROM file_optional").fetchone()[0]
        contents["content.json"] = new_content
        assert contents.db.execute("SELECT COUNT(*) FROM file_optional").fetchone()[0] > num_optional_files_before

        # Remove file
        new_content = copy.deepcopy(contents["content.json"])
        del new_content["files_optional"]["testfile"]
        num_optional_files_before = contents.db.execute("SELECT COUNT(*) FROM file_optional").fetchone()[0]
        contents["content.json"] = new_content
        assert contents.db.execute("SELECT COUNT(*) FROM file_optional").fetchone()[0] < num_optional_files_before

    def testDeleteContent(self, site):
        contents = site.content_manager.contents
        num_optional_files_before = contents.db.execute("SELECT COUNT(*) FROM file_optional").fetchone()[0]
        del contents["content.json"]
        assert contents.db.execute("SELECT COUNT(*) FROM file_optional").fetchone()[0] < num_optional_files_before

    def testVerifyFiles(self, site):
        contents = site.content_manager.contents

        # Add new file
        new_content = copy.deepcopy(contents["content.json"])
        new_content["files_optional"]["testfile"] = {
            "size": 1234,
            "sha512": "aaaabbbbcccc"
        }
        contents["content.json"] = new_content
        file_row = contents.db.execute("SELECT * FROM file_optional WHERE inner_path = 'testfile'").fetchone()
        assert not file_row["is_downloaded"]

        # Write file from outside of ZeroNet
        site.storage.open("testfile", "wb").write(b"A" * 1234)  # For quick check hash does not matter only file size

        hashfield_len_before = len(site.content_manager.hashfield)
        site.storage.verifyFiles(quick_check=True)
        assert len(site.content_manager.hashfield) == hashfield_len_before + 1

        file_row = contents.db.execute("SELECT * FROM file_optional WHERE inner_path = 'testfile'").fetchone()
        assert file_row["is_downloaded"]

        # Delete file outside of ZeroNet
        site.storage.delete("testfile")
        site.storage.verifyFiles(quick_check=True)
        file_row = contents.db.execute("SELECT * FROM file_optional WHERE inner_path = 'testfile'").fetchone()
        assert not file_row["is_downloaded"]

    def testVerifyFilesSameHashId(self, site):
        contents = site.content_manager.contents

        new_content = copy.deepcopy(contents["content.json"])

        # Add two files with same hashid (first 4 character)
        new_content["files_optional"]["testfile1"] = {
            "size": 1234,
            "sha512": "aaaabbbbcccc"
        }
        new_content["files_optional"]["testfile2"] = {
            "size": 2345,
            "sha512": "aaaabbbbdddd"
        }
        contents["content.json"] = new_content

        assert site.content_manager.hashfield.getHashId("aaaabbbbcccc") == site.content_manager.hashfield.getHashId("aaaabbbbdddd")

        # Write files from outside of ZeroNet (For quick check hash does not matter only file size)
        site.storage.open("testfile1", "wb").write(b"A" * 1234)
        site.storage.open("testfile2", "wb").write(b"B" * 2345)

        site.storage.verifyFiles(quick_check=True)

        # Make sure that both is downloaded
        assert site.content_manager.isDownloaded("testfile1")
        assert site.content_manager.isDownloaded("testfile2")
        assert site.content_manager.hashfield.getHashId("aaaabbbbcccc") in site.content_manager.hashfield

        # Delete one of the files
        site.storage.delete("testfile1")
        site.storage.verifyFiles(quick_check=True)
        assert not site.content_manager.isDownloaded("testfile1")
        assert site.content_manager.isDownloaded("testfile2")
        assert site.content_manager.hashfield.getHashId("aaaabbbbdddd") in site.content_manager.hashfield

    def testIsPinned(self, site):
        assert not site.content_manager.isPinned("data/img/zerotalk-upvote.png")
        site.content_manager.setPin("data/img/zerotalk-upvote.png", True)
        assert site.content_manager.isPinned("data/img/zerotalk-upvote.png")

        assert len(site.content_manager.cache_is_pinned) == 1
        site.content_manager.cache_is_pinned = {}
        assert site.content_manager.isPinned("data/img/zerotalk-upvote.png")

    def testBigfilePieceReset(self, site):
        site.bad_files = {
            "data/fake_bigfile.mp4|0-1024": 10,
            "data/fake_bigfile.mp4|1024-2048": 10,
            "data/fake_bigfile.mp4|2048-3064": 10
        }
        site.onFileDone("data/fake_bigfile.mp4|0-1024")
        assert site.bad_files["data/fake_bigfile.mp4|1024-2048"] == 1
        assert site.bad_files["data/fake_bigfile.mp4|2048-3064"] == 1

    def testOptionalDelete(self, site):
        contents = site.content_manager.contents

        site.content_manager.setPin("data/img/zerotalk-upvote.png", True)
        site.content_manager.setPin("data/img/zeroid.png", False)
        new_content = copy.deepcopy(contents["content.json"])
        del new_content["files_optional"]["data/img/zerotalk-upvote.png"]
        del new_content["files_optional"]["data/img/zeroid.png"]

        assert site.storage.isFile("data/img/zerotalk-upvote.png")
        assert site.storage.isFile("data/img/zeroid.png")

        site.storage.writeJson("content.json", new_content)
        site.content_manager.loadContent("content.json", force=True)

        assert not site.storage.isFile("data/img/zeroid.png")
        assert site.storage.isFile("data/img/zerotalk-upvote.png")

    def testOptionalRename(self, site):
        contents = site.content_manager.contents

        site.content_manager.setPin("data/img/zerotalk-upvote.png", True)
        new_content = copy.deepcopy(contents["content.json"])
        new_content["files_optional"]["data/img/zerotalk-upvote-new.png"] = new_content["files_optional"]["data/img/zerotalk-upvote.png"]
        del new_content["files_optional"]["data/img/zerotalk-upvote.png"]

        assert site.storage.isFile("data/img/zerotalk-upvote.png")
        assert site.content_manager.isPinned("data/img/zerotalk-upvote.png")

        site.storage.writeJson("content.json", new_content)
        site.content_manager.loadContent("content.json", force=True)

        assert not site.storage.isFile("data/img/zerotalk-upvote.png")
        assert not site.content_manager.isPinned("data/img/zerotalk-upvote.png")
        assert site.content_manager.isPinned("data/img/zerotalk-upvote-new.png")
        assert site.storage.isFile("data/img/zerotalk-upvote-new.png")

```

# `plugins/PeerDb/PeerDbPlugin.py`

This is a Python class that appears to be a database plugin for a web application. It is using the Flask-SQLAlchemy library and the SQLite database.

The `ContentDbPlugin` class is responsible for the persistence of the database. It initializes the site, loads all peers, and saves them. It also has a method `saveAllPeers` that is a function of the `savePeers` method which will save all peers.

The `savePeers` method is using a `with` statement to ensure that the connection is closed in case an exception is raised. It uses the `greenlet_manager` property to spawn the task of loading and saving the peers. It also, if `spawn` is `True`, it will run the function every hour.

The `getCursor` method is using the `executemany` method to execute the query and return multiple results. It is passed the function `self.iteratePeers` as a parameter.

The `iteratePeers` method is responsible for loading all peers and is a part of the `savePeers` method. It is not defined in this class, but it appears to be responsible for querying the database for all peers, and then inserting the data into the `peer` table.

It is using the `time.time()` method to get the current time and subtract it from the start time of the hour, in order to not update the same site at the same time.


```py
import time
import sqlite3
import random
import atexit

import gevent
from Plugin import PluginManager


@PluginManager.registerTo("ContentDb")
class ContentDbPlugin(object):
    def __init__(self, *args, **kwargs):
        atexit.register(self.saveAllPeers)
        super(ContentDbPlugin, self).__init__(*args, **kwargs)

    def getSchema(self):
        schema = super(ContentDbPlugin, self).getSchema()

        schema["tables"]["peer"] = {
            "cols": [
                ["site_id", "INTEGER REFERENCES site (site_id) ON DELETE CASCADE"],
                ["address", "TEXT NOT NULL"],
                ["port", "INTEGER NOT NULL"],
                ["hashfield", "BLOB"],
                ["reputation", "INTEGER NOT NULL"],
                ["time_added", "INTEGER NOT NULL"],
                ["time_found", "INTEGER NOT NULL"]
            ],
            "indexes": [
                "CREATE UNIQUE INDEX peer_key ON peer (site_id, address, port)"
            ],
            "schema_changed": 2
        }

        return schema

    def loadPeers(self, site):
        s = time.time()
        site_id = self.site_ids.get(site.address)
        res = self.execute("SELECT * FROM peer WHERE site_id = :site_id", {"site_id": site_id})
        num = 0
        num_hashfield = 0
        for row in res:
            peer = site.addPeer(str(row["address"]), row["port"])
            if not peer:  # Already exist
                continue
            if row["hashfield"]:
                peer.hashfield.replaceFromBytes(row["hashfield"])
                num_hashfield += 1
            peer.time_added = row["time_added"]
            peer.time_found = row["time_found"]
            peer.reputation = row["reputation"]
            if row["address"].endswith(".onion"):
                peer.reputation = peer.reputation / 2 - 1 # Onion peers less likely working
            num += 1
        if num_hashfield:
            site.content_manager.has_optional_files = True
        site.log.debug("%s peers (%s with hashfield) loaded in %.3fs" % (num, num_hashfield, time.time() - s))

    def iteratePeers(self, site):
        site_id = self.site_ids.get(site.address)
        for key, peer in list(site.peers.items()):
            address, port = key.rsplit(":", 1)
            if peer.has_hashfield:
                hashfield = sqlite3.Binary(peer.hashfield.tobytes())
            else:
                hashfield = ""
            yield (site_id, address, port, hashfield, peer.reputation, int(peer.time_added), int(peer.time_found))

    def savePeers(self, site, spawn=False):
        if spawn:
            # Save peers every hour (+random some secs to not update very site at same time)
            site.greenlet_manager.spawnLater(60 * 60 + random.randint(0, 60), self.savePeers, site, spawn=True)
        if not site.peers:
            site.log.debug("Peers not saved: No peers found")
            return
        s = time.time()
        site_id = self.site_ids.get(site.address)
        cur = self.getCursor()
        try:
            cur.execute("DELETE FROM peer WHERE site_id = :site_id", {"site_id": site_id})
            cur.executemany(
                "INSERT INTO peer (site_id, address, port, hashfield, reputation, time_added, time_found) VALUES (?, ?, ?, ?, ?, ?, ?)",
                self.iteratePeers(site)
            )
        except Exception as err:
            site.log.error("Save peer error: %s" % err)
        site.log.debug("Peers saved in %.3fs" % (time.time() - s))

    def initSite(self, site):
        super(ContentDbPlugin, self).initSite(site)
        site.greenlet_manager.spawnLater(0.5, self.loadPeers, site)
        site.greenlet_manager.spawnLater(60*60, self.savePeers, site, spawn=True)

    def saveAllPeers(self):
        for site in list(self.sites.values()):
            try:
                self.savePeers(site)
            except Exception as err:
                site.log.error("Save peer error: %s" % err)

```

# `plugins/PeerDb/__init__.py`

这段代码的作用是导入了一个名为 "PeerDbPlugin" 的类，可能是一个数据库插件，用于在 PeerDB(可能是一个数据库引擎)中进行数据操作。具体来说，这个插件允许在 PeerDB 中使用 SQL 查询语言来查询和修改数据。在代码中，我们可能看到了一些定义变量和函数，这些变量和函数可以用于配置、查询或执行数据库操作。但这些都是推测，因为我们不知道代码的上下文和确切的目的。


```py
from . import PeerDbPlugin


```

# `plugins/Sidebar/ConsolePlugin.py`

这段代码定义了一个名为 `WsLogStreamer` 的类，用于将 `logging.StreamHandler` 与 WebSocket 进行集成。具体来说，它实现了以下几个方法：

1. `__init__` 方法：初始化函数，根据传入的 `stream_id` 和 `ui_websocket` 参数，加载预设过滤规则。如果过滤规则不是安全的，则会抛出异常。
2. `format` 方法：用于将 `record` 对象转换为适合输出到 WebSocket 的格式。首先，会根据传递的过滤规则检查字符串是否匹配，如果不是，则返回 `False`。否则，会将记录的 `line` 添加到输出到 WebSocket 的数据中。
3. `emit` 方法：将 `record` 对象发送到 WebSocket 的 `logLineAdd` 方法中。
4. `stop` 方法：在 `__init__` 方法中实现，用于停止记录并从 WebSocket 中清除所有记录。

`WsLogStreamer` 的作用是帮助用户将日志信息记录到 WebSocket 中。由于它使用了 `logging.StreamHandler` 和 `re` 库，因此可以记录多种类型的日志信息，包括标准日志信息和包含 `__private__` 修饰的日志信息。此外，由于它使用了 `ui_websocket` 提供的 WebSocket 客户端，因此可以确保记录的日志信息实时地发送到 WebSocket 中。


```py
import re
import logging

from Plugin import PluginManager
from Config import config
from Debug import Debug
from util import SafeRe
from util.Flag import flag


class WsLogStreamer(logging.StreamHandler):
    def __init__(self, stream_id, ui_websocket, filter):
        self.stream_id = stream_id
        self.ui_websocket = ui_websocket

        if filter:
            if not SafeRe.isSafePattern(filter):
                raise Exception("Not a safe prex pattern")
            self.filter_re = re.compile(".*" + filter)
        else:
            self.filter_re = None
        return super(WsLogStreamer, self).__init__()

    def emit(self, record):
        if self.ui_websocket.ws.closed:
            self.stop()
            return
        line = self.format(record)
        if self.filter_re and not self.filter_re.match(line):
            return False

        self.ui_websocket.cmd("logLineAdd", {"stream_id": self.stream_id, "lines": [line]})

    def stop(self):
        logging.getLogger('').removeHandler(self)


```

This is a Python class that represents a console log stream. It has an action method for adding a log streamer and a remove method for removing a log streamer.

The log streamer is added by the addLogStreamer method. This method takes a stream ID and a filter. It creates a log streamer by calling the addLogStreamer method and passing the stream ID and filter to it. The log streamer is then added to the log streamer dictionary, which is stored in the log\_streamers attribute.

The remove log streamer is removed by the removeLogStreamer method. This method takes a stream ID to identify the log streamer to remove. It calls the stop method on the log streamer object and removes it from the log streamer dictionary.

The actionConsoleLogStream method is a wrapper for the addLogStreamer method. It takes a target object and a filter to apply. It passes the target object to the addLogStreamer method and returns a success or failure message.

The actionConsoleLogStreamRemove method is a wrapper for the removeLogStreamer method. It takes a target object and the stream ID to remove. It calls the removeLogStreamer method on the log streamer object and returns an error message if anything went wrong or a success message.


```py
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    def __init__(self, *args, **kwargs):
        self.log_streamers = {}
        return super(UiWebsocketPlugin, self).__init__(*args, **kwargs)

    @flag.no_multiuser
    @flag.admin
    def actionConsoleLogRead(self, to, filter=None, read_size=32 * 1024, limit=500):
        log_file_path = "%s/debug.log" % config.log_dir
        log_file = open(log_file_path, encoding="utf-8")
        log_file.seek(0, 2)
        end_pos = log_file.tell()
        log_file.seek(max(0, end_pos - read_size))
        if log_file.tell() != 0:
            log_file.readline()  # Partial line junk

        pos_start = log_file.tell()
        lines = []
        if filter:
            assert SafeRe.isSafePattern(filter)
            filter_re = re.compile(".*" + filter)

        last_match = False
        for line in log_file:
            if not line.startswith("[") and last_match:  # Multi-line log entry
                lines.append(line.replace(" ", "&nbsp;"))
                continue

            if filter and not filter_re.match(line):
                last_match = False
                continue
            last_match = True
            lines.append(line)

        num_found = len(lines)
        lines = lines[-limit:]

        return {"lines": lines, "pos_end": log_file.tell(), "pos_start": pos_start, "num_found": num_found}

    def addLogStreamer(self, stream_id, filter=None):
        logger = WsLogStreamer(stream_id, self, filter)
        logger.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)-8s %(name)s %(message)s'))
        logger.setLevel(logging.getLevelName("DEBUG"))

        logging.getLogger('').addHandler(logger)
        return logger

    @flag.no_multiuser
    @flag.admin
    def actionConsoleLogStream(self, to, filter=None):
        stream_id = to
        self.log_streamers[stream_id] = self.addLogStreamer(stream_id, filter)
        self.response(to, {"stream_id": stream_id})

    @flag.no_multiuser
    @flag.admin
    def actionConsoleLogStreamRemove(self, to, stream_id):
        try:
            self.log_streamers[stream_id].stop()
            del self.log_streamers[stream_id]
            return "ok"
        except Exception as err:
            return {"error": Debug.formatException(err)}

```

# `plugins/Sidebar/SidebarPlugin.py`

这段代码的作用是执行以下任务：

1. 导入需要的库：re、os、html、sys、math、time、json、io、urllib、urllib.parse。
2. 从指定文件夹（可能是用户输入）读取所有文本文件，并解析里面的链接。
3. 将文本文件中的链接转换为阿拉伯数字。
4. 将文本文件中的链接与标签之间的关系（如title、rate等）保存到配置文件中。
5. 将解析出的链接和标签信息存储到数据库中。
6. 使用gevent库的 gevent 函数来异步处理这些链接。
7. 将解析出的链接和标签信息暴露给应用程序使用。


```py
import re
import os
import html
import sys
import math
import time
import json
import io
import urllib
import urllib.parse

import gevent

import util
from Config import config
```

这段代码的作用是定义了几个 Python 类，包括 PluginManager、Debug、Translate 和 helper，以及一个名为 ZipStream 的 ZipStream 类的别名。

具体来说，这段代码做以下几件事情：

1. 导入了一些 Python 类，包括 os 和 zipfile。
2. 定义了一个名为 plugin_dir 的类变量，该变量是一个字符串，表示插件文件的目录。
3. 定义了一个名为 media_dir 的类变量，该变量是一个字符串，表示媒体文件目录，该目录下面会有多个子目录，每个子目录下都有一个名为 "media" 的目录。
4. 定义了一个名为 local_cache 的类变量，该变量是一个字典，用于存储本地变量。
5. 定义了一个名为 Translate 的类，该类继承自 pygame.locals 的 helpers.py 文件，然后重写了其中的几个方法，包括 _ __init__、_ __call__ 和 _ __get__。
6. 定义了一个名为 ZipStream 的类，该类继承自 zipfile.ZipFile，然后重写了其中的几个方法，包括 open 和 seek。
7. 在插件目录下创建了一个名为 "languages" 的目录，并在里面创建了一个名为 "en" 的语言文件夹。

这段代码的主要作用是定义了几个 Python 类，以及一个名为 ZipStream 的 ZipStream 类的别名，然后通过这些类提供了一些方便的游戏开发中使用的功能，例如本地变量缓存、调试输出、媒体文件处理和压缩等。


```py
from Plugin import PluginManager
from Debug import Debug
from Translate import Translate
from util import helper
from util.Flag import flag
from .ZipStream import ZipStream

plugin_dir = os.path.dirname(__file__)
media_dir = plugin_dir + "/media"

loc_cache = {}
if "_" not in locals():
    _ = Translate(plugin_dir + "/languages/")


```

This is a Python class that implements the `IHttpRequestPlugin` interface for media files. It is used to serve media files from a server and provides functionality for handling different media file formats, such as JavaScript and CSS files, and serving the files through a zip file.

The class has the following methods:

* `__init__(self, http, server, site_manager, action_manager, media_dir, debug)`: Initializes the media plugin, including the configuration settings.
* `do_action(name, method, args, **kwargs)`: Handles the action for a given action, including the execution of the action and its arguments.
* `execute_action(action, **kwargs)`: Executes the action for the given action, passing in the required arguments.
* `actionFile(path, method=None, **kwargs)`: Handles the action file for the given action, passing in the required arguments.
* `getActionResponse(response, **kwargs)`: Retrieves the response for the given action.
* `actionZip()`: Downloads and returns the zip file for the media.

The class also includes several utility methods, such as `uriToJs()` and `jsToJson()`, which are not implemented in this class but are passed down from the parent class.


```py
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # Inject our resources to end of original file streams
    def actionUiMedia(self, path):
        if path == "/uimedia/all.js" or path == "/uimedia/all.css":
            # First yield the original file and header
            body_generator = super(UiRequestPlugin, self).actionUiMedia(path)
            for part in body_generator:
                yield part

            # Append our media file to the end
            ext = re.match(".*(js|css)$", path).group(1)
            plugin_media_file = "%s/all.%s" % (media_dir, ext)
            if config.debug:
                # If debugging merge *.css to all.css and *.js to all.js
                from Debug import DebugMedia
                DebugMedia.merge(plugin_media_file)
            if ext == "js":
                yield _.translateData(open(plugin_media_file).read()).encode("utf8")
            else:
                for part in self.actionFile(plugin_media_file, send_header=False):
                    yield part
        elif path.startswith("/uimedia/globe/"):  # Serve WebGL globe files
            file_name = re.match(".*/(.*)", path).group(1)
            plugin_media_file = "%s_globe/%s" % (media_dir, file_name)
            if config.debug and path.endswith("all.js"):
                # If debugging merge *.css to all.css and *.js to all.js
                from Debug import DebugMedia
                DebugMedia.merge(plugin_media_file)
            for part in self.actionFile(plugin_media_file):
                yield part
        else:
            for part in super(UiRequestPlugin, self).actionUiMedia(path):
                yield part

    def actionZip(self):
        address = self.get["address"]
        site = self.server.site_manager.get(address)
        if not site:
            return self.error404("Site not found")

        title = site.content_manager.contents.get("content.json", {}).get("title", "")
        filename = "%s-backup-%s.zip" % (title, time.strftime("%Y-%m-%d_%H_%M"))
        filename_quoted = urllib.parse.quote(filename)
        self.sendHeader(content_type="application/zip", extra_headers={'Content-Disposition': 'attachment; filename="%s"' % filename_quoted})

        return self.streamZip(site.storage.getPath("."))

    def streamZip(self, dir_path):
        zs = ZipStream(dir_path)
        while 1:
            data = zs.read()
            if not data:
                break
            yield data


```

This is a Flask-Restplus implementation of an admin API for managing private keys and automatic downloads.

The `PrivateKey` class is used to store the private key for each address index in the content JSON file. The `CryptBitcoin.hdPrivatekey` method is used to generate the private key from the master seed.

The `CryptBitcoin.privatekeyToAddress` method is used to convert the private key to an address.

The `actionUserSetSitePrivatekey` and `actionSiteSetAutodownloadoptional` functions allow users to set the site's private key and enable automatic downloads, respectively.

The `actionDbReload` and `actionDbRebuild` functions are used to reload and rebuild the database, respectively.

Note: This implementation is for educational and demonstration purposes only and should not be used in a production environment.


```py
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    def sidebarRenderPeerStats(self, body, site):
        connected = len([peer for peer in list(site.peers.values()) if peer.connection and peer.connection.connected])
        connectable = len([peer_id for peer_id in list(site.peers.keys()) if not peer_id.endswith(":0")])
        onion = len([peer_id for peer_id in list(site.peers.keys()) if ".onion" in peer_id])
        local = len([peer for peer in list(site.peers.values()) if helper.isPrivateIp(peer.ip)])
        peers_total = len(site.peers)

        # Add myself
        if site.isServing():
            peers_total += 1
            if any(site.connection_server.port_opened.values()):
                connectable += 1
            if site.connection_server.tor_manager.start_onions:
                onion += 1

        if peers_total:
            percent_connected = float(connected) / peers_total
            percent_connectable = float(connectable) / peers_total
            percent_onion = float(onion) / peers_total
        else:
            percent_connectable = percent_connected = percent_onion = 0

        if local:
            local_html = _("<li class='color-yellow'><span>{_[Local]}:</span><b>{local}</b></li>")
        else:
            local_html = ""

        peer_ips = [peer.key for peer in site.getConnectablePeers(20, allow_private=False)]
        peer_ips.sort(key=lambda peer_ip: ".onion:" in peer_ip)
        copy_link = "http://127.0.0.1:43110/%s/?zeronet_peers=%s" % (
            site.content_manager.contents.get("content.json", {}).get("domain", site.address),
            ",".join(peer_ips)
        )

        body.append(_("""
            <li>
             <label>
              {_[Peers]}
              <small class="label-right"><a href='{copy_link}' id='link-copypeers' class='link-right'>{_[Copy to clipboard]}</a></small>
             </label>
             <ul class='graph'>
              <li style='width: 100%' class='total back-black' title="{_[Total peers]}"></li>
              <li style='width: {percent_connectable:.0%}' class='connectable back-blue' title='{_[Connectable peers]}'></li>
              <li style='width: {percent_onion:.0%}' class='connected back-purple' title='{_[Onion]}'></li>
              <li style='width: {percent_connected:.0%}' class='connected back-green' title='{_[Connected peers]}'></li>
             </ul>
             <ul class='graph-legend'>
              <li class='color-green'><span>{_[Connected]}:</span><b>{connected}</b></li>
              <li class='color-blue'><span>{_[Connectable]}:</span><b>{connectable}</b></li>
              <li class='color-purple'><span>{_[Onion]}:</span><b>{onion}</b></li>
              {local_html}
              <li class='color-black'><span>{_[Total]}:</span><b>{peers_total}</b></li>
             </ul>
            </li>
        """.replace("{local_html}", local_html)))

    def sidebarRenderTransferStats(self, body, site):
        recv = float(site.settings.get("bytes_recv", 0)) / 1024 / 1024
        sent = float(site.settings.get("bytes_sent", 0)) / 1024 / 1024
        transfer_total = recv + sent
        if transfer_total:
            percent_recv = recv / transfer_total
            percent_sent = sent / transfer_total
        else:
            percent_recv = 0.5
            percent_sent = 0.5

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

    def sidebarRenderFileStats(self, body, site):
        body.append(_("""
            <li>
             <label>
              {_[Files]}
              <a href='/list/{site.address}' class='link-right link-outline' id="browse-files">{_[Browse files]}</a>
              <small class="label-right">
               <a href='/ZeroNet-Internal/Zip?address={site.address}' id='link-zip' class='link-right' download='site.zip'>{_[Save as .zip]}</a>
              </small>
             </label>
             <ul class='graph graph-stacked'>
        """))

        extensions = (
            ("html", "yellow"),
            ("css", "orange"),
            ("js", "purple"),
            ("Image", "green"),
            ("json", "darkblue"),
            ("User data", "blue"),
            ("Other", "white"),
            ("Total", "black")
        )
        # Collect stats
        size_filetypes = {}
        size_total = 0
        contents = site.content_manager.listContents()  # Without user files
        for inner_path in contents:
            content = site.content_manager.contents[inner_path]
            if "files" not in content or content["files"] is None:
                continue
            for file_name, file_details in list(content["files"].items()):
                size_total += file_details["size"]
                ext = file_name.split(".")[-1]
                size_filetypes[ext] = size_filetypes.get(ext, 0) + file_details["size"]

        # Get user file sizes
        size_user_content = site.content_manager.contents.execute(
            "SELECT SUM(size) + SUM(size_files) AS size FROM content WHERE ?",
            {"not__inner_path": contents}
        ).fetchone()["size"]
        if not size_user_content:
            size_user_content = 0
        size_filetypes["User data"] = size_user_content
        size_total += size_user_content

        # The missing difference is content.json sizes
        if "json" in size_filetypes:
            size_filetypes["json"] += max(0, site.settings["size"] - size_total)
        size_total = size_other = site.settings["size"]

        # Bar
        for extension, color in extensions:
            if extension == "Total":
                continue
            if extension == "Other":
                size = max(0, size_other)
            elif extension == "Image":
                size = size_filetypes.get("jpg", 0) + size_filetypes.get("png", 0) + size_filetypes.get("gif", 0)
                size_other -= size
            else:
                size = size_filetypes.get(extension, 0)
                size_other -= size
            if size_total == 0:
                percent = 0
            else:
                percent = 100 * (float(size) / size_total)
            percent = math.floor(percent * 100) / 100  # Floor to 2 digits
            body.append(
                """<li style='width: %.2f%%' class='%s back-%s' title="%s"></li>""" %
                (percent, _[extension], color, _[extension])
            )

        # Legend
        body.append("</ul><ul class='graph-legend'>")
        for extension, color in extensions:
            if extension == "Other":
                size = max(0, size_other)
            elif extension == "Image":
                size = size_filetypes.get("jpg", 0) + size_filetypes.get("png", 0) + size_filetypes.get("gif", 0)
            elif extension == "Total":
                size = size_total
            else:
                size = size_filetypes.get(extension, 0)

            if extension == "js":
                title = "javascript"
            else:
                title = extension

            if size > 1024 * 1024 * 10:  # Format as mB is more than 10mB
                size_formatted = "%.0fMB" % (size / 1024 / 1024)
            else:
                size_formatted = "%.0fkB" % (size / 1024)

            body.append("<li class='color-%s'><span>%s:</span><b>%s</b></li>" % (color, _[title], size_formatted))

        body.append("</ul></li>")

    def sidebarRenderSizeLimit(self, body, site):
        free_space = helper.getFreeSpace() / 1024 / 1024
        size = float(site.settings["size"]) / 1024 / 1024
        size_limit = site.getSizeLimit()
        percent_used = size / size_limit

        body.append(_("""
            <li>
             <label>{_[Size limit]} <small>({_[limit used]}: {percent_used:.0%}, {_[free space]}: {free_space:,.0f}MB)</small></label>
             <input type='text' class='text text-num' value="{size_limit}" id='input-sitelimit'/><span class='text-post'>MB</span>
             <a href='#Set' class='button' id='button-sitelimit'>{_[Set]}</a>
            </li>
        """))

    def sidebarRenderOptionalFileStats(self, body, site):
        size_total = float(site.settings["size_optional"])
        size_downloaded = float(site.settings["optional_downloaded"])

        if not size_total:
            return False

        percent_downloaded = size_downloaded / size_total

        size_formatted_total = size_total / 1024 / 1024
        size_formatted_downloaded = size_downloaded / 1024 / 1024

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

        return True

    def sidebarRenderOptionalFileSettings(self, body, site):
        if self.site.settings.get("autodownloadoptional"):
            checked = "checked='checked'"
        else:
            checked = ""

        body.append(_("""
            <li>
             <label>{_[Help distribute added optional files]}</label>
             <input type="checkbox" class="checkbox" id="checkbox-autodownloadoptional" {checked}/><div class="checkbox-skin"></div>
        """))

        if hasattr(config, "autodownload_bigfile_size_limit"):
            autodownload_bigfile_size_limit = int(site.settings.get("autodownload_bigfile_size_limit", config.autodownload_bigfile_size_limit))
            body.append(_("""
                <div class='settings-autodownloadoptional'>
                 <label>{_[Auto download big file size limit]}</label>
                 <input type='text' class='text text-num' value="{autodownload_bigfile_size_limit}" id='input-autodownload_bigfile_size_limit'/><span class='text-post'>MB</span>
                 <a href='#Set' class='button' id='button-autodownload_bigfile_size_limit'>{_[Set]}</a>
                 <a href='#Download+previous' class='button' id='button-autodownload_previous'>{_[Download previous files]}</a>
                </div>
            """))
        body.append("</li>")

    def sidebarRenderBadFiles(self, body, site):
        body.append(_("""
            <li>
             <label>{_[Needs to be updated]}:</label>
             <ul class='filelist'>
        """))

        i = 0
        for bad_file, tries in site.bad_files.items():
            i += 1
            body.append(_("""<li class='color-red' title="{bad_file_path} ({tries})">{bad_filename}</li>""", {
                "bad_file_path": bad_file,
                "bad_filename": helper.getFilename(bad_file),
                "tries": _.pluralize(tries, "{} try", "{} tries")
            }))
            if i > 30:
                break

        if len(site.bad_files) > 30:
            num_bad_files = len(site.bad_files) - 30
            body.append(_("""<li class='color-red'>{_[+ {num_bad_files} more]}</li>""", nested=True))

        body.append("""
             </ul>
            </li>
        """)

    def sidebarRenderDbOptions(self, body, site):
        if site.storage.db:
            inner_path = site.storage.getInnerPath(site.storage.db.db_path)
            size = float(site.storage.getSize(inner_path)) / 1024
            feeds = len(site.storage.db.schema.get("feeds", {}))
        else:
            inner_path = _["No database found"]
            size = 0.0
            feeds = 0

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

    def sidebarRenderIdentity(self, body, site):
        auth_address = self.user.getAuthAddress(self.site.address, create=False)
        rules = self.site.content_manager.getRules("data/users/%s/content.json" % auth_address)
        if rules and rules.get("max_size"):
            quota = rules["max_size"] / 1024
            try:
                content = site.content_manager.contents["data/users/%s/content.json" % auth_address]
                used = len(json.dumps(content)) + sum([file["size"] for file in list(content["files"].values())])
            except:
                used = 0
            used = used / 1024
        else:
            quota = used = 0

        body.append(_("""
            <li>
             <label>{_[Identity address]} <small>({_[limit used]}: {used:.2f}kB / {quota:.2f}kB)</small></label>
             <div class='flex'>
              <span class='input text disabled'>{auth_address}</span>
              <a href='#Change' class='button' id='button-identity'>{_[Change]}</a>
             </div>
            </li>
        """))

    def sidebarRenderControls(self, body, site):
        auth_address = self.user.getAuthAddress(self.site.address, create=False)
        if self.site.settings["serving"]:
            class_pause = ""
            class_resume = "hidden"
        else:
            class_pause = "hidden"
            class_resume = ""

        body.append(_("""
            <li>
             <label>{_[Site control]}</label>
             <a href='#Update' class='button noupdate' id='button-update'>{_[Update]}</a>
             <a href='#Pause' class='button {class_pause}' id='button-pause'>{_[Pause]}</a>
             <a href='#Resume' class='button {class_resume}' id='button-resume'>{_[Resume]}</a>
             <a href='#Delete' class='button noupdate' id='button-delete'>{_[Delete]}</a>
            </li>
        """))

        donate_key = site.content_manager.contents.get("content.json", {}).get("donate", True)
        site_address = self.site.address
        body.append(_("""
            <li>
             <label>{_[Site address]}</label><br>
             <div class='flex'>
              <span class='input text disabled'>{site_address}</span>
        """))
        if donate_key == False or donate_key == "":
            pass
        elif (type(donate_key) == str or type(donate_key) == str) and len(donate_key) > 0:
            body.append(_("""
             </div>
            </li>
            <li>
             <label>{_[Donate]}</label><br>
             <div class='flex'>
             {donate_key}
            """))
        else:
            body.append(_("""
              <a href='bitcoin:{site_address}' class='button' id='button-donate'>{_[Donate]}</a>
            """))
        body.append(_("""
             </div>
            </li>
        """))

    def sidebarRenderOwnedCheckbox(self, body, site):
        if self.site.settings["own"]:
            checked = "checked='checked'"
        else:
            checked = ""

        body.append(_("""
            <h2 class='owned-title'>{_[This is my site]}</h2>
            <input type="checkbox" class="checkbox" id="checkbox-owned" {checked}/><div class="checkbox-skin"></div>
        """))

    def sidebarRenderOwnSettings(self, body, site):
        title = site.content_manager.contents.get("content.json", {}).get("title", "")
        description = site.content_manager.contents.get("content.json", {}).get("description", "")

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

    def sidebarRenderContents(self, body, site):
        has_privatekey = bool(self.user.getSiteData(site.address, create=False).get("privatekey"))
        if has_privatekey:
            tag_privatekey = _("{_[Private key saved.]} <a href='#Forget+private+key' id='privatekey-forget' class='link-right'>{_[Forget]}</a>")
        else:
            tag_privatekey = _("<a href='#Add+private+key' id='privatekey-add' class='link-right'>{_[Add saved private key]}</a>")

        body.append(_("""
            <li>
             <label>{_[Content publishing]} <small class='label-right'>{tag_privatekey}</small></label>
        """.replace("{tag_privatekey}", tag_privatekey)))

        # Choose content you want to sign
        body.append(_("""
             <div class='flex'>
              <input type='text' class='text' value="content.json" id='input-contents'/>
              <a href='#Sign-and-Publish' id='button-sign-publish' class='button'>{_[Sign and publish]}</a>
              <a href='#Sign-or-Publish' id='menu-sign-publish'>\u22EE</a>
             </div>
        """))

        contents = ["content.json"]
        contents += list(site.content_manager.contents.get("content.json", {}).get("includes", {}).keys())
        body.append(_("<div class='contents'>{_[Choose]}: "))
        for content in contents:
            body.append(_("<a href='{content}' class='contents-content'>{content}</a> "))
        body.append("</div>")
        body.append("</li>")

    @flag.admin
    def actionSidebarGetHtmlTag(self, to):
        site = self.site

        body = []

        body.append("<div>")
        body.append("<a href='#Close' class='close'>&times;</a>")
        body.append("<h1>%s</h1>" % html.escape(site.content_manager.contents.get("content.json", {}).get("title", ""), True))

        body.append("<div class='globe loading'></div>")

        body.append("<ul class='fields'>")

        self.sidebarRenderPeerStats(body, site)
        self.sidebarRenderTransferStats(body, site)
        self.sidebarRenderFileStats(body, site)
        self.sidebarRenderSizeLimit(body, site)
        has_optional = self.sidebarRenderOptionalFileStats(body, site)
        if has_optional:
            self.sidebarRenderOptionalFileSettings(body, site)
        self.sidebarRenderDbOptions(body, site)
        self.sidebarRenderIdentity(body, site)
        self.sidebarRenderControls(body, site)
        if site.bad_files:
            self.sidebarRenderBadFiles(body, site)

        self.sidebarRenderOwnedCheckbox(body, site)
        body.append("<div class='settings-owned'>")
        self.sidebarRenderOwnSettings(body, site)
        self.sidebarRenderContents(body, site)
        body.append("</div>")
        body.append("</ul>")
        body.append("</div>")

        body.append("<div class='menu template'>")
        body.append("<a href='#'' class='menu-item template'>Template</a>")
        body.append("</div>")

        self.response(to, "".join(body))

    def downloadGeoLiteDb(self, db_path):
        import gzip
        import shutil
        from util import helper

        if config.offline:
            return False

        self.log.info("Downloading GeoLite2 City database...")
        self.cmd("progress", ["geolite-info", _["Downloading GeoLite2 City database (one time only, ~20MB)..."], 0])
        db_urls = [
            "https://raw.githubusercontent.com/aemr3/GeoLite2-Database/master/GeoLite2-City.mmdb.gz",
            "https://raw.githubusercontent.com/texnikru/GeoLite2-Database/master/GeoLite2-City.mmdb.gz"
        ]
        for db_url in db_urls:
            downloadl_err = None
            try:
                # Download
                response = helper.httpRequest(db_url)
                data_size = response.getheader('content-length')
                data_recv = 0
                data = io.BytesIO()
                while True:
                    buff = response.read(1024 * 512)
                    if not buff:
                        break
                    data.write(buff)
                    data_recv += 1024 * 512
                    if data_size:
                        progress = int(float(data_recv) / int(data_size) * 100)
                        self.cmd("progress", ["geolite-info", _["Downloading GeoLite2 City database (one time only, ~20MB)..."], progress])
                self.log.info("GeoLite2 City database downloaded (%s bytes), unpacking..." % data.tell())
                data.seek(0)

                # Unpack
                with gzip.GzipFile(fileobj=data) as gzip_file:
                    shutil.copyfileobj(gzip_file, open(db_path, "wb"))

                self.cmd("progress", ["geolite-info", _["GeoLite2 City database downloaded!"], 100])
                time.sleep(2)  # Wait for notify animation
                self.log.info("GeoLite2 City database is ready at: %s" % db_path)
                return True
            except Exception as err:
                download_err = err
                self.log.error("Error downloading %s: %s" % (db_url, err))
                pass
        self.cmd("progress", [
            "geolite-info",
            _["GeoLite2 City database download error: {}!<br>Please download manually and unpack to data dir:<br>{}"].format(download_err, db_urls[0]),
            -100
        ])

    def getLoc(self, geodb, ip):
        global loc_cache

        if ip in loc_cache:
            return loc_cache[ip]
        else:
            try:
                loc_data = geodb.get(ip)
            except:
                loc_data = None

            if not loc_data or "location" not in loc_data:
                loc_cache[ip] = None
                return None

            loc = {
                "lat": loc_data["location"]["latitude"],
                "lon": loc_data["location"]["longitude"],
            }
            if "city" in loc_data:
                loc["city"] = loc_data["city"]["names"]["en"]

            if "country" in loc_data:
                loc["country"] = loc_data["country"]["names"]["en"]

            loc_cache[ip] = loc
            return loc

    @util.Noparallel()
    def getGeoipDb(self):
        db_name = 'GeoLite2-City.mmdb'

        sys_db_paths = []
        if sys.platform == "linux":
            sys_db_paths += ['/usr/share/GeoIP/' + db_name]

        data_dir_db_path = os.path.join(config.data_dir, db_name)

        db_paths = sys_db_paths + [data_dir_db_path]

        for path in db_paths:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path

        self.log.info("GeoIP database not found at [%s]. Downloading to: %s",
                " ".join(db_paths), data_dir_db_path)
        if self.downloadGeoLiteDb(data_dir_db_path):
            return data_dir_db_path
        return None

    def getPeerLocations(self, peers):
        import maxminddb

        db_path = self.getGeoipDb()
        if not db_path:
            self.log.debug("Not showing peer locations: no GeoIP database")
            return False

        geodb = maxminddb.open_database(db_path)

        peers = list(peers.values())
        # Place bars
        peer_locations = []
        placed = {}  # Already placed bars here
        for peer in peers:
            # Height of bar
            if peer.connection and peer.connection.last_ping_delay:
                ping = round(peer.connection.last_ping_delay * 1000)
            else:
                ping = None
            loc = self.getLoc(geodb, peer.ip)

            if not loc:
                continue
            # Create position array
            lat, lon = loc["lat"], loc["lon"]
            latlon = "%s,%s" % (lat, lon)
            if latlon in placed and helper.getIpType(peer.ip) == "ipv4":  # Dont place more than 1 bar to same place, fake repos using ip address last two part
                lat += float(128 - int(peer.ip.split(".")[-2])) / 50
                lon += float(128 - int(peer.ip.split(".")[-1])) / 50
                latlon = "%s,%s" % (lat, lon)
            placed[latlon] = True
            peer_location = {}
            peer_location.update(loc)
            peer_location["lat"] = lat
            peer_location["lon"] = lon
            peer_location["ping"] = ping

            peer_locations.append(peer_location)

        # Append myself
        for ip in self.site.connection_server.ip_external_list:
            my_loc = self.getLoc(geodb, ip)
            if my_loc:
                my_loc["ping"] = 0
                peer_locations.append(my_loc)

        return peer_locations

    @flag.admin
    @flag.async_run
    def actionSidebarGetPeers(self, to):
        try:
            peer_locations = self.getPeerLocations(self.site.peers)
            globe_data = []
            ping_times = [
                peer_location["ping"]
                for peer_location in peer_locations
                if peer_location["ping"]
            ]
            if ping_times:
                ping_avg = sum(ping_times) / float(len(ping_times))
            else:
                ping_avg = 0

            for peer_location in peer_locations:
                if peer_location["ping"] == 0:  # Me
                    height = -0.135
                elif peer_location["ping"]:
                    height = min(0.20, math.log(1 + peer_location["ping"] / ping_avg, 300))
                else:
                    height = -0.03

                globe_data += [peer_location["lat"], peer_location["lon"], height]

            self.response(to, globe_data)
        except Exception as err:
            self.log.debug("sidebarGetPeers error: %s" % Debug.formatException(err))
            self.response(to, {"error": str(err)})

    @flag.admin
    @flag.no_multiuser
    def actionSiteSetOwned(self, to, owned):
        if self.site.address == config.updatesite:
            return {"error": "You can't change the ownership of the updater site"}

        self.site.settings["own"] = bool(owned)
        self.site.updateWebsocket(owned=owned)
        return "ok"

    @flag.admin
    @flag.no_multiuser
    def actionSiteRecoverPrivatekey(self, to):
        from Crypt import CryptBitcoin

        site_data = self.user.sites[self.site.address]
        if site_data.get("privatekey"):
            return {"error": "This site already has saved privated key"}

        address_index = self.site.content_manager.contents.get("content.json", {}).get("address_index")
        if not address_index:
            return {"error": "No address_index in content.json"}

        privatekey = CryptBitcoin.hdPrivatekey(self.user.master_seed, address_index)
        privatekey_address = CryptBitcoin.privatekeyToAddress(privatekey)

        if privatekey_address == self.site.address:
            site_data["privatekey"] = privatekey
            self.user.save()
            self.site.updateWebsocket(recover_privatekey=True)
            return "ok"
        else:
            return {"error": "Unable to deliver private key for this site from current user's master_seed"}

    @flag.admin
    @flag.no_multiuser
    def actionUserSetSitePrivatekey(self, to, privatekey):
        site_data = self.user.sites[self.site.address]
        site_data["privatekey"] = privatekey
        self.site.updateWebsocket(set_privatekey=bool(privatekey))
        self.user.save()

        return "ok"

    @flag.admin
    @flag.no_multiuser
    def actionSiteSetAutodownloadoptional(self, to, owned):
        self.site.settings["autodownloadoptional"] = bool(owned)
        self.site.worker_manager.removeSolvedFileTasks()

    @flag.no_multiuser
    @flag.admin
    def actionDbReload(self, to):
        self.site.storage.closeDb()
        self.site.storage.getDb()

        return self.response(to, "ok")

    @flag.no_multiuser
    @flag.admin
    def actionDbRebuild(self, to):
        try:
            self.site.storage.rebuildDb()
        except Exception as err:
            return self.response(to, {"error": str(err)})


        return self.response(to, "ok")

```