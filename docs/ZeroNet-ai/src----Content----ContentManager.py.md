# `ZeroNet\src\Content\ContentManager.py`

```py
# 导入所需的模块
import json
import time
import re
import os
import copy
import base64
import sys

# 导入并使用 gevent 模块
import gevent

# 导入自定义模块
from Debug import Debug
from Crypt import CryptHash
from Config import config
from util import helper
from util import Diff
from util import SafeRe
from Peer import PeerHashfield
from .ContentDbDict import ContentDbDict
from Plugin import PluginManager

# 自定义异常类
class VerifyError(Exception):
    pass

class SignError(Exception):
    pass

# 使用插件管理器装饰器
@PluginManager.acceptPlugins
class ContentManager(object):

    # 初始化方法
    def __init__(self, site):
        # 设置属性
        self.site = site
        self.log = self.site.log
        self.contents = ContentDbDict(site)
        self.hashfield = PeerHashfield()
        self.has_optional_files = False

    # 加载所有 content.json 文件
    def loadContents(self):
        # 如果内容为空，则从文件系统加载文件
        if len(self.contents) == 0:
            self.log.info("ContentDb not initialized, load files from filesystem...")
            self.loadContent(add_bad_files=False, delete_removed_files=False)
        # 获取总大小并设置到站点设置中
        self.site.settings["size"], self.site.settings["size_optional"] = self.getTotalSize()

        # 加载 hashfield 缓存
        if "hashfield" in self.site.settings.get("cache", {}):
            self.hashfield.frombytes(base64.b64decode(self.site.settings["cache"]["hashfield"]))
            del self.site.settings["cache"]["hashfield"]
        # 如果存在 content.json 文件且可选大小大于 0，则更新坏文件
        elif self.contents.get("content.json") and self.site.settings["size_optional"] > 0:
            self.site.storage.updateBadFiles()  # No hashfield cache created yet
        # 设置是否存在可选文件的标志
        self.has_optional_files = bool(self.hashfield)

        # 初始化内容数据库
        self.contents.db.initSite(self.site)
    # 获取文件变化情况，即在旧文件列表中而不在新文件列表中的文件
    def getFileChanges(self, old_files, new_files):
        # 找出在旧文件列表中而不在新文件列表中的文件，组成删除文件的字典
        deleted = {key: val for key, val in old_files.items() if key not in new_files}
        # 找出在旧文件列表中而不在新文件列表中的文件的哈希值，组成删除文件哈希值的字典
        deleted_hashes = {val.get("sha512"): key for key, val in old_files.items() if key not in new_files}
        # 找出在新文件列表中而不在旧文件列表中的文件，组成新增文件的字典
        added = {key: val for key, val in new_files.items() if key not in old_files}
        # 初始化重命名文件的字典
        renamed = {}
        # 遍历新增文件列表，查找是否有相同哈希值的文件，若有则认为是重命名文件
        for relative_path, node in added.items():
            hash = node.get("sha512")
            if hash in deleted_hashes:
                # 获取重命名文件的旧文件名和新文件名，更新重命名文件的字典，并从删除文件的字典中删除对应的旧文件名
                relative_path_old = deleted_hashes[hash]
                renamed[relative_path_old] = relative_path
                del(deleted[relative_path_old])
        # 返回删除文件列表和重命名文件字典
        return list(deleted), renamed

    # 加载 content.json 到 self.content
    # 返回：变化的文件 ["index.html", "data/messages.json"]，删除的文件 ["old.jpg"]
    # 从指定路径中删除内容
    def removeContent(self, inner_path):
        # 获取内部路径的目录名
        inner_dir = helper.getDirname(inner_path)
        try:
            # 获取内容字典中指定路径的内容
            content = self.contents[inner_path]
            # 获取必需文件和可选文件的字典，并合并成一个新的字典
            files = dict(
                content.get("files", {}),
                **content.get("files_optional", {})
            )
        except Exception as err:
            # 如果出现异常，记录错误日志
            self.log.debug("Error loading %s for removeContent: %s" % (inner_path, Debug.formatException(err)))
            files = {}
        # 将content.json文件标记为True
        files["content.json"] = True
        # 删除不在content.json中的文件
        for file_relative_path in files:
            # 获取文件的内部路径
            file_inner_path = inner_dir + file_relative_path
            try:
                # 删除文件
                self.site.storage.delete(file_inner_path)
                self.log.debug("Deleted file: %s" % file_inner_path)
            except Exception as err:
                # 如果出现异常，记录错误日志
                self.log.debug("Error deleting file %s: %s" % (file_inner_path, err))
        try:
            # 删除指定目录
            self.site.storage.deleteDir(inner_dir)
        except Exception as err:
            # 如果出现异常，记录错误日志
            self.log.debug("Error deleting dir %s: %s" % (inner_dir, err))

        try:
            # 从内容字典中删除指定路径的内容
            del self.contents[inner_path]
        except Exception as err:
            # 如果出现异常，记录错误日志
            self.log.debug("Error key from contents: %s" % inner_path)

    # 获取站点的总大小
    # 返回：32819（文件大小，单位为kb）
    def getTotalSize(self, ignore=None):
        return self.contents.db.getTotalSize(self.site, ignore)

    # 列出修改过的内容
    def listModified(self, after=None, before=None):
        return self.contents.db.listModified(self.site, after=after, before=before)
    # 列出指定内部路径的内容，默认为 "content.json"，如果 user_files 为 True，则列出用户文件
    def listContents(self, inner_path="content.json", user_files=False):
        # 如果指定的内部路径不在内容中，则返回空列表
        if inner_path not in self.contents:
            return []
        # 初始化返回结果列表，包含指定内部路径
        back = [inner_path]
        # 获取内容内部目录
        content_inner_dir = helper.getDirname(inner_path)
        # 遍历内容中指定内部路径的包含文件
        for relative_path in list(self.contents[inner_path].get("includes", {}).keys()):
            # 获取包含文件的内部路径
            include_inner_path = content_inner_dir + relative_path
            # 递归调用 listContents 函数，将结果添加到返回结果列表中
            back += self.listContents(include_inner_path)
        # 返回结果列表
        return back

    # 返回给定修改日期的文件是否已归档
    def isArchived(self, inner_path, modified):
        # 使用正则表达式匹配内部路径的目录和文件名
        match = re.match(r"(.*)/(.*?)/", inner_path)
        # 如果匹配失败，则返回 False
        if not match:
            return False
        # 获取用户内容的内部路径
        user_contents_inner_path = match.group(1) + "/content.json"
        # 获取相对目录
        relative_directory = match.group(2)

        # 获取用户内容文件的信息
        file_info = self.getFileInfo(user_contents_inner_path)
        # 如果文件信息存在
        if file_info:
            # 获取归档前的时间
            time_archived_before = file_info.get("archived_before", 0)
            # 获取目录归档的时间
            time_directory_archived = file_info.get("archived", {}).get(relative_directory, 0)
            # 如果修改日期早于等于归档前的时间或者早于等于目录归档的时间，则返回 True，否则返回 False
            if modified <= time_archived_before or modified <= time_directory_archived:
                return True
            else:
                return False
        else:
            return False

    # 返回给定内部路径和哈希值是否已下载
    def isDownloaded(self, inner_path, hash_id=None):
        # 如果没有指定哈希值
        if not hash_id:
            # 获取文件信息
            file_info = self.getFileInfo(inner_path)
            # 如果文件信息不存在或者文件信息中没有 sha512 字段，则返回 False
            if not file_info or "sha512" not in file_info:
                return False
            # 获取文件信息中的 sha512 哈希值对应的哈希 ID
            hash_id = self.hashfield.getHashId(file_info["sha512"])
        # 返回哈希 ID 是否在哈希字段中
        return hash_id in self.hashfield

    # 是否自签名后修改
    # 检查文件是否被修改
    def isModified(self, inner_path):
        # 记录当前时间
        s = time.time()
        # 如果文件路径以"content.json"结尾
        if inner_path.endswith("content.json"):
            # 尝试验证文件是否有效，如果有效则标记为未修改，否则标记为已修改
            try:
                is_valid = self.verifyFile(inner_path, self.site.storage.open(inner_path), ignore_same=False)
                if is_valid:
                    is_modified = False
                else:
                    is_modified = True
            # 如果验证出错，则标记为已修改
            except VerifyError:
                is_modified = True
        # 如果文件路径不以"content.json"结尾
        else:
            # 尝试验证文件是否有效，如果有效则标记为未修改，否则标记为已修改
            try:
                self.verifyFile(inner_path, self.site.storage.open(inner_path), ignore_same=False)
                is_modified = False
            # 如果验证出错，则标记为已修改
            except VerifyError:
                is_modified = True
        # 返回文件是否被修改的标记
        return is_modified

    # 从self.contents中找到文件信息行
    # 返回: { "sha512": "c29d73d...21f518", "size": 41 , "content_inner_path": "content.json"}
    # 获取文件的规则
    # 返回: 文件的规则，如果不允许则返回False
    # 获取指定内部路径的规则，如果没有指定内容，则默认为空
    def getRules(self, inner_path, content=None):
        # 如果内部路径不是以"content.json"结尾，则先查找文件content.json
        if not inner_path.endswith("content.json"):  # Find the files content.json first
            # 获取文件信息
            file_info = self.getFileInfo(inner_path)
            # 如果文件信息不存在，则返回False，表示文件未找到
            if not file_info:
                return False  # File not found
            # 更新内部路径为文件信息中的content_inner_path
            inner_path = file_info["content_inner_path"]

        # 如果内部路径是"content.json"，表示根目录下的content.json
        if inner_path == "content.json":  # Root content.json
            # 创建空的规则字典
            rules = {}
            # 获取有效签名者列表
            rules["signers"] = self.getValidSigners(inner_path, content)
            # 返回规则字典
            return rules

        # 将内部路径按"/"分割，获取父目录和相对于content.json的文件名
        dirs = inner_path.split("/")  # Parent dirs of content.json
        inner_path_parts = [dirs.pop()]  # Filename relative to content.json
        inner_path_parts.insert(0, dirs.pop())  # Dont check in self dir
        # 循环查找父目录中的content.json文件
        while True:
            # 构建父目录中的content.json文件路径
            content_inner_path = "%s/content.json" % "/".join(dirs)
            # 获取父目录中的content.json内容
            parent_content = self.contents.get(content_inner_path.strip("/"))
            # 如果父目录内容存在并且包含"includes"字段，则返回对应的规则
            if parent_content and "includes" in parent_content:
                return parent_content["includes"].get("/".join(inner_path_parts))
            # 如果父目录内容存在并且包含"user_contents"字段，则调用getUserContentRules方法获取规则
            elif parent_content and "user_contents" in parent_content:
                return self.getUserContentRules(parent_content, inner_path, content)
            else:  # No inner path in this dir, lets try the parent dir
                # 如果当前目录中没有内部路径，则尝试父目录
                if dirs:
                    inner_path_parts.insert(0, dirs.pop())
                else:  # No more parent dirs
                    break

        # 如果未找到规则，则返回False
        return False

    # 获取用户文件的规则
    # 返回：文件的规则，如果不允许则返回False
    # 获取更改文件的差异
    # 获取指定内部路径的差异内容
    def getDiffs(self, inner_path, limit=30 * 1024, update_files=True):
        # 如果指定的内部路径不在内容中，则返回空字典
        if inner_path not in self.contents:
            return {}
        # 初始化差异字典
        diffs = {}
        # 获取内容内部路径的目录名
        content_inner_path_dir = helper.getDirname(inner_path)
        # 遍历内容内部路径下的文件
        for file_relative_path in self.contents[inner_path].get("files", {}):
            # 获取文件的内部路径
            file_inner_path = content_inner_path_dir + file_relative_path
            # 如果存在新版本文件
            if self.site.storage.isFile(file_inner_path + "-new"):
                # 计算新旧版本文件的差异
                diffs[file_relative_path] = Diff.diff(
                    list(self.site.storage.open(file_inner_path)),
                    list(self.site.storage.open(file_inner_path + "-new")),
                    limit=limit
                )
                # 如果需要更新文件，则删除旧文件，重命名新文件
                if update_files:
                    self.site.storage.delete(file_inner_path)
                    self.site.storage.rename(file_inner_path + "-new", file_inner_path)
            # 如果存在旧版本文件
            if self.site.storage.isFile(file_inner_path + "-old"):
                # 计算新旧版本文件的差异
                diffs[file_relative_path] = Diff.diff(
                    list(self.site.storage.open(file_inner_path + "-old")),
                    list(self.site.storage.open(file_inner_path)),
                    limit=limit
                )
                # 如果需要更新文件，则删除旧版本文件
                if update_files:
                    self.site.storage.delete(file_inner_path + "-old")
        # 返回差异字典
        return diffs
    # 计算文件的哈希值，并返回包含文件相对路径和哈希值的字典
    def hashFile(self, dir_inner_path, file_relative_path, optional=False):
        back = {}  # 创建一个空字典
        file_inner_path = dir_inner_path + "/" + file_relative_path  # 拼接文件的内部路径

        file_path = self.site.storage.getPath(file_inner_path)  # 获取文件的完整路径
        file_size = os.path.getsize(file_path)  # 获取文件大小
        sha512sum = CryptHash.sha512sum(file_path)  # 计算文件的 sha512 哈希值
        if optional and not self.hashfield.hasHash(sha512sum):  # 如果可选并且哈希字段中没有该哈希值
            self.optionalDownloaded(file_inner_path, self.hashfield.getHashId(sha512sum), file_size, own=True)  # 调用 optionalDownloaded 方法

        back[file_relative_path] = {"sha512": sha512sum, "size": os.path.getsize(file_path)}  # 将文件相对路径、sha512 哈希值和文件大小添加到字典中
        return back  # 返回包含文件相对路径和哈希值的字典

    # 检查相对路径是否有效
    def isValidRelativePath(self, relative_path):
        if ".." in relative_path.replace("\\", "/").split("/"):  # 如果相对路径中包含 ".."
            return False
        elif len(relative_path) > 255:  # 如果相对路径长度大于 255
            return False
        elif relative_path[0] in ("/", "\\"):  # 如果相对路径以 "/" 或 "\" 开头
            return False
        elif relative_path[-1] in (".", " "):  # 如果相对路径以 "." 或 " " 结尾
            return False
        elif re.match(r".*(^|/)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]|CONOUT\$|CONIN\$)(\.|/|$)", relative_path, re.IGNORECASE):  # 如果相对路径匹配 Windows 下的受保护文件名
            return False
        else:
            return re.match(r"^[^\x00-\x1F\"*:<>?\\|]+$", relative_path)  # 如果相对路径符合正则表达式要求

    # 清理路径中的非法字符
    def sanitizePath(self, inner_path):
        return re.sub("[\x00-\x1F\"*:<>?\\|]", "", inner_path)  # 使用正则表达式替换路径中的非法字符
    # 根据指定目录路径和忽略模式，生成文件哈希
    def hashFiles(self, dir_inner_path, ignore_pattern=None, optional_pattern=None):
        # 存储文件节点的字典
        files_node = {}
        # 存储可选文件节点的字典
        files_optional_node = {}
        # 获取数据库文件的内部路径
        db_inner_path = self.site.storage.getDbFile()
        # 如果目录内部路径不为空且不是有效的相对路径，则忽略并记录错误信息
        if dir_inner_path and not self.isValidRelativePath(dir_inner_path):
            ignored = True
            self.log.error("- [ERROR] Only ascii encoded directories allowed: %s" % dir_inner_path)

        # 遍历指定目录下的文件路径，根据忽略模式进行过滤
        for file_relative_path in self.site.storage.walk(dir_inner_path, ignore_pattern):
            # 获取文件名
            file_name = helper.getFilename(file_relative_path)

            ignored = optional = False
            # 如果文件名为 "content.json"，则忽略
            if file_name == "content.json":
                ignored = True
            # 如果文件名以 "." 开头，或以 "-old" 或 "-new" 结尾，则忽略
            elif file_name.startswith(".") or file_name.endswith("-old") or file_name.endswith("-new"):
                ignored = True
            # 如果文件路径不是有效的相对路径，则忽略并记录错误信息
            elif not self.isValidRelativePath(file_relative_path):
                ignored = True
                self.log.error("- [ERROR] Invalid filename: %s" % file_relative_path)
            # 如果目录内部路径为空且数据库内部路径存在且文件路径以数据库内部路径开头，则忽略
            elif dir_inner_path == "" and db_inner_path and file_relative_path.startswith(db_inner_path):
                ignored = True
            # 如果可选模式存在且文件路径与可选模式匹配，则标记为可选文件
            elif optional_pattern and SafeRe.match(optional_pattern, file_relative_path):
                optional = True

            # 如果被忽略，则记录跳过信息
            if ignored:  # Ignore content.json, defined regexp and files starting with .
                self.log.info("- [SKIPPED] %s" % file_relative_path)
            else:
                # 如果是可选文件，则记录可选信息并更新可选文件节点字典
                if optional:
                    self.log.info("- [OPTIONAL] %s" % file_relative_path)
                    files_optional_node.update(
                        self.hashFile(dir_inner_path, file_relative_path, optional=True)
                    )
                # 否则记录文件信息并更新文件节点字典
                else:
                    self.log.info("- %s" % file_relative_path)
                    files_node.update(
                        self.hashFile(dir_inner_path, file_relative_path)
                    )
        # 返回文件节点字典和可选文件节点字典
        return files_node, files_optional_node

    # 创建并签名一个 content.json
    # 返回：如果 filewrite = False，则返回新内容
    # content.json 文件的有效签名者
    # 返回：["1KRxE1s3oDyNDawuYWpzbLUwNm8oDbeEp6", "13ReyhCsjhpuCVahn1DHdf6eMqqEVev162"]
    def getValidSigners(self, inner_path, content=None):
        valid_signers = []
        if inner_path == "content.json":  # 根目录下的 content.json
            if "content.json" in self.contents and "signers" in self.contents["content.json"]:
                valid_signers += self.contents["content.json"]["signers"][:]
        else:
            rules = self.getRules(inner_path, content)
            if rules and "signers" in rules:
                valid_signers += rules["signers"]

        if self.site.address not in valid_signers:
            valid_signers.append(self.site.address)  # 站点地址始终有效
        return valid_signers

    # 返回：content.json 所需的有效签名数量
    def getSignsRequired(self, inner_path, content=None):
        return 1  # 待办事项：多重签名

    def verifyCertSign(self, user_address, user_auth_type, user_name, issuer_address, sign):
        from Crypt import CryptBitcoin
        cert_subject = "%s#%s/%s" % (user_address, user_auth_type, user_name)
        return CryptBitcoin.verify(cert_subject, issuer_address, sign)
    # 验证证书的有效性
    def verifyCert(self, inner_path, content):
        # 获取规则
        rules = self.getRules(inner_path, content)

        # 如果没有规则，则抛出验证错误
        if not rules:
            raise VerifyError("No rules for this file")

        # 如果没有证书签发者和证书签发者模式，则不需要证书
        if not rules.get("cert_signers") and not rules.get("cert_signers_pattern"):
            return True  # Does not need cert

        # 如果 content 中缺少 cert_user_id，则抛出验证错误
        if "cert_user_id" not in content:
            raise VerifyError("Missing cert_user_id")

        # 如果 cert_user_id 中 @ 的数量不等于 1，则抛出验证错误
        if content["cert_user_id"].count("@") != 1:
            raise VerifyError("Invalid domain in cert_user_id")

        # 将 cert_user_id 按 @ 分割为 name 和 domain
        name, domain = content["cert_user_id"].rsplit("@", 1)
        # 获取证书地址
        cert_address = rules["cert_signers"].get(domain)
        # 如果没有证书地址，则判断是否符合证书签发者模式，如果符合则使用 domain 作为证书地址，否则抛出验证错误
        if not cert_address:  # Unknown Cert signer
            if rules.get("cert_signers_pattern") and SafeRe.match(rules["cert_signers_pattern"], domain):
                cert_address = domain
            else:
                raise VerifyError("Invalid cert signer: %s" % domain)

        # 验证证书签名
        return self.verifyCertSign(rules["user_address"], content["cert_auth_type"], name, cert_address, content["cert_sign"])

    # 检查 content.json 内容是否有效
    # 返回：True 或 False
    # 验证内容是否包含指定规则的方法
    def verifyContentInclude(self, inner_path, content, content_size, content_size_optional):
        # 获取指定路径和内容的规则
        rules = self.getRules(inner_path, content)
        # 如果没有规则，则抛出验证错误
        if not rules:
            raise VerifyError("No rules")

        # 检查内容是否超过最大大小限制
        if rules.get("max_size") is not None:  # 包含大小限制
            if content_size > rules["max_size"]:
                raise VerifyError("Include too large %sB > %sB" % (content_size, rules["max_size"))

        # 检查可选内容是否超过最大大小限制
        if rules.get("max_size_optional") is not None:  # 包含可选文件大小限制
            if content_size_optional > rules["max_size_optional"]:
                raise VerifyError("Include optional files too large %sB > %sB" % (
                    content_size_optional, rules["max_size_optional"])
                )

        # 检查文件名是否符合规则
        if rules.get("files_allowed"):
            for file_inner_path in list(content["files"].keys()):
                if not SafeRe.match(r"^%s$" % rules["files_allowed"], file_inner_path):
                    raise VerifyError("File not allowed: %s" % file_inner_path)

        # 检查可选文件名是否符合规则
        if rules.get("files_allowed_optional"):
            for file_inner_path in list(content.get("files_optional", {}).keys()):
                if not SafeRe.match(r"^%s$" % rules["files_allowed_optional"], file_inner_path):
                    raise VerifyError("Optional file not allowed: %s" % file_inner_path)

        # 检查内容是否包含允许的内容
        if rules.get("includes_allowed") is False and content.get("includes"):
            raise VerifyError("Includes not allowed")

        return True  # 所有验证通过

    # 验证文件的有效性
    # 返回: None = 与之前相同, False = 无效, True = 有效
    def optionalDelete(self, inner_path):
        self.site.storage.delete(inner_path)
    # 检查是否已下载了可选文件，如果未指定文件大小，则获取文件大小
    def optionalDownloaded(self, inner_path, hash_id, size=None, own=False):
        if size is None:
            size = self.site.storage.getSize(inner_path)
    
        # 将哈希 ID 添加到哈希字段中
        done = self.hashfield.appendHashId(hash_id)
        # 更新已下载的可选文件大小
        self.site.settings["optional_downloaded"] += size
        # 返回操作是否完成
        return done
    
    # 检查是否已移除了可选文件，如果未指定文件大小，则获取文件大小
    def optionalRemoved(self, inner_path, hash_id, size=None):
        if size is None:
            size = self.site.storage.getSize(inner_path)
        # 将哈希 ID 从哈希字段中移除
        done = self.hashfield.removeHashId(hash_id)
    
        # 更新已下载的可选文件大小
        self.site.settings["optional_downloaded"] -= size
        # 返回操作是否完成
        return done
    
    # 对可选文件进行重命名操作
    def optionalRenamed(self, inner_path_old, inner_path_new):
        # 返回操作是否完成
        return True
```