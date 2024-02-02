# `ZeroNet\update.py`

```py
# 导入必要的模块
import os
import sys
import json
import re
import shutil

# 定义更新函数
def update():
    # 从配置文件中导入配置信息
    from Config import config
    # 解析配置信息，静默模式
    config.parse(silent=True)

    # 如果存在源更新目录
    if getattr(sys, 'source_update_dir', False):
        # 如果源更新目录不存在，则创建
        if not os.path.isdir(sys.source_update_dir):
            os.makedirs(sys.source_update_dir)
        # 去除源更新目录末尾的斜杠
        source_path = sys.source_update_dir.rstrip("/")
    else:
        # 获取当前工作目录，并去除末尾的斜杠
        source_path = os.getcwd().rstrip("/")

    # 如果配置的发布类型以"bundle_linux"开头
    if config.dist_type.startswith("bundle_linux"):
        # 获取运行时路径
        runtime_path = os.path.normpath(os.path.dirname(sys.executable) + "/../..")
    else:
        # 获取运行时路径
        runtime_path = os.path.dirname(sys.executable)

    # 更新站点路径为数据目录加上更新站点名称
    updatesite_path = config.data_dir + "/" + config.updatesite

    # 读取sites.json文件，获取更新站点的坏文件信息
    sites_json = json.load(open(config.data_dir + "/sites.json"))
    updatesite_bad_files = sites_json.get(config.updatesite, {}).get("cache", {}).get("bad_files", {})
    # 打印更新站点路径、坏文件数量、源路径、运行时路径、发布类型
    print(
        "Update site path: %s, bad_files: %s, source path: %s, runtime path: %s, dist type: %s" %
        (updatesite_path, len(updatesite_bad_files), source_path, runtime_path, config.dist_type)
    )

    # 读取更新站点的content.json文件，获取文件列表
    updatesite_content_json = json.load(open(updatesite_path + "/content.json"))
    inner_paths = list(updatesite_content_json.get("files", {}).keys())
    inner_paths += list(updatesite_content_json.get("files_optional", {}).keys())

    # 只保留在ZeroNet目录中的文件
    inner_paths = [inner_path for inner_path in inner_paths if re.match("^(core|bundle)", inner_path)]

    # 检查插件
    plugins_enabled = []
    plugins_disabled = []
    if os.path.isdir("%s/plugins" % source_path):
        for dir in os.listdir("%s/plugins" % source_path):
            if dir.startswith("disabled-"):
                plugins_disabled.append(dir.replace("disabled-", ""))
            else:
                plugins_enabled.append(dir)
        # 打印已启用的插件和已禁用的插件
        print("Plugins enabled:", plugins_enabled, "disabled:", plugins_disabled)

    # 初始化更新路径字典
    update_paths = {}
    # 遍历内部路径列表
    for inner_path in inner_paths:
        # 如果路径中包含 ".."，则跳过
        if ".." in inner_path:
            continue
        # 将路径中的反斜杠替换为斜杠，并去除两侧的斜杠，确保路径为 Unix 风格
        inner_path = inner_path.replace("\\", "/").strip("/")
        # 打印"."，不换行
        print(".", end=" ")
        # 如果路径以 "core" 开头
        if inner_path.startswith("core"):
            # 目标路径为源路径加上去除 "core/" 后的内部路径
            dest_path = source_path + "/" + re.sub("^core/", "", inner_path)
        # 如果路径以配置的 dist_type 开头
        elif inner_path.startswith(config.dist_type):
            # 目标路径为运行时路径加上去除 "bundle[^/]+/" 后的内部路径
            dest_path = runtime_path + "/" + re.sub("^bundle[^/]+/", "", inner_path)
        else:
            # 否则跳过当前路径
            continue

        # 如果目标路径为空，则跳过当前路径
        if not dest_path:
            continue

        # 保持插件的禁用/启用状态
        match = re.match(re.escape(source_path) + "/plugins/([^/]+)", dest_path)
        if match:
            # 获取插件名称
            plugin_name = match.group(1).replace("disabled-", "")
            # 如果插件名称在启用插件列表中
            if plugin_name in plugins_enabled:
                # 将目标路径中的 "plugins/disabled-" 替换为 "plugins/"
                dest_path = dest_path.replace("plugins/disabled-" + plugin_name, "plugins/" + plugin_name)
            # 如果插件名称在禁用插件列表中
            elif plugin_name in plugins_disabled:
                # 将目标路径中的 "plugins/" 替换为 "plugins/disabled-"
                dest_path = dest_path.replace("plugins/" + plugin_name, "plugins/disabled-" + plugin_name)
            # 打印 "P"，不换行
            print("P", end=" ")

        # 获取目标路径的目录
        dest_dir = os.path.dirname(dest_path)
        # 如果目标路径的目录存在且不是一个目录，则创建目录
        if dest_dir and not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)

        # 如果目标路径的目录不等于去除两侧斜杠后的目标路径
        if dest_dir != dest_path.strip("/"):
            # 更新路径字典，将更新站点路径加上内部路径作为键，目标路径作为值
            update_paths[updatesite_path + "/" + inner_path] = dest_path

    # 初始化计数器
    num_ok = 0
    num_rename = 0
    num_error = 0
    # 遍历更新路径字典，获取源路径和目标路径
    for path_from, path_to in update_paths.items():
        # 打印源路径和目标路径
        print("-", path_from, "->", path_to)
        # 如果源路径不是文件，打印"Missing file"并继续下一次循环
        if not os.path.isfile(path_from):
            print("Missing file")
            continue

        # 读取源路径文件的二进制数据
        data = open(path_from, "rb").read()

        # 尝试将数据写入目标路径文件，如果成功则增加成功计数
        try:
            open(path_to, 'wb').write(data)
            num_ok += 1
        # 如果写入出现异常，尝试进行文件重命名并再次写入
        except Exception as err:
            try:
                # 打印错误信息并尝试进行文件重命名
                print("Error writing: %s. Renaming old file as workaround..." % err)
                path_to_tmp = path_to + "-old"
                # 如果已存在重命名后的文件，删除它
                if os.path.isfile(path_to_tmp):
                    os.unlink(path_to_tmp)
                # 将目标路径文件重命名为临时文件
                os.rename(path_to, path_to_tmp)
                # 增加重命名计数
                num_rename += 1
                # 再次将数据写入目标路径文件
                open(path_to, 'wb').write(data)
                # 复制临时文件的权限到目标文件
                shutil.copymode(path_to_tmp, path_to)
                # 打印写入成功信息
                print("Write done after rename!")
                # 增加成功计数
                num_ok += 1
            # 如果重命名后仍然出现异常，打印错误信息并增加错误计数
            except Exception as err:
                print("Write error after rename: %s" % err)
                num_error += 1
    # 打印更新文件数、重命名文件数和错误文件数
    print("* Updated files: %s, renamed: %s, error: %s" % (num_ok, num_rename, num_error))
# 如果当前模块是主程序，则执行以下代码
if __name__ == "__main__":
    # 将当前文件所在目录的 src 目录添加到系统路径中，以便相对于 src 目录进行导入
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))  # Imports relative to src

    # 调用 update 函数进行更新操作
    update()
```