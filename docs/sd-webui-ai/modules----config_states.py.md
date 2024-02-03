# `stable-diffusion-webui\modules\config_states.py`

```py
"""
Supports saving and restoring webui and extensions from a known working set of commits
"""

# 导入所需的模块
import os
import json
import tqdm

# 导入 datetime 模块中的 datetime 类和 git 模块
from datetime import datetime
import git

# 导入自定义模块中的 shared, extensions, errors
from modules import shared, extensions, errors
# 导入自定义模块中的 script_path, config_states_dir
from modules.paths_internal import script_path, config_states_dir

# 创建一个空字典用于存储所有配置状态
all_config_states = {}


# 列出所有配置状态
def list_config_states():
    global all_config_states

    # 清空字典
    all_config_states.clear()
    # 确保配置状态目录存在
    os.makedirs(config_states_dir, exist_ok=True)

    # 创建一个空列表用于存储配置状态
    config_states = []
    # 遍历配置状态目录下的所有文件
    for filename in os.listdir(config_states_dir):
        # 如果文件以 .json 结尾
        if filename.endswith(".json"):
            # 获取文件路径
            path = os.path.join(config_states_dir, filename)
            try:
                # 尝试打开文件并加载 JSON 数据
                with open(path, "r", encoding="utf-8") as f:
                    j = json.load(f)
                    # 确保 JSON 数据中包含 "created_at" 键
                    assert "created_at" in j, '"created_at" does not exist'
                    # 将文件路径添加到 JSON 数据中
                    j["filepath"] = path
                    # 将 JSON 数据添加到配置状态列表中
                    config_states.append(j)
            except Exception as e:
                # 打印错误信息
                print(f'[ERROR]: Config states {path}, {e}')

    # 根据 "created_at" 键对配置状态列表进行排序
    config_states = sorted(config_states, key=lambda cs: cs["created_at"], reverse=True)

    # 遍历排序后的配置状态列表
    for cs in config_states:
        # 将时间戳转换为可读的时间格式
        timestamp = datetime.fromtimestamp(cs["created_at"]).strftime('%Y-%m-%d %H:%M:%S')
        # 获取配置状态的名称
        name = cs.get("name", "Config")
        # 构建完整的配置状态名称
        full_name = f"{name}: {timestamp}"
        # 将完整的配置状态名称作为键，配置状态数据作为值存储到字典中
        all_config_states[full_name] = cs

    # 返回所有配置状态字典
    return all_config_states


# 获取 webui 配置信息
def get_webui_config():
    # 初始化 webui_repo 变量为 None
    webui_repo = None

    try:
        # 如果脚本路径下存在 .git 文件夹
        if os.path.exists(os.path.join(script_path, ".git")):
            # 获取 webui 仓库信息
            webui_repo = git.Repo(script_path)
    except Exception:
        # 报告错误信息
        errors.report(f"Error reading webui git info from {script_path}", exc_info=True)

    # 初始化 webui_remote, webui_commit_hash, webui_commit_date, webui_branch 变量为 None
    webui_remote = None
    webui_commit_hash = None
    webui_commit_date = None
    webui_branch = None
    # 检查 webui_repo 是否存在且不是一个空仓库
    if webui_repo and not webui_repo.bare:
        try:
            # 获取 webui_repo 的远程地址
            webui_remote = next(webui_repo.remote().urls, None)
            # 获取 webui_repo 的最新提交
            head = webui_repo.head.commit
            # 获取 webui_repo 最新提交的提交日期
            webui_commit_date = webui_repo.head.commit.committed_date
            # 获取 webui_repo 最新提交的哈希值
            webui_commit_hash = head.hexsha
            # 获取 webui_repo 当前所在分支的名称
            webui_branch = webui_repo.active_branch.name

        except Exception:
            # 如果出现异常，将 webui_remote 设为 None
            webui_remote = None

    # 返回包含远程地址、提交哈希值、提交日期和分支名称的字典
    return {
        "remote": webui_remote,
        "commit_hash": webui_commit_hash,
        "commit_date": webui_commit_date,
        "branch": webui_branch,
    }
# 获取扩展配置信息的函数
def get_extension_config():
    # 创建空字典用于存储扩展配置信息
    ext_config = {}

    # 遍历所有扩展，读取信息并存储到字典中
    for ext in extensions.extensions:
        ext.read_info_from_repo()

        # 创建包含扩展信息的字典条目
        entry = {
            "name": ext.name,
            "path": ext.path,
            "enabled": ext.enabled,
            "is_builtin": ext.is_builtin,
            "remote": ext.remote,
            "commit_hash": ext.commit_hash,
            "commit_date": ext.commit_date,
            "branch": ext.branch,
            "have_info_from_repo": ext.have_info_from_repo
        }

        # 将扩展信息添加到扩展配置字典中
        ext_config[ext.name] = entry

    # 返回扩展配置字典
    return ext_config


# 获取整体配置信息的函数
def get_config():
    # 获取当前时间戳作为创建时间
    creation_time = datetime.now().timestamp()
    # 获取 WebUI 配置信息
    webui_config = get_webui_config()
    # 获取扩展配置信息
    ext_config = get_extension_config()

    # 返回整体配置信息字典
    return {
        "created_at": creation_time,
        "webui": webui_config,
        "extensions": ext_config
    }


# 恢复 WebUI 配置信息的函数
def restore_webui_config(config):
    print("* Restoring webui state...")

    # 检查配置中是否包含 WebUI 配置信息
    if "webui" not in config:
        print("Error: No webui data saved to config")
        return

    # 获取 WebUI 配置信息
    webui_config = config["webui"]

    # 检查是否包含提交哈希信息
    if "commit_hash" not in webui_config:
        print("Error: No commit saved to webui config")
        return

    # 获取 WebUI 提交哈希和仓库信息
    webui_commit_hash = webui_config.get("commit_hash", None)
    webui_repo = None

    try:
        # 尝试读取 WebUI 仓库信息
        if os.path.exists(os.path.join(script_path, ".git")):
            webui_repo = git.Repo(script_path)
    except Exception:
        errors.report(f"Error reading webui git info from {script_path}", exc_info=True)
        return

    try:
        # 尝试从远程仓库拉取最新代码并重置到指定提交
        webui_repo.git.fetch(all=True)
        webui_repo.git.reset(webui_commit_hash, hard=True)
        print(f"* Restored webui to commit {webui_commit_hash}.")
    except Exception:
        errors.report(f"Error restoring webui to commit{webui_commit_hash}")


# 恢复扩展配置信息的函数
def restore_extension_config(config):
    print("* Restoring extension state...")

    # 检查配置中是否包含扩展配置信息
    if "extensions" not in config:
        print("Error: No extension data saved to config")
        return

    # 获取扩展配置信息
    ext_config = config["extensions"]
    # 存储处理结果的列表
    results = []
    # 存储禁用的扩展列表
    disabled = []

    # 遍历所有扩展
    for ext in tqdm.tqdm(extensions.extensions):
        # 如果是内置扩展，则跳过
        if ext.is_builtin:
            continue

        # 从仓库中读取扩展信息
        ext.read_info_from_repo()
        # 获取当前提交哈希值
        current_commit = ext.commit_hash

        # 如果扩展不在配置中
        if ext.name not in ext_config:
            # 标记为禁用，并添加到禁用列表中
            ext.disabled = True
            disabled.append(ext.name)
            # 添加处理结果到结果列表中
            results.append((ext, current_commit[:8], False, "Saved extension state not found in config, marking as disabled"))
            continue

        # 获取扩展在配置中的条目
        entry = ext_config[ext.name]

        # 如果配置中包含提交哈希值
        if "commit_hash" in entry and entry["commit_hash"]:
            try:
                # 拉取并重置到指定提交哈希值
                ext.fetch_and_reset_hard(entry["commit_hash"])
                ext.read_info_from_repo()
                # 如果当前提交哈希值与配置中的提交哈希值不同
                if current_commit != entry["commit_hash"]:
                    results.append((ext, current_commit[:8], True, entry["commit_hash"][:8]))
            except Exception as ex:
                results.append((ext, current_commit[:8], False, ex))
        else:
            results.append((ext, current_commit[:8], False, "No commit hash found in config"))

        # 如果配置中未启用该扩展
        if not entry.get("enabled", False):
            ext.disabled = True
            disabled.append(ext.name)
        else:
            ext.disabled = False

    # 更新全局禁用扩展列表
    shared.opts.disabled_extensions = disabled
    # 保存配置
    shared.opts.save(shared.config_filename)

    # 打印处理结果
    print("* Finished restoring extensions. Results:")
    for ext, prev_commit, success, result in results:
        if success:
            print(f"  + {ext.name}: {prev_commit} -> {result}")
        else:
            print(f"  ! {ext.name}: FAILURE ({result})")
```