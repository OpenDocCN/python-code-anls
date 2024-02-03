# `stable-diffusion-webui\modules\ui_extensions.py`

```py
# 导入所需的模块
import json
import os
import threading
import time
from datetime import datetime, timezone

import git

# 导入 gradio 模块并重命名为 gr
import gradio as gr
import html
import shutil
import errno

# 导入自定义模块
from modules import extensions, shared, paths, config_states, errors, restart
from modules.paths_internal import config_states_dir
from modules.call_queue import wrap_gradio_gpu_call

# 初始化可用扩展列表
available_extensions = {"extensions": []}
# 定义主题样式
STYLE_PRIMARY = ' style="color: var(--primary-400)"'

# 检查是否禁用扩展访问
def check_access():
    assert not shared.cmd_opts.disable_extension_access, "extension access disabled because of command line flags"

# 应用更改并重新启动程序
def apply_and_restart(disable_list, update_list, disable_all):
    check_access()

    # 解析禁用列表
    disabled = json.loads(disable_list)
    assert type(disabled) == list, f"wrong disable_list data for apply_and_restart: {disable_list}"

    # 解析更新列表
    update = json.loads(update_list)
    assert type(update) == list, f"wrong update_list data for apply_and_restart: {update_list}"

    if update:
        # 保存更新前的配置状态
        save_config_state("Backup (pre-update)")

    update = set(update)

    # 遍历扩展列表，获取更新并重置
    for ext in extensions.extensions:
        if ext.name not in update:
            continue

        try:
            ext.fetch_and_reset_hard()
        except Exception:
            errors.report(f"Error getting updates for {ext.name}", exc_info=True)

    # 更新禁用扩展和禁用所有扩展的状态
    shared.opts.disabled_extensions = disabled
    shared.opts.disable_all_extensions = disable_all
    shared.opts.save(shared.config_filename)

    # 如果可以重新启动程序，则重新启动
    if restart.is_restartable():
        restart.restart_program()
    else:
        restart.stop_program()

# 保存配置状态
def save_config_state(name):
    current_config_state = config_states.get_config()
    if not name:
        name = "Config"
    current_config_state["name"] = name
    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    filename = os.path.join(config_states_dir, f"{timestamp}_{name}.json")
    print(f"Saving backup of webui/extension state to {filename}.")
    # 使用 UTF-8 编码打开文件，准备写入 JSON 格式的当前配置状态
    with open(filename, "w", encoding="utf-8") as f:
        # 将当前配置状态以缩进格式写入文件
        json.dump(current_config_state, f, indent=4, ensure_ascii=False)
    # 列出所有配置状态
    config_states.list_config_states()
    # 获取下一个配置状态的键，如果没有则使用 "Current"
    new_value = next(iter(config_states.all_config_states.keys()), "Current")
    # 创建新的选择列表，包括 "Current" 和所有配置状态的键
    new_choices = ["Current"] + list(config_states.all_config_states.keys())
    # 返回下拉框更新后的值和选项，以及保存成功的提示信息
    return gr.Dropdown.update(value=new_value, choices=new_choices), f"<span>Saved current webui/extension state to \"{filename}\"</span>"
# 恢复配置状态的函数，根据确认状态、配置状态名称和恢复类型进行操作
def restore_config_state(confirmed, config_state_name, restore_type):
    # 如果配置状态名称为"Current"，返回选择一个配置进行恢复
    if config_state_name == "Current":
        return "<span>Select a config to restore from.</span>"
    # 如果未确认，返回取消
    if not confirmed:
        return "<span>Cancelled.</span>"

    # 检查访问权限
    check_access()

    # 获取指定配置状态
    config_state = config_states.all_config_states[config_state_name]

    # 打印恢复 webui 状态的备份信息
    print(f"*** Restoring webui state from backup: {restore_type} ***")

    # 如果恢复类型为"extensions"或"both"，设置恢复配置状态文件路径并保存
    if restore_type == "extensions" or restore_type == "both":
        shared.opts.restore_config_state_file = config_state["filepath"]
        shared.opts.save(shared.config_filename)

    # 如果恢复类型为"webui"或"both"，恢复 webui 配置状态
    if restore_type == "webui" or restore_type == "both":
        config_states.restore_webui_config(config_state)

    # 请求重启应用
    shared.state.request_restart()

    return ""


# 检查更新的函数，根据任务ID和禁用列表进行操作
def check_updates(id_task, disable_list):
    # 检查访问权限
    check_access()

    # 解析禁用列表为列表类型，如果不是列表类型则报错
    disabled = json.loads(disable_list)
    assert type(disabled) == list, f"wrong disable_list data for apply_and_restart: {disable_list}"

    # 获取需要检查更新的扩展列表
    exts = [ext for ext in extensions.extensions if ext.remote is not None and ext.name not in disabled]
    shared.state.job_count = len(exts)

    # 遍历扩展列表，设置状态文本信息并检查更新
    for ext in exts:
        shared.state.textinfo = ext.name

        try:
            ext.check_updates()
        except FileNotFoundError as e:
            if 'FETCH_HEAD' not in str(e):
                raise
        except Exception:
            errors.report(f"Error checking updates for {ext.name}", exc_info=True)

        shared.state.nextjob()

    return extension_table(), ""


# 生成提交链接的函数，根据提交哈希、远程仓库和文本生成链接
def make_commit_link(commit_hash, remote, text=None):
    # 如果文本为空，默认为提交哈希的前8位
    if text is None:
        text = commit_hash[:8]
    # 如果远程仓库以"https://github.com/"开头
    if remote.startswith("https://github.com/"):
        # 如果远程仓库以".git"结尾，去除".git"
        if remote.endswith(".git"):
            remote = remote[:-4]
        # 生成提交链接
        href = remote + "/commit/" + commit_hash
        return f'<a href="{href}" target="_blank">{text}</a>'
    else:
        return text


# 扩展表格的函数
def extension_table():
    # 生成时间戳注释
    code = f"""<!-- {time.time()} -->
    # 创建一个表格，用于展示扩展信息
    <table id="extensions">
        <thead>
            <tr>
                <th>
                    # 创建一个复选框，根据所有扩展是否启用来确定是否选中
                    <input class="gr-check-radio gr-checkbox all_extensions_toggle" type="checkbox" {'checked="checked"' if all(ext.enabled for ext in extensions.extensions) else ''} onchange="toggle_all_extensions(event)" />
                    # 缩写标题，提示用户使用复选框来启用或禁用扩展
                    <abbr title="Use checkbox to enable the extension; it will be enabled or disabled when you click apply button">Extension</abbr>
                </th>
                <th>URL</th>
                <th>Branch</th>
                <th>Version</th>
                <th>Date</th>
                <th>
                    # 缩写标题，提示用户使用复选框来标记扩展以进行更新
                    <abbr title="Use checkbox to mark the extension for update; it will be updated when you click apply button">Update</abbr>
                </th>
            </tr>
        </thead>
        <tbody>
    """
    # 遍历扩展列表中的每一个扩展对象
    for ext in extensions.extensions:
        # 将当前扩展对象的信息从仓库中读取
        ext.read_info_from_repo()

        # 生成远程链接的 HTML 标签
        remote = f"""<a href="{html.escape(ext.remote or '')}" target="_blank">{html.escape("built-in" if ext.is_builtin else ext.remote or '')}</a>"""

        # 根据扩展是否可更新，生成不同的状态显示
        if ext.can_update:
            ext_status = f"""<label><input class="gr-check-radio gr-checkbox" name="update_{html.escape(ext.name)}" checked="checked" type="checkbox">{html.escape(ext.status)}</label>"""
        else:
            ext_status = ext.status

        # 根据条件设置样式
        style = ""
        if shared.cmd_opts.disable_extra_extensions and not ext.is_builtin or shared.opts.disable_all_extensions == "extra" and not ext.is_builtin or shared.cmd_opts.disable_all_extensions or shared.opts.disable_all_extensions == "all":
            style = STYLE_PRIMARY

        # 如果扩展有提交哈希和远程链接，则生成提交链接
        version_link = ext.version
        if ext.commit_hash and ext.remote:
            version_link = make_commit_link(ext.commit_hash, ext.remote, ext.version)

        # 将扩展信息添加到代码块中
        code += f"""
            <tr>
                <td><label{style}><input class="gr-check-radio gr-checkbox extension_toggle" name="enable_{html.escape(ext.name)}" type="checkbox" {'checked="checked"' if ext.enabled else ''} onchange="toggle_extension(event)" />{html.escape(ext.name)}</label></td>
                <td>{remote}</td>
                <td>{ext.branch}</td>
                <td>{version_link}</td>
                <td>{datetime.fromtimestamp(ext.commit_date) if ext.commit_date else ""}</td>
                <td{' class="extension_status"' if ext.remote is not None else ''}>{ext_status}</td>
            </tr>
    """

    # 添加表格结束标签到代码块中
    code += """
        </tbody>
    </table>
    """

    # 返回生成的代码块
    return code
# 更新配置状态表格，根据状态名称获取相应配置信息
def update_config_states_table(state_name):
    # 如果状态名称为"Current"，则获取当前配置信息
    if state_name == "Current":
        config_state = config_states.get_config()
    else:
        # 否则获取指定状态名称的配置信息
        config_state = config_states.all_config_states[state_name]

    # 获取配置名称、创建日期和文件路径
    config_name = config_state.get("name", "Config")
    created_date = datetime.fromtimestamp(config_state["created_at"]).strftime('%Y-%m-%d %H:%M:%S')
    filepath = config_state.get("filepath", "<unknown>")

    try:
        # 获取 WebUI 配置信息
        webui_remote = config_state["webui"]["remote"] or ""
        webui_branch = config_state["webui"]["branch"]
        webui_commit_hash = config_state["webui"]["commit_hash"] or "<unknown>"
        webui_commit_date = config_state["webui"]["commit_date"]
        # 格式化提交日期
        if webui_commit_date:
            webui_commit_date = time.asctime(time.gmtime(webui_commit_date))
        else:
            webui_commit_date = "<unknown>"

        # 创建远程链接、提交链接和日期链接
        remote = f"""<a href="{html.escape(webui_remote)}" target="_blank">{html.escape(webui_remote or '')}</a>"""
        commit_link = make_commit_link(webui_commit_hash, webui_remote)
        date_link = make_commit_link(webui_commit_hash, webui_remote, webui_commit_date)

        # 获取当前 WebUI 配置信息
        current_webui = config_states.get_webui_config()

        # 设置样式
        style_remote = ""
        style_branch = ""
        style_commit = ""
        if current_webui["remote"] != webui_remote:
            style_remote = STYLE_PRIMARY
        if current_webui["branch"] != webui_branch:
            style_branch = STYLE_PRIMARY
        if current_webui["commit_hash"] != webui_commit_hash:
            style_commit = STYLE_PRIMARY

        # 创建配置信息的 HTML 代码块
        code = f"""<!-- {time.time()} -->
<h2>Config Backup: {config_name}</h2>
<div><b>Filepath:</b> {filepath}</div>
<div><b>Created at:</b> {created_date}</div>
<h2>WebUI State</h2>
<table id="config_state_webui">
    <thead>
        <tr>
            <th>URL</th>
            <th>Branch</th>
            <th>Commit</th>
            <th>Date</th>
        </tr>
    </thead>
    <tbody>
        <!-- 表格主体部分 -->
        <tr>
            <!-- 表格行 -->
            <td>
                <!-- 第一列，显示远程信息 -->
                <label{style_remote}>{remote}</label>
            </td>
            <td>
                <!-- 第二列，显示 webui 分支信息 -->
                <label{style_branch}>{webui_branch}</label>
            </td>
            <td>
                <!-- 第三列，显示提交链接信息 -->
                <label{style_commit}>{commit_link}</label>
            </td>
            <td>
                <!-- 第四列，显示日期链接信息 -->
                <label{style_commit}>{date_link}</label>
            </td>
        </tr>
    </tbody>
# 结束表格标签
</table>
# 添加 Extension State 标题
<h2>Extension State</h2>
# 创建配置状态扩展表格
<table id="config_state_extensions">
    <thead>
        <tr>
            <th>Extension</th>
            <th>URL</th>
            <th>Branch</th>
            <th>Commit</th>
            <th>Date</th>
        </tr>
    </thead>
    <tbody>
"""

        # 添加表格结束标签
        code += """    </tbody>
</table>"""

    # 捕获异常并处理
    except Exception as e:
        # 打印错误信息
        print(f"[ERROR]: Config states {filepath}, {e}")
        # 生成包含错误信息的 HTML 代码
        code = f"""<!-- {time.time()} -->
<h2>Config Backup: {config_name}</h2>
<div><b>Filepath:</b> {filepath}</div>
<div><b>Created at:</b> {created_date}</div>
<h2>This file is corrupted</h2>"""

    # 返回生成的 HTML 代码
    return code


# 标准化 Git URL
def normalize_git_url(url):
    # 如果 URL 为 None，则返回空字符串
    if url is None:
        return ""

    # 去除 URL 中的 .git 后缀
    url = url.replace(".git", "")
    return url


# 从 URL 中获取扩展目录名
def get_extension_dirname_from_url(url):
    # 使用 / 分割 URL，获取最后一个部分作为目录名
    *parts, last_part = url.split('/')
    return normalize_git_url(last_part)


# 从 URL 安装扩展
def install_extension_from_url(dirname, url, branch_name=None):
    # 检查访问权限
    check_access()

    # 如果目录名为字符串，则去除首尾空格
    if isinstance(dirname, str):
        dirname = dirname.strip()
    # 如果 URL 为字符串，则去除首尾空格
    if isinstance(url, str):
        url = url.strip()

    # 断言 URL 不为空
    assert url, 'No URL specified'

    # 如果目录名为 None 或空字符串，则从 URL 中获取目录名
    if dirname is None or dirname == "":
        dirname = get_extension_dirname_from_url(url)

    # 拼接目标目录路径
    target_dir = os.path.join(extensions.extensions_dir, dirname)
    # 断言目标目录不存在
    assert not os.path.exists(target_dir), f'Extension directory already exists: {target_dir}'

    # 标准化 URL
    normalized_url = normalize_git_url(url)
    # 如果已经安装了具有相同 URL 的扩展，则抛出异常
    if any(x for x in extensions.extensions if normalize_git_url(x.remote) == normalized_url):
        raise Exception(f'Extension with this URL is already installed: {url}')

    # 临时目录路径
    tmpdir = os.path.join(paths.data_path, "tmp", dirname)
    # 尝试删除临时目录
    try:
        shutil.rmtree(tmpdir, True)
        # 如果没有指定分支，则使用默认分支
        if not branch_name:
            # 从指定 URL 克隆仓库到临时目录，使用过滤器只获取文件内容
            with git.Repo.clone_from(url, tmpdir, filter=['blob:none']) as repo:
                # 拉取远程仓库内容
                repo.remote().fetch()
                # 更新子模块
                for submodule in repo.submodules:
                    submodule.update()
        else:
            # 从指定 URL 克隆仓库到临时目录，使用过滤器只获取文件内容，并指定分支
            with git.Repo.clone_from(url, tmpdir, filter=['blob:none'], branch=branch_name) as repo:
                # 拉取远程仓库内容
                repo.remote().fetch()
                # 更新子模块
                for submodule in repo.submodules:
                    submodule.update()
        try:
            # 尝试将临时目录重命名为目标目录
            os.rename(tmpdir, target_dir)
        except OSError as err:
            if err.errno == errno.EXDEV:
                # 跨设备链接，典型情况是在 Docker 中或者 tmp/ 和 extensions/ 在不同文件系统上
                # 由于无法使用重命名操作，使用较慢但更灵活的 shutil.move() 方法
                shutil.move(tmpdir, target_dir)
            else:
                # 其他情况，如空间不足、权限等，重新抛出异常以便处理
                raise err

        # 导入 launch 模块
        import launch
        # 运行扩展安装器，安装到目标目录
        launch.run_extension_installer(target_dir)

        # 列出已安装的扩展
        extensions.list_extensions()
        # 返回安装结果信息
        return [extension_table(), html.escape(f"Installed into {target_dir}. Use Installed tab to restart.")]
    finally:
        # 最终无论如何都要删除临时目录
        shutil.rmtree(tmpdir, True)
# 从指定 URL 安装扩展，返回安装结果和消息
def install_extension_from_index(url, hide_tags, sort_column, filter_text):
    # 从 URL 安装扩展并获取结果表格和消息
    ext_table, message = install_extension_from_url(None, url)

    # 刷新可用扩展列表，并获取结果代码
    code, _ = refresh_available_extensions_from_data(hide_tags, sort_column, filter_text)

    # 返回结果代码、扩展表格、消息和空字符串
    return code, ext_table, message, ''


# 刷新可用扩展列表
def refresh_available_extensions(url, hide_tags, sort_column):
    # 声明全局变量可用扩展列表
    global available_extensions

    # 使用 urllib 请求指定 URL 并读取响应内容
    import urllib.request
    with urllib.request.urlopen(url) as response:
        text = response.read()

    # 将响应内容解析为 JSON 格式，更新可用扩展列表
    available_extensions = json.loads(text)

    # 刷新可用扩展列表并获取结果代码和标签
    code, tags = refresh_available_extensions_from_data(hide_tags, sort_column)

    # 返回 URL、结果代码、更新复选框组件的选项、空字符串和空字符串
    return url, code, gr.CheckboxGroup.update(choices=tags), '', ''


# 根据标签刷新可用扩展列表
def refresh_available_extensions_for_tags(hide_tags, sort_column, filter_text):
    # 刷新可用扩展列表并获取结果代码
    code, _ = refresh_available_extensions_from_data(hide_tags, sort_column, filter_text)

    # 返回结果代码和空字符串
    return code, ''


# 搜索扩展
def search_extensions(filter_text, hide_tags, sort_column):
    # 刷新可用扩展列表并获取结果代码
    code, _ = refresh_available_extensions_from_data(hide_tags, sort_column, filter_text)

    # 返回结果代码和空字符串
    return code, ''


# 排序顺序列表
sort_ordering = [
    # (reverse, order_by_function)
    (True, lambda x: x.get('added', 'z')),
    (False, lambda x: x.get('added', 'z')),
    (False, lambda x: x.get('name', 'z')),
    (True, lambda x: x.get('name', 'z')),
    (False, lambda x: 'z'),
    (True, lambda x: x.get('commit_time', '')),
    (True, lambda x: x.get('created_at', '')),
    (True, lambda x: x.get('stars', 0)),
]


# 获取日期信息
def get_date(info: dict, key):
    try:
        # 尝试将日期字符串转换为日期对象，并格式化输出
        return datetime.strptime(info.get(key), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone().strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ''


# 根据数据刷新可用扩展列表
def refresh_available_extensions_from_data(hide_tags, sort_column, filter_text=""):
    # 获取可用扩展列表
    extlist = available_extensions["extensions"]
    # 获取已安装扩展的名称集合
    installed_extensions = {extension.name for extension in extensions.extensions}
    # 从 extensions.extensions 中获取每个扩展的远程地址，通过 normalize_git_url 函数规范化，存储在 installed_extension_urls 集合中
    installed_extension_urls = {normalize_git_url(extension.remote) for extension in extensions.extensions if extension.remote is not None}
    
    # 从 available_extensions 中获取标签信息，初始化 tags 字典
    tags = available_extensions.get("tags", {})
    # 将需要隐藏的标签转换为集合，初始化 tags_to_hide 集合
    tags_to_hide = set(hide_tags)
    # 初始化隐藏计数器
    hidden = 0
    
    # 生成包含时间戳的 HTML 代码
    code = f"""<!-- {time.time()} -->
    <table id="available_extensions">
        <thead>
            <tr>
                <th>Extension</th>
                <th>Description</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # 根据排序列索引获取排序方式和排序函数
    sort_reverse, sort_function = sort_ordering[sort_column if 0 <= sort_column < len(sort_ordering) else 0]
    
    # 添加 HTML 代码片段
    code += """
        </tbody>
    </table>
    """
    
    # 如果有隐藏的扩展，则在 HTML 代码中添加提示信息
    if hidden > 0:
        code += f"<p>Extension hidden: {hidden}</p>"
    
    # 返回生成的 HTML 代码和标签列表
    return code, list(tags)
# 预加载扩展的 Git 元数据信息
def preload_extensions_git_metadata():
    # 遍历所有扩展，从仓库中读取信息
    for extension in extensions.extensions:
        extension.read_info_from_repo()

# 创建用户界面
def create_ui():
    # 导入 UI 模块
    import modules.ui
    # 列出配置状态
    config_states.list_config_states()
    # 创建一个新线程，用于预加载扩展的 Git 元数据信息
    threading.Thread(target=preload_extensions_git_metadata).start()
    # 返回用户界面对象
    return ui
```