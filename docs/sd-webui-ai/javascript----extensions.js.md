# `stable-diffusion-webui\javascript\extensions.js`

```
# 应用扩展的更改，根据传入的参数来决定是否禁用所有扩展
function extensions_apply(_disabled_list, _update_list, disable_all) {
    # 初始化禁用和更新列表
    var disable = [];
    var update = [];

    # 遍历所有扩展复选框
    gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach(function(x) {
        # 如果复选框名称以"enable_"开头且未选中，则将其添加到禁用列表中
        if (x.name.startsWith("enable_") && !x.checked) {
            disable.push(x.name.substring(7));
        }
        # 如果复选框名称以"update_"开头且选中，则将其添加到更新列表中
        if (x.name.startsWith("update_") && x.checked) {
            update.push(x.name.substring(7));
        }
    });

    # 重新加载页面
    restart_reload();

    # 返回禁用列表、更新列表和是否禁用所有扩展的 JSON 字符串
    return [JSON.stringify(disable), JSON.stringify(update), disable_all];
}

# 检查扩展的状态
function extensions_check() {
    # 初始化禁用列表
    var disable = [];

    # 遍历所有扩展复选框
    gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach(function(x) {
        # 如果复选框名称以"enable_"开头且未选中，则将其添加到禁用列表中
        if (x.name.startsWith("enable_") && !x.checked) {
            disable.push(x.name.substring(7));
        }
    });

    # 将所有扩展状态设置为"Loading..."
    gradioApp().querySelectorAll('#extensions .extension_status').forEach(function(x) {
        x.innerHTML = "Loading...";
    });

    # 生成随机 ID
    var id = randomId();
    # 请求进度
    requestProgress(id, gradioApp().getElementById('extensions_installed_html'), null, function() {

    });

    # 返回随机 ID 和禁用列表的 JSON 字符串
    return [id, JSON.stringify(disable)];
}

# 从索引安装扩展
function install_extension_from_index(button, url) {
    # 禁用按钮并将其值设置为"Installing..."
    button.disabled = "disabled";
    button.value = "Installing...";

    # 获取要安装的扩展 URL，并更新输入框
    var textarea = gradioApp().querySelector('#extension_to_install textarea');
    textarea.value = url;
    updateInput(textarea);

    # 触发安装扩展按钮的点击事件
    gradioApp().querySelector('#install_extension_button').click();
}

# 确认恢复配置状态
function config_state_confirm_restore(_, config_state_name, config_restore_type) {
    # 如果配置状态名称为"Current"，则返回 false、配置状态名称和配置恢复类型
    if (config_state_name == "Current") {
        return [false, config_state_name, config_restore_type];
    }
    # 初始化恢复信息字符串
    let restored = "";
    # 根据配置恢复类型设置恢复信息字符串
    if (config_restore_type == "extensions") {
        restored = "all saved extension versions";
    } else if (config_restore_type == "webui") {
        restored = "the webui version";
    } else {
        restored = "the webui version and all saved extension versions";
    }
}
    # 弹出确认对话框，询问用户是否确定要从当前状态恢复
    let confirmed = confirm("Are you sure you want to restore from this state?\nThis will reset " + restored + ".");
    # 如果用户确认恢复操作
    if (confirmed) {
        # 重新加载页面
        restart_reload();
        # 获取所有扩展状态元素，并将其内容设置为"Loading..."
        gradioApp().querySelectorAll('#extensions .extension_status').forEach(function(x) {
            x.innerHTML = "Loading...";
        });
    }
    # 返回确认结果、配置状态名称和恢复类型
    return [confirmed, config_state_name, config_restore_type];
# 定义一个函数，用于切换所有扩展的选中状态
function toggle_all_extensions(event) {
    # 获取所有扩展的复选框元素，并对其进行遍历
    gradioApp().querySelectorAll('#extensions .extension_toggle').forEach(function(checkbox_el) {
        # 将复选框的选中状态设置为与事件目标相同
        checkbox_el.checked = event.target.checked;
    });
}

# 定义一个函数，用于切换单个扩展的选中状态
function toggle_extension() {
    # 初始化所有扩展是否已经全部切换的变量为真
    let all_extensions_toggled = true;
    # 遍历所有扩展的复选框元素
    for (const checkbox_el of gradioApp().querySelectorAll('#extensions .extension_toggle')) {
        # 如果有任何一个复选框未被选中，则将所有扩展是否已经全部切换的变量设为假，并跳出循环
        if (!checkbox_el.checked) {
            all_extensions_toggled = false;
            break;
        }
    }
    # 将所有扩展的全选复选框的选中状态设置为所有扩展是否已经全部切换的变量
    gradioApp().querySelector('#extensions .all_extensions_toggle').checked = all_extensions_toggled;
}
```