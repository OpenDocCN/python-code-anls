# `stable-diffusion-webui\javascript\localStorage.js`

```py
# 将键值对存储到本地存储中
function localSet(k, v) {
    try:
        # 尝试将键值对存储到本地存储中
        localStorage.setItem(k, v);
    except (e):
        # 如果存储失败，则输出警告信息
        console.warn(`Failed to save ${k} to localStorage: ${e}`);
}

# 从本地存储中获取指定键的值
function localGet(k, def) {
    try:
        # 尝试从本地存储中获取指定键的值
        return localStorage.getItem(k);
    except (e):
        # 如果获取失败，则输出警告信息
        console.warn(`Failed to load ${k} from localStorage: ${e}`);
    
    # 如果获取失败，则返回默认值
    return def;
}

# 从本地存储中移除指定键值对
function localRemove(k) {
    try:
        # 尝试从本地存储中移除指定键值对
        return localStorage.removeItem(k);
    except (e):
        # 如果移除失败，则输出警告信息
        console.warn(`Failed to remove ${k} from localStorage: ${e}`);
}
```