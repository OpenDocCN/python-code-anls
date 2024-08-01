# `.\DB-GPT-src\docs\static\redirect.js`

```py
# 当页面加载完成后执行的函数
window.onload = function() {
    # 检查当前页面的路径是否为根路径或空路径
    if (window.location.pathname == '/' || window.location.pathname == "") {
        # 如果是，则重定向到 '/docs/overview' 页面
        window.location.href = '/docs/overview';
    }
};
```