# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\web_backend\js\mpl_tornado.js`

```py
/* This .js file contains functions for matplotlib's built-in
   tornado-based server, that are not relevant when embedding WebAgg
   in another web application. */
/* 这个 .js 文件包含了为 matplotlib 的基于 tornado 的内置服务器编写的函数，
   当嵌入到其他网络应用程序中时，这些函数并不相关。 */

/* exported mpl_ondownload */
/* 导出 mpl_ondownload 函数，使其可以在其他地方使用 */
function mpl_ondownload(figure, format) {
    // 打开一个新窗口以下载图形，使用图形的 ID 和指定的格式
    window.open(figure.id + '/download.' + format, '_blank');
}
```