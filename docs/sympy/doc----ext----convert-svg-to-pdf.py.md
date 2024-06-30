# `D:\src\scipysrc\sympy\doc\ext\convert-svg-to-pdf.py`

```
"""
    Converts SVG images to PDF using chrome in case the builder does not
    support SVG images natively (e.g. LaTeX).

"""

# 导入必要的模块和类
from sphinx.transforms.post_transforms.images import ImageConverter
from sphinx.util import logging
import os
import platform
from typing import Any  # NOQA
from sphinx.application import Sphinx  # NOQA

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


# 创建 Converter 类，继承自 ImageConverter 类
class Converter(ImageConverter):
    # 定义转换规则，将 SVG 图像转换为 PDF 格式
    conversion_rules = [
        ('image/svg+xml', 'application/pdf'),
    ]

    # 判断转换器是否可用的方法
    def is_available(self) -> bool:
        """Confirms if converter is available or not."""
        return True

    # 返回 Chrome 浏览器可执行文件的路径或命令
    def chrome_command(self) -> str | None:
        # 在 Windows 平台上检查 Chrome 的可执行文件位置
        if platform.win32_ver()[0]:
            if os.system("where chrome") == 0:
                return "chrome"
            # 尝试查找默认安装路径下的 Chrome 可执行文件
            path = os.path.join(os.environ["PROGRAMW6432"], "Google\\Chrome\\Application\\chrome.exe")
            if os.path.exists(path):
                return f'"{path}"'
            return None
        # 在其他平台上检查 Chrome 的可执行文件位置
        if os.system("chrome --version") == 0:
            return "chrome"
        # macOS 上返回 Chrome 的默认安装路径
        if platform.mac_ver()[0]:
            return "'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'"
        # Linux 上返回 Chrome 的命令名
        elif platform.libc_ver()[0]:
            return "google-chrome"
        return None

    # 返回 Chromium 浏览器可执行文件的路径或命令
    def chromium_command(self) -> str | None:
        # 在 Windows 平台上检查 Chromium 的可执行文件位置
        if platform.win32_ver()[0]:
            if os.system("where chromium") == 0:
                return "chromium"
            # 尝试查找默认安装路径下的 Chromium 可执行文件
            path = os.path.join(os.environ["PROGRAMW6432"], "Chromium\\Application\\chrome.exe")
            if os.path.exists(path):
                return f'"{path}"'
            return None
        # 在其他平台上检查 Chromium 的可执行文件位置
        if os.system("chromium --version") == 0:
            return "chromium"
        # macOS 上返回 Chromium 的默认安装路径
        if platform.mac_ver()[0]:
            path = "/Applications/Chromium.app/Contents/MacOS/Chromium"
            if os.path.exists(path):
                return path
        # Linux 上返回 Chromium 的命令名
        elif platform.libc_ver()[0]:
            if os.system("chromium-browser --version") == 0:
                return "chromium-browser"
        return None

    # 运行命令来将 SVG 文件转换为 PDF
    def command_runner(self, chrome: str | None, _to: str, temp_name: str) -> int:
        # 如果没有提供有效的 Chrome 或 Chromium 路径，则返回错误
        if not chrome:
            return 1
        # 构建用于将 SVG 转换为 PDF 的命令
        command = f'{chrome} --headless --disable-gpu --disable-software-rasterizer --print-to-pdf={_to} {temp_name}'
        # 记录命令到日志
        logger.error(command)
        # 执行命令并返回执行结果状态码
        return os.system(command)
    # 定义一个方法 convert，用于将 SVG 图像转换为 PDF 格式
    def convert(self, _from: str, _to: str) -> bool:
        """Converts the image from SVG to PDF using chrome."""
        # 打开源 SVG 文件，并读取其内容
        with open(_from, 'r') as f:
            svg = f.read()

        # 构建 HTML 字符串，包含 SVG 内容，并根据 SVG 大小设置页面大小
        HTML = "<html><head><style>body {margin: 0; }</style><script>function init() {const element = document.querySelector('svg');const positionInfo = element.getBoundingClientRect();const height = positionInfo.height;const width = positionInfo.width;const style = document.createElement('style');style.innerHTML = `@page {margin: 0; size: ${width}px ${height}px}`;document.head.appendChild(style); }window.onload = init;</script></head><body>%s</body></html>" % (svg)
        # 生成临时的 HTML 文件，将 HTML 字符串写入其中
        temp_name = f'{_from}.html'
        with open(temp_name, 'w') as f:
            f.write(HTML)

        # 获取 Chromium 命令，并执行转换命令
        chromium = self.chromium_command()
        code = self.command_runner(chromium, _to, temp_name)
        # 如果转换失败，尝试使用 Chrome 进行转换
        if code != 0:
            chrome = self.chrome_command()
            code = self.command_runner(chrome, _to, temp_name)
        # 如果两者都失败，则记录错误并退出程序
        if code != 0:
            logger.error('Fail to convert svg to pdf. Make sure Chromium or Chrome is installed.')
            exit(1)
        # 转换成功，返回 True
        return True
# 定义一个函数 `setup`，接受一个类型为 `Sphinx` 的参数 `app`，返回一个包含配置信息的字典
def setup(app: Sphinx) -> dict[str, Any]:
    # 向 Sphinx 应用程序添加一个后处理转换器 `Converter`
    app.add_post_transform(Converter)

    # 返回一个包含以下信息的字典作为配置：
    # - 版本信息为 `'builtin'`
    # - 支持并行读取安全性设置为 `True`
    # - 支持并行写入安全性设置为 `True`
    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```