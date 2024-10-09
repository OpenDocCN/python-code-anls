# `.\MinerU\magic_pdf\libs\language.py`

```
# 导入操作系统相关模块
import os
# 导入 Unicode 数据处理模块
import unicodedata

# 检查环境变量 "FTLANG_CACHE" 是否存在
if not os.getenv("FTLANG_CACHE"):
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在目录
    current_dir = os.path.dirname(current_file_path)
    # 获取根目录，即当前目录的上级目录
    root_dir = os.path.dirname(current_dir)
    # 构建 fasttext 语言检测的缓存目录路径
    ftlang_cache_dir = os.path.join(root_dir, 'resources', 'fasttext-langdetect')
    # 设置环境变量 "FTLANG_CACHE" 为缓存目录路径
    os.environ["FTLANG_CACHE"] = str(ftlang_cache_dir)
    # print(os.getenv("FTLANG_CACHE"))  # 可选：打印当前缓存路径

# 从 fast_langdetect 模块导入语言检测函数
from fast_langdetect import detect_language


# 定义检测语言的函数，输入为文本，输出为语言代码
def detect_lang(text: str) -> str:
    # 如果输入文本长度为零，返回空字符串
    if len(text) == 0:
        return ""
    try:
        # 尝试检测文本语言并转换为大写
        lang_upper = detect_language(text)
    except:
        # 处理异常：去除文本中的控制字符
        html_no_ctrl_chars = ''.join([l for l in text if unicodedata.category(l)[0] not in ['C', ]])
        # 尝试再次检测清理后的文本语言
        lang_upper = detect_language(html_no_ctrl_chars)
    try:
        # 将检测到的语言转换为小写
        lang = lang_upper.lower()
    except:
        # 如果转换失败，返回空字符串
        lang = ""
    # 返回最终检测到的语言代码
    return lang


# 检查是否为主程序执行
if __name__ == '__main__':
    # 打印环境变量 "FTLANG_CACHE"
    print(os.getenv("FTLANG_CACHE"))
    # 打印检测到的语言结果
    print(detect_lang("This is a test."))
    print(detect_lang("<html>This is a test</html>"))
    print(detect_lang("这个是中文测试。"))
    print(detect_lang("<html>这个是中文测试。</html>"))
```