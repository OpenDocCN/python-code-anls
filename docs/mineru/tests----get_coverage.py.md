# `.\MinerU\tests\get_coverage.py`

```
# 文档字符串，表示该模块的功能
"""
get cov
"""
# 导入 BeautifulSoup 库，用于解析 HTML 内容
from bs4 import BeautifulSoup

# 定义函数获取覆盖率
def get_covrage():
    """获取覆盖率"""
    # 读取本地 HTML 文件内容
    html_content = open("htmlcov/index.html", "r", encoding="utf-8").read()
    # 使用 BeautifulSoup 解析 HTML 内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 查找具有 'pc_cov' 类名的 span 标签
    pc_cov_span = soup.find('span', class_='pc_cov')

    # 提取该标签中的文本并去掉前后空格
    percentage_value = pc_cov_span.text.strip()
    # 将提取的百分比值转换为浮点数，去掉百分号
    percentage_float = float(percentage_value.rstrip('%'))
    # 打印浮点数表示的覆盖率
    print ("percentage_float:", percentage_float)
    # 确保覆盖率不低于 0.2，若低于则抛出异常
    assert percentage_float >= 0.2

# 检查是否为主程序运行，若是则调用 get_covrage 函数
if __name__ == '__main__':
    get_covrage()
```