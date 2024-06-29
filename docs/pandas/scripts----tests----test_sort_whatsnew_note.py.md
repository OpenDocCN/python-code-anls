# `D:\src\scipysrc\pandas\scripts\tests\test_sort_whatsnew_note.py`

```
# 导入排序函数 `sort_whatsnew_note` 从 `scripts.sort_whatsnew_note` 模块
from scripts.sort_whatsnew_note import sort_whatsnew_note

# 定义测试函数，不返回任何值 (`None`)
def test_sort_whatsnew_note() -> None:
    # 定义测试用例的内容字符串
    content = (
        ".. _whatsnew_200:\n"
        "\n"
        "What's new in 2.0.0 (March XX, 2023)\n"
        "------------------------------------\n"
        "\n"
        "Timedelta\n"
        "^^^^^^^^^\n"
        "- Bug in :meth:`Timedelta.round` (:issue:`51494`)\n"
        "- Bug in :class:`TimedeltaIndex` (:issue:`51575`)\n"
        "\n"
    )
    # 定义预期的排序后的内容字符串
    expected = (
        ".. _whatsnew_200:\n"
        "\n"
        "What's new in 2.0.0 (March XX, 2023)\n"
        "------------------------------------\n"
        "\n"
        "Timedelta\n"
        "^^^^^^^^^\n"
        "- Bug in :class:`TimedeltaIndex` (:issue:`51575`)\n"
        "- Bug in :meth:`Timedelta.round` (:issue:`51494`)\n"
        "\n"
    )
    # 调用排序函数 `sort_whatsnew_note` 处理内容字符串，并获取处理结果
    result = sort_whatsnew_note(content)
    # 使用断言验证处理结果是否与预期相符
    assert result == expected
```