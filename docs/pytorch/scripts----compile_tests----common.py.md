# `.\pytorch\scripts\compile_tests\common.py`

```
import functools
import os
import warnings

try:
    import lxml.etree

    # 创建一个 XML 解析器对象，用于处理大型 XML 文件
    p = lxml.etree.XMLParser(huge_tree=True)
    # 创建一个部分应用函数，使用 lxml 库的 parse 函数，并传入上面创建的解析器
    parse = functools.partial(lxml.etree.parse, parser=p)
except ImportError:
    import xml.etree.ElementTree as ET

    # 如果 lxml 库导入失败，则使用标准库中的 ElementTree
    parse = ET.parse
    # 发出警告，提示用户安装 lxml 库以提升脚本性能
    warnings.warn(
        "lxml was not found. `pip install lxml` to make this script run much faster"
    )


def open_test_results(directory):
    # 存储所有找到的 XML 文件
    xmls = []
    # 遍历指定目录及其子目录中的所有文件
    for root, _, files in os.walk(directory):
        for file in files:
            # 仅处理以 ".xml" 结尾的文件
            if file.endswith(".xml"):
                # 解析 XML 文件并将其添加到 xmls 列表中
                tree = parse(f"{root}/{file}")
                xmls.append(tree)
    return xmls


def get_testcases(xmls):
    # 获取所有测试用例
    testcases = []
    for xml in xmls:
        root = xml.getroot()
        # 找到所有名为 "testcase" 的元素并加入 testcases 列表
        testcases.extend(list(root.iter("testcase")))
    return testcases


def find(testcase, condition):
    # 获取测试用例的所有子元素，并检查第一个元素是否为测试用例本身
    children = list(testcase.iter())
    assert children[0] is testcase
    # 去除第一个元素后，剩余的元素作为条件函数的输入，判断是否满足条件
    children = children[1:]
    return condition(children)


def skipped_test(testcase):
    # 判断测试用例是否被跳过
    def condition(children):
        tags = [child.tag for child in children]
        if "skipped" in tags:
            return True
        return False

    return find(testcase, condition)


def passed_test(testcase):
    # 判断测试用例是否通过
    def condition(children):
        if len(children) == 0:
            return True
        tags = [child.tag for child in children]
        if "skipped" in tags:
            return False
        if "failed" in tags:
            return False
        return True

    return find(testcase, condition)


def key(testcase):
    # 生成测试用例的唯一标识符
    file = testcase.attrib.get("file", "UNKNOWN")
    classname = testcase.attrib["classname"]
    name = testcase.attrib["name"]
    return "::".join([file, classname, name])


def get_passed_testcases(xmls):
    # 获取所有通过的测试用例
    testcases = get_testcases(xmls)
    passed_testcases = [testcase for testcase in testcases if passed_test(testcase)]
    return passed_testcases


def get_excluded_testcases(xmls):
    # 获取所有被排除的测试用例
    testcases = get_testcases(xmls)
    excluded_testcases = [t for t in testcases if excluded_testcase(t)]
    return excluded_testcases


def excluded_testcase(testcase):
    # 判断测试用例是否被排除
    def condition(children):
        for child in children:
            if child.tag == "skipped":
                if "Policy: we don't run" in child.attrib["message"]:
                    return True
        return False

    return find(testcase, condition)


def is_unexpected_success(testcase):
    # 判断测试用例是否出现了意外的成功
    def condition(children):
        for child in children:
            if child.tag != "failure":
                continue
            is_unexpected_success = (
                "unexpected success" in child.attrib["message"].lower()
            )
            if is_unexpected_success:
                return True
        return False

    return find(testcase, condition)


MSG = "This test passed, maybe we can remove the skip from dynamo_test_failures.py"


def is_passing_skipped_test(testcase):
    # 定义一个名为 condition 的函数，接受一个 children 列表作为参数
    def condition(children):
        # 遍历 children 列表中的每个 child
        for child in children:
            # 如果 child 的标签不是 "skipped"，则跳过本次循环
            if child.tag != "skipped":
                continue
            # 检查 child 的属性中是否包含特定的消息 MSG
            has_passing_skipped_test_msg = MSG in child.attrib["message"]
            # 如果找到符合条件的消息，返回 True
            if has_passing_skipped_test_msg:
                return True
        # 如果没有找到符合条件的 child，返回 False
        return False

    # 调用外部定义的 find 函数，传入参数 testcase 和定义的 condition 函数
    return find(testcase, condition)
# 定义函数 is_failure，用于判断测试用例是否失败
def is_failure(testcase):
    # 定义内部函数 condition，用于检查测试用例中是否存在失败条件
    def condition(children):
        # 遍历子元素列表
        for child in children:
            # 如果子元素标签不是 "failure"，则继续下一个子元素
            if child.tag != "failure":
                continue
            # 检查子元素的消息属性是否包含 "unexpected success"（不期望的成功）
            is_unexpected_success = (
                "unexpected success" in child.attrib["message"].lower()
            )
            # 如果不是不期望的成功，则返回 True，表示存在失败
            if not is_unexpected_success:
                return True
        # 如果未找到任何失败条件，则返回 False，表示未失败
        return False

    # 调用 find 函数，传入测试用例和内部定义的 condition 函数进行判断
    return find(testcase, condition)
```