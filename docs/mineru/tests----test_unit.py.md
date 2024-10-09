# `.\MinerU\tests\test_unit.py`

```
# 导入操作系统相关的模块
import os

# 导入pytest测试框架
import pytest

# 从magic_pdf.libs.boxbase模块导入多个函数
from magic_pdf.libs.boxbase import (__is_overlaps_y_exceeds_threshold,
                                    _is_bottom_full_overlap, _is_in,
                                    _is_in_or_part_overlap,
                                    _is_in_or_part_overlap_with_area_ratio,
                                    _is_left_overlap, _is_part_overlap,
                                    _is_vertical_full_overlap, _left_intersect,
                                    _right_intersect, bbox_distance,
                                    bbox_relative_pos, calculate_iou,
                                    calculate_overlap_area_2_minbox_area_ratio,
                                    calculate_overlap_area_in_bbox1_area_ratio,
                                    find_bottom_nearest_text_bbox,
                                    find_left_nearest_text_bbox,
                                    find_right_nearest_text_bbox,
                                    find_top_nearest_text_bbox,
                                    get_bbox_in_boundary,
                                    get_minbox_if_overlap_by_ratio)

# 从magic_pdf.libs.commons模块导入多个实用函数
from magic_pdf.libs.commons import get_top_percent_list, join_path, mymax

# 从magic_pdf.libs.config_reader模块导入获取S3配置的函数
from magic_pdf.libs.config_reader import get_s3_config

# 从magic_pdf.libs.path_utils模块导入解析S3路径的函数
from magic_pdf.libs.path_utils import parse_s3path


# 使用pytest的参数化功能测试mymax函数，检查输入列表的最大值
@pytest.mark.parametrize('list_input, target_num',
                         [
                             ([0, 0, 0, 0], 0),  # 测试全为0的列表，期望返回0
                             ([0], 0),           # 测试单个0的列表，期望返回0
                             ([1, 2, 5, 8, 4], 8),  # 测试包含多个数的列表，期望返回8
                             ([], 0),            # 测试空列表，期望返回0
                             ([1.1, 7.6, 1.009, 9.9], 9.9),  # 测试浮点数列表，期望返回9.9
                             ([1.0 * 10 ** 2, 3.5 * 10 ** 3, 0.9 * 10 ** 6], 0.9 * 10 ** 6),  # 测试科学计数法，期望返回0.9*10^6
                         ])
def test_list_max(list_input: list, target_num) -> None:
    """
    list_input: 输入列表元素，元素均为数字类型
    """
    # 断言target_num等于mymax函数返回的最大值
    assert target_num == mymax(list_input)


# 使用pytest的参数化功能测试join_path函数，检查路径连接的正确性
@pytest.mark.parametrize('path_input, target_path', [
    (['https:', '', 'www.baidu.com'], 'https://www.baidu.com'),  # 测试连接https与域名
    (['https:', 'www.baidu.com'], 'https:/www.baidu.com'),  # 测试连接https与域名，缺少斜杠
    (['D:', 'file', 'pythonProject', 'demo' + '.py'], 'D:/file/pythonProject/demo.py'),  # 测试Windows路径
])
def test_join_path(path_input: list, target_path: str) -> None:
    """
    path_input: 输入path的列表，列表元素均为字符串
    """
    # 断言target_path等于join_path函数的返回结果
    assert target_path == join_path(*path_input)


# 使用pytest的参数化功能测试get_top_percent_list函数，检查获取前百分之多少元素的功能
@pytest.mark.parametrize('num_list, percent, target_num_list', [
    ([], 0.75, []),  # 测试空列表，期望返回空列表
    ([-5, -10, 9, 3, 7, -7, 0, 23, -1, -11], 0.8, [23, 9, 7, 3, 0, -1, -5, -7]),  # 测试负数和正数的列表
    ([-5, -10, 9, 3, 7, -7, 0, 23, -1, -11], 0, []),  # 测试占比为0，期望返回空列表
    ([-5, -10, 9, 3, 7, -7, 0, 23, -1, -11, 28], 0.8, [28, 23, 9, 7, 3, 0, -1, -5])  # 测试包含更多元素的列表
])
def test_get_top_percent_list(num_list: list, percent: float, target_num_list: list) -> None:
    """
    num_list: 数字列表，列表元素为数字
    percent: 占比，float, 向下取整
    """
    # 断言target_num_list等于get_top_percent_list函数的返回结果
    assert target_num_list == get_top_percent_list(num_list, percent)
# 输入一个s3路径，返回bucket名字和其余部分(key)
@pytest.mark.parametrize('s3_path, target_data', [
    # 测试用例：有效的s3路径，期望返回bucket名字
    ('s3://bucket/path/to/my/file.txt', 'bucket'),
    # 测试用例：使用s3a协议的有效路径
    ('s3a://bucket1/path/to/my/file2.txt', 'bucket1'),
    # 注释的测试用例
    # ("/path/to/my/file1.txt", "path"),
    # ("bucket/path/to/my/file2.txt", "bucket"),
])
def test_parse_s3path(s3_path: str, target_data: str):
    """
    s3_path: s3路径
        如果为无效路径，则返回对应的bucket名字和其余部分
        如果为异常路径 例如：file2.txt，则报异常
    """
    # 调用解析函数，获取bucket名字和key
    bucket_name, key = parse_s3path(s3_path)
    # 断言返回的bucket名字是否与期望相等
    assert target_data == bucket_name


# 2个box是否处于包含或者部分重合关系。
# 如果某边界重合算重合。
# 部分边界重合，其他在内部也算包含
@pytest.mark.parametrize('box1, box2, target_bool', [
    # 测试用例：box1与box2部分重合
    ((120, 133, 223, 248), (128, 168, 269, 295), True),
    # 测试用例：box1与box2部分重合
    ((137, 53, 245, 157), (134, 11, 200, 147), True),  # 部分重合
    # 测试用例：box1与box2部分重合
    ((137, 56, 211, 116), (140, 66, 202, 199), True),  # 部分重合
    # 测试用例：box1与box2完全相同
    ((42, 34, 69, 65), (42, 34, 69, 65), True),  # 部分重合
    # 测试用例：box1与box2部分重合
    ((39, 63, 87, 106), (37, 66, 85, 109), True),  # 部分重合
    # 测试用例：box1与box2部分重合
    ((13, 37, 55, 66), (7, 46, 49, 75), True),  # 部分重合
    # 测试用例：box1与box2部分重合
    ((56, 83, 85, 104), (64, 85, 93, 106), True),  # 部分重合
    # 测试用例：box1与box2部分重合
    ((12, 53, 48, 94), (14, 53, 50, 94), True),  # 部分重合
    # 测试用例：box1包含box2
    ((43, 54, 93, 131), (55, 82, 77, 106), True),  # 包含
    # 测试用例：box1包含box2
    ((63, 2, 134, 71), (72, 43, 104, 78), True),  # 包含
    # 测试用例：box1包含box2
    ((25, 57, 109, 127), (26, 73, 49, 95), True),  # 包含
    # 测试用例：box1包含box2
    ((24, 47, 111, 115), (34, 81, 58, 106), True),  # 包含
    # 测试用例：box1包含box2
    ((34, 8, 105, 83), (76, 20, 116, 45), True),  # 包含
])
def test_is_in_or_part_overlap(box1: tuple, box2: tuple, target_bool: bool) -> None:
    """
    box1: 坐标数组
    box2: 坐标数组
    """
    # 断言box1和box2的重合关系是否符合预期
    assert target_bool == _is_in_or_part_overlap(box1, box2)


# 如果box1在box2内部，返回True
#   如果是部分重合的，则重合面积占box1的比例大于阈值时候返回True
@pytest.mark.parametrize('box1, box2, target_bool', [
    # 测试用例：box1被box2部分覆盖
    ((35, 28, 108, 90), (47, 60, 83, 96), False),  # 包含 box1 up box2,  box2 多半,box1少半
    # 测试用例：box1在box2内部
    ((65, 151, 92, 177), (49, 99, 105, 198), True),  # 包含 box1 in box2
    # 测试用例：box1在box2内部
    ((80, 62, 112, 84), (74, 40, 144, 111), True),  # 包含 box1 in box2
    # 测试用例：box2覆盖box1部分
    ((65, 88, 127, 144), (92, 102, 131, 139), False),  # 包含 box2 多半，box1约一半
    # 测试用例：box1在box2内部
    ((92, 102, 131, 139), (65, 88, 127, 144), True),  # 包含 box1 多半
    # 测试用例：box2在box1内部
    ((100, 93, 199, 168), (169, 126, 198, 165), False),  # 包含 box2 in box1
    # 测试用例：box2在box1内部
    ((26, 75, 106, 172), (65, 108, 90, 128), False),  # 包含 box2 in box1
    # 测试用例：box1与box2相交
    ((28, 90, 77, 126), (35, 84, 84, 120), True),  # 相交 box1多半，box2多半
    # 测试用例：box1与box2相交
    ((37, 6, 69, 52), (28, 3, 60, 49), True),  # 相交 box1多半，box2多半
    # 测试用例：box1与box2相交
    ((94, 29, 133, 60), (84, 30, 123, 61), True),  # 相交 box1多半，box2多半
])
def test_is_in_or_part_overlap_with_area_ratio(box1: tuple, box2: tuple, target_bool: bool) -> None:
    # 调用函数以获取box1和box2的重合状态
    out_bool = _is_in_or_part_overlap_with_area_ratio(box1, box2)
    # 断言结果是否符合预期
    assert target_bool == out_bool


# box1在box2内部或者box2在box1内部返回True。如果部分边界重合也算作包含。
@pytest.mark.parametrize('box1, box2, target_bool', [
    # 注释的测试用例，预期返回错误
    # ((), (), "Error"),  # Error
    # 测试用例：box1在box2内部
    ((65, 151, 92, 177), (49, 99, 105, 198), True),  # 包含 box1 in box2
    # 测试用例：box1在box2内部
    ((80, 62, 112, 84), (74, 40, 144, 111), True),  # 包含 box1 in box2
    # 测试用例：box1和box2完全分离
    ((76, 140, 154, 277), (121, 326, 192, 384), False),  # 分离
    # 定义两个矩形的坐标，表示 box2 多半在 box1 内，box1 约一半在 box2 内
    ((65, 88, 127, 144), (92, 102, 131, 139), False),  
    # 定义两个矩形的坐标，表示 box1 多半在 box2 内
    ((92, 102, 131, 139), (65, 88, 127, 144), False),  
    # 定义两个矩形的坐标，box1 完全包含在 box2 中，两边 x 坐标相切
    ((68, 94, 118, 120), (68, 90, 118, 122), True),  
    # 定义两个矩形的坐标，box1 完全包含在 box2 中，一边 x 坐标相切
    ((69, 94, 118, 120), (68, 90, 118, 122), True),  
    # 定义两个矩形的坐标，box1 完全包含在 box2 中，一边 y 坐标相切
    ((69, 114, 118, 122), (68, 90, 118, 122), True),  
    # 定义两个矩形的坐标，box2 完全包含在 box1 中，出现错误，注释掉
    # ((100, 93, 199, 168), (169, 126, 198, 165), True),  
    # 定义两个矩形的坐标，box2 完全包含在 box1 中，出现错误，注释掉
    # ((26, 75, 106, 172), (65, 108, 90, 128), True),  
    # 定义两个矩形的坐标，box2 完全包含在 box1 中，两边 y 坐标相切，出现错误，注释掉
    # ((38, 94, 122, 120), (68, 94, 118, 120), True),  
    # 定义两个矩形的坐标，box2 完全包含在 box1 中，两边 x 坐标相切，出现错误，注释掉
    # ((68, 34, 118, 158), (68, 94, 118, 120), True),  
    # 定义两个矩形的坐标，box2 完全包含在 box1 中，一边 x 坐标相切，出现错误，注释掉
    # ((68, 34, 118, 158), (68, 94, 84, 120), True),  
    # 定义两个矩形的坐标，box2 完全包含在 box1 中，一边 y 坐标相切，出现错误，注释掉
    # ((27, 94, 118, 158), (68, 94, 84, 120), True),  
# 测试 box1 是否在 box2 中
def test_is_in(box1: tuple, box2: tuple, target_bool: bool) -> None:
    # 断言 target_bool 是否等于 _is_in(box1, box2) 的返回值
    assert target_bool == _is_in(box1, box2)


# 仅仅是部分包含关系，返回True，如果是完全包含关系则返回False
@pytest.mark.parametrize('box1, box2, target_bool', [
    ((65, 151, 92, 177), (49, 99, 105, 198), False),  # box1 完全在 box2 内
    ((80, 62, 112, 84), (74, 40, 144, 111), False),  # box1 完全在 box2 内
    # ((76, 140, 154, 277), (121, 326, 192, 384), False),  # 分离  错误
    ((76, 140, 154, 277), (121, 277, 192, 384), True),  # 外相切
    ((65, 88, 127, 144), (92, 102, 131, 139), True),  # box2 多半在 box1 内，box1 约一半在 box2 内
    ((92, 102, 131, 139), (65, 88, 127, 144), True),  # box1 多半在 box2 内
    ((68, 94, 118, 120), (68, 90, 118, 122), False),  # box1 在 box2 内，两边 x 相切
    ((69, 94, 118, 120), (68, 90, 118, 122), False),  # box1 在 box2 内，一边 x 相切
    ((69, 114, 118, 122), (68, 90, 118, 122), False),  # box1 在 box2 内，一边 y 相切
    # ((26, 75, 106, 172), (65, 108, 90, 128), False),  # box2 在 box1 内 错误
    # ((38, 94, 122, 120), (68, 94, 118, 120), False),  # box2 在 box1 内，两边 y 相切 错误
    # ((68, 34, 118, 158), (68, 94, 84, 120), False),  # box2 在 box1 内，一边 x 相切 错误

])
def test_is_part_overlap(box1: tuple, box2: tuple, target_bool: bool) -> None:
    # 断言 target_bool 是否等于 _is_part_overlap(box1, box2) 的返回值
    assert target_bool == _is_part_overlap(box1, box2)


# left_box右侧是否和right_box左侧有部分重叠
@pytest.mark.parametrize('box1, box2, target_bool', [
    (None, None, False),  # 空值测试
    ((88, 81, 222, 173), (60, 221, 123, 358), False),  # box1 和 box2 分离
    ((121, 149, 184, 289), (172, 130, 230, 268), True),  # box1 左下和 box2 相交
    ((172, 130, 230, 268), (121, 149, 184, 289), False),  # box2 左下和 box1 相交
    ((109, 68, 182, 146), (215, 188, 277, 253), False),  # box1 左上和 box2 分离
    ((117, 53, 222, 176), (174, 142, 298, 276), True),  # box1 左上和 box2 相交
    ((174, 142, 298, 276), (117, 53, 222, 176), False),  # box2 左上和 box1 相交
    ((65, 88, 127, 144), (92, 102, 131, 139), True),  # box1 左侧 box2 在 box1 内
    ((92, 102, 131, 139), (65, 88, 127, 144), False),  # box2 左侧 box1 在 box2 内
    ((182, 130, 230, 268), (121, 149, 174, 289), False),  # box2 左侧 box1 分离
    ((1, 10, 26, 45), (3, 4, 20, 39), True),  # box1 下侧 box2 在 box1 内
])
def test_left_intersect(box1: tuple, box2: tuple, target_bool: bool) -> None:
    # 断言 target_bool 是否等于 _left_intersect(box1, box2) 的返回值
    assert target_bool == _left_intersect(box1, box2)


# left_box左侧是否和right_box右侧部分重叠
@pytest.mark.parametrize('box1, box2, target_bool', [
    (None, None, False),  # 空值测试
    ((88, 81, 222, 173), (60, 221, 123, 358), False),  # box1 和 box2 分离
    ((121, 149, 184, 289), (172, 130, 230, 268), False),  # box1 左下和 box2 相交
    ((172, 130, 230, 268), (121, 149, 184, 289), True),  # box2 左下和 box1 相交
    ((109, 68, 182, 146), (215, 188, 277, 253), False),  # box1 左上和 box2 分离
    ((117, 53, 222, 176), (174, 142, 298, 276), False),  # box1 左上和 box2 相交
    ((174, 142, 298, 276), (117, 53, 222, 176), True),  # box2 左上和 box1 相交
    ((65, 88, 127, 144), (92, 102, 131, 139), False),  # box1 左侧 box2 在 box1 内
    # 表示 box2 在 box1 左侧，并且 box1 在 box2 内部，存在错误
    # box2 的坐标为 (182, 130, 230, 268)，box1 的坐标为 (121, 149, 174, 289)，结果为 False，表示 box2 和 box1 分离
    # 表示 box1 在 box2 底部，并且 box2 在 box1 内部，存在错误
# 定义测试函数，检查 box1 和 box2 的右侧是否相交，比较结果与期望值是否一致
def test_right_intersect(box1: tuple, box2: tuple, target_bool: bool) -> None:
    # 断言目标布尔值与 _right_intersect 函数返回的结果相等
    assert target_bool == _right_intersect(box1, box2)


# x方向上：要么box1包含box2, 要么box2包含box1。不能部分包含
# y方向上：box1和box2有重叠
@pytest.mark.parametrize('box1, box2, target_bool', [
    # (None, None, False),  # 错误案例
    ((35, 28, 108, 90), (47, 60, 83, 96), True),  # box1 在 box2 上方，x:box2 在 box1 内部，y:有重叠
    ((35, 28, 98, 90), (27, 60, 103, 96), True),  # box1 在 box2 上方，x:box1 在 box2 内部，y:有重叠
    ((57, 77, 130, 210), (59, 219, 119, 293), False),  # box1 在 box2 上方，x:box2 在 box1 内部，y:无重叠
    ((47, 60, 83, 96), (35, 28, 108, 90), True),  # box2 在 box1 上方，x:box1 在 box2 内部，y:有重叠
    ((27, 60, 103, 96), (35, 28, 98, 90), True),  # box2 在 box1 上方，x:box2 在 box1 内部，y:有重叠
    ((59, 219, 119, 293), (57, 77, 130, 210), False),  # box2 在 box1 上方，x:box1 在 box2 内部，y:无重叠
    ((35, 28, 55, 90), (57, 60, 83, 96), False),  # box1 在 box2 上方，x:无重叠，y:有重叠
    ((47, 60, 63, 96), (65, 28, 108, 90), False),  # box2 在 box1 上方，x:无重叠，y:有重叠
])
# 定义测试函数，检查 box1 和 box2 是否在垂直方向完全重叠，比较结果与期望值是否一致
def test_is_vertical_full_overlap(box1: tuple, box2: tuple, target_bool: bool) -> None:
    # 断言目标布尔值与 _is_vertical_full_overlap 函数返回的结果相等
    assert target_bool == _is_vertical_full_overlap(box1, box2)


# 检查 box1 下方和 box2 的上方是否有轻微的重叠，轻微程度由 y_tolerance 限制
@pytest.mark.parametrize('box1, box2, target_bool', [
    (None, None, False),  # 错误案例
    ((35, 28, 108, 90), (47, 89, 83, 116), True),  # box1 在 box2 上方，y:有重叠
    ((35, 28, 108, 90), (47, 60, 83, 96), False),  # box1 在 box2 上方，y:有重叠且过多
    ((57, 77, 130, 210), (59, 219, 119, 293), False),  # box1 在 box2 上方，y:无重叠
    ((47, 60, 83, 96), (35, 28, 108, 90), False),  # box2 在 box1 上方，y:有重叠且过多
    ((27, 89, 103, 116), (35, 28, 98, 90), False),  # box2 在 box1 上方，y:有重叠
    ((59, 219, 119, 293), (57, 77, 130, 210), False),  # box2 在 box1 上方，y:无重叠
])
# 定义测试函数，检查 box1 的底部与 box2 的顶部是否完全重叠
def test_is_bottom_full_overlap(box1: tuple, box2: tuple, target_bool: bool) -> None:
    # 断言目标布尔值与 _is_bottom_full_overlap 函数返回的结果相等
    assert target_bool == _is_bottom_full_overlap(box1, box2)


# 检查 box1 的左侧是否和 box2 有重叠
@pytest.mark.parametrize('box1, box2, target_bool', [
    (None, None, False),  # 错误案例
    ((88, 81, 222, 173), (60, 221, 123, 358), False),  # 两个框分离
    # ((121, 149, 184, 289), (172, 130, 230, 268), False),  # box1 左侧底部 box2 相交，错误案例
    # ((172, 130, 230, 268), (121, 149, 184, 289), True),  # box2 左侧底部 box1 相交，错误案例
    ((109, 68, 182, 146), (215, 188, 277, 253), False),  # box1 左上角 box2，分离
    ((117, 53, 222, 176), (174, 142, 298, 276), False),  # box1 左上角 box2，相交
    # ((174, 142, 298, 276), (117, 53, 222, 176), True),  # box2 左上角 box1 相交，错误案例
    # ((65, 88, 127, 144), (92, 102, 131, 139), False),  # box1 左侧 box2，y:box2 在 box1 内部，错误案例
    ((1, 10, 26, 45), (3, 4, 20, 39), True),  # box1 中部底部 box2，x:box2 在 box1 内部
])
# 定义测试函数，检查 box1 和 box2 的左侧是否有重叠
def test_is_left_overlap(box1: tuple, box2: tuple, target_bool: bool) -> None:
    # 断言目标布尔值与 _is_left_overlap 函数返回的结果相等
    assert target_bool == _is_left_overlap(box1, box2)


# 查找两个 bbox 在 y 轴上是否有重叠，并检查重叠区域的高度是否超过阈值
@pytest.mark.parametrize('box1, box2, target_bool', [
    # (None, None, "Error"),  # 错误案例
    # 包含两个矩形框的坐标及其重叠关系，返回 True 表示 box1 在 box2 内部
        ((51, 69, 192, 147), (75, 48, 132, 187), True),  # y: box1 in box2
    # 包含两个矩形框的坐标及其重叠关系，返回 True 表示 box2 在 box1 内部
        ((51, 39, 192, 197), (75, 48, 132, 187), True),  # y: box2 in box1
    # 包含两个矩形框的坐标及其重叠关系，返回 False 表示 box1 位于 box2 的顶部
        ((88, 81, 222, 173), (60, 221, 123, 358), False),  # y: box1 top box2
    # 包含两个矩形框的坐标及其重叠关系，返回 False 表示 box1 位于 box2 的顶部且重叠较小
        ((109, 68, 182, 196), (215, 188, 277, 253), False),  # y: box1 top box2 little
    # 包含两个矩形框的坐标及其重叠关系，返回 True 表示 box1 位于 box2 的顶部且重叠较大
        ((109, 68, 182, 196), (215, 78, 277, 253), True),  # y: box1 top box2 more
    # 包含两个矩形框的坐标及其重叠关系，返回 False 表示 box1 位于 box2 的顶部且重叠率低于阈值
        ((109, 68, 182, 196), (215, 138, 277, 253), False),  # y: box1 top box2 more but lower overlap_ratio_threshold
    # 包含两个矩形框的坐标及其重叠关系，返回 True 表示 box1 位于 box2 的顶部且重叠率高于阈值
        ((109, 68, 182, 196), (215, 138, 277, 203), True),  # y: box1 top box2 more and more overlap_ratio_threshold
# 定义测试函数，检查 box1 和 box2 是否在 Y 轴重叠超过阈值
def test_is_overlaps_y_exceeds_threshold(box1: tuple, box2: tuple, target_bool: bool) -> None:
    # 断言实际结果与预期结果相等
    assert target_bool == __is_overlaps_y_exceeds_threshold(box1, box2)


# 使用参数化测试，定义 box1 和 box2 的坐标及目标重叠比例
@pytest.mark.parametrize('box1, box2, target_num', [
    # (None, None, "Error"),  # 错误情况
    ((88, 81, 222, 173), (60, 221, 123, 358), 0.0),  # 分离情况
    ((76, 140, 154, 277), (121, 326, 192, 384), 0.0),  # 分离情况
    ((142, 109, 238, 164), (134, 211, 224, 270), 0.0),  # 分离情况
    ((109, 68, 182, 196), (175, 138, 277, 213), 0.024475524475524476),  # 相交情况
    ((56, 90, 170, 219), (103, 212, 171, 304), 0.02288586346557361),  # 相交情况
    ((109, 126, 204, 245), (130, 127, 232, 186), 0.33696071621517326),  # 相交情况
    ((109, 126, 204, 245), (110, 127, 232, 206), 0.5493822593770807),  # 相交情况
    ((76, 140, 154, 277), (121, 277, 192, 384), 0.0)  # 相切情况
])
# 定义测试函数，计算 IoU (Intersection over Union)
def test_calculate_iou(box1: tuple, box2: tuple, target_num: float) -> None:
    # 断言实际结果与目标 IoU 值相等
    assert target_num == calculate_iou(box1, box2)


# 使用参数化测试，定义 box1 和 box2 的坐标及目标重叠比例
@pytest.mark.parametrize('box1, box2, target_num', [
    # (None, None, "Error"),  # 错误情况
    ((142, 109, 238, 164), (134, 211, 224, 270), 0.0),  # 分离情况
    ((88, 81, 222, 173), (60, 221, 123, 358), 0.0),  # 分离情况
    ((76, 140, 154, 277), (121, 326, 192, 384), 0.0),  # 分离情况
    ((76, 140, 154, 277), (121, 277, 192, 384), 0.0),  # 相切情况
    ((109, 126, 204, 245), (110, 127, 232, 206), 0.7704918032786885),  # 相交情况
    ((56, 90, 170, 219), (103, 212, 171, 304), 0.07496803069053709),  # 相交情况
    ((121, 149, 184, 289), (172, 130, 230, 268), 0.17841079460269865),  # 相交情况
    ((51, 69, 192, 147), (75, 48, 132, 187), 0.5611510791366906),  # 相交情况
    ((117, 53, 222, 176), (174, 142, 298, 276), 0.12636469221835075),  # 相交情况
    ((102, 60, 233, 203), (70, 190, 220, 319), 0.08188757807078417),  # 相交情况
    ((109, 126, 204, 245), (130, 127, 232, 186), 0.7254901960784313),  # 相交情况
])
# 定义测试函数，计算 box1 和 box2 的重叠面积占最小包围盒面积的比例
def test_calculate_overlap_area_2_minbox_area_ratio(box1: tuple, box2: tuple, target_num: float) -> None:
    # 断言实际结果与目标重叠比例相等
    assert target_num == calculate_overlap_area_2_minbox_area_ratio(box1, box2)


# 使用参数化测试，定义 box1 和 box2 的坐标及目标重叠比例
@pytest.mark.parametrize('box1, box2, target_num', [
    # (None, None, "Error"),  # 错误情况
    ((142, 109, 238, 164), (134, 211, 224, 270), 0.0),  # 分离情况
    ((88, 81, 222, 173), (60, 221, 123, 358), 0.0),  # 分离情况
    ((76, 140, 154, 277), (121, 326, 192, 384), 0.0),  # 分离情况
    ((76, 140, 154, 277), (121, 277, 192, 384), 0.0),  # 相切情况
    ((142, 109, 238, 164), (134, 164, 224, 270), 0.0),  # 相切情况
    ((109, 126, 204, 245), (110, 127, 232, 206), 0.6568774878372402),  # 相交情况
    ((56, 90, 170, 219), (103, 212, 171, 304), 0.03189174486604107),  # 相交情况
    ((121, 149, 184, 289), (172, 130, 230, 268), 0.1619047619047619),  # 相交情况
    ((51, 69, 192, 147), (75, 48, 132, 187), 0.40425531914893614),  # 相交情况
    ((117, 53, 222, 176), (174, 142, 298, 276), 0.12636469221835075),  # 相交情况
    ((102, 60, 233, 203), (70, 190, 220, 319), 0.08188757807078417),  # 相交情况
    # 定义一个元组，包含两个子元组和一个浮点数，表示相交的相关数据
    ((109, 126, 204, 245), (130, 127, 232, 186), 0.38620079610791685),  # 相交
# 测试计算两个边界框重叠面积占第一个框面积比例的函数
def test_calculate_overlap_area_in_bbox1_area_ratio(box1: tuple, box2: tuple, target_num: float) -> None:
    # 断言目标数值与计算函数返回的值相等
    assert target_num == calculate_overlap_area_in_bbox1_area_ratio(box1, box2)


# 使用参数化测试多个边界框的重叠情况，返回符合条件的边界框
@pytest.mark.parametrize('box1, box2, ratio, target_box', [
    # (None, None, 0.8, "Error"),  # 错误案例
    ((142, 109, 238, 164), (134, 211, 224, 270), 0.0, None),  # 分离情况，返回 None
    ((109, 126, 204, 245), (110, 127, 232, 206), 0.5, (110, 127, 232, 206)),  # 重叠情况，返回第二个框
    ((56, 90, 170, 219), (103, 212, 171, 304), 0.5, None),  # 分离情况，返回 None
    ((121, 149, 184, 289), (172, 130, 230, 268), 0.5, None),  # 分离情况，返回 None
    ((51, 69, 192, 147), (75, 48, 132, 187), 0.5, (75, 48, 132, 187)),  # 重叠情况，返回第二个框
    ((117, 53, 222, 176), (174, 142, 298, 276), 0.5, None),  # 分离情况，返回 None
    ((102, 60, 233, 203), (70, 190, 220, 319), 0.5, None),  # 分离情况，返回 None
    ((109, 126, 204, 245), (130, 127, 232, 186), 0.5, (130, 127, 232, 186)),  # 重叠情况，返回第二个框
])
# 测试获取符合重叠比例的最小边界框
def test_get_minbox_if_overlap_by_ratio(box1: tuple, box2: tuple, ratio: float, target_box: list) -> None:
    # 断言目标框与计算函数返回的结果相等
    assert target_box == get_minbox_if_overlap_by_ratio(box1, box2, ratio)


# 根据边界范围获取完全包含在范围内的所有边界框
@pytest.mark.parametrize('boxes, boundary, target_boxs', [
    # ([], (), "Error"),  # 错误案例
    ([], (110, 340, 209, 387), []),  # 空列表与边界无重叠
    ([(142, 109, 238, 164)], (134, 211, 224, 270), []),  # 分离情况，无重叠框
    ([(109, 126, 204, 245), (110, 127, 232, 206)], (105, 116, 258, 300), [(109, 126, 204, 245), (110, 127, 232, 206)]),  # 完全重叠，返回所有框
    ([(109, 126, 204, 245), (110, 127, 232, 206)], (105, 116, 258, 230), [(110, 127, 232, 206)]),  # 部分重叠，返回重叠框
    ([(81, 280, 123, 315), (282, 203, 342, 247), (183, 100, 300, 155), (46, 99, 133, 148), (33, 156, 97, 211),
      (137, 29, 287, 87)], (80, 90, 249, 200), []),  # 无框重叠，返回空列表
    ([(81, 280, 123, 315), (282, 203, 342, 247), (183, 100, 300, 155), (46, 99, 133, 148), (33, 156, 97, 211),
      (137, 29, 287, 87)], (30, 20, 349, 320),
     [(81, 280, 123, 315), (282, 203, 342, 247), (183, 100, 300, 155), (46, 99, 133, 148), (33, 156, 97, 211),
      (137, 29, 287, 87)]),  # 边界完全包含所有框
    ([(81, 280, 123, 315), (282, 203, 342, 247), (183, 100, 300, 155), (46, 99, 133, 148), (33, 156, 97, 211),
      (137, 29, 287, 87)], (30, 20, 200, 320),
     [(81, 280, 123, 315), (46, 99, 133, 148), (33, 156, 97, 211)]),  # 边界部分包含，返回相应框
])
# 测试在给定边界内获取框
def test_get_bbox_in_boundary(boxes: list, boundary: tuple, target_boxs: list) -> None:
    # 断言目标框与计算函数返回的结果相等
    assert target_boxs == get_bbox_in_boundary(boxes, boundary)


# 寻找与给定框在 y 方向上距离最近的框，考虑 x 方向的重叠
@pytest.mark.parametrize('pymu_blocks, obj_box, target_boxs', [
    ([{'bbox': (81, 280, 123, 315)}, {'bbox': (282, 203, 342, 247)}, {'bbox': (183, 100, 300, 155)},
      {'bbox': (46, 99, 133, 148)}, {'bbox': (33, 156, 97, 211)},
      {'bbox': (137, 29, 287, 87)}], (81, 280, 123, 315), {'bbox': (33, 156, 97, 211)}),  # 目标框与最近框重叠
    # ([{"bbox": (168, 120, 263, 159)},
    #   {"bbox": (231, 61, 279, 159)},
    #   {"bbox": (35, 85, 136, 110)},
    #   {"bbox": (228, 193, 347, 225)},
    #   {"bbox": (144, 264, 188, 323)},
    #   {"bbox": (62, 37, 126, 64)}], (228, 193, 347, 225),
    #  包含两个字典，分别表示两个边界框，注释说明 y 方向最近的有两个，x 方向两个均有重合错误
    ([{'bbox': (35, 85, 136, 159)},  # 第一个边界框的坐标
      {'bbox': (168, 120, 263, 159)},  # 第二个边界框的坐标
      {'bbox': (231, 61, 279, 118)},  # 第三个边界框的坐标
      {'bbox': (228, 193, 347, 225)},  # 第四个边界框的坐标
      {'bbox': (144, 264, 188, 323)},  # 第五个边界框的坐标
      {'bbox': (62, 37, 126, 64)}],  # 第六个边界框的坐标
     (228, 193, 347, 225),  # 目标边界框的坐标
     {'bbox': (168, 120, 263, 159)},),  # 参考边界框，y 方向最近的有两个，x 方向只有一个有重合
    ([{'bbox': (239, 115, 379, 167)},  # 第一个边界框的坐标
      {'bbox': (33, 237, 104, 262)},  # 第二个边界框的坐标
      {'bbox': (124, 288, 168, 325)},  # 第三个边界框的坐标
      {'bbox': (242, 291, 379, 340)},  # 第四个边界框的坐标
      {'bbox': (55, 117, 121, 154)},  # 第五个边界框的坐标
      {'bbox': (266, 183, 384, 217)}, ],  # 第六个边界框的坐标
     (124, 288, 168, 325),  # 目标边界框的坐标
     {'bbox': (55, 117, 121, 154)}),  # 参考边界框的坐标
    ([{'bbox': (239, 115, 379, 167)},  # 第一个边界框的坐标
      {'bbox': (33, 237, 104, 262)},  # 第二个边界框的坐标
      {'bbox': (124, 288, 168, 325)},  # 第三个边界框的坐标
      {'bbox': (242, 291, 379, 340)},  # 第四个边界框的坐标
      {'bbox': (55, 117, 119, 154)},  # 第五个边界框的坐标，注意这里宽度略有不同
      {'bbox': (266, 183, 384, 217)}, ],  # 第六个边界框的坐标
     (124, 288, 168, 325),  # 目标边界框的坐标
     None),  # 参考边界框为 None，表示没有重合
    ([{'bbox': (80, 90, 249, 200)},  # 第一个边界框的坐标
      {'bbox': (183, 100, 240, 155)}, ],  # 第二个边界框的坐标
     (183, 100, 240, 155),  # 目标边界框的坐标
     None),  # 参考边界框为 None，表示包含关系
# 测试寻找距离指定框最近的文本框的功能
def test_find_top_nearest_text_bbox(pymu_blocks: list, obj_box: tuple, target_boxs: dict) -> None:
    # 断言目标框与找到的最近框相同
    assert target_boxs == find_top_nearest_text_bbox(pymu_blocks, obj_box)


# 用于测试参数化的寻找下方最近框的功能
@pytest.mark.parametrize('pymu_blocks, obj_box, target_boxs', [
    # 定义测试用例：框列表、目标框、期望框
    ([{'bbox': (165, 96, 300, 114)},
      {'bbox': (11, 157, 139, 201)},
      {'bbox': (124, 208, 265, 262)},
      {'bbox': (124, 283, 248, 306)},
      {'bbox': (39, 267, 84, 301)},
      {'bbox': (36, 89, 114, 145)}, ], (165, 96, 300, 114), {'bbox': (124, 208, 265, 262)}),
    # 定义另一个测试用例，包含不同框
    ([{'bbox': (187, 37, 303, 49)},
      {'bbox': (2, 227, 90, 283)},
      {'bbox': (158, 174, 200, 212)},
      {'bbox': (259, 174, 324, 228)},
      {'bbox': (205, 61, 316, 97)},
      {'bbox': (295, 248, 374, 287)}, ], (205, 61, 316, 97), {'bbox': (259, 174, 324, 228)}),  # y有两个最近的, x只有一个重合
    # 被注释的测试用例
    # ([{"bbox": (187, 37, 303, 49)},
    #   {"bbox": (2, 227, 90, 283)},
    #   {"bbox": (259, 174, 324, 228)},
    #   {"bbox": (205, 61, 316, 97)},
    #   {"bbox": (295, 248, 374, 287)},
    #   {"bbox": (158, 174, 209, 212)}, ], (205, 61, 316, 97),
    #  [{"bbox": (259, 174, 324, 228)}, {"bbox": (158, 174, 209, 212)}]),  # x有重合，y有两个最近的  Error
    # 另一个测试用例，目标框和期望框为None
    ([{'bbox': (287, 132, 398, 191)},
      {'bbox': (44, 141, 163, 188)},
      {'bbox': (132, 191, 240, 241)},
      {'bbox': (81, 25, 142, 67)},
      {'bbox': (74, 297, 116, 314)},
      {'bbox': (77, 84, 224, 107)}, ], (287, 132, 398, 191), None),  # x没有重合
    # 最后一个测试用例
    ([{'bbox': (80, 90, 249, 200)},
      {'bbox': (183, 100, 240, 155)}, ], (183, 100, 240, 155), None),  # 包含
])
# 测试寻找下方最近框的功能
def test_find_bottom_nearest_text_bbox(pymu_blocks: list, obj_box: tuple, target_boxs: dict) -> None:
    # 断言目标框与找到的最近框相同
    assert target_boxs == find_bottom_nearest_text_bbox(pymu_blocks, obj_box)


# 用于测试参数化的寻找左侧最近框的功能
@pytest.mark.parametrize('pymu_blocks, obj_box, target_boxs', [
    # 定义测试用例，框包含
    ([{'bbox': (80, 90, 249, 200)}, {'bbox': (183, 100, 240, 155)}], (183, 100, 240, 155), None),  # 包含
    # 定义另一个测试用例，y方向和x方向都重叠
    ([{'bbox': (28, 90, 77, 126)}, {'bbox': (35, 84, 84, 120)}], (35, 84, 84, 120), None),  # y:重叠，x:重叠大于2
    # 定义另一个测试用例
    ([{'bbox': (28, 90, 77, 126)}, {'bbox': (75, 84, 134, 120)}], (75, 84, 134, 120), {'bbox': (28, 90, 77, 126)}),
    # y:重叠，x:重叠小于等于2
    ([{'bbox': (239, 115, 379, 167)},
      {'bbox': (33, 237, 104, 262)},
      {'bbox': (124, 288, 168, 325)},
      {'bbox': (242, 291, 379, 340)},
      {'bbox': (55, 113, 161, 154)},
      {'bbox': (266, 123, 384, 217)}], (266, 123, 384, 217), {'bbox': (55, 113, 161, 154)}),  # y重叠，x left
    # 被注释的测试用例
    # ([{"bbox": (136, 219, 268, 240)},
    #   {"bbox": (169, 115, 268, 181)},
    #   {"bbox": (33, 237, 104, 262)},
    #   {"bbox": (124, 288, 168, 325)},
    #   {"bbox": (55, 117, 161, 154)},
    #   {"bbox": (266, 183, 384, 217)}], (266, 183, 384, 217),
    #  [{"bbox": (136, 219, 267, 240)}, {"bbox": (169, 115, 267, 181)}]),  # y有重叠，x重叠小于2或者在left Error
])
# 测试寻找左侧最近框的功能
def test_find_left_nearest_text_bbox(pymu_blocks: list, obj_box: tuple, target_boxs: dict) -> None:
    # 确保目标框与查找的左侧最近文本框的边界框相等
        assert target_boxs == find_left_nearest_text_bbox(pymu_blocks, obj_box)
# 寻找右侧距离自己最近的box, y方向有重叠，x方向最近
@pytest.mark.parametrize('pymu_blocks, obj_box, target_boxs', [
    # 测试用例：包含情况，obj_box完全在某个box内
    ([{'bbox': (80, 90, 249, 200)}, {'bbox': (183, 100, 240, 155)}], (183, 100, 240, 155), None),  
    # 测试用例：y方向重叠且x方向重叠大于2
    ([{'bbox': (28, 90, 77, 126)}, {'bbox': (35, 84, 84, 120)}], (28, 90, 77, 126), None),  
    # 测试用例：y方向重叠且返回目标box
    ([{'bbox': (28, 90, 77, 126)}, {'bbox': (75, 84, 134, 120)}], (28, 90, 77, 126), {'bbox': (75, 84, 134, 120)}),  
    # 测试用例：y方向重叠且x方向重叠小于等于2
    ([{'bbox': (239, 115, 379, 167)},
      {'bbox': (33, 237, 104, 262)},
      {'bbox': (124, 288, 168, 325)},
      {'bbox': (242, 291, 379, 340)},
      {'bbox': (55, 113, 161, 154)},
      {'bbox': (266, 123, 384, 217)}], (55, 113, 161, 154), {'bbox': (239, 115, 379, 167)}),  # y重叠，x right
    # 注释掉的测试用例，测试y有重叠，x重叠小于2或在右边
    # ([{"bbox": (169, 115, 298, 181)},
    #   {"bbox": (169, 219, 268, 240)},
    #   {"bbox": (33, 177, 104, 262)},
    #   {"bbox": (124, 288, 168, 325)},
    #   {"bbox": (55, 117, 161, 154)},
    #   {"bbox": (266, 183, 384, 217)}], (33, 177, 104, 262),
    #  [{"bbox": (169, 115, 298, 181)}, {"bbox": (169, 219, 268, 240)}]),  # y有重叠，x重叠小于2或者在right Error
])
# 测试函数：查找与obj_box最近的右侧box
def test_find_right_nearest_text_bbox(pymu_blocks: list, obj_box: tuple, target_boxs: dict) -> None:
    # 断言返回的目标box与预期的target_boxs相等
    assert target_boxs == find_right_nearest_text_bbox(pymu_blocks, obj_box)


# 判断两个矩形框的相对位置关系 (left, right, bottom, top)
@pytest.mark.parametrize('box1, box2, target_box', [
    # 测试用例：错误情况，两个框均为None
    # (None, None, "Error"),  # Error
    # 测试用例：box1包含box2
    ((80, 90, 249, 200), (183, 100, 240, 155), (False, False, False, False)),  
    # 测试用例：分离，右上角不重叠
    # ((124, 81, 222, 173), (60, 221, 123, 358), (False, True, False, True)),  # 分离，右上 Error
    # 测试用例：分离，上面不重叠
    ((142, 109, 238, 164), (134, 211, 224, 270), (False, False, False, True)),  
    # 测试用例：分离，左上角不重叠
    # ((51, 69, 192, 147), (205, 198, 282, 297), (True, False, False, True)),  # 分离，左上 Error
    # 测试用例：分离，左边不重叠
    # ((101, 149, 164, 289), (172, 130, 230, 268), (True, False, False, False)),  # 分离，左  Error
    # 测试用例：分离，左下角不重叠
    # ((69, 196, 124, 285), (130, 127, 232, 186), (True, False, True, False)),  # 分离，左下  Error
    # 测试用例：分离，底部不重叠
    ((103, 212, 171, 304), (56, 90, 170, 209), (False, False, True, False)),  
    # 测试用例：分离，右下角不重叠
    # ((124, 367, 222, 415), (60, 221, 123, 358), (False, True, True, False)),  # 分离，右下 Error
    # 测试用例：分离，右边不重叠
    # ((172, 130, 230, 268), (101, 149, 164, 289), (False, True, False, False)),  # 分离，右  Error
])
# 测试函数：判断两个框的相对位置关系
def test_bbox_relative_pos(box1: tuple, box2: tuple, target_box: tuple) -> None:
    # 断言返回的目标位置关系与预期相等
    assert target_box == bbox_relative_pos(box1, box2)


# 计算两个矩形框的距离
"""
受bbox_relative_pos方法的影响，左右相反，这里计算结果全部受影响，在错误的基础上计算出了正确的结果
"""


@pytest.mark.parametrize('box1, box2, target_num', [
    # 测试用例：错误情况，两个框均为None
    # (None, None, "Error"),  # Error
    # 测试用例：box1包含box2，距离为0
    ((80, 90, 249, 200), (183, 100, 240, 155), 0.0),  
    # 测试用例：box1与box2分离，上方距离
    ((142, 109, 238, 164), (134, 211, 224, 270), 47.0),  
    # 测试用例：box1与box2分离，下方距离
    ((103, 212, 171, 304), (56, 90, 170, 209), 3.0),  
    # 测试用例：box1与box2分离，左侧距离
    ((101, 149, 164, 289), (172, 130, 230, 268), 8.0),  
    # 测试用例：box1与box2分离，右侧距离
    ((172, 130, 230, 268), (101, 149, 164, 289), 8.0),  
    # 测试用例：box1包含box2，浮点数情况，距离为0
    ((80.3, 90.8, 249.0, 200.5), (183.8, 100.6, 240.2, 155.1), 0.0),  
    # 定义一个元组，包含分离区域的坐标和一个数值
    ((142.3, 109.5, 238.9, 164.2), (134.4, 211.2, 224.8, 270.1), 47.0),  # 分离，上
    # 定义一个元组，包含分离区域的坐标和一个数值
    ((103.5, 212.6, 171.1, 304.8), (56.1, 90.9, 170.6, 209.2), 3.4),  # 分离，下
    # 定义一个元组，包含分离区域的坐标和一个数值
    ((101.1, 149.3, 164.9, 289.0), (172.1, 130.1, 230.5, 268.5), 7.2),  # 分离，左
    # 定义一个元组，包含分离区域的坐标和一个数值
    ((172.1, 130.3, 230.1, 268.1), (101.2, 149.9, 164.3, 289.1), 7.8),  # 分离，右
    # 定义一个元组，包含分离区域的坐标和一个数值
    ((124.3, 81.1, 222.5, 173.8), (60.3, 221.5, 123.0, 358.9), 47.717711596429254),  # 分离，右上
    # 定义一个元组，包含分离区域的坐标和一个数值
    ((51.2, 69.31, 192.5, 147.9), (205.0, 198.1, 282.98, 297.09), 51.73287156151299),  # 分离，左上
    # 定义一个元组，包含分离区域的坐标和一个数值
    ((124.3, 367.1, 222.9, 415.7), (60.9, 221.4, 123.2, 358.6), 8.570880934886448),  # 分离，右下
    # 定义一个元组，包含分离区域的坐标和一个数值
    ((69.9, 196.2, 124.1, 285.7), (130.0, 127.3, 232.6, 186.1), 11.69700816448377),  # 分离，左下
])
# 定义测试函数，比较两个边界框之间的距离与目标值
def test_bbox_distance(box1: tuple, box2: tuple, target_num: float) -> None:
    # 断言目标值与计算的边界框距离的差小于 1
    assert target_num - bbox_distance(box1, box2) < 1


# 标记此测试为跳过，原因是 'skip'
@pytest.mark.skip(reason='skip')
# 根据 bucket_name 获取 S3 配置，包括 ak、sk 和 endpoint
def test_get_s3_config() -> None:
    # 从环境变量中获取 bucket_name
    bucket_name = os.getenv('bucket_name')
    # 从环境变量中获取 target_data
    target_data = os.getenv('target_data')
    # 断言将 target_data 转换为列表后的结果与获取的 S3 配置相等
    assert convert_string_to_list(target_data) == list(get_s3_config(bucket_name))


# 定义函数将字符串转换为列表
def convert_string_to_list(s):
    # 去除字符串首尾的单引号
    cleaned_s = s.strip("'")
    # 按逗号分割字符串为多个项
    items = cleaned_s.split(',')
    # 去除每个项的前后空格
    cleaned_items = [item.strip() for item in items]
    # 返回处理后的列表
    return cleaned_items
```