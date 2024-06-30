# `D:\src\scipysrc\sympy\sympy\parsing\tests\test_autolev.py`

```
# 导入标准库中的 os 模块
import os

# 从 sympy 库中导入特定函数和模块
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.parsing.autolev import parse_autolev

# 导入 antlr4 模块，并且如果导入失败，则将 disabled 标志设为 True
antlr4 = import_module("antlr4")
if not antlr4:
    disabled = True

# 获取当前脚本文件所在目录的上两级目录路径
FILE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
)


def _test_examples(in_filename, out_filename, test_name=""):
    # 构建输入文件的完整路径
    in_file_path = os.path.join(FILE_DIR, 'autolev', 'test-examples',
                                in_filename)
    # 构建正确输出文件的完整路径
    correct_file_path = os.path.join(FILE_DIR, 'autolev', 'test-examples',
                                     out_filename)
    # 打开输入文件，使用 parse_autolev 函数解析文件内容并生成代码
    with open(in_file_path) as f:
        generated_code = parse_autolev(f, include_numeric=True)

    # 打开正确输出文件，并逐行比较生成的代码与正确的代码是否一致
    with open(correct_file_path) as f:
        for idx, line1 in enumerate(f):
            # 如果遇到以 "#" 开头的行，则停止比较
            if line1.startswith("#"):
                break
            try:
                # 获取生成代码的第 idx 行并去除末尾的空白字符
                line2 = generated_code.split('\n')[idx]
                # 断言生成代码的当前行与正确代码的当前行相等
                assert line1.rstrip() == line2.rstrip()
            except Exception:
                # 如果断言失败，则抛出 AssertionError，指明行号
                msg = 'mismatch in ' + test_name + ' in line no: {0}'
                raise AssertionError(msg.format(idx+1))


def test_rule_tests():
    # 定义测试文件列表
    l = ["ruletest1", "ruletest2", "ruletest3", "ruletest4", "ruletest5",
         "ruletest6", "ruletest7", "ruletest8", "ruletest9", "ruletest10",
         "ruletest11", "ruletest12"]

    # 对于列表中的每个文件，构建输入和输出文件的路径，然后执行测试
    for i in l:
        in_filepath = i + ".al"
        out_filepath = i + ".py"
        _test_examples(in_filepath, out_filepath, i)


def test_pydy_examples():
    # 定义示例文件列表
    l = ["mass_spring_damper", "chaos_pendulum", "double_pendulum",
         "non_min_pendulum"]

    # 对于列表中的每个文件，构建输入和输出文件的路径，然后执行测试
    for i in l:
        in_filepath = os.path.join("pydy-example-repo", i + ".al")
        out_filepath = os.path.join("pydy-example-repo", i + ".py")
        _test_examples(in_filepath, out_filepath, i)


def test_autolev_tutorial():
    # 构建 Autolev 教程文件夹的路径
    dir_path = os.path.join(FILE_DIR, 'autolev', 'test-examples',
                            'autolev-tutorial')

    # 如果路径存在，则对指定文件执行测试
    if os.path.isdir(dir_path):
        l = ["tutor1", "tutor2", "tutor3", "tutor4", "tutor5", "tutor6",
             "tutor7"]
        for i in l:
            in_filepath = os.path.join("autolev-tutorial", i + ".al")
            out_filepath = os.path.join("autolev-tutorial", i + ".py")
            _test_examples(in_filepath, out_filepath, i)


def test_dynamics_online():
    # 构建在线动力学文件夹的路径
    dir_path = os.path.join(FILE_DIR, 'autolev', 'test-examples',
                            'dynamics-online')
    # 如果给定的路径是一个目录
    if os.path.isdir(dir_path):
        # 定义不同章节的子目录和文件名后缀列表
        ch1 = ["1-4", "1-5", "1-6", "1-7", "1-8", "1-9_1", "1-9_2", "1-9_3"]
        ch2 = ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9",
               "circular"]
        ch3 = ["3-1_1", "3-1_2", "3-2_1", "3-2_2", "3-2_3", "3-2_4", "3-2_5",
               "3-3"]
        ch4 = ["4-1_1", "4-2_1", "4-4_1", "4-4_2", "4-5_1", "4-5_2"]
        
        # 将章节名称和对应的列表组成元组列表
        chapters = [(ch1, "ch1"), (ch2, "ch2"), (ch3, "ch3"), (ch4, "ch4")]
        
        # 遍历章节列表
        for ch, name in chapters:
            # 遍历每个章节下的文件名后缀列表
            for i in ch:
                # 构建输入文件路径和输出文件路径
                in_filepath = os.path.join("dynamics-online", name, i + ".al")
                out_filepath = os.path.join("dynamics-online", name, i + ".py")
                # 调用 _test_examples 函数进行测试，传入输入和输出文件路径以及文件名后缀
                _test_examples(in_filepath, out_filepath, i)
# 定义测试函数，用于验证 Autolev 示例的计算结果是否正确
def test_output_01():
    """Autolev example calculates the position, velocity, and acceleration of a
    point and expresses in a single reference frame.
    
    Autolev 示例计算点的位置、速度和加速度，并将其表达在单一参考坐标系中。
    """

    # 检查是否导入了 antlr4 模块，如果没有则跳过测试
    if not antlr4:
        skip('Test skipped: antlr4 is not installed.')

    # 定义 Autolev 示例输入
    autolev_input = """\
FRAMES C,D,F
VARIABLES FD'',DC''
CONSTANTS R,L
POINTS O,E
SIMPROT(F,D,1,FD)
SIMPROT(D,C,2,DC)
W_C_F>=EXPRESS(W_C_F>,F)
P_O_E>=R*D2>-L*C1>
P_O_E>=EXPRESS(P_O_E>,D)
V_E_F>=EXPRESS(DT(P_O_E>,F),D)
A_E_F>=EXPRESS(DT(V_E_F>,F),D)\
"""

    # 解析 Autolev 输入为 SymPy 表达式
    sympy_input = parse_autolev(autolev_input)

    # 定义全局和局部变量字典，执行 Autolev 输入的代码
    g = {}
    l = {}
    exec(sympy_input, g, l)

    # 计算角速度 W_C_F>
    w_c_f = l['frame_c'].ang_vel_in(l['frame_f'])
    # 计算位置 P_O_E>
    p_o_e = l['point_e'].pos_from(l['point_o'])
    # 计算速度 V_E_F>
    v_e_f = l['point_e'].vel(l['frame_f'])
    # 计算加速度 A_E_F>
    a_e_f = l['point_e'].acc(l['frame_f'])

    # 断言：验证计算结果与预期值相等
    # 检查计算得到的角速度与预期的角速度是否相等
    expected_w_c_f = (l['fd'].diff()*l['frame_f'].x +
                      cos(l['fd'])*l['dc'].diff()*l['frame_f'].y +
                      sin(l['fd'])*l['dc'].diff()*l['frame_f'].z)
    assert (w_c_f - expected_w_c_f).simplify() == 0

    # 检查计算得到的位置与预期的位置是否相等
    expected_p_o_e = (-l['l']*cos(l['dc'])*l['frame_d'].x +
                      l['r']*l['frame_d'].y +
                      l['l']*sin(l['dc'])*l['frame_d'].z)
    assert (p_o_e - expected_p_o_e).simplify() == 0

    # 检查计算得到的速度与预期的速度是否相等
    expected_v_e_f = (l['l']*sin(l['dc'])*l['dc'].diff()*l['frame_d'].x -
                      l['l']*sin(l['dc'])*l['fd'].diff()*l['frame_d'].y +
                      (l['r']*l['fd'].diff() +
                       l['l']*cos(l['dc'])*l['dc'].diff())*l['frame_d'].z)
    assert (v_e_f - expected_v_e_f).simplify() == 0
    # 计算期望的加速度-欧拉参数-框架导数（a_e_f）并与预期值（expected_a_e_f）进行比较
    expected_a_e_f = (l['l']*(cos(l['dc'])*l['dc'].diff()**2 +
                              sin(l['dc'])*l['dc'].diff().diff())*l['frame_d'].x +
                      (-l['r']*l['fd'].diff()**2 -
                       2*l['l']*cos(l['dc'])*l['dc'].diff()*l['fd'].diff() -
                       l['l']*sin(l['dc'])*l['fd'].diff().diff())*l['frame_d'].y +
                      (l['r']*l['fd'].diff().diff() +
                       l['l']*cos(l['dc'])*l['dc'].diff().diff() -
                       l['l']*sin(l['dc'])*l['dc'].diff()**2 -
                       l['l']*sin(l['dc'])*l['fd'].diff()**2)*l['frame_d'].z)
    # 使用简化（simplify）方法检查计算出的值与预期值的差是否为零，并使用断言进行验证
    assert (a_e_f - expected_a_e_f).simplify() == 0
```