# `arknights-mower\menu.py`

```py
# 导入 json 模块
import json
# 从 multiprocessing 模块中导入 Pipe, Process, freeze_support
from multiprocessing import Pipe, Process, freeze_support
# 导入 time 模块
import time
# 导入 PySimpleGUI 模块，并使用别名 sg
import PySimpleGUI as sg
# 导入 os 模块
import os
# 从 ruamel.yaml 模块中导入 YAML
from ruamel.yaml import YAML

# 从 arknights_mower.utils.conf 模块中导入 load_conf, save_conf, load_plan, write_plan
from arknights_mower.utils.conf import load_conf, save_conf, load_plan, write_plan
# 从 arknights_mower.utils.log 模块中导入 logger
from arknights_mower.utils.log import logger
# 从 arknights_mower.data 模块中导入 agent_list
from arknights_mower.data import agent_list
# 从 arknights_mower.__main__ 模块中导入 main
from arknights_mower.__main__ import main
# 从 arknights_mower.__init__ 模块中导入 __version__
from arknights_mower.__init__ import __version__
# 从 arknights_mower.utils.update 模块中导入 compere_version, update_version, download_version
from arknights_mower.utils.update import compere_version, update_version, download_version

# 创建 YAML 对象
yaml = YAML()
# 初始化 conf 变量为空字典
conf = {}
# 初始化 plan 变量为空字典
plan = {}
# 初始化 current_plan 变量为空字典
current_plan = {}
# 初始化 operators 变量为空字典
operators = {}
# 初始化全局变量 window
global window
# 初始化 buffer 变量为空字符串
buffer = ''
# 初始化 line 变量为 0
line = 0
# 初始化 half_line_index 变量为 0
half_line_index = 0

# 定义一个函数 build_plan，参数为 url
def build_plan(url):
    # 声明全局变量 plan
    global plan
    # 声明全局变量 current_plan
    global current_plan
    # 尝试加载计划文件
    try:
        plan = load_plan(url)
        # 获取默认计划
        current_plan = plan[plan['default']]
        # 设置配置中的计划文件路径
        conf['planFile'] = url
        # 循环遍历3x3的房间按钮，更新按钮文本和颜色
        for i in range(1, 4):
            for j in range(1, 4):
                window[f'btn_room_{str(i)}_{str(j)}'].update('待建造', button_color=('white', '#4f4945'))
        # 遍历当前计划的键值对
        for key in current_plan:
            # 如果值的类型为列表，则转换为新格式的字典
            if type(current_plan[key]).__name__ == 'list':  # 兼容旧版格式
                current_plan[key] = {'plans': current_plan[key], 'name': ''}
            # 根据不同的建筑类型更新按钮文本和颜色
            elif current_plan[key]['name'] == '贸易站':
                window['btn_' + key].update('贸易站', button_color=('#4f4945', '#33ccff'))
            elif current_plan[key]['name'] == '制造站':
                window['btn_' + key].update('制造站', button_color=('#4f4945', '#ffcc00'))
            elif current_plan[key]['name'] == '发电站':
                window['btn_' + key].update('发电站', button_color=('#4f4945', '#ccff66'))
        # 更新窗口中的各个配置项
        window['plan_radio_ling_xi_' + str(plan['conf']['ling_xi'])].update(True)
        window['plan_int_max_resting_count'].update(plan['conf']['max_resting_count'])
        window['plan_conf_exhaust_require'].update(plan['conf']['exhaust_require'])
        window['plan_conf_workaholic'].update(plan['conf']['workaholic'])
        window['plan_conf_rest_in_full'].update(plan['conf']['rest_in_full'])
        window['plan_conf_resting_priority'].update(plan['conf']['resting_priority'])
    # 捕获异常并记录错误信息
    except Exception as e:
        logger.error(e)
        println('json格式错误！')
# 主页面
def menu():
    global window  # 声明全局变量 window
    global buffer  # 声明全局变量 buffer
    global conf  # 声明全局变量 conf
    global plan  # 声明全局变量 plan
    conf = load_conf()  # 调用 load_conf 函数，将返回值赋给 conf
    plan = load_plan(conf['planFile'])  # 调用 load_plan 函数，将返回值赋给 plan
    sg.theme('LightBlue2')  # 设置界面主题为 'LightBlue2'
    # --------主页
    package_type_title = sg.Text('服务器:', size=10)  # 创建文本标签对象 package_type_title
    package_type_1 = sg.Radio('官服', 'package_type', default=conf['package_type'] == 1,
                              key='radio_package_type_1', enable_events=True)  # 创建单选按钮对象 package_type_1
    package_type_2 = sg.Radio('BiliBili服', 'package_type', default=conf['package_type'] == 2,
                              key='radio_package_type_2', enable_events=True)  # 创建单选按钮对象 package_type_2
    adb_title = sg.Text('adb连接地址:', size=10)  # 创建文本标签对象 adb_title
    adb = sg.InputText(conf['adb'], size=60, key='conf_adb', enable_events=True)  # 创建输入框对象 adb
    # 黑名单
    free_blacklist_title = sg.Text('宿舍黑名单:', size=10)  # 创建文本标签对象 free_blacklist_title
    free_blacklist = sg.InputText(conf['free_blacklist'], size=60, key='conf_free_blacklist', enable_events=True)  # 创建输入框对象 free_blacklist
    # 排班表json
    plan_title = sg.Text('排班表:', size=10)  # 创建文本标签对象 plan_title
    plan_file = sg.InputText(conf['planFile'], readonly=True, size=60, key='planFile', enable_events=True)  # 创建只读输入框对象 plan_file
    plan_select = sg.FileBrowse('...', size=(3, 1), file_types=(("JSON files", "*.json"),))  # 创建文件浏览按钮对象 plan_select
    # 总开关
    on_btn = sg.Button('开始执行', key='on')  # 创建按钮对象 on_btn
    off_btn = sg.Button('立即停止', key='off', visible=False, button_color='red')  # 创建按钮对象 off_btn
    # 日志栏
    output = sg.Output(size=(150, 25), key='log', text_color='#808069', font=('微软雅黑', 9))  # 创建输出框对象 output

    # --------排班表设置页面
    # 宿舍区
    central = sg.Button('控制中枢', key='btn_central', size=(18, 3), button_color='#303030')  # 创建按钮对象 central
    dormitory_1 = sg.Button('宿舍', key='btn_dormitory_1', size=(18, 2), button_color='#303030')  # 创建按钮对象 dormitory_1
    dormitory_2 = sg.Button('宿舍', key='btn_dormitory_2', size=(18, 2), button_color='#303030')  # 创建按钮对象 dormitory_2
    dormitory_3 = sg.Button('宿舍', key='btn_dormitory_3', size=(18, 2), button_color='#303030')  # 创建按钮对象 dormitory_3
    dormitory_4 = sg.Button('宿舍', key='btn_dormitory_4', size=(18, 2), button_color='#303030')  # 创建按钮对象 dormitory_4
    central_area = sg.Column([[central], [dormitory_1], [dormitory_2], [dormitory_3], [dormitory_4]])  # 创建列对象 central_area
    # 制造站区
    # 创建一个按钮，显示文本为'待建造'，键为'btn_room_1_1'，大小为12x2，按钮颜色为'#4f4945'
    room_1_1 = sg.Button('待建造', key='btn_room_1_1', size=(12, 2), button_color='#4f4945')
    # 创建一个按钮，显示文本为'待建造'，键为'btn_room_1_2'，大小为12x2，按钮颜色为'#4f4945'
    room_1_2 = sg.Button('待建造', key='btn_room_1_2', size=(12, 2), button_color='#4f4945')
    # 创建一个按钮，显示文本为'待建造'，键为'btn_room_1_3'，大小为12x2，按钮颜色为'#4f4945'
    room_1_3 = sg.Button('待建造', key='btn_room_1_3', size=(12, 2), button_color='#4f4945')
    # 创建一个按钮，显示文本为'待建造'，键为'btn_room_2_1'，大小为12x2，按钮颜色为'#4f4945'
    room_2_1 = sg.Button('待建造', key='btn_room_2_1', size=(12, 2), button_color='#4f4945')
    # 创建一个按钮，显示文本为'待建造'，键为'btn_room_2_2'，大小为12x2，按钮颜色为'#4f4945'
    room_2_2 = sg.Button('待建造', key='btn_room_2_2', size=(12, 2), button_color='#4f4945')
    # 创建一个按钮，显示文本为'待建造'，键为'btn_room_2_3'，大小为12x2，按钮颜色为'#4f4945'
    room_2_3 = sg.Button('待建造', key='btn_room_2_3', size=(12, 2), button_color='#4f4945')
    # 创建一个按钮，显示文本为'待建造'，键为'btn_room_3_1'，大小为12x2，按钮颜色为'#4f4945'
    room_3_1 = sg.Button('待建造', key='btn_room_3_1', size=(12, 2), button_color='#4f4945')
    # 创建一个按钮，显示文本为'待建造'，键为'btn_room_3_2'，大小为12x2，按钮颜色为'#4f4945'
    room_3_2 = sg.Button('待建造', key='btn_room_3_2', size=(12, 2), button_color='#4f4945')
    # 创建一个按钮，显示文本为'待建造'，键为'btn_room_3_3'，大小为12x2，按钮颜色为'#4f4945'
    room_3_3 = sg.Button('待建造', key='btn_room_3_3', size=(12, 2), button_color='#4f4945')
    # 创建一个包含按钮的列，用于左侧区域显示
    left_area = sg.Column([[room_1_1, room_1_2, room_1_3],
                           [room_2_1, room_2_2, room_2_3],
                           [room_3_1, room_3_2, room_3_3]])
    # 创建一个按钮，显示文本为'会客室'，键为'btn_meeting'，大小为24x2，按钮颜色为'#303030'
    meeting = sg.Button('会客室', key='btn_meeting', size=(24, 2), button_color='#303030')
    # 创建一个按钮，显示文本为'加工站'，键为'btn_factory'，大小为24x2，按钮颜色为'#303030'
    factory = sg.Button('加工站', key='btn_factory', size=(24, 2), button_color='#303030')
    # 创建一个按钮，显示文本为'办公室'，键为'btn_contact'，大小为24x2，按钮颜色为'#303030'
    contact = sg.Button('办公室', key='btn_contact', size=(24, 2), button_color='#303030')
    # 创建一个包含按钮的列，用于右侧区域显示
    right_area = sg.Column([[meeting], [factory], [contact]])

    # 创建一个设置布局，包含一个下拉框，用于选择设施类别
    setting_layout = [
        [sg.Column([[sg.Text('设施类别:'), sg.InputCombo(['贸易站', '制造站', '发电站'], size=12, key='station_type')]],
                   key='station_type_col', visible=False)]]
    # 排班表设置标签
    # 循环5次，创建包含干员、组、替换输入框的布局，并添加到setting_layout列表中
    for i in range(1, 6):
        set_area = sg.Column([[sg.Text('干员：'),
                               sg.Combo(['Free'] + agent_list, size=20, key='agent' + str(i), change_submits=True),
                               sg.Text('组：'),
                               sg.InputText('', size=15, key='group' + str(i)),
                               sg.Text('替换：'),
                               sg.InputText('', size=30, key='replacement' + str(i))
                               ]], key='setArea' + str(i), visible=False)
        setting_layout.append([set_area])
    # 添加保存和清空按钮到setting_layout列表中
    setting_layout.append(
        [sg.Button('保存', key='savePlan', visible=False), sg.Button('清空', key='clearPlan', visible=False)])
    # 创建包含setting_layout的列布局，设置水平居中和垂直底部对齐，水平扩展
    setting_area = sg.Column(setting_layout, element_justification="center",
                             vertical_alignment="bottom",
                             expand_x=True)

    # --------高级设置页面
    # 创建显示当前版本的文本
    current_version = sg.Text('当前版本：' + __version__, size=25)
    # 创建检测更新按钮
    btn_check_update = sg.Button('检测更新', key='check_update')
    # 创建更新消息的文本
    update_msg = sg.Text('', key='update_msg')
    # 创建运行模式标题文本
    run_mode_title = sg.Text('运行模式：', size=25)
    # 创建换班模式单选按钮
    run_mode_1 = sg.Radio('换班模式', 'run_mode', default=conf['run_mode'] == 1,
                          key='radio_run_mode_1', enable_events=True)
    # 创建仅跑单模式单选按钮
    run_mode_2 = sg.Radio('仅跑单模式', 'run_mode', default=conf['run_mode'] == 2,
                          key='radio_run_mode_2', enable_events=True)
    # 创建令夕模式标题文本
    ling_xi_title = sg.Text('令夕模式（令夕上班时起作用）：', size=25)
    # 创建感知信息单选按钮
    ling_xi_1 = sg.Radio('感知信息', 'ling_xi', default=plan['conf']['ling_xi'] == 1,
                         key='plan_radio_ling_xi_1', enable_events=True)
    # 创建人间烟火单选按钮
    ling_xi_2 = sg.Radio('人间烟火', 'ling_xi', default=plan['conf']['ling_xi'] == 2,
                         key='plan_radio_ling_xi_2', enable_events=True)
    # 创建均衡模式单选按钮
    ling_xi_3 = sg.Radio('均衡模式', 'ling_xi', default=plan['conf']['ling_xi'] == 3,
                         key='plan_radio_ling_xi_3', enable_events=True)
    # 创建线索收集标题文本
    enable_party_title = sg.Text('线索收集：', size=25)
    # 创建一个单选按钮，用于启用或禁用某项功能，根据配置文件中的值来设置默认选项
    enable_party_1 = sg.Radio('启用', 'enable_party', default=conf['enable_party'] == 1,
                              key='radio_enable_party_1', enable_events=True)
    # 创建一个单选按钮，用于启用或禁用某项功能，根据配置文件中的值来设置默认选项
    enable_party_0 = sg.Radio('禁用', 'enable_party', default=conf['enable_party'] == 0,
                              key='radio_enable_party_0', enable_events=True)
    # 创建一个文本标签，用于显示最大组人数的标题
    max_resting_count_title = sg.Text('最大组人数：', size=25, key='max_resting_count_title')
    # 创建一个输入框，用于输入最大组人数的值，根据配置文件中的值来设置默认值
    max_resting_count = sg.InputText(plan['conf']['max_resting_count'], size=5,
                                     key='plan_int_max_resting_count', enable_events=True)
    # 创建一个文本标签，用于显示无人机使用阈值的标题
    drone_count_limit_title = sg.Text('无人机使用阈值：', size=25, key='drone_count_limit_title')
    # 创建一个输入框，用于输入无人机使用阈值的值，根据配置文件中的值来设置默认值
    drone_count_limit = sg.InputText(conf['drone_count_limit'], size=5,
                                     key='int_drone_count_limit', enable_events=True)
    # 创建一个文本标签，用于显示跑单前置延时的标题
    run_order_delay_title = sg.Text('跑单前置延时(分钟)：', size=25, key='run_order_delay_title')
    # 创建一个输入框，用于输入跑单前置延时的值，根据配置文件中的值来设置默认值
    run_order_delay = sg.InputText(conf['run_order_delay'], size=5,
                                   key='float_run_order_delay', enable_events=True)
    # 创建一个文本标签，用于显示无人机使用房间的标题
    drone_room_title = sg.Text('无人机使用房间（room_X_X）：', size=25, key='drone_room_title')
    # 创建一个文本标签，用于显示搓玉补货房间的标题
    reload_room_title = sg.Text('搓玉补货房间（逗号分隔房间名）：', size=25, key='reload_room_title')
    # 创建一个输入框，用于输入无人机使用房间的值，根据配置文件中的值来设置默认值
    drone_room = sg.InputText(conf['drone_room'], size=15,
                              key='conf_drone_room', enable_events=True)
    # 创建一个输入框，用于输入搓玉补货房间的值，根据配置文件中的值来设置默认值
    reload_room = sg.InputText(conf['reload_room'], size=30,
                               key='conf_reload_room', enable_events=True)
    # 创建一个文本标签，用于显示需要回满心情的干员的标题
    rest_in_full_title = sg.Text('需要回满心情的干员：', size=25)
    # 创建一个输入框，用于输入需要回满心情的干员的值，根据配置文件中的值来设置默认值
    rest_in_full = sg.InputText(plan['conf']['rest_in_full'], size=60,
                                key='plan_conf_rest_in_full', enable_events=True)
    # 创建一个文本标签，用于显示需用尽心情的干员的标题
    exhaust_require_title = sg.Text('需用尽心情的干员：', size=25)
    # 创建一个输入框，用于输入需用尽心情的干员的值，根据配置文件中的值来设置默认值
    exhaust_require = sg.InputText(plan['conf']['exhaust_require'], size=60,
                                   key='plan_conf_exhaust_require', enable_events=True)
    # 创建一个文本标签，显示“0心情工作的干员：”，设置字体大小为25
    workaholic_title = sg.Text('0心情工作的干员：', size=25)
    # 创建一个文本输入框，显示计划配置中的'workaholic'值，设置大小为60，设置关键字为'plan_conf_workaholic'，并启用事件
    workaholic = sg.InputText(plan['conf']['workaholic'], size=60,
                              key='plan_conf_workaholic', enable_events=True)

    # 创建一个文本标签，显示“宿舍低优先级干员：”，设置字体大小为25
    resting_priority_title = sg.Text('宿舍低优先级干员：', size=25)
    # 创建一个文本输入框，显示计划配置中的'resting_priority'值，设置大小为60，设置关键字为'plan_conf_resting_priority'，并启用事件
    resting_priority = sg.InputText(plan['conf']['resting_priority'], size=60,
                                    key='plan_conf_resting_priority', enable_events=True)

    # 创建一个复选框，显示“启动mower时自动开始任务”，默认值为conf['start_automatically']，设置关键字为'conf_start_automatically'，并启用事件
    start_automatically = sg.Checkbox('启动mower时自动开始任务', default=conf['start_automatically'],
                                      key='conf_start_automatically', enable_events=True)
    # --------外部调用设置页面
    # mail
    # 创建一个单选框，显示“启用”，默认值为conf['mail_enable'] == 1，设置关键字为'radio_mail_enable_1'，并启用事件
    mail_enable_1 = sg.Radio('启用', 'mail_enable', default=conf['mail_enable'] == 1,
                             key='radio_mail_enable_1', enable_events=True)
    # 创建一个单选框，显示“禁用”，默认值为conf['mail_enable'] == 0，设置关键字为'radio_mail_enable_0'，并启用事件
    mail_enable_0 = sg.Radio('禁用', 'mail_enable', default=conf['mail_enable'] == 0,
                             key='radio_mail_enable_0', enable_events=True)
    # 创建一个文本标签，显示“QQ邮箱”，设置字体大小为25
    account_title = sg.Text('QQ邮箱', size=25)
    # 创建一个文本输入框，显示conf['account']值，设置大小为60，设置关键字为'conf_account'，并启用事件
    account = sg.InputText(conf['account'], size=60, key='conf_account', enable_events=True)
    # 创建一个文本标签，显示“授权码”，设置字体大小为25
    pass_code_title = sg.Text('授权码', size=25)
    # 创建一个文本输入框，显示conf['pass_code']值，设置大小为60，设置关键字为'conf_pass_code'，并启用事件，密码字符为'*'
    pass_code = sg.Input(conf['pass_code'], size=60, key='conf_pass_code', enable_events=True, password_char='*')
    # 创建一个框架，显示“邮件提醒”，包含单选框、文本标签和文本输入框
    mail_frame = sg.Frame('邮件提醒',
                          [[mail_enable_1, mail_enable_0], [account_title, account], [pass_code_title, pass_code]])
    # maa
    # 创建一个单选框，显示“启用”，默认值为conf['maa_enable'] == 1，设置关键字为'radio_maa_enable_1'，并启用事件
    maa_enable_1 = sg.Radio('启用', 'maa_enable', default=conf['maa_enable'] == 1,
                            key='radio_maa_enable_1', enable_events=True)
    # 创建一个单选框，显示“禁用”，默认值为conf['maa_enable'] == 0，设置关键字为'radio_maa_enable_0'，并启用事件
    maa_enable_0 = sg.Radio('禁用', 'maa_enable', default=conf['maa_enable'] == 0,
                            key='radio_maa_enable_0', enable_events=True)
    # 创建一个文本标签，显示“MAA启动间隔(小时)：”，设置字体大小为15
    maa_gap_title = sg.Text('MAA启动间隔(小时)：', size=15)
    # 创建一个文本输入框，显示conf['maa_gap']值，设置大小为5，设置关键字为'float_maa_gap'，并启用事件
    maa_gap = sg.InputText(conf['maa_gap'], size=5, key='float_maa_gap', enable_events=True)
    # 创建一个文本标签，显示“信用商店优先购买（逗号分隔）：”，设置字体大小为25，设置关键字为'mall_buy_title'
    maa_mall_buy_title = sg.Text('信用商店优先购买（逗号分隔）：', size=25, key='mall_buy_title')
    # 创建一个文本输入框，用于输入maa_mall_buy的值，大小为30，关键字为'conf_maa_mall_buy'，并且支持事件
    maa_mall_buy = sg.InputText(conf['maa_mall_buy'], size=30,
                               key='conf_maa_mall_buy', enable_events=True)
    # 创建一个复选框，用于设置公招三星的时间为7:40而非9:00，默认为conf['maa_recruitment_time']的值，关键字为'conf_maa_recruitment_time'，并且支持事件
    maa_recruitment_time = sg.Checkbox('公招三星设置7:40而非9:00', default=conf['maa_recruitment_time'],
                                      key='conf_maa_recruitment_time', enable_events=True)
    # 创建一个复选框，用于设置仅公招四星，默认为conf['maa_recruit_only_4']的值，关键字为'conf_maa_recruit_only_4'，并且支持事件
    maa_recruit_only_4 = sg.Checkbox('仅公招四星', default=conf['maa_recruit_only_4'],
                                       key='conf_maa_recruit_only_4', enable_events=True)
    # 创建一个文本，用于显示信用商店黑名单的标题，大小为25，关键字为'mall_blacklist_title'
    maa_mall_blacklist_title = sg.Text('信用商店黑名单（逗号分隔）：', size=25, key='mall_blacklist_title')
    # 创建一个文本输入框，用于输入maa_mall_blacklist的值，大小为30，关键字为'conf_maa_mall_blacklist'，并且支持事件
    maa_mall_blacklist = sg.InputText(conf['maa_mall_blacklist'], size=30,
                                key='conf_maa_mall_blacklist', enable_events=True)
    # 创建一个文本，用于显示肉鸽的标题，大小为10
    maa_rg_title = sg.Text('肉鸽：', size=10)
    # 创建一个单选按钮，用于设置肉鸽是否启用，默认为conf['maa_rg_enable']等于1，关键字为'radio_maa_rg_enable_1'，并且支持事件
    maa_rg_enable_1 = sg.Radio('启用', 'maa_rg_enable', default=conf['maa_rg_enable'] == 1,
                               key='radio_maa_rg_enable_1', enable_events=True)
    # 创建一个单选按钮，用于设置肉鸽是否禁用，默认为conf['maa_rg_enable']等于0，关键字为'radio_maa_rg_enable_0'，并且支持事件
    maa_rg_enable_0 = sg.Radio('禁用', 'maa_rg_enable', default=conf['maa_rg_enable'] == 0,
                               key='radio_maa_rg_enable_0', enable_events=True)
    # 创建一个文本，用于显示肉鸽任务休眠时间的标题，大小为25
    maa_rg_sleep = sg.Text('肉鸽任务休眠时间(如8:30-23:30)', size=25)
    # 创建一个文本输入框，用于输入maa_rg_sleep_min的值，大小为5，关键字为'conf_maa_rg_sleep_min'，并且支持事件
    maa_rg_sleep_min = sg.InputText(conf['maa_rg_sleep_min'], size=5, key='conf_maa_rg_sleep_min', enable_events=True)
    # 创建一个文本输入框，用于输入maa_rg_sleep_max的值，大小为5，关键字为'conf_maa_rg_sleep_max'，并且支持事件
    maa_rg_sleep_max = sg.InputText(conf['maa_rg_sleep_max'], size=5, key='conf_maa_rg_sleep_max', enable_events=True)
    # 创建一个文本，用于显示MAA地址的标题，大小为25
    maa_path_title = sg.Text('MAA地址', size=25)
    # 创建一个文本输入框，用于输入maa_path的值，大小为60，关键字为'conf_maa_path'，并且支持事件
    maa_path = sg.InputText(conf['maa_path'], size=60, key='conf_maa_path', enable_events=True)
    # 创建一个文本，用于显示adb地址的标题，大小为25
    maa_adb_path_title = sg.Text('adb地址', size=25)
    # 创建一个文本输入框，用于输入maa_adb_path的值，大小为60，关键字为'conf_maa_adb_path'，并且支持事件
    maa_adb_path = sg.InputText(conf['maa_adb_path'], size=60, key='conf_maa_adb_path', enable_events=True)
    # 创建一个文本，用于显示周计划的标题，大小为25
    maa_weekly_plan_title = sg.Text('周计划', size=25)
    # 创建一个二维列表，用于存储布局信息
    maa_layout = [[maa_enable_1, maa_enable_0, maa_gap_title, maa_gap, maa_recruitment_time, maa_recruit_only_4],
                  [maa_mall_buy_title, maa_mall_buy, maa_mall_blacklist_title, maa_mall_blacklist],
                  [maa_rg_title, maa_rg_enable_1, maa_rg_enable_0, maa_rg_sleep, maa_rg_sleep_min, maa_rg_sleep_max],
                  [maa_path_title, maa_path], [maa_adb_path_title, maa_adb_path],
                  [maa_weekly_plan_title]]
    # 遍历配置文件中的maa_weekly_plan，将其内容添加到maa_layout中
    for i, v in enumerate(conf['maa_weekly_plan']):
        maa_layout.append([
            sg.Text(f"-- {v['weekday']}:", size=15),  # 添加星期几的文本
            sg.Text('关卡:', size=5),  # 添加关卡文本
            sg.InputText(",".join(v['stage']), size=15, key='maa_weekly_plan_stage_' + str(i), enable_events=True),  # 添加关卡输入框
            sg.Text('理智药:', size=10),  # 添加理智药文本
            sg.Spin([l for l in range(0, 999)], initial_value=v['medicine'], size=5,
                    key='maa_weekly_plan_medicine_' + str(i), enable_events=True, readonly=True)  # 添加理智药选择框
        ])
    # 创建一个名为"MAA"的框架，包含maa_layout中的内容
    maa_frame = sg.Frame('MAA', maa_layout)
    # 创建一个名为"主页"的选项卡，包含一些其他的元素
    main_tab = sg.Tab('  主页  ', [[package_type_title, package_type_1, package_type_2],
                                 [adb_title, adb],
                                 [free_blacklist_title, free_blacklist],
                                 [plan_title, plan_file, plan_select],
                                 [output],
                                 [on_btn, off_btn]])
    # 创建一个名为"排班表"的选项卡，包含一些其他的元素
    plan_tab = sg.Tab('  排班表 ', [[left_area, central_area, right_area], [setting_area]], element_justification="center")
    # 创建一个名为 setting_tab 的选项卡，包含一系列高级设置的控件
    setting_tab = sg.Tab('  高级设置 ',
                         [
                             [current_version, btn_check_update, update_msg],  # 显示当前版本、检查更新按钮和更新信息
                             [run_mode_title, run_mode_1, run_mode_2],  # 显示运行模式标题和两个运行模式选项
                             [ling_xi_title, ling_xi_1, ling_xi_2, ling_xi_3],  # 显示灵犀标题和三个灵犀选项
                             [enable_party_title, enable_party_1, enable_party_0],  # 显示启用派对标题和两个启用派对选项
                             [max_resting_count_title, max_resting_count, sg.Text('', size=16), run_order_delay_title,
                              run_order_delay],  # 显示最大休息次数、运行顺序延迟等控件
                             [drone_room_title, drone_room, sg.Text('', size=7), drone_count_limit_title,
                              drone_count_limit],  # 显示无人机机房、无人机数量限制等控件
                             [reload_room_title, reload_room],  # 显示重新加载机房控件
                             [rest_in_full_title, rest_in_full],  # 显示满员休息控件
                             [exhaust_require_title, exhaust_require],  # 显示耗尽需求控件
                             [workaholic_title, workaholic],  # 显示工作狂控件
                             [resting_priority_title, resting_priority],  # 显示休息优先级控件
                             [start_automatically],  # 显示自动启动控件
                         ], pad=((10, 10), (10, 10)))  # 设置选项卡的内边距

    # 创建一个名为 other_tab 的选项卡，包含外部调用的控件
    other_tab = sg.Tab('  外部调用 ',
                       [[mail_frame], [maa_frame]], pad=((10, 10), (10, 10)))  # 显示邮件框架和 maa 框架

    # 创建一个名为 window 的窗口，包含多个选项卡，并设置字体、不可调整大小
    window = sg.Window('Mower', [[sg.TabGroup([[main_tab, plan_tab, setting_tab, other_tab]], border_width=0,
                                              tab_border_width=0, focus_color='#bcc8e5',
                                              selected_background_color='#d4dae8', background_color='#aab6d3',
                                              tab_background_color='#aab6d3')]], font='微软雅黑', finalize=True,
                       resizable=False)

    # 根据配置文件中的计划文件构建计划
    build_plan(conf['planFile'])

    # 初始化 btn 为 None
    btn = None

    # 为基建布局左边的站点排序绑定事件
    bind_scirpt()

    # 创建一个名为 drag_task 的拖动任务
    drag_task = DragTask()

    # 如果配置文件中设置为自动启动，则启动程序
    if conf['start_automatically']:
        start()

    # 关闭窗口
    window.close()

    # 保存配置文件
    save_conf(conf)

    # 将计划写入到配置文件中的计划文件
    write_plan(plan, conf['planFile'])
# 启动函数，设置全局变量，更新窗口状态，清空内容，创建进程通信管道，启动主线程，执行长时间操作
def start():
    global main_thread, child_conn
    window['on'].update(visible=False)  # 更新窗口状态，隐藏“on”按钮
    window['off'].update(visible=True)  # 更新窗口状态，显示“off”按钮
    clear()  # 清空内容
    parent_conn, child_conn = Pipe()  # 创建进程通信管道
    main_thread = Process(target=main, args=(conf, plan, operators, child_conn), daemon=True)  # 创建主线程进程
    main_thread.start()  # 启动主线程
    window.perform_long_operation(lambda: recv(parent_conn), 'recv')  # 执行长时间操作，接收父进程通信


# 绑定脚本函数，为按钮绑定鼠标事件和键盘事件
def bind_scirpt():
    for i in range(3):
        for j in range(3):
            event = f'btn_room_{str(i + 1)}_{str(j + 1)}'
            window[event].bind("<B1-Motion>", "-motion-script")  # 绑定鼠标拖拽事件
            window[event].bind("<ButtonRelease-1>", "-ButtonRelease-script")  # 绑定鼠标松开事件
            window[event].bind("<Enter>", "-Enter-script")  # 绑定鼠标进入事件
    for i in range(5):
        event = f'agent{str(i + 1)}'
        window[event].bind("<Key>", "-agent_change")  # 绑定键盘按键事件


# 运行脚本函数，根据事件和拖拽任务执行相应操作
def run_script(event, drag_task):
    # logger.info(f"{event}:{drag_task}")
    if event.endswith('-motion'):  # 判断是否为拖拽事件
        if drag_task.step == 0 or drag_task.step == 2:  # 判断拖拽任务步骤
            drag_task.btn = event[:event.rindex('-')]  # 记录初始按钮
            drag_task.step = 1  # 初始化键位，并推进任务步骤
    elif event.endswith('-ButtonRelease'):  # 判断是否为松开按钮事件
        if drag_task.step == 1:
            drag_task.step = 2  # 推进任务步骤
    elif event.endswith('-Enter'):  # 判断是否为进入元素事件
        if drag_task.step == 2:
            drag_task.new_btn = event[:event.rindex('-')]  # 记录需交换的按钮
            switch_plan(drag_task)  # 执行交换计划
            drag_task.clear()  # 清空拖拽任务
        else:
            drag_task.clear()  # 清空拖拽任务


# 执行交换计划函数，根据拖拽任务的按钮交换计划中的键值对
def switch_plan(drag_task):
    key1 = drag_task.btn[4:]  # 获取按钮对应的键
    key2 = drag_task.new_btn[4:]  # 获取新按钮对应的键
    value1 = current_plan[key1] if key1 in current_plan else None  # 获取键对应的值
    value2 = current_plan[key2] if key2 in current_plan else None  # 获取新键对应的值
    if value1 is not None:
        current_plan[key2] = value1  # 将值1赋给新键
    elif key2 in current_plan:
        current_plan.pop(key2)  # 移除新键
    if value2 is not None:
        current_plan[key1] = value2  # 将值2赋给原键
    # 如果 key1 存在于 current_plan 中，则将其移除
    elif key1 in current_plan:
        current_plan.pop(key1)
    # 将修改后的 plan 写入到指定的 planFile 中
    write_plan(plan, conf['planFile'])
    # 根据指定的 planFile 构建计划
    build_plan(conf['planFile'])
# 初始化按钮事件处理函数，根据事件参数中的房间键值进行初始化操作
def init_btn(event):
    # 从事件参数中获取房间键值
    room_key = event[4:]
    # 如果当前计划中存在该房间键值，则获取该房间的名称和计划列表，否则为空字符串和空列表
    station_name = current_plan[room_key]['name'] if room_key in current_plan.keys() else ''
    plans = current_plan[room_key]['plans'] if room_key in current_plan.keys() else []
    # 如果房间键值以'room'开头，则更新窗口中的站点类型列和站点类型，并设置设施干员需求数量为3
    if room_key.startswith('room'):
        window['station_type_col'].update(visible=True)
        window['station_type'].update(station_name)
        visible_cnt = 3  # 设施干员需求数量
    else:
        # 根据不同的房间键值设置不同的可见数量和更新窗口中的站点类型列和站点类型
        if room_key == 'meeting':
            visible_cnt = 2
        elif room_key == 'factory' or room_key == 'contact':
            visible_cnt = 1
        else:
            visible_cnt = 5
        window['station_type_col'].update(visible=False)
        window['station_type'].update('')
    # 更新保存计划按钮和清除计划按钮为可见状态
    window['savePlan'].update(visible=True)
    window['clearPlan'].update(visible=True)
    # 遍历1到5的范围，根据可见数量更新窗口中的设置区域、干员、组别和替补信息
    for i in range(1, 6):
        if i > visible_cnt:
            window['setArea' + str(i)].update(visible=False)
            window['agent' + str(i)].update('')
            window['group' + str(i)].update('')
            window['replacement' + str(i)].update('')
        else:
            window['setArea' + str(i)].update(visible=True)
            window['agent' + str(i)].update(plans[i - 1]['agent'] if len(plans) >= i else '')
            window['group' + str(i)].update(plans[i - 1]['group'] if len(plans) >= i else '')
            window['replacement' + str(i)].update(','.join(plans[i - 1]['replacement']) if len(plans) >= i else '')


# 保存按钮事件处理函数，创建包含站点类型和计划列表的字典
def save_btn(btn):
    plan1 = {'name': window['station_type'].get(), 'plans': []}
    # 遍历范围为1到5的数字
    for i in range(1, 6):
        # 获取窗口中名为'agent' + str(i)的值
        agent = window['agent' + str(i)].get()
        # 获取窗口中名为'group' + str(i)的值
        group = window['group' + str(i)].get()
        # 获取窗口中名为'replacement' + str(i)的值，将中文逗号替换为英文逗号后，以逗号分割成列表，并过滤掉空值
        replacement = list(filter(None, window['replacement' + str(i)].get().replace('，', ',').split(',')))
        # 如果agent不为空
        if agent != '':
            # 将agent、group和replacement添加到plan1字典的'plans'列表中
            plan1['plans'].append({'agent': agent, 'group': group, 'replacement': replacement})
        # 如果btn以'btn_dormitory'开头
        elif btn.startswith('btn_dormitory'):  # 宿舍
            # 将'Free'代理、空组和空替换列表添加到plan1字典的'plans'列表中
            plan1['plans'].append({'agent': 'Free', 'group': '', 'replacement': []})
    # 将plan1字典赋值给current_plan字典的键为btn[4:]的值
    current_plan[btn[4:]] = plan1
    # 将plan写入到配置文件中
    write_plan(plan, conf['planFile'])
    # 构建计划
    build_plan(conf['planFile'])
# 清除按钮操作，从当前计划中移除对应的按钮
def clear_btn(btn):
    if btn[4:] in current_plan:  # 检查按钮对应的计划是否存在
        current_plan.pop(btn[4:])  # 从当前计划中移除对应的按钮
    init_btn(btn)  # 初始化按钮
    write_plan(plan, conf['planFile'])  # 将计划写入文件
    build_plan(conf['planFile'])  # 构建计划


# 检查更新
def check_update():
    try:
        newest_version = compere_version()  # 比较版本号，获取最新版本号
        if newest_version:  # 如果有新版本
            window['update_msg'].update('检测到有新版本'+newest_version+',正在下载...',text_color='black')  # 更新提示信息
            download_version(newest_version)  # 下载最新版本
            window['update_msg'].update('下载完毕！',text_color='green')  # 下载完成提示
        else:  # 如果没有新版本
            window['update_msg'].update('已是最新版！',text_color='green')  # 已是最新版本提示
    except Exception as e:  # 捕获异常
        logger.error(e)  # 记录错误日志
        window['update_msg'].update('更新失败！',text_color='red')  # 更新失败提示
        return None  # 返回空值
    window['check_update'].update(disabled=False)  # 更新按钮状态
    return newest_version  # 返回最新版本号


# 接收推送
def recv(pipe):
    try:
        while True:
            msg = pipe.recv()  # 接收消息
            if msg['type'] == 'log':  # 如果消息类型为日志
                println(msg['data'])  # 打印日志
            elif msg['type'] == 'operators':  # 如果消息类型为操作员
                global operators  # 声明全局变量
                operators = msg['data']  # 更新操作员信息
    except EOFError:  # 捕获文件结束异常
        pipe.close()  # 关闭管道


# 打印消息
def println(msg):
    global buffer  # 声明全局变量
    global line  # 声明全局变量
    global half_line_index  # 声明全局变量
    maxLen = 500  # 最大行数
    buffer = f'{buffer}\n{time.strftime("%m-%d %H:%M:%S")} {msg}'.strip()  # 格式化消息并添加到缓冲区
    window['log'].update(value=buffer)  # 更新日志窗口
    if line == maxLen // 2:  # 如果行数达到最大行数的一半
        half_line_index = len(buffer)  # 更新一半行数的索引
    if line >= maxLen:  # 如果行数达到最大行数
        buffer = buffer[half_line_index:]  # 截取一半行数之后的内容
        line = maxLen // 2  # 更新行数为最大行数的一半
    else:  # 如果行数未达到最大行数
        line += 1  # 行数加一


# 清空输出栏
def clear():
    global buffer  # 声明全局变量
    global line  # 声明全局变量
    buffer = ''  # 清空缓冲区
    window['log'].update(value=buffer)  # 更新日志窗口
    line = 0  # 行数归零


# 拖动任务类
class DragTask:
    def __init__(self):
        self.btn = None  # 初始化按钮为None
        self.new_btn = None  # 初始化新按钮为None
        self.step = 0  # 初始化步骤为0

    def clear(self):
        self.btn = None  # 清空按钮
        self.new_btn = None  # 清空新按钮
        self.step = 0  # 步骤归零

    def __repr__(self):
        return f"btn:{self.btn},new_btn:{self.new_btn},step:{self.step}"  # 返回字符串表示


if __name__ == '__main__':
    freeze_support()  # 冻结支持
    menu()  # 菜单操作
```