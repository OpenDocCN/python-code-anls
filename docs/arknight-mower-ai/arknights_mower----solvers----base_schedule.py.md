# `arknights-mower\arknights_mower\solvers\base_schedule.py`

```
# 导入必要的模块和库
from __future__ import annotations
import copy
import subprocess
import time
import sys
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 导入自定义模块
from ..command import recruit
from ..data import agent_list
from ..utils import character_recognize, detector, segment
from ..utils.digit_reader import DigitReader
from ..utils.operators import Operators, Operator, Dormitory
from ..utils.recruit import filter_result
from ..utils.scheduler_task import SchedulerTask
from ..utils import typealias as tp
from ..utils.device import Device
from ..utils.log import logger
from ..utils.pipe import push_operators
from ..utils.recognize import RecognizeError, Recognizer, Scene
from ..utils.solver import BaseSolver
from ..utils.datetime import get_server_weekday, the_same_time
from paddleocr import PaddleOCR
import cv2

# 导入自定义模块
from arknights_mower.__main__ import format_time
from arknights_mower.utils.asst import Asst, Message
import json
from arknights_mower.utils.email import task_template, maa_template, recruit_template

# 初始化 OCR 变量
ocr = None

# 定义枚举类型 ArrangeOrder
class ArrangeOrder(Enum):
    STATUS = 1
    SKILL = 2
    FEELING = 3
    TRUST = 4

# 定义 ArrangeOrder 对应的坐标
arrange_order_res = {
    ArrangeOrder.STATUS: (1560 / 2496, 96 / 1404),
    ArrangeOrder.SKILL: (1720 / 2496, 96 / 1404),
    ArrangeOrder.FEELING: (1880 / 2496, 96 / 1404),
    ArrangeOrder.TRUST: (2050 / 2496, 96 / 1404),
}

# 初始化舞台掉落字典
stage_drop = {}

# 初始化公招选择tag相关字典
recruit_tags_delected = {}
recruit_tags_selected = {}
recruit_results = {}
recruit_special_tags = {}

# 定义基础调度求解器类
class BaseSchedulerSolver(BaseSolver):
    """
    收集基建的产物：物资、赤金、信赖
    """
    package_name = ''
    # 初始化函数，接受设备和识别器两个参数，默认为 None
    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        # 调用父类的初始化函数，传入设备和识别器参数
        super().__init__(device, recog)
        # 初始化操作数据为 None
        self.op_data = None
        # 设置最大休息次数为 4
        self.max_resting_count = 4
        # 初始化派对时间为 None
        self.party_time = None
        # 初始化无人机时间为 None
        self.drone_time = None
        # 初始化重新加载时间为 None
        self.reload_time = None
        # 初始化重新加载房间为 None
        self.reload_room = None
        # 设置运行顺序延迟为 10
        self.run_order_delay = 10
        # 设置线索数量限制为 9
        self.clue_count_limit = 9
        # 启用派对功能
        self.enable_party = True
        # 初始化数字阅读器
        self.digit_reader = DigitReader()
        # 设置错误标志为 False
        self.error = False
        # 初始化线索数量为 0
        self.clue_count = 0
        # 初始化任务列表为空
        self.tasks = []
        # 初始化 MAA 配置为空字典
        self.maa_config = {}
        # 初始化免费线索为 None
        self.free_clue = None
        # 初始化信用战斗为 None
        self.credit_fight = None
        # 设置空闲时退出游戏为 False
        self.exit_game_when_idle = False

    # 运行函数
    def run(self) -> None:
        """
        :param clue_collect: bool, 是否收取线索
        """
        # 设置错误标志为 False
        self.error = False
        # 处理错误
        self.handle_error(True)
        # 如果任务列表不为空
        if len(self.tasks) > 0:
            # 找到时间最近的一次单个任务
            self.task = self.tasks[0]
        else:
            # 否则任务为空
            self.task = None
        # 如果派对时间不为 None 并且派对时间早于当前时间
        if self.party_time is not None and self.party_time < datetime.now():
            # 将派对时间设为 None
            self.party_time = None
        # 如果免费线索不为 None 并且免费线索不等于服务器当前星期
        if self.free_clue is not None and self.free_clue != get_server_weekday():
            # 将免费线索设为 None
            self.free_clue = None
        # 如果信用战斗不为 None 并且信用战斗不等于服务器当前星期
        if self.credit_fight is not None and self.credit_fight != get_server_weekday():
            # 将信用战斗设为 None
            self.credit_fight = None
        # 设置待办任务为 False
        self.todo_task = False
        # 设置收集通知为 False
        self.collect_notification = False
        # 设置计划为 False
        self.planned = False
        # 如果操作数据为 None 或者操作数据的操作员为 None
        if self.op_data is None or self.op_data.operators is None:
            # 初始化操作员
            self.initialize_operators()
        # 修正操作数据的宿舍
        self.op_data.correct_dorm()
        # 遍历工作狂代理的名字
        for name in self.op_data.workaholic_agent:
            # 如果名字不在免费黑名单中
            if name not in self.free_blacklist:
                # 将名字添加到免费黑名单中
                self.free_blacklist.append(name)
        # 调用父类的运行函数
        return super().run()
    # 定义状态转换方法
    def transition(self) -> None:
        # 更新识别结果
        self.recog.update()
        # 如果当前场景是INDEX，则点击index_infrastructure元素
        if self.get_infra_scene() == Scene.INDEX:
            self.tap_element('index_infrastructure')
        # 如果当前场景是INFRA_MAIN，则执行infra_main方法
        elif self.get_infra_scene() == Scene.INFRA_MAIN:
            return self.infra_main()
        # 如果当前场景是INFRA_TODOLIST，则执行todo_list方法
        elif self.get_infra_scene() == Scene.INFRA_TODOLIST:
            return self.todo_list()
        # 如果当前场景是INFRA_DETAILS，则返回上一级
        elif self.get_infra_scene() == Scene.INFRA_DETAILS:
            self.back()
        # 如果当前场景是LOADING，则等待加载完成
        elif self.get_infra_scene() == Scene.LOADING:
            self.waiting_solver(Scene.LOADING)
        # 如果当前场景是CONNECTING，则等待连接完成
        elif self.get_infra_scene() == Scene.CONNECTING:
            self.waiting_solver(Scene.CONNECTING)
        # 如果存在导航元素，则点击nav_infrastructure元素
        elif self.get_navigation():
            self.tap_element('nav_infrastructure')
        # 如果当前场景是INFRA_ARRANGE_ORDER，则点击arrange_blue_yes元素
        elif self.get_infra_scene() == Scene.INFRA_ARRANGE_ORDER:
            self.tap_element('arrange_blue_yes')
        # 如果当前场景是UNKNOWN或者当前场景不是UNKNOWN，则返回到INDEX场景，重设上次房间为空，并记录日志
        elif self.get_infra_scene() == Scene.UNKNOWN or self.scene() != Scene.UNKNOWN:
            self.back_to_index()
            self.last_room = ''
            logger.info("重设上次房间为空")
        # 如果以上条件都不满足，则抛出识别错误
        else:
            raise RecognizeError('Unknown scene')

    # 查找下一个任务
    def find_next_task(self, compare_time=None, task_type='', compare_type='<'):
        # 如果比较类型是'='，则返回下一个符合条件的任务
        if compare_type == '=':
            return next((e for e in self.tasks if the_same_time(e.time, compare_time) and (
                True if task_type == '' else task_type in e.type)), None)
        # 如果比较类型是'>'，则返回下一个符合条件的任务
        elif compare_type == '>':
            return next((e for e in self.tasks if (True if compare_time is None else e.time > compare_time) and (
                True if task_type == '' else task_type in e.type)), None)
        # 如果比较类型是其他，则返回下一个符合条件的任务
        else:
            return next((e for e in self.tasks if (True if compare_time is None else e.time < compare_time) and (
                True if task_type == '' else task_type in e.type)), None)
    # 处理错误情况，force 参数默认为 False
    def handle_error(self, force=False):
        # 如果当前场景为未知场景
        if self.scene() == Scene.UNKNOWN:
            # 退出设备，并指定要退出的应用程序包名
            self.device.exit(self.package_name)
        # 如果存在错误或者 force 参数为 True
        if self.error or force:
            # 如果没有任何时间小于当前时间的任务才生成空任务
            if self.find_next_task(datetime.now()) is None:
                # 记录调试信息，生成一次空任务来执行纠错
                logger.debug("由于出现错误情况，生成一次空任务来执行纠错")
                # 将空任务添加到任务列表中
                self.tasks.append(SchedulerTask())
            # 如果没有任何时间小于当前时间减去 10 分钟的任务，则清空任务
            if self.find_next_task(datetime.now() - timedelta(seconds=900)) is not None:
                # 记录信息，检测到执行超过 15 分钟的任务，清空全部任务
                logger.info("检测到执行超过15分钟的任务，清空全部任务")
                # 清空任务列表
                self.tasks = []
        # 如果不存在错误并且 force 参数为 False
        elif self.find_next_task(datetime.now() + timedelta(hours=2.5)) is None:
            # 记录调试信息，2.5 小时内没有其他任务，生成一个空任务
            logger.debug("2.5小时内没有其他任务，生成一个空任务")
            # 将一个时间为当前时间加上 2.5 小时的空任务添加到任务列表中
            self.tasks.append(SchedulerTask(time=datetime.now() + timedelta(hours=2.5)))
        # 返回 True
        return True
    # 定义一个方法，用于规划菲亚梅塔的任务
    def plan_fia(self):
        # 获取菲亚梅塔的计划和房间信息
        fia_plan, fia_room = self.check_fia()
        # 如果房间和计划都不为空
        if fia_room is not None and fia_plan is not None:
            # 获取当前时间
            current_time = self.task.time
            # 创建候选列表
            candidate_lst = []
            # 复制最后一位的当前信息
            last_candidate = copy.deepcopy(self.op_data.operators[self.op_data.operators['菲亚梅塔'].replacement[-1]])
            plan_last = True
            # 遍历替换列表中除了最后一位的干员名
            for name in self.op_data.operators['菲亚梅塔'].replacement[:-1]:
                # 如果干员名存在于操作员数据中
                if name in self.op_data.operators:
                    # 如果心情消耗速率在0到2之间
                    if not 0 < self.op_data.operators[name].depletion_rate < 2:
                        # 记录日志，提示心情消耗速率缺失或不在合理范围内
                        logger.info(f'{name}的心情消耗速率缺失或不在合理范围内')
                        plan_last = False
                    # 复制除去最后一位的当前信息
                    data = copy.deepcopy(self.op_data.operators[name])
                    data.mood = data.current_mood()
                    candidate_lst.append(data)
            # 跳过当前任务
            self.skip()
            # 根据心情值排序候选列表
            candidate_lst.sort(key=lambda x: (x.mood - x.lower_limit) / (x.upper_limit - x.lower_limit), reverse=False)
            # 打印候选列表和最后一位干员的信息
            print(candidate_lst)
            print(last_candidate)
            # 获取候选列表中第一个干员的名字
            name = candidate_lst[0].name
            # 如果计划为真或者候选列表中第一个干员的心情值大于等于20，并且最后一位干员的当前房间不是"dorm"
            if (plan_last or candidate_lst[0].current_mood() >= 20) and not last_candidate.current_room.startswith("dorm"):
                # 获取最后一位干员的心情值
                mood = last_candidate.current_mood()
                # 判断是否是最低心情值
                is_lowest = mood < candidate_lst[0].current_mood()
                # 记录调试日志，包括最后一位干员的名字和心情值
                logger.debug(f'{last_candidate.name},mood:{mood}')
                # 如果是最低心情值
                if is_lowest:
                    # 如果计划为真并且可以预测菲亚梅塔的情况
                    if plan_last and self.op_data.predict_fia(copy.deepcopy(candidate_lst), mood):
                        # 使用最后一位干员的名字
                        name = last_candidate.name
                    # 如果计划不为真
                    elif not plan_last:
                        # 使用最后一位干员的名字
                        name = last_candidate.name
            # 将任务添加到任务列表中
            self.tasks.append(SchedulerTask(time=current_time, task_plan={fia_room: [name, '菲亚梅塔']}))
    # 初始化运营商信息
    def initialize_operators(self):
        # 获取当前计划
        plan = self.current_plan
        # 初始化运营商数据
        self.op_data = Operators(self.agent_base_config, self.max_resting_count, plan)
        # 初始化并验证运营商数据
        return self.op_data.init_and_validate()
    
    # 检查是否存在菲亚梅塔运营商，并且所在房间以'dormitory'开头
    def check_fia(self):
        if '菲亚梅塔' in self.op_data.operators.keys() and self.op_data.operators['菲亚梅塔'].room.startswith('dormitory'):
            return self.op_data.operators['菲亚梅塔'].replacement, self.op_data.operators['菲亚梅塔'].room
        return None, None
    
    # 获取插拔时间
    def get_run_roder_time(self, room):
        logger.info('基建：读取插拔时间')
        # 点击进入该房间
        self.enter_room(room)
        # 进入房间详情
        error_count = 0
        while self.find('bill_accelerate') is None:
            if error_count > 5:
                raise Exception('未成功进入无人机界面')
            self.tap((self.recog.w * 0.05, self.recog.h * 0.95), interval=1)
            error_count += 1
        # 读取插拔时间
        execute_time = self.double_read_time((int(self.recog.w * 650 / 2496), int(self.recog.h * 660 / 1404),
                                              int(self.recog.w * 815 / 2496), int(self.recog.h * 710 / 1404)),
                                             use_digit_reader=True)
        # 计算下一次进行插拔的时间
        execute_time = execute_time - timedelta(seconds=(60 * self.run_order_delay))
        logger.info('下一次进行插拔的时间为：' + execute_time.strftime("%H:%M:%S"))
        logger.info('返回基建主界面')
        self.back(interval=2, rebuild=False)
        self.back(interval=2)
        return execute_time
    
    # 读取时间并返回时间对象
    def double_read_time(self, cord, upperLimit=None, use_digit_reader=False):
        self.recog.update()
        time_in_seconds = self.read_time(cord, upperLimit, use_digit_reader)
        if time_in_seconds is None:
            return datetime.now()
        execute_time = datetime.now() + timedelta(seconds=(time_in_seconds))
        return execute_time
    
    # 初始化PaddleOCR
    def initialize_paddle(self):
        global ocr
        if ocr is None:
            ocr = PaddleOCR(enable_mkldnn=False, use_angle_cls=False)
    # 从屏幕截图中读取信息
    def read_screen(self, img, type="mood", limit=24, cord=None):
        # 如果给定了坐标范围，截取对应区域的图像
        if cord is not None:
            img = img[cord[1]:cord[3], cord[0]:cord[2]]
        # 如果类型是心情或时间，对心情图片进行复制以提高准确率
        if 'mood' in type or type == "time":
            for x in range(0, 4):
                img = cv2.vconcat([img, img])
        try:
            # 初始化 PaddleOCR
            self.initialize_paddle()
            # 使用 OCR 进行图像识别
            rets = ocr.ocr(img, cls=True)
            line_conf = []
            # 遍历识别结果
            for idx in range(len(rets[0])):
                res = rets[0][idx]
                if 'mood' in type:
                    # 过滤不符合规范的结果
                    if ('/' + str(limit)) in res[1][0]:
                        new_string = res[1][0].replace('/' + str(limit), '')
                        if len(new_string) > 0:
                            line_conf.append(res[1])
                else:
                    line_conf.append(res[1])
            logger.debug(line_conf)
            # 如果没有识别到有效结果
            if len(line_conf) == 0:
                if 'mood' in type:
                    return -1
                elif 'name' in type:
                    logger.debug("使用老版识别")
                    return character_recognize.agent_name(img, self.recog.h)
                else:
                    return ""
            x = [i[0] for i in line_conf]
            # 获取出现次数最多的字符串
            __str = max(set(x), key=x.count)
            # 如果类型是心情
            if "mood" in type:
                if '.' in __str:
                    __str = __str.replace(".", "")
                number = int(__str[0:__str.index('/')])
                return number
            # 如果类型是时间
            elif 'time' in type:
                if '.' in __str:
                    __str = __str.replace(".", ":")
            # 如果类型是姓名且结果不在代理人列表中
            elif 'name' in type and __str not in agent_list:
                logger.debug("使用老版识别")
                __str = character_recognize.agent_name(img, self.recog.h)
            logger.debug(__str)
            return __str
        except Exception as e:
            logger.exception(e)
            return limit + 1
    # 读取时间信息
    def read_time(self, cord, upperlimit, error_count=0, use_digit_reader=False):
        # 刷新图片
        self.recog.update()
        # 如果使用数字识别器，则调用数字识别器获取时间字符串
        if use_digit_reader:
            time_str = self.digit_reader.get_time(self.recog.gray)
        else:
            # 否则从屏幕读取时间信息
            time_str = self.read_screen(self.recog.img, type='time', cord=cord)
        try:
            # 尝试将时间字符串按照小时、分钟、秒进行分割
            h, m, s = str(time_str).split(':')
            # 如果分钟或秒数大于60，则抛出异常
            if int(m) > 60 or int(s) > 60:
                raise Exception(f"读取错误")
            # 计算总秒数
            res = int(h) * 3600 + int(m) * 60 + int(s)
            # 如果存在上限，并且结果超过上限，则抛出异常
            if upperlimit is not None and res > upperlimit:
                raise Exception(f"超过读取上限")
            else:
                return res
        except:
            # 捕获异常，记录错误日志
            logger.error("读取失败")
            # 如果错误次数超过3次，则记录异常日志并返回None
            if error_count > 3:
                logger.exception(f"读取失败{error_count}次超过上限")
                return None
            else:
                # 否则递归调用自身，增加错误次数
                return self.read_time(cord, upperlimit, error_count + 1, use_digit_reader)

    # 处理基建 Todo 列表
    def todo_list(self) -> None:
        """ 处理基建 Todo 列表 """
        # 标记是否已经点击
        tapped = False
        # 查找干员信赖
        trust = self.find('infra_collect_trust')
        if trust is not None:
            logger.info('基建：干员信赖')
            self.tap(trust)
            tapped = True
        # 查找订单交付
        bill = self.find('infra_collect_bill')
        if bill is not None:
            logger.info('基建：订单交付')
            self.tap(bill)
            tapped = True
        # 查找可收获
        factory = self.find('infra_collect_factory')
        if factory is not None:
            logger.info('基建：可收获')
            self.tap(factory)
            tapped = True
        # 如果没有找到任何可点击的任务，则点击屏幕左下角
        if not tapped:
            self.tap((self.recog.w * 0.05, self.recog.h * 0.95))
            self.todo_task = True
    # 定义一个方法用于分享线索
    def share_clue(self):
        # 声明全局变量
        global x1, x2, x3, x4, y0, y1, y2
        # 初始化变量
        x1, x2, x3, x4 = 0, 0, 0, 0
        y0, y1, y2 = 0, 0, 0

        # 记录日志信息
        logger.info('基建：赠送线索')
        # 进入会客室
        self.enter_room('meeting')

        # 关闭掉房间总览
        error_count = 0
        while self.find('clue_func') is None:
            if error_count > 5:
                raise Exception('未成功进入线索详情界面')
            self.tap((self.recog.w * 0.1, self.recog.h * 0.9), interval=3)
            error_count += 1
        # 识别右侧按钮
        (x0, y0), (x1, y1) = self.find('clue_func', strict=True)

        # 点击按钮
        self.tap(((x0 + x1) // 2, (y0 + y1 * 3) // 4), interval=3, rebuild=True)
        # 检查是否处于连接状态，若是则等待
        if self.get_infra_scene() == Scene.CONNECTING:
            if not self.waiting_solver(Scene.CONNECTING, sleep_time=2):
                return
        # 识别条形码
        self.recog_bar()
        self.recog_view(only_y2=False)
        for i in range(1, 8):
            # 切换阵营
            self.tap(self.switch_camp(i))
            # 获得和线索视图有关的数据
            self.recog_view()
            ori_results = self.ori_clue()
            if len(ori_results) > 1:
                last_ori = ori_results[0]
                self.tap(((last_ori[0][0] + last_ori[2][0]) / 2, (last_ori[0][1] + last_ori[2][1]) / 2), interval=1)
                self.tap((self.recog.w * 0.93, self.recog.h * 0.15), interval=3)
                logger.info(f'赠送线索 {i} -->给一位随机的幸运儿')
                self.clue_count -= 1
                break
            else:
                continue
        # 检查是否处于连接状态，若是则等待
        if self.get_infra_scene() == Scene.CONNECTING:
            if not self.waiting_solver(Scene.CONNECTING, sleep_time=2):
                return
        # 点击返回按钮
        self.tap((self.recog.w * 0.95, self.recog.h * 0.05), interval=3)
        # 返回上一级页面
        self.back()
        self.back()
    # 在当前位置放置线索，参数为上一个方向
    def place_clue(self, last_ori):
        # 初始化错误计数
        error_count = 0
        # 当未找到未选择的线索时
        while self.find('clue_unselect') is None:
            # 如果错误计数超过3次，则抛出异常
            if error_count > 3:
                raise Exception('未成功放置线索')
            # 点击线索位置的中心点
            self.tap(((last_ori[0][0] + last_ori[2][0]) / 2, (last_ori[0][1] + last_ori[2][1]) / 2), interval=1)
            # 更新识别结果
            self.recog.update()
            # 如果识别到连接中的场景
            if self.get_infra_scene() == Scene.CONNECTING:
                # 如果等待解决者连接超时，则返回
                if not self.waiting_solver(Scene.CONNECTING, sleep_time=2):
                    return
            # 错误计数加一
            error_count += 1

    # 切换阵营，参数为阵营ID，返回值为坐标元组
    def switch_camp(self, id: int) -> tuple[int, int]:
        x = ((id + 0.5) * x2 + (8 - id - 0.5) * x1) // 8
        y = (y0 + y1) // 2
        return x, y

    # 识别阵营选择栏
    def recog_bar(self) -> None:
        global x1, x2, y0, y1
        # 获取阵营选择栏的位置
        (x1, y0), (x2, y1) = self.find('clue_nav', strict=True)
        # 调整位置以适应实际情况
        while int(self.recog.img[y0, x1 - 1].max()) - int(self.recog.img[y0, x1].max()) <= 1:
            x1 -= 1
        while int(self.recog.img[y0, x2].max()) - int(self.recog.img[y0, x2 - 1].max()) <= 1:
            x2 += 1
        while abs(int(self.recog.img[y1 + 1, x1].max()) - int(self.recog.img[y1, x1].max())) <= 1:
            y1 += 1
        y1 += 1
        # 输出调整后的位置信息
        logger.debug(f'recog_bar: x1:{x1}, x2:{x2}, y0:{y0}, y1:{y1}')

    # 识别与线索视图相关的其他数据
    def recog_view(self, only_y2: bool = True) -> None:
        global x1, x2, x3, x4, y0, y1, y2
        # 获取线索底部位置
        y2 = self.recog.h
        while self.recog.img[y2 - 1, x1:x2].ptp() <= 24:
            y2 -= 1
        # 如果只需要线索底部位置，则直接返回
        if only_y2:
            logger.debug(f'recog_view: y2:{y2}')
            return y2
        # 获取右边黑色 mask 边缘位置
        x3 = self.recog_view_mask_right()
        # 区分单个线索的位置
        x4 = (54 * x1 + 25 * x2) // 79
        # 输出位置信息
        logger.debug(f'recog_view: y2:{y2}, x3:{x3}, x4:{x4}')
    def recog_view_mask_right(self) -> int:
        """ 识别线索视图中右边黑色 mask 边缘的位置 """
        # 将 x3 初始化为 x2
        x3 = x2
        # 进入循环，直到条件不满足
        while True:
            # 初始化最大绝对值为 0
            max_abs = 0
            # 遍历 y1 到 y2 之间的每一个 y 坐标
            for y in range(y1, y2):
                # 计算当前像素点与前一个像素点的颜色值差的绝对值，并更新最大绝对值
                max_abs = max(max_abs,
                              abs(int(self.recog.img[y, x3 - 1, 0]) - int(self.recog.img[y, x3 - 2, 0])))
            # 如果最大绝对值小于等于 5，则将 x3 减 1
            if max_abs <= 5:
                x3 -= 1
            else:
                break
        # 初始化标志为 False
        flag = False
        # 遍历 y1 到 y2 之间的每一个 y 坐标
        for y in range(y1, y2):
            # 如果当前像素点与前一个像素点的颜色值差的绝对值等于最大绝对值，则将标志设为 True
            if int(self.recog.img[y, x3 - 1, 0]) - int(self.recog.img[y, x3 - 2, 0]) == max_abs:
                flag = True
        # 如果标志为 False
        if not flag:
            # 在 ((x1 + x2) // 2, y1 + 10) 处进行点击操作，不重建
            self.tap(((x1 + x2) // 2, y1 + 10), rebuild=False)
            # 将 x3 初始化为 x2
            x3 = x2
            # 进入循环，直到条件不满足
            while True:
                # 初始化最大绝对值为 0
                max_abs = 0
                # 遍历 y1 到 y2 之间的每一个 y 坐标
                for y in range(y1, y2):
                    # 计算当前像素点与前一个像素点的颜色值差的绝对值，并更新最大绝对值
                    max_abs = max(max_abs,
                                  abs(int(self.recog.img[y, x3 - 1, 0]) - int(self.recog.img[y, x3 - 2, 0])))
                # 如果最大绝对值小于等于 5，则将 x3 减 1
                if max_abs <= 5:
                    x3 -= 1
                else:
                    break
            # 初始化标志为 False
            flag = False
            # 遍历 y1 到 y2 之间的每一个 y 坐标
            for y in range(y1, y2):
                # 如果当前像素点与前一个像素点的颜色值差的绝对值等于最大绝对值，则将标志设为 True
                if int(self.recog.img[y, x3 - 1, 0]) - int(self.recog.img[y, x3 - 2, 0]) == max_abs:
                    flag = True
            # 如果标志为 False
            if not flag:
                # 将 x3 设为 None
                x3 = None
        # 返回 x3
        return x3

    def get_clue_mask(self) -> None:
        """ 界面内是否有被选中的线索 """
        try:
            # 初始化 mask 为空列表
            mask = []
            # 遍历 y1 到 y2 之间的每一个 y 坐标
            for y in range(y1, y2):
                # 如果当前像素点与前一个像素点的颜色值差大于 20，并且当前像素点与前一个像素点的颜色值差的范围为 0，则将 y 添加到 mask 中
                if int(self.recog.img[y, x3 - 1, 0]) - int(self.recog.img[y, x3 - 2, 0]) > 20 and np.ptp(
                        self.recog.img[y, x3 - 2]) == 0:
                    mask.append(y)
            # 如果 mask 的长度大于 0
            if len(mask) > 0:
                # 计算 mask 中元素的平均值，并返回
                logger.debug(np.average(mask))
                return np.average(mask)
            else:
                # 返回 None
                return None
        except Exception as e:
            # 抛出识别错误异常
            raise RecognizeError(e)
    def clear_clue_mask(self) -> None:
        """ 清空界面内被选中的线索 """
        try:
            while True:
                mask = False
                for y in range(y1, y2):
                    # 检查像素值的差异，如果大于20并且像素值范围为0，则执行下一步操作
                    if int(self.recog.img[y, x3 - 1, 0]) - int(self.recog.img[y, x3 - 2, 0]) > 20 and np.ptp(
                            self.recog.img[y, x3 - 2]) == 0:
                        # 点击坐标(x3-2, y+1)，并重新构建界面
                        self.tap((x3 - 2, y + 1), rebuild=True)
                        mask = True
                        break
                if mask:
                    continue
                break
        except Exception as e:
            # 抛出识别错误异常
            raise RecognizeError(e)

    def ori_clue(self):
        """ 获取界面内有多少线索 """
        clues = []
        y3 = y1
        status = -2
        for y in range(y1, y2):
            # 检查像素值范围，如果小于192，则执行下一步操作
            if self.recog.img[y, x4 - 5:x4 + 5].max() < 192:
                if status == -1:
                    status = 20
                if status > 0:
                    status -= 1
                if status == 0:
                    status = -2
                    # 获取(x1, x2, y3, y-20)范围内的多边形线索
                    clues.append(segment.get_poly(x1, x2, y3, y - 20))
                    y3 = y - 20 + 5
            else:
                status = -1
        if status != -2:
            # 获取(x1, x2, y3, y2)范围内的多边形线索
            clues.append(segment.get_poly(x1, x2, y3, y2)

        # 忽视一些只有一半的线索，将线索转换为列表形式
        clues = [x.tolist() for x in clues if x[1][1] - x[0][1] >= self.recog.h / 5]
        logger.debug(clues)
        return clues
    # 定义一个方法，用于进入指定房间并返回房间的位置
    def enter_room(self, room: str) -> tp.Rectangle:
        """ 获取房间的位置并进入 """
        # 初始化成功标志和重试次数
        success = False
        retry = 3
        # 循环直到成功或者重试次数用尽
        while not success:
            try:
                # 获取基建各个房间的位置
                base_room = segment.base(self.recog.img, self.find('control_central', strict=True))
                # 将画面外的部分删去
                _room = base_room[room]

                # 对房间位置进行边界处理
                for i in range(4):
                    _room[i, 0] = max(_room[i, 0], 0)
                    _room[i, 0] = min(_room[i, 0], self.recog.w)
                    _room[i, 1] = max(_room[i, 1], 0)
                    _room[i, 1] = min(_room[i, 1], self.recog.h)

                # 点击进入房间
                self.tap(_room[0], interval=3)
                # 循环直到成功进入房间
                while self.find('control_central') is not None:
                    self.tap(_room[0], interval=3)
                success = True
            except Exception as e:
                # 出现异常时减少重试次数，返回基建主界面，等待基建主界面出现
                retry -= 1
                self.back_to_infrastructure()
                self.wait_for_scene(Scene.INFRA_MAIN, "get_infra_scene")
                # 如果重试次数用尽，则抛出异常
                if retry <= 0:
                    raise e

    # 定义一个方法，用于获取最佳的布置顺序
    def get_arrange_order(self) -> ArrangeOrder:
        # 初始化最佳分数和最佳顺序
        best_score, best_order = 0, None
        # 遍历所有布置顺序
        for order in ArrangeOrder:
            # 获取当前布置顺序的分数
            score = self.recog.score(arrange_order_res[order][0])
            # 如果分数不为空且大于最佳分数，则更新最佳分数和最佳顺序
            if score is not None and score[0] > best_score:
                best_score, best_order = score[0], order
        # 返回最佳顺序
        logger.debug((best_score, best_order))
        return best_order
    # 切换指定位置的元素顺序
    def switch_arrange_order(self, index: int, asc="false") -> None:
        # 点击指定位置的元素，根据参数决定是否升序或降序
        self.tap((self.recog.w * arrange_order_res[ArrangeOrder(index)][0],
                  self.recog.h * arrange_order_res[ArrangeOrder(index)][1]), interval=0, rebuild=False)
        # 如果位置小于4，则点击下一个位置的元素
        if index < 4:
            self.tap((self.recog.w * arrange_order_res[ArrangeOrder(index + 1)][0],
                      self.recog.h * arrange_order_res[ArrangeOrder(index)][1]), interval=0, rebuild=False)
        # 如果位置大于等于4，则点击上一个位置的元素
        else:
            self.tap((self.recog.w * arrange_order_res[ArrangeOrder(index - 1)][0],
                      self.recog.h * arrange_order_res[ArrangeOrder(index)][1]), interval=0, rebuild=False)
        # 点击回到原来位置的元素，间隔0.2秒
        self.tap((self.recog.w * arrange_order_res[ArrangeOrder(index)][0],
                  self.recog.h * arrange_order_res[ArrangeOrder(index)][1]), interval=0.2, rebuild=True)
        # 如果asc参数不为"false"，则再次点击回到原来位置的元素，间隔0.2秒，实现倒序
        if asc != "false":
            self.tap((self.recog.w * arrange_order_res[ArrangeOrder(index)][0],
                      self.recog.h * arrange_order_res[ArrangeOrder(index)][1]), interval=0.2, rebuild=True)
    # 扫描干员列表，识别干员并返回识别结果
    def scan_agant(self, agent: list[str], error_count=0, max_agent_count=-1):
        try:
            # 更新识别器
            self.recog.update()
            # 识别干员并返回识别结果，顺序是从左往右从上往下
            ret = character_recognize.agent(self.recog.img)  
            # 提取识别出来的干员的名字
            select_name = []
            for y in ret:
                name = y[0]
                if name in agent:
                    select_name.append(name)
                    # 点击干员位置
                    self.tap((y[1][0]), interval=0)
                    # 从干员列表中移除已选择的干员
                    agent.remove(name)
                    # 如果是按照个数选择 Free
                    if max_agent_count != -1:
                        if len(select_name) >= max_agent_count:
                            return select_name, ret
            return select_name, ret
        except Exception as e:
            # 错误计数加一
            error_count += 1
            if error_count < 3:
                # 记录异常信息
                logger.exception(e)
                # 等待3秒
                self.sleep(3)
                # 重新扫描干员
                return self.scan_agant(agent, error_count, max_agent_count)
            else:
                # 错误次数超过3次，抛出异常
                raise e

    # 获取干员的排序信息
    def get_order(self, name):
        if name in self.agent_base_config and "ArrangeOrder" in self.agent_base_config[name]:
            return True, self.agent_base_config[name]["ArrangeOrder"]
        return False, self.agent_base_config["Default"]["ArrangeOrder"]

    # 过滤干员详情
    def detail_filter(self, turn_on, type="not_in_dorm"):
        # 记录日志
        logger.info(f'开始 {("打开" if turn_on else "关闭")} {type} 筛选')
        # 点击筛选按钮
        self.tap((self.recog.w * 0.95, self.recog.h * 0.05), interval=1)
        if type == "not_in_dorm":
            # 查找非宿舍中的干员
            not_in_dorm = self.find('arrange_non_check_in', score=0.9)
            # 如果需要打开或关闭非宿舍中的干员筛选
            if turn_on ^ (not_in_dorm is None):
                # 点击非宿舍中的干员筛选按钮
                self.tap((self.recog.w * 0.3, self.recog.h * 0.5), interval=0.5)
        # 确认筛选
        self.tap((self.recog.w * 0.8, self.recog.h * 0.8), interval=0.5)
    # 向左滑动指定次数
    def swipe_left(self, right_swipe, w, h):
        for _ in range(right_swipe):
            self.swipe_only((w // 2, h // 2), (w // 2, 0), interval=0.5)
        return 0

    # 读取准确的情绪值
    def read_accurate_mood(self, img, cord):
        try:
            # 裁剪图像
            img = img[cord[1]:cord[3], cord[0]:cord[2]]
            # 将图像转换为灰度图
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 对灰度图进行高斯模糊处理
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

            # 对图像进行阈值处理，以分离进度条区域
            contours, hierarchy = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 计算进度条的边界框
            x, y, w, h = cv2.boundingRect(contours[0])

            # 裁剪进度条区域
            progress_bar = img[y:y + h, x:x + w]

            # 将进度条转换为灰度图
            gray_pb = cv2.cvtColor(progress_bar, cv2.COLOR_BGR2GRAY)

            # 对进度条进行阈值处理，以分离灰色填充
            ret, thresh_pb = cv2.threshold(gray_pb, 137, 255, cv2.THRESH_BINARY)

            # 计算进度条区域中有色像素与总像素数的比率
            total_pixels = w * h
            colored_pixels = cv2.countNonZero(thresh_pb)
            return colored_pixels / total_pixels * 24

        except Exception:
            return 24

    # 刷新当前房间
    @push_operators
    def refresh_current_room(self, room, current_index=None):
        _current_room = self.op_data.get_current_room(room, current_index=current_index)
        if _current_room is None:
            self.get_agent_from_room(room)
            _current_room = self.op_data.get_current_room(room, True)
        return _current_room
    # 跳过指定任务，如果未指定任务名，则默认跳过所有任务
    def skip(self, task_names='All'):
        # 如果未指定任务名，则默认为 ['planned', 'collect_notification', 'todo_task']
        if task_names == 'All':
            task_names = ['planned', 'collect_notification', 'todo_task']
        # 如果 'planned' 在任务名列表中，则设置 self.planned 为 True
        if 'planned' in task_names:
            self.planned = True
        # 如果 'todo_task' 在任务名列表中，则设置 self.todo_task 为 True
        if 'todo_task' in task_names:
            self.todo_task = True
        # 如果 'collect_notification' 在任务名列表中，则设置 self.collect_notification 为 True
        if 'collect_notification' in task_names:
            self.collect_notification = True

    # 重新加载任务
    def reload(self):
        error = False
        # 遍历需要重新加载的房间列表
        for room in self.reload_room:
            try:
                # 进入指定房间
                self.enter_room(room)
                # 点击指定位置，执行补货操作
                self.tap((self.recog.w * 0.05, self.recog.h * 0.95), interval=0.5)
                # 补货
                self.tap((self.recog.w * 0.75, self.recog.h * 0.3), interval=0.5)
                self.tap((self.recog.w * 0.75, self.recog.h * 0.9), interval=0.5)
                # 如果当前场景为连接中，则等待连接完成
                if self.get_infra_scene() == Scene.CONNECTING:
                    if not self.waiting_solver(Scene.CONNECTING, sleep_time=2):
                        return
                # 返回上一级场景
                self.back()
                self.back()
            except Exception as e:
                # 记录错误日志
                logger.error(e)
                error = True
                # 更新识别结果
                self.recog.update()
                back_count = 0
                # 当前场景不为 INFRA_MAIN 时，返回上一级场景
                while self.get_infra_scene() != Scene.INFRA_MAIN:
                    self.back()
                    self.recog.update()
                    back_count += 1
                    # 如果返回次数超过3次，则抛出异常
                    if back_count > 3:
                        raise e
        # 如果没有错误发生，则更新重新加载时间为当前时间
        if not error:
            self.reload_time = datetime.now()

    # 回调类型为 Asst.CallBackType
    @Asst.CallBackType
    # 记录消息、详细信息和参数到日志
    def log_maa(msg, details, arg):
        # 创建消息对象
        m = Message(msg)
        # 解析详细信息为 JSON 格式
        d = json.loads(details.decode('utf-8'))
        # 记录详细信息到日志
        logger.debug(d)
        # 记录消息到日志
        logger.debug(m)
        # 记录参数到日志
        logger.debug(arg)
        # 如果详细信息中包含 "what" 并且值为 "StageDrops"
        if "what" in d and d["what"] == "StageDrops":
            # 添加详细信息中的 "drops" 到全局变量 stage_drop 的 "details" 列表中
            global stage_drop
            stage_drop["details"].append(d["details"]["drops"])
            # 更新全局变量 stage_drop 的 "summary" 值为详细信息中的 "stats"
            stage_drop["summary"] = d["details"]["stats"]
        # 如果详细信息中包含 "what" 并且值为 "RecruitTagsSelected"
        elif "what" in d and d["what"] == "RecruitTagsSelected":
            # 添加详细信息中的 "tags" 到全局变量 recruit_tags_selected 的 "tags" 列表中
            global recruit_tags_selected
            recruit_tags_selected["tags"].append(d["details"]["tags"])
        # 如果详细信息中包含 "what" 并且值为 "RecruitResult"
        elif "what" in d and d["what"] == "RecruitResult":
            # 添加详细信息中的 "tags"、"level" 和 "result" 到全局变量 recruit_results 的 "results" 列表中
            global recruit_results
            temp_dict = {
                "tags": d["details"]["tags"],
                "level": d["details"]["level"],
                "result": d["details"]["result"],
            }
            recruit_results["results"].append(temp_dict)
        # 如果详细信息中包含 "what" 并且值为 "RecruitSpecialTag"
        elif "what" in d and d["what"] == "RecruitSpecialTag":
            # 添加详细信息中的 "tags" 到全局变量 recruit_special_tags 的 "tags" 列表中
            global recruit_special_tags
            recruit_special_tags["tags"].append(d["details"]["tags"])

    # 初始化MAA
    def initialize_maa(self):
        # 若需要获取详细执行信息，请传入 callback 参数
        # 例如 asst = Asst(callback=my_callback)
        # 加载MAA配置文件
        Asst.load(path=self.maa_config['maa_path'])
        # 创建MAA对象，并传入log_maa作为回调函数
        self.MAA = Asst(callback=self.log_maa)
        # 初始化阶段列表
        self.stages = []
        # 设置MAA实例选项
        self.MAA.set_instance_option(2, self.maa_config['touch_option'])
        # 连接到MAA设备
        if self.MAA.connect(self.maa_config['maa_adb_path'], self.device.client.device_id,
                            self.maa_config["conn_preset"]):
            logger.info("MAA 连接成功")
        else:
            logger.info("MAA 连接失败")
            raise Exception("MAA 连接失败")
    # 如果邮件功能未开启，则记录日志并返回
        # logger.info('邮件功能未开启')
        # return
    
        # 创建一个多部分 MIME 消息
        # msg = MIMEMultipart()
        # 将正文和子类型附加到消息上
        # msg.attach(MIMEText(body, subtype))
        # 设置邮件主题
        # msg['Subject'] = self.email_config['subject'] + subject
        # 设置发件人
        # msg['From'] = self.email_config['account']
    
        # 当重试次数大于 0 时，循环发送邮件
        # while retry_times > 0:
        #     尝试连接到 QQ 邮箱的 SMTP 服务器
        #     s = smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=10.0)
        #     # 登录邮箱
        #     s.login(self.email_config['account'], self.email_config['pass_code'])
        #     # 开始发送邮件
        #     s.sendmail(self.email_config['account'], self.email_config['receipts'], msg.as_string())
        #     记录日志，表示邮件发送成功
        #     logger.info("邮件发送成功")
        #     退出循环
        #     break
        # 如果发送邮件过程中出现异常，则记录日志并减少重试次数，然后休眠 3 秒
        # except Exception as e:
        #     logger.error("邮件发送失败")
        #     记录异常信息
        #     logger.exception(e)
        #     减少重试次数
        #     retry_times -= 1
        #     休眠 3 秒
        #     time.sleep(3)
```