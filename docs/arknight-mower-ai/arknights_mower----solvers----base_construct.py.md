# `arknights-mower\arknights_mower\solvers\base_construct.py`

```
# 导入未来版本的注解特性
from __future__ import annotations

# 导入枚举类型
from enum import Enum

# 导入 numpy 库，并使用别名 np
import numpy as np

# 从上级目录的 data 模块中导入 base_room_list
from ..data import base_room_list

# 从上级目录的 utils 模块中导入 character_recognize, detector, segment 和 typealias
from ..utils import character_recognize, detector, segment
from ..utils import typealias as tp

# 从上级目录的 utils.device 模块中导入 Device
from ..utils.device import Device

# 从上级目录的 utils.log 模块中导入 logger
from ..utils.log import logger

# 从上级目录的 utils.recognize 模块中导入 RecognizeError, Recognizer 和 Scene
from ..utils.recognize import RecognizeError, Recognizer, Scene

# 从上级目录的 utils.solver 模块中导入 BaseSolver
from ..utils.solver import BaseSolver

# 定义枚举类型 ArrangeOrder
class ArrangeOrder(Enum):
    STATUS = 1
    SKILL = 2
    FEELING = 3
    TRUST = 4

# 定义 arrange_order_res 字典，存储 ArrangeOrder 对应的值
arrange_order_res = {
    ArrangeOrder.STATUS: ('arrange_status', 0.1),
    ArrangeOrder.SKILL: ('arrange_skill', 0.35),
    ArrangeOrder.FEELING: ('arrange_feeling', 0.65),
    ArrangeOrder.TRUST: ('arrange_trust', 0.9),
}

# 定义 BaseConstructSolver 类，继承自 BaseSolver 类
class BaseConstructSolver(BaseSolver):
    """
    收集基建的产物：物资、赤金、信赖
    """

    # 初始化方法
    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)

    # 运行方法
    def run(self, arrange: dict[str, tp.BasePlan] = None, clue_collect: bool = False, drone_room: str = None, fia_room: str = None) -> None:
        """
        :param arrange: dict(room_name: [agent,...]), 基建干员安排
        :param clue_collect: bool, 是否收取线索
        :param drone_room: str, 是否使用无人机加速
        :param fia_room: str, 是否使用菲亚梅塔恢复心情
        """
        # 设置实例变量
        self.arrange = arrange
        self.clue_collect = clue_collect
        self.drone_room = drone_room
        self.fia_room = fia_room
        self.todo_task = False   # 基建 Todo 是否未被处理

        # 输出日志信息
        logger.info('Start: 基建')
        # 调用父类的 run 方法
        super().run()
    # 定义状态转换方法
    def transition(self) -> None:
        # 如果当前场景是首页
        if self.scene() == Scene.INDEX:
            # 点击首页的基建按钮
            self.tap_element('index_infrastructure')
        # 如果当前场景是基建主页
        elif self.scene() == Scene.INFRA_MAIN:
            # 调用基建主页方法
            return self.infra_main()
        # 如果当前场景是基建待办事项列表
        elif self.scene() == Scene.INFRA_TODOLIST:
            # 调用待办事项列表方法
            return self.todo_list()
        # 如果当前场景是基建详情页
        elif self.scene() == Scene.INFRA_DETAILS:
            # 如果找到安排签到按钮
            if self.find('arrange_check_in_on'):
                # 点击安排签到按钮
                self.tap_element('arrange_check_in_on')
            # 返回上一级页面
            self.back()
        # 如果当前场景是加载中
        elif self.scene() == Scene.LOADING:
            # 等待3秒
            self.sleep(3)
        # 如果当前场景是连接中
        elif self.scene() == Scene.CONNECTING:
            # 等待3秒
            self.sleep(3)
        # 如果存在导航栏
        elif self.get_navigation():
            # 点击基建导航按钮
            self.tap_element('nav_infrastructure')
        # 如果当前场景是基建安排订单
        elif self.scene() == Scene.INFRA_ARRANGE_ORDER:
            # 点击蓝色确认按钮
            self.tap_element('arrange_blue_yes')
        # 如果当前场景不是未知场景
        elif self.scene() != Scene.UNKNOWN:
            # 返回到首页
            self.back_to_index()
        # 否则
        else:
            # 抛出识别错误
            raise RecognizeError('Unknown scene')

    # 定义基建主页方法
    def infra_main(self) -> None:
        """ 位于基建首页 """
        # 如果找不到控制中枢
        if self.find('control_central') is None:
            # 返回上一级页面
            self.back()
            return
        # 如果有线索收集
        if self.clue_collect:
            # 调用线索方法
            self.clue()
            self.clue_collect = False
        # 如果有无人机房间
        elif self.drone_room is not None:
            # 调用无人机方法
            self.drone(self.drone_room)
            self.drone_room = None
        # 如果有采购部房间
        elif self.fia_room is not None:
            # 调用采购部方法
            self.fia(self.fia_room)
            self.fia_room = None
        # 如果有安排订单
        elif self.arrange is not None:
            # 调用代理安排方法
            self.agent_arrange(self.arrange)
            self.arrange = None
        # 如果没有待办任务
        elif not self.todo_task:
            # 处理基建待办事项
            notification = detector.infra_notification(self.recog.img)
            # 如果没有通知
            if notification is None:
                # 等待1秒
                self.sleep(1)
                notification = detector.infra_notification(self.recog.img)
            # 如果有通知
            if notification is not None:
                # 点击通知
                self.tap(notification)
            else:
                # 设置待办任务为True
                self.todo_task = True
        # 否则
        else:
            # 返回True
            return True
    # 处理基建 Todo 列表
    def todo_list(self) -> None:
        # 初始化 tapped 变量
        tapped = False
        # 查找干员信赖任务
        trust = self.find('infra_collect_trust')
        # 如果找到了干员信赖任务
        if trust is not None:
            # 输出日志信息
            logger.info('基建：干员信赖')
            # 点击干员信赖任务
            self.tap(trust)
            # 更新 tapped 变量
            tapped = True
        # 查找订单交付任务
        bill = self.find('infra_collect_bill')
        # 如果找到了订单交付任务
        if bill is not None:
            # 输出日志信息
            logger.info('基建：订单交付')
            # 点击订单交付任务
            self.tap(bill)
            # 更新 tapped 变量
            tapped = True
        # 查找可收获任务
        factory = self.find('infra_collect_factory')
        # 如果找到了可收获任务
        if factory is not None:
            # 输出日志信息
            logger.info('基建：可收获')
            # 点击可收获任务
            self.tap(factory)
            # 更新 tapped 变量
            tapped = True
        # 如果没有找到任何任务
        if not tapped:
            # 点击屏幕底部中央位置
            self.tap((self.recog.w*0.05, self.recog.h*0.95))
            # 更新 todo_task 变量
            self.todo_task = True

    # 切换阵营
    def switch_camp(self, id: int) -> tuple[int, int]:
        # 计算 x 坐标
        x = ((id+0.5) * x2 + (8-id-0.5) * x1) // 8
        # 计算 y 坐标
        y = (y0 + y1) // 2
        # 返回坐标元组
        return x, y

    # 识别阵营选择栏
    def recog_bar(self) -> None:
        # 声明全局变量
        global x1, x2, y0, y1

        # 查找阵营选择栏的位置
        (x1, y0), (x2, y1) = self.find('clue_nav', strict=True)
        # 调整 x1 的位置
        while int(self.recog.img[y0, x1-1].max()) - int(self.recog.img[y0, x1].max()) <= 1:
            x1 -= 1
        # 调整 x2 的位置
        while int(self.recog.img[y0, x2].max()) - int(self.recog.img[y0, x2-1].max()) <= 1:
            x2 += 1
        # 调整 y1 的位置
        while abs(int(self.recog.img[y1+1, x1].max()) - int(self.recog.img[y1, x1].max())) <= 1:
            y1 += 1
        y1 += 1

        # 输出调整后的位置信息
        logger.debug(f'recog_bar: x1:{x1}, x2:{x2}, y0:{y0}, y1:{y1}')
    def recog_view(self, only_y2: bool = True) -> None:
        """ 识别另外一些和线索视图有关的数据 """
        global x1, x2, x3, x4, y0, y1, y2

        # y2: 线索底部
        y2 = self.recog.h
        while self.recog.img[y2-1, x1:x2].ptp() <= 24:
            y2 -= 1
        if only_y2:
            logger.debug(f'recog_view: y2:{y2}')
            return y2
        # x3: 右边黑色 mask 边缘
        x3 = self.recog_view_mask_right()
        # x4: 用来区分单个线索
        x4 = (54 * x1 + 25 * x2) // 79

        logger.debug(f'recog_view: y2:{y2}, x3:{x3}, x4:{x4}')

    def recog_view_mask_right(self) -> int:
        """ 识别线索视图中右边黑色 mask 边缘的位置 """
        x3 = x2
        while True:
            max_abs = 0
            for y in range(y1, y2):
                max_abs = max(max_abs,
                              abs(int(self.recog.img[y, x3-1, 0]) - int(self.recog.img[y, x3-2, 0])))
            if max_abs <= 5:
                x3 -= 1
            else:
                break
        flag = False
        for y in range(y1, y2):
            if int(self.recog.img[y, x3-1, 0]) - int(self.recog.img[y, x3-2, 0]) == max_abs:
                flag = True
        if not flag:
            self.tap(((x1+x2)//2, y1+10), rebuild=False)
            x3 = x2
            while True:
                max_abs = 0
                for y in range(y1, y2):
                    max_abs = max(max_abs,
                                  abs(int(self.recog.img[y, x3-1, 0]) - int(self.recog.img[y, x3-2, 0])))
                if max_abs <= 5:
                    x3 -= 1
                else:
                    break
            flag = False
            for y in range(y1, y2):
                if int(self.recog.img[y, x3-1, 0]) - int(self.recog.img[y, x3-2, 0]) == max_abs:
                    flag = True
            if not flag:
                x3 = None
        return x3
    def get_clue_mask(self) -> None:
        """ 界面内是否有被选中的线索 """
        try:
            mask = []  # 创建一个空列表用于存储被选中的线索
            for y in range(y1, y2):  # 遍历指定范围内的 y 坐标
                if int(self.recog.img[y, x3-1, 0]) - int(self.recog.img[y, x3-2, 0]) > 20 and np.ptp(self.recog.img[y, x3-2]) == 0:
                    # 判断条件，如果满足则将当前 y 坐标添加到 mask 列表中
                    mask.append(y)
            if len(mask) > 0:  # 如果 mask 列表不为空
                logger.debug(np.average(mask))  # 记录 mask 列表中 y 坐标的平均值
                return np.average(mask)  # 返回 y 坐标的平均值
            else:  # 如果 mask 列表为空
                return None  # 返回空值
        except Exception as e:  # 捕获异常
            raise RecognizeError(e)  # 抛出自定义的 RecognizeError 异常

    def clear_clue_mask(self) -> None:
        """ 清空界面内被选中的线索 """
        try:
            while True:  # 无限循环
                mask = False  # 初始化 mask 变量为 False
                for y in range(y1, y2):  # 遍历指定范围内的 y 坐标
                    if int(self.recog.img[y, x3-1, 0]) - int(self.recog.img[y, x3-2, 0]) > 20 and np.ptp(self.recog.img[y, x3-2]) == 0:
                        # 判断条件，如果满足则执行以下操作
                        self.tap((x3-2, y+1), rebuild=True)  # 在指定位置进行点击操作
                        mask = True  # 将 mask 变量设置为 True
                        break  # 跳出当前循环
                if mask:  # 如果 mask 变量为 True
                    continue  # 继续下一次循环
                break  # 跳出循环
        except Exception as e:  # 捕获异常
            raise RecognizeError(e)  # 抛出自定义的 RecognizeError 异常

    def ori_clue(self):
        """ 获取界面内有多少线索 """
        clues = []  # 创建一个空列表用于存储线索
        y3 = y1  # 初始化 y3 变量为 y1
        status = -2  # 初始化 status 变量为 -2
        for y in range(y1, y2):  # 遍历指定范围内的 y 坐标
            if self.recog.img[y, x4-5:x4+5].max() < 192:  # 判断条件
                if status == -1:  # 判断条件
                    status = 20  # 设置 status 变量的值为 20
                if status > 0:  # 判断条件
                    status -= 1  # status 变量减 1
                if status == 0:  # 判断条件
                    status = -2  # 设置 status 变量的值为 -2
                    clues.append(segment.get_poly(x1, x2, y3, y-20))  # 将获取的线索添加到 clues 列表中
                    y3 = y-20+5  # 更新 y3 变量的值
            else:  # 如果条件不满足
                status = -1  # 设置 status 变量的值为 -1
        if status != -2:  # 判断条件
            clues.append(segment.get_poly(x1, x2, y3, y2))  # 将获取的线索添加到 clues 列表中

        # 忽视一些只有一半的线索
        clues = [x.tolist() for x in clues if x[1][1] - x[0][1] >= self.recog.h / 5]  # 过滤掉长度小于界面高度五分之一的线索
        logger.debug(clues)  # 记录 clues 列表
        return clues  # 返回线索列表
    def enter_room(self, room: str) -> tp.Rectangle:
        """ 获取房间的位置并进入 """

        # 获取基建各个房间的位置
        base_room = segment.base(self.recog.img, self.find('control_central', strict=True))

        # 将画面外的部分删去
        room = base_room[room]
        for i in range(4):
            room[i, 0] = max(room[i, 0], 0)  # 确保房间位置不超出画面左边界
            room[i, 0] = min(room[i, 0], self.recog.w)  # 确保房间位置不超出画面右边界
            room[i, 1] = max(room[i, 1], 0)  # 确保房间位置不超出画面上边界
            room[i, 1] = min(room[i, 1], self.recog.h)  # 确保房间位置不超出画面下边界

        # 点击进入
        self.tap(room[0], interval=3)  # 点击进入房间，间隔3秒
        while self.find('control_central') is not None:  # 当在房间内时，持续点击进入
            self.tap(room[0], interval=3)

    def drone(self, room: str):
        logger.info('基建：无人机加速')

        # 点击进入该房间
        self.enter_room(room)

        # 进入房间详情
        self.tap((self.recog.w*0.05, self.recog.h*0.95), interval=3)  # 点击进入房间详情，间隔3秒

        accelerate = self.find('factory_accelerate')
        if accelerate:
            logger.info('制造站加速')
            self.tap(accelerate)  # 点击加速按钮
            self.tap_element('all_in')  # 点击全部加速
            self.tap(accelerate, y_rate=1)  # 点击加速按钮，y轴位置比例为1

        else:
            accelerate = self.find('bill_accelerate')
            while accelerate:
                logger.info('贸易站加速')
                self.tap(accelerate)  # 点击加速按钮
                self.tap_element('all_in')  # 点击全部加速
                self.tap((self.recog.w*0.75, self.recog.h*0.8), interval=3)  # 点击相对位置为0.75, 0.8的按钮，间隔3秒

                st = accelerate[1]   # 起点
                ed = accelerate[0]   # 终点
                # 0.95, 1.05 are offset compensations
                self.swipe_noinertia(st, (ed[0]*0.95-st[0]*1.05, 0), rebuild=True)  # 无惯性滑动
                accelerate = self.find('bill_accelerate')  # 继续查找加速按钮

        logger.info('返回基建主界面')
        self.back(interval=2, rebuild=False)  # 返回基建主界面，间隔2秒
        self.back(interval=2)  # 再次返回基建主界面，间隔2秒
    # 获取最佳排列顺序及其得分
    def get_arrange_order(self) -> ArrangeOrder:
        best_score, best_order = 0, None
        # 遍历所有排列顺序
        for order in ArrangeOrder:
            # 计算当前排列顺序的得分
            score = self.recog.score(arrange_order_res[order][0])
            # 如果得分不为空且大于最佳得分，则更新最佳得分和最佳排列顺序
            if score is not None and score[0] > best_score:
                best_score, best_order = score[0], order
        # 输出最佳得分和最佳排列顺序的调试信息
        logger.debug((best_score, best_order))
        # 返回最佳排列顺序
        return best_order

    # 切换排列顺序
    def switch_arrange_order(self, order: ArrangeOrder) -> None:
        # 点击指定排列顺序的元素
        self.tap_element(arrange_order_res[order][0], x_rate=arrange_order_res[order][1], judge=False)

    # 排列顺序
    def arrange_order(self, order: ArrangeOrder) -> None:
        # 如果当前排列顺序不是目标排列顺序，则切换到目标排列顺序
        if self.get_arrange_order() != order:
            self.switch_arrange_order(order)

    # 统计线索
    # def clue_statis(self):

    #     clues = {'all': {}, 'own': {}}

    #     self.recog_bar()
    #     self.tap(((x1*7+x2)//8, y0//2), rebuild=False)
    #     self.tap(((x1*7.5+x2*0.5)//8, (y0+y1)//2), rebuild=False)
    #     self.recog_view(only_y2=False)

    #     if x3 is None:
    #         return clues

    #     for i in range(1, 8):

    #         self.tap((((i+0.5)*x2+(8-i-0.5)*x1)//8, (y0+y1)//2), rebuild=False)
    #         self.clear_clue_mask()
    #         self.recog_view()

    #         count = 0
    #         if y2 < self.recog.h - 20:
    #             count = len(self.ori_clue())
    #         else:
    #             while True:
    #                 restart = False
    #                 count = 0
    #                 ret = self.ori_clue()
    #                 while True:

    #                     y4 = 0
    #                     for poly in ret:
    #                         count += 1
    #                         y4 = poly[0, 1]

    #                     self.tap((x4, y4+10), rebuild=False)
    #                     self.device.swipe([(x4, y4), (x4, y1+10), (0, y1+10)], duration=(y4-y1-10)*3)
    #                     self.sleep(1, rebuild=False)
    # 获取线索遮罩
    mask = self.get_clue_mask()
    # 如果存在线索遮罩，则清除线索遮罩
    if mask is not None:
        self.clear_clue_mask()
    # 获取原始线索
    ret = self.ori_clue()

    # 如果不存在线索遮罩或者线索遮罩不在原始线索范围内
    if mask is None or not (ret[0][0, 1] <= mask <= ret[-1][1, 1]):
        # 重新开始
        restart = True
        break

    # 如果线索遮罩在第一个原始线索范围内
    if ret[0][0, 1] <= mask <= ret[0][1, 1]:
        count -= 1
        continue
    else:
        # 遍历原始线索
        for poly in ret:
            if mask < poly[0, 1]:
                count += 1
        break

    # 如果需要重新开始
    if restart:
        self.swipe((x4, y1+10), (0, 1000),
                   duration=500, interval=3, rebuild=False)
        continue
    break

# 更新线索数量
clues['all'][i] = count

# 点击屏幕中间位置
self.tap(((x1+x2)//2, y0//2), rebuild=False)

# 循环点击8个位置
for i in range(1, 8):
    self.tap((((i+0.5)*x2+(8-i-0.5)*x1)//8, (y0+y1)//2), rebuild=False)

    # 清除线索遮罩并识别视图
    self.clear_clue_mask()
    self.recog_view()

    count = 0
    # 如果y2小于视图高度-20
    if y2 < self.recog.h - 20:
        count = len(self.ori_clue())
    else:
        while True:
            restart = False
            count = 0
            ret = self.ori_clue()
            while True:
                # 获取最后一个原始线索的y坐标
                y4 = 0
                for poly in ret:
                    count += 1
                    y4 = poly[0, 1]

                # 点击屏幕指定位置并滑动
                self.tap((x4, y4+10), rebuild=False)
                self.device.swipe([(x4, y4), (x4, y1+10), (0, y1+10)], duration=(y4-y1-10)*3)
                self.sleep(1, rebuild=False)
    # 获取线索的遮罩
    # 如果存在遮罩，则清除遮罩
    # 获取原始线索
    # 如果不存在遮罩或者遮罩不在原始线索范围内，则重新开始
    # 如果遮罩在第一个线索的范围内，则减少计数并继续
    # 否则，遍历线索并增加计数，直到找到遮罩所在的线索
    # 如果需要重新开始，则重新滑动屏幕并继续
    # 更新线索字典中的计数值
    # 返回线索字典
```