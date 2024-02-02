# `arknights-mower\arknights_mower\solvers\operation.py`

```py
# 导入未来版本的注解特性
from __future__ import annotations

# 导入时间模块
import time
# 导入异常追踪模块
import traceback

# 从上级目录的 data 模块中导入 chapter_list, level_list, weekly_zones, zone_list
from ..data import chapter_list, level_list, weekly_zones, zone_list
# 从上级目录的 ocr 模块中导入 ocrhandle
from ..ocr import ocrhandle
# 从上级目录的 utils 模块中导入 config, typealias as tp
from ..utils import config
from ..utils import typealias as tp
# 从上级目录的 utils.image 模块中导入 scope2slice
from ..utils.image import scope2slice
# 从上级目录的 utils.log 模块中导入 logger
from ..utils.log import logger
# 从上级目录的 utils.recognize 模块中导入 RecognizeError, Scene
from ..utils.recognize import RecognizeError, Scene
# 从上级目录的 utils.solver 模块中导入 BaseSolver, StrategyError
from ..utils.solver import BaseSolver, StrategyError

# 定义常量 BOTTOM_TAP_NUMER 为 8
BOTTOM_TAP_NUMER = 8

# 定义一个自定义异常类 LevelUnopenError，继承自内置异常类 Exception
class LevelUnopenError(Exception):
    pass

# 定义一个类 OpeSolver，继承自 BaseSolver 类
class OpeSolver(BaseSolver):
    """
    自动作战策略
    """

    # 初始化方法，接受 device 和 recog 两个参数
    def __init__(self, device=None, recog=None):
        # 调用父类的初始化方法
        super().__init__(device, recog)
    def run(self, level: str = None, times: int = -1, potion: int = 0, originite: int = 0, eliminate: int = 0, plan: list = None):
        """
        :param level: str, 指定关卡，默认为前往上一次关卡或当前界面关卡
        :param times: int, 作战的次数上限，-1 为无限制，默认为 -1
        :param potion: int, 使用药剂恢复体力的次数上限，-1 为无限制，默认为 0
        :param originite: int, 使用源石恢复体力的次数上限，-1 为无限制，默认为 0
        :param eliminate: int, 是否优先处理未完成的每周剿灭，0 为忽略剿灭，1 为优先剿灭，2 为优先剿灭但只消耗代理卡，默认为 0
        :param plan: [[str, int]...], 指定多个关卡以及次数，优先级高于 level

        :return remain_plan: [[str, int]...], 未完成的计划
        """
        # 如果同时指定了 level 和 plan，则报错并返回
        if level is not None and plan is not None:
            logger.error('不可同时指定 level 和 plan')
            return
        # 如果指定了 plan，则检查每个关卡是否合法
        if plan is not None:
            for x in plan:
                if x[0] != 'pre_ope' and (x[0] not in level_list.keys() or level_list[x[0]]['ap_cost'] == 0):
                    logger.error(f'不支持关卡 {x[0]}，请重新指定')
                    return
        # 如果只指定了 level，则检查关卡是否合法
        if level is not None:
            if level not in level_list.keys() or level_list[level]['ap_cost'] == 0:
                logger.error(f'不支持关卡 {level}，请重新指定')
                return
            plan = [[level, times]]
        # 如果没有指定 plan，则默认为上一次作战关卡
        if plan is None:
            plan = [['pre_ope', times]]  # 上一次作战关卡

        # 设置各个参数的值
        self.level = level
        self.times = times
        self.potion = potion
        self.originite = originite
        self.eliminate = eliminate
        self.plan = plan

        # 设置相关状态的初始值
        self.recover_state = 0  # 有关体力恢复的状态，0 为未知，1 为体力药剂恢复中，2 为源石恢复中（防止网络波动）
        self.eliminate_state = 0  # 有关每周剿灭的状态，0 为未知，1 为未完成，2 为已完成，3 为未完成但无代理卡可用
        self.wait_pre = 10  # 作战时每次等待的时长，普通关卡为 10s，剿灭关卡为 60s
        self.wait_start = 0  # 作战时第一次等待的时长
        self.wait_total = 0  # 作战时累计等待的时长
        self.level_choosed = plan[0][0] == 'pre_ope'  # 是否已经选定关卡
        self.unopen = []  # 未开放的关卡
        self.failed = False  # 作战代理是否正常运作

        # 输出日志信息
        logger.info('Start: 作战')
        logger.debug(f'plan: {plan}')
        # 调用父类的 run 方法
        super().run()
        # 返回未完成的计划
        return self.plan + self.unopen
    # 切换当前计划，将计划列表中的第一个元素移除
    def switch_plan(self) -> None:
        self.plan = self.plan[1:]
        # 重置等待开始时间
        self.wait_start = 0
        # 重置关卡选择状态
        self.level_choosed = False

    # 主要终端操作
    def terminal_main(self) -> bool:
        # 如果剿灭状态不为3
        if self.eliminate_state != 3:
            # 查找需要执行的剿灭任务
            eliminate_todo = self.find('terminal_eliminate')
            # 检查每周剿灭完成情况
            if eliminate_todo is not None:
                self.eliminate_state = 1
            else:
                self.eliminate_state = 2
            # 如果每周剿灭未完成且设定为优先处理
            if self.eliminate and eliminate_todo is not None:
                # 点击执行剿灭任务
                self.tap(eliminate_todo)
                return
        try:
            # 选择关卡
            self.choose_level(self.plan[0][0])
        except LevelUnopenError:
            # 如果关卡未开放，记录错误日志并切换计划
            logger.error(f'关卡 {self.plan[0][0]} 未开放，请重新指定')
            self.unopen.append(self.plan[0])
            self.switch_plan()
            return
        # 设置关卡选择状态为已完成
        self.level_choosed = True
    # 操作前的准备工作，返回布尔值
    def operator_before(self) -> bool:
        # 如果关卡未选定，退回到终端主界面选择关卡
        if not self.level_choosed:
            self.get_navigation()
            self.tap_element('nav_terminal')
            return
        # 如果代理出现过失误，终止作战
        if self.failed:
            return True
        # 激活代理作战
        agency = self.find('ope_agency')
        if agency is not None:
            self.tap(agency)
            return
        # 重置普通关卡等待时长
        if self.wait_pre != 10:
            self.wait_start = 0
            self.wait_pre = 10
        self.wait_total = 0
        # 点击开始作战
        # ope_start_SN 对应三周年活动愚人船的 UI
        ope_start = self.find('ope_start', judge=False) or self.find('ope_start_SN', judge=False)
        if ope_start:
            self.tap(ope_start)
            # 确定可以开始作战后扣除相应的消耗药剂或者源石
            if self.recover_state == 1:
                logger.info('use potion to recover sanity')
                self.potion -= 1
            elif self.recover_state == 2:
                logger.info('use originite to recover sanity')
                self.originite -= 1
            self.recover_state = 0
        else:
            # 与预期不符，等待一阵并重新截图
            self.sleep(1)
    # 检查每周剿灭完成情况，如果未知则返回到终端主界面选择关卡
    def operator_before_elimi(self) -> bool:
        if self.eliminate_state == 0:
            self.get_navigation()
            self.tap_element('nav_terminal')
            return
        # 如果每周剿灭已完成但仍然在剿灭关卡前，则只可能是 pre_ope 为剿灭关卡，此时应该退出
        if self.eliminate_state == 2:
            logger.warning('检测到关卡为剿灭，但每周剿灭任务已完成')
            return True
        # 如果剿灭代理卡已经用完但仍然在剿灭关卡前，则只可能是 pre_ope 为剿灭关卡，此时应该退出
        if self.eliminate_state == 3:
            logger.warning('检测到关卡为剿灭，但剿灭代理卡已经用完')
            return True
        # 代理出现过失误，终止作战
        if self.failed:
            return True
        # 激活代理作战
        agency = self.find('ope_elimi_agency')
        if agency is not None:
            self.tap(agency)
            return
        agency = self.find('ope_agency')
        if agency is not None:
            self.tap(agency)
            return
        # 若只想用代理卡，但此时代理卡已经用光，则退回到终端主界面选择关卡
        if self.eliminate == 2 and self.find('ope_elimi_agenct_used') is None:
            self.eliminate_state = 3
            self.get_navigation()
            self.tap_element('nav_terminal')
            return
        # 重置剿灭关卡等待时长
        if self.wait_pre != 60:
            self.wait_start = 0
            self.wait_pre = 60
        self.wait_total = 0
        # 点击开始作战
        self.tap_element('ope_start')
        # 确定可以开始作战后扣除相应的消耗药剂或者源石
        if self.recover_state == 1:
            logger.info('use potion to recover sanity')
            self.potion -= 1
        elif self.recover_state == 2:
            logger.info('use originite to recover sanity')
            self.originite -= 1
        self.recover_state = 0
    # 定义一个方法，用于处理进行中的操作
    def ope_ongoing(self) -> None:
        # 如果等待总时间小于等于等待开始时间
        if self.wait_total < self.wait_start:
            # 如果等待总时间为0
            if self.wait_total == 0:
                # 记录日志，等待等待开始时间秒
                logger.info(f'等待 {self.wait_start} 秒')
            # 等待总时间增加等待间隔时间
            self.wait_total += self.wait_pre
            # 如果等待总时间等于等待开始时间
            if self.wait_total == self.wait_start:
                # 调用sleep方法等待等待间隔时间
                self.sleep(self.wait_pre)
            else:
                # 否则，直接调用time.sleep方法等待等待间隔时间
                time.sleep(self.wait_pre)
        else:
            # 否则，记录日志，等待等待间隔时间秒
            logger.info(f'等待 {self.wait_pre} 秒')
            # 等待总时间增加等待间隔时间
            self.wait_total += self.wait_pre
            # 调用sleep方法等待等待间隔时间
            self.sleep(self.wait_pre)

    # 定义一个方法，用于处理操作完成后的操作
    def ope_finish(self) -> None:
        # 更新等待开始时间
        if self.wait_total > 0:
            # 如果等待开始时间为0
            if self.wait_start == 0:
                # 等待开始时间为等待总时间减去等待间隔时间
                self.wait_start = self.wait_total - self.wait_pre
            else:
                # 否则，等待开始时间为等待开始时间加上等待间隔时间和等待总时间减去等待间隔时间的最小值
                self.wait_start = min(
                    self.wait_start + self.wait_pre, self.wait_total - self.wait_pre)
            # 等待总时间重置为0
            self.wait_total = 0
        # 如果关卡已选定，则扣除任务次数
        if self.level_choosed:
            self.plan[0][1] -= 1
        # 若每周剿灭未完成，则剿灭完成状态变为未知
        if self.eliminate_state == 1:
            self.eliminate_state = 0
        # 随便点击某处退出结算界面
        self.tap((self.recog.w // 2, 10))

    # 定义一个方法，用于处理剿灭完成后的操作
    def ope_finish_elimi(self) -> None:
        # 每周剿灭完成情况变为未知
        self.eliminate_state = 0
        # 随便点击某处退出结算界面
        self.tap((self.recog.w // 2, 10))
    # 恢复药剂的方法，返回布尔值
    def recover_potion(self) -> bool:
        # 如果药剂数量为0
        if self.potion == 0:
            # 如果源石数量不为0
            if self.originite != 0:
                # 转而去使用源石恢复
                self.tap_element('ope_recover_originite')
                return
            # 关闭恢复界面
            self.back()
            return True
        # 如果正在恢复状态
        elif self.recover_state:
            # 正在恢复中，防止网络波动
            self.sleep(3)
        else:
            # 选择药剂恢复体力
            if self.find('ope_recover_potion_empty') is not None:
                # 使用次数未归零但已经没有药剂可以恢复体力了
                logger.info('The potions have been used up.')
                self.potion = 0
                return
            self.tap_element('ope_recover_potion_choose', 0.9, 0.75, judge=False)
            # 修改状态
            self.recover_state = 1

    # 恢复源石的方法，返回布尔值
    def recover_originite(self) -> bool:
        # 如果源石数量为0
        if self.originite == 0:
            # 如果药剂数量不为0
            if self.potion != 0:
                # 转而去使用药剂恢复
                self.tap_element('ope_recover_potion')
                return
            # 关闭恢复界面
            self.back()
            return True
        # 如果正在恢复状态
        elif self.recover_state:
            # 正在恢复中，防止网络波动
            self.sleep(3)
        else:
            # 选择源石恢复体力
            if self.find('ope_recover_originite_empty') is not None:
                # 使用次数未归零但已经没有源石可以恢复体力了
                logger.info('The originites have been used up.')
                self.originite = 0
                return
            self.tap_element('ope_recover_originite_choose', 0.9, 0.85, judge=False)
            # 修改状态
            self.recover_state = 2

    # 使用 OCR 进行等级识别，返回识别结果和等级列表
    def ocr_level(self) -> list:
        # 使用 OCR 进行等级识别
        ocr = ocrhandle.predict(self.recog.img)
        # 过滤出等级列表中的识别结果
        ocr = list(filter(lambda x: x[1] in level_list.keys(), ocr))
        # 对等级列表进行排序
        levels = sorted([x[1] for x in ocr])
        return ocr, levels

    # 切换底部按钮的方法，不返回任何值
    def switch_bottom(self, id: int) -> None:
        # 计算底部按钮的位置
        id = id * 2 + 1
        bottom = self.recog.h - 10
        self.tap((self.recog.w//BOTTOM_TAP_NUMER//2*id, bottom))
    def choose_zone_theme(self, zone: list, scope: tp.Scope) -> None:
        """ 识别主题曲区域 """
        # 定位 Chapter 编号
        ocr = []  # 初始化 OCR 结果列表
        act_id = 999  # 初始化 act_id 变量为 999
        while act_id != zone['chapterIndex']:  # 当 act_id 不等于 zone['chapterIndex'] 时循环
            _act_id = act_id  # 将当前 act_id 赋值给 _act_id
            act_id = -1  # 将 act_id 初始化为 -1
            for x in ocr:  # 遍历 OCR 结果列表
                if zone['chapterIndex'] < _act_id:  # 如果 zone['chapterIndex'] 小于 _act_id
                    if x[1].upper().replace(' ', '') == chapter_list[_act_id-1].replace(' ', ''):  # 如果 OCR 结果中的文本去除空格转为大写等于 chapter_list 中的对应文本
                        self.tap(x[2])  # 调用 tap 方法
                        break  # 跳出循环
                else:  # 否则
                    if x[1].upper().replace(' ', '') == chapter_list[_act_id+1].replace(' ', ''):  # 如果 OCR 结果中的文本去除空格转为大写等于 chapter_list 中的对应文本
                        self.tap(x[2])  # 调用 tap 方法
                        break  # 跳出循环
            ocr = ocrhandle.predict(self.recog.img[scope2slice(scope)])  # 使用 OCR 模块预测图片中的文本
            for x in ocr:  # 遍历 OCR 结果列表
                if x[1][:7].upper() == 'EPISODE' and len(x[1]) == 9:  # 如果文本的前7个字符转为大写等于 'EPISODE' 且文本长度为9
                    try:  # 尝试执行以下代码
                        episode = int(x[1][-2:])  # 将文本倒数第二到最后转为整数赋值给 episode
                        act_id = zone_list[f'main_{episode}']['chapterIndex']  # 获取 zone_list 中对应键的值赋值给 act_id
                        break  # 跳出循环
                    except Exception:  # 如果出现异常
                        raise RecognizeError('Unknown episode')  # 抛出 RecognizeError 异常，提示未知的 episode
            if act_id == -1 or _act_id == act_id:  # 如果 act_id 等于 -1 或者 _act_id 等于 act_id
                raise RecognizeError('Unknown error')  # 抛出 RecognizeError 异常，提示未知错误

        # 定位 Episode 编号
        cover = self.find(f'main_{episode}')  # 调用 find 方法，找到对应的封面
        while zone['zoneIndex'] < episode:  # 当 zone['zoneIndex'] 小于 episode 时循环
            self.swipe_noinertia((cover[0][0], cover[0][1]),  # 调用 swipe_noinertia 方法，进行滑动操作
                                 (cover[1][0] - cover[0][0], 0))
            episode -= 1  # episode 减一
        while episode < zone['zoneIndex']:  # 当 episode 小于 zone['zoneIndex'] 时循环
            self.swipe_noinertia((cover[1][0], cover[0][1]),  # 调用 swipe_noinertia 方法，进行滑动操作
                                 (cover[0][0] - cover[1][0], 0))
            episode += 1  # episode 加一
        self.tap(cover)  # 调用 tap 方法，点击封面
    def choose_zone_supple(self, zone: list, scope: tp.Scope) -> None:
        """ 识别别传/插曲区域 """
        # 设置尝试次数
        try_times = 5
        # 创建空字典用于存储区域索引
        zoneIndex = {}
        # 遍历区域列表，将区域名称去除特殊符号后作为键，区域索引作为值存入字典
        for x in zone_list.values():
            zoneIndex[x['name'].replace('·', '')] = x['zoneIndex']
        # 循环尝试识别区域，直到尝试次数用尽
        while try_times:
            try_times -= 1
            # 使用 OCR 模块识别指定范围内的图像内容
            ocr = ocrhandle.predict(self.recog.img[scope2slice(scope)])
            # 创建空集合用于存储识别出的区域索引
            zones = set()
            # 遍历 OCR 结果，将识别出的区域索引加入集合
            for x in ocr:
                if x[1] in zoneIndex.keys():
                    zones.add(zoneIndex[x[1]])
            # 输出识别出的区域索引
            logger.debug(zones)
            # 如果指定区域的索引在识别结果中，则执行相应操作
            if zone['zoneIndex'] in zones:
                for x in ocr:
                    if x[1] == zone['name'].replace('·', ''):
                        self.tap(x[2])
                        self.tap_element('enter')
                        return
                # 如果指定区域的名称未在识别结果中，则抛出识别错误
                raise RecognizeError
            else:
                st, ed = None, None
                # 遍历 OCR 结果，找到识别出的区域索引的起始和结束位置
                for x in ocr:
                    if x[1] in zoneIndex.keys() and zoneIndex[x[1]] == min(zones):
                        ed = x[2][0]
                    elif x[1] in zoneIndex.keys() and zoneIndex[x[1]] == max(zones):
                        st = x[2][0]
                # 输出起始和结束位置
                logger.debug((st, ed))
                # 执行惯性滑动操作
                self.swipe_noinertia(st, (0, ed[1]-st[1]))
    def choose_zone_resource(self, zone: list) -> None:
        """ 识别资源收集区域 """
        # 使用 OCR 模块对当前图像进行识别
        ocr = ocrhandle.predict(self.recog.img)
        # 过滤出不可进入或者本日16:00开启的区域
        unable = list(filter(lambda x: x[1] in ['不可进入', '本日16:00开启'], ocr))
        # 过滤出符合本周区域的信息
        ocr = list(filter(lambda x: x[1] in weekly_zones, ocr))
        # 对符合本周区域的信息进行排序
        weekly = sorted([x[1] for x in ocr])
        # 当选择的区域不在本周区域内时，进行滑动操作直到找到为止
        while zone['name'] not in weekly:
            _weekly = weekly
            self.swipe((self.recog.w // 4, self.recog.h // 4),
                       (self.recog.w // 16, 0))
            ocr = ocrhandle.predict(self.recog.img)
            unable = list(filter(lambda x: x[1] in ['不可进入', '本日16:00开启'], ocr))
            ocr = list(filter(lambda x: x[1] in weekly_zones, ocr))
            weekly = sorted([x[1] for x in ocr])
            if _weekly == weekly:
                break
        # 当选择的区域不在本周区域内时，进行滑动操作直到找到为止
        while zone['name'] not in weekly:
            _weekly = weekly
            self.swipe((self.recog.w // 4, self.recog.h // 4),
                       (-self.recog.w // 16, 0))
            ocr = ocrhandle.predict(self.recog.img)
            unable = list(filter(lambda x: x[1] in ['不可进入', '本日16:00开启'], ocr))
            ocr = list(filter(lambda x: x[1] in weekly_zones, ocr))
            weekly = sorted([x[1] for x in ocr])
            if _weekly == weekly:
                break
        # 如果选择的区域不在本周区域内，则抛出异常
        if zone['name'] not in weekly:
            raise RecognizeError('Not as expected')
        # 遍历 OCR 结果，找到选择的区域并进行相应操作
        for x in ocr:
            if x[1] == zone['name']:
                # 检查选择的区域是否有不可进入的情况，如果有则抛出异常
                for item in unable:
                    if x[2][0][0] < item[2][0][0] < x[2][1][0]:
                        raise LevelUnopenError
                # 点击选择的区域
                self.tap(x[2])
                ocr = ocrhandle.predict(self.recog.img)
                # 检查选择的区域是否为关卡尚未开放，如果是则抛出异常
                unable = list(filter(lambda x: x[1] == '关卡尚未开放', ocr))
                if len(unable):
                    raise LevelUnopenError
                break
```