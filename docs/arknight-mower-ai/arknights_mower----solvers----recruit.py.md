# `arknights-mower\arknights_mower\solvers\recruit.py`

```
# 导入未来版本的注解特性
from __future__ import annotations

# 导入 combinations 函数
from itertools import combinations

# 从上级目录的 data 模块中导入 recruit_agent, recruit_tag, recruit_agent_list
from ..data import recruit_agent, recruit_tag, recruit_agent_list

# 从上级目录的 ocr 模块中导入 ocr_rectify, ocrhandle
from ..ocr import ocr_rectify, ocrhandle

# 从上级目录的 utils 模块中导入 segment, Device, recruit_template, recruit_rarity, logger, RecognizeError, Recognizer, Scene, BaseSolver
from ..utils import segment
from ..utils.device import Device
from ..utils.email import recruit_template, recruit_rarity
from ..utils.log import logger
from ..utils.recognize import RecognizeError, Recognizer, Scene
from ..utils.solver import BaseSolver


# 定义公招标签组合的可能性数据类
class RecruitPoss(object):
    """ 记录公招标签组合的可能性数据 """

    def __init__(self, choose: int, max: int = 0, min: int = 7) -> None:
        self.choose = choose  # 标签选择（按位），第 6 个标志位表示是否选满招募时限，0 为选满，1 为选 03:50
        self.max = max  # 等级上限
        self.min = min  # 等级下限
        self.poss = 0  # 可能性
        self.lv2a3 = False  # 是否包含等级为 2 和 3 的干员
        self.ls = []  # 可能的干员列表

    def __lt__(self, another: RecruitPoss) -> bool:
        return (self.poss) < (another.poss)

    def __str__(self) -> str:
        return "%s,%s,%s,%s,%s" % (self.choose, self.max, self.min, self.poss, self.ls)

    def __repr__(self) -> str:
        return "%s,%s,%s,%s,%s" % (self.choose, self.max, self.min, self.poss, self.ls)


# 定义公招求解器类
class RecruitSolver(BaseSolver):
    """
    自动进行公招
    """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)

        self.result_agent = {}  # 存储招募结果的字典
        self.agent_choose = {}  # 存储干员选择的字典
        self.recruit_config = {}  # 存储招募配置的字典

        self.recruit_pos = -1  # 招募位置初始化为 -1
    # 定义一个方法，用于运行公招任务
    def run(self, priority: list[str] = None, email_config={}, maa_config={}) -> None:
        """
        :param priority: list[str], 优先考虑的公招干员，默认为高稀有度优先
        """
        # 设置优先考虑的公招干员
        self.priority = priority
        # 初始化招募次数为0
        self.recruiting = 0
        # 默认含有招募票
        self.has_ticket = True
        # 默认可以刷新
        self.can_refresh = True
        # 设置邮件配置
        self.email_config = email_config

        # 调整公招参数
        self.add_recruit_param(maa_config)

        # 输出日志信息
        logger.info('Start: 公招')
        # 清空结果干员字典
        self.result_agent.clear()

        # 初始化结果干员字典和干员选择字典
        self.result_agent = {}
        self.agent_choose = {}

        # 尝试运行公招任务，捕获异常并记录日志
        try:
            super().run()
        except Exception as e:
            logger.error(e)

        # 输出干员选择和结果干员的调试信息
        logger.debug(self.agent_choose)
        logger.debug(self.result_agent)

        # 如果有结果干员，则输出结果汇总信息
        if self.result_agent:
            logger.info(f"上次公招结果汇总{self.result_agent}")

        # 如果有干员选择，则输出公招标签信息
        if self.agent_choose:
            logger.info(f'公招标签：{self.agent_choose}')
        # 如果有干员选择或结果干员，则发送邮件通知
        if self.agent_choose or self.result_agent:
            self.send_email(recruit_template.render(recruit_results=self.agent_choose,
                                                    recruit_get_agent=self.result_agent,
                                                    title_text="公招汇总"), "公招汇总通知", "html")

    # 定义一个方法，用于添加公招参数
    def add_recruit_param(self, maa_config):
        # 如果招募设置为空，则抛出异常
        if not maa_config:
            raise Exception("招募设置为空")

        # 根据招募时间设置招募时间参数
        if maa_config['recruitment_time']:
            recruitment_time = 460
        else:
            recruitment_time = 540

        # 设置招募配置参数
        self.recruit_config = {
            "recruit_only_4": maa_config['recruit_only_4'],
            "recruitment_time": {
                "3": recruitment_time,
                "4": 540
            }
        }
    # 定义状态转换函数，返回布尔值
    def transition(self) -> bool:
        # 如果当前场景是INDEX
        if self.scene() == Scene.INDEX:
            # 点击index_recruit元素
            self.tap_element('index_recruit')
        # 如果当前场景是RECRUIT_MAIN
        elif self.scene() == Scene.RECRUIT_MAIN:
            # 对招募主界面进行分割
            segments = segment.recruit(self.recog.img)
            # 标记是否已经点击
            tapped = False
            # 遍历分割后的区域
            for idx, seg in enumerate(segments):
                # 重置招募位置为-1
                self.recruit_pos = -1
                # 如果正在招募并且当前位置已经招募过，则继续下一次循环
                if self.recruiting & (1 << idx) != 0:
                    continue
                # 如果点击了recruit_finish元素
                if self.tap_element('recruit_finish', scope=seg, detected=True):
                    # 设置招募位置为当前位置
                    self.recruit_pos = idx
                    tapped = True
                    break
                # 如果没有招募券并且不能刷新，则继续下一次循环
                if not self.has_ticket and not self.can_refresh:
                    continue
                # 在当前区域找到职业需求
                required = self.find('job_requirements', scope=seg)
                # 如果没有职业需求，则设置招募位置为当前位置，点击当前位置，标记为已招募
                if required is None:
                    self.recruit_pos = idx
                    self.tap(seg)
                    tapped = True
                    self.recruiting |= (1 << idx)
                    break
            # 如果没有点击，则返回True
            if not tapped:
                return True
        # 如果当前场景是RECRUIT_TAGS
        elif self.scene() == Scene.RECRUIT_TAGS:
            # 调用recruit_tags函数
            return self.recruit_tags()
        # 如果当前场景是SKIP
        elif self.scene() == Scene.SKIP:
            # 点击skip元素
            self.tap_element('skip')
        # 如果当前场景是RECRUIT_AGENT
        elif self.scene() == Scene.RECRUIT_AGENT:
            # 调用recruit_result函数
            return self.recruit_result()
        # 如果当前场景是MATERIEL
        elif self.scene() == Scene.MATERIEL:
            # 点击materiel_ico元素
            self.tap_element('materiel_ico')
        # 如果当前场景是LOADING
        elif self.scene() == Scene.LOADING:
            # 等待3秒
            self.sleep(3)
        # 如果当前场景是CONNECTING
        elif self.scene() == Scene.CONNECTING:
            # 等待3秒
            self.sleep(3)
        # 如果有导航
        elif self.get_navigation():
            # 点击nav_recruit元素
            self.tap_element('nav_recruit')
        # 如果当前场景不是UNKNOWN
        elif self.scene() != Scene.UNKNOWN:
            # 返回到INDEX场景
            self.back_to_index()
        # 否则，抛出识别错误
        else:
            raise RecognizeError('Unknown scene')
    def recruit_result(self) -> bool:
        """ 识别公招招募到的干员 """
        """ 卡在首次获得 挖个坑"""
        # 初始化干员名称为 None
        agent = None
        # 使用 OCR 模块对图片进行识别
        ocr = ocrhandle.predict(self.recog.img)
        # 遍历 OCR 结果
        for x in ocr:
            # 如果识别结果以'的信物'结尾，则提取干员名称
            if x[1][-3:] == '的信物':
                agent = x[1][:-3]
                agent_ocr = x
                break
        # 如果未能识别到干员名称，则记录警告日志
        if agent is None:
            logger.warning('未能识别到干员名称')
        else:
            # 如果干员名称不在招募干员字典中，则进行纠正
            if agent not in recruit_agent.keys():
                agent_with_suf = [x + '的信物' for x in recruit_agent.keys()]
                agent = ocr_rectify(
                    self.recog.img, agent_ocr, agent_with_suf, '干员名称')[:-3]
            # 如果干员名称在招募干员字典中，则记录日志
            if agent in recruit_agent.keys():
                if 2 <= recruit_agent[agent]['stars'] <= 4:
                    logger.info(f'获得干员：{agent}')
                else:
                    logger.critical(f'获得干员：{agent}')

        if agent is not None:
            # 将招募结果添加到结果字典中
            self.result_agent[str(self.recruit_pos + 1)] = agent

        # 点击屏幕中央位置
        self.tap((self.recog.w // 2, self.recog.h // 2))

    def recruit_str(self, recruit_result: dict):
        # 如果招募结果为空，则返回默认字符串
        if not recruit_result:
            return "随机三星干员"
        # 初始化结果字符串
        result_str = "{"
        # 遍历招募结果字典
        for key in recruit_result:
            # 构造临时字符串
            temp_str = "{[" + ",".join(list(key))
            temp_str = temp_str + "],level:"
            temp_str = temp_str + str(recruit_result[key]["level"]) + ",agent:"
            temp_str = temp_str + str(recruit_result[key]["agent"]) + "},"
            result_str = result_str + temp_str

        return result_str
```