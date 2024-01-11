# `arknights-mower\arknights_mower\utils\solver.py`

```
# 导入未来版本的注解特性
from __future__ import annotations

# 导入用于发送邮件的模块
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 导入时间、异常追踪和抽象方法相关的模块
import time
import traceback
from abc import abstractmethod

# 导入自定义的类型别名和其他模块
from ..utils import typealias as tp
from . import config, detector
from .device import Device, KeyCode
from .log import logger
from .recognize import RecognizeError, Recognizer, Scene


# 自定义异常类，用于表示策略错误
class StrategyError(Exception):
    """ Strategy Error """
    pass


# 基础求解器类，提供基本操作
class BaseSolver:
    """ Base class, provide basic operation """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        # 初始化方法，接受设备和识别器对象作为参数
        # 如果设备为空且识别器不为空，则抛出运行时错误
        if device is None and recog is not None:
            raise RuntimeError
        # 根据参数初始化设备和识别器对象
        self.device = device if device is not None else Device()
        self.recog = recog if recog is not None else Recognizer(self.device)
        # 检查当前焦点
        self.device.check_current_focus()
        # 更新识别器
        self.recog.update()

    def run(self) -> None:
        # 运行方法，执行状态转换
        retry_times = config.MAX_RETRYTIME
        result = None
        while retry_times > 0:
            try:
                # 执行状态转换
                result = self.transition()
                if result:
                    return result
            except RecognizeError as e:
                # 捕获识别错误并记录日志
                logger.warning(f'识别出了点小差错 qwq: {e}')
                self.recog.save_screencap('failure')
                retry_times -= 1
                self.sleep(3)
                continue
            except StrategyError as e:
                # 捕获策略错误并记录日志
                logger.error(e)
                logger.debug(traceback.format_exc())
                return
            except Exception as e:
                # 捕获其他异常并抛出
                raise e
            retry_times = config.MAX_RETRYTIME

    @abstractmethod
    def transition(self) -> bool:
        # 抽象方法，表示从一个状态到另一个状态的转换
        return True  # 表示任务完成
    # 获取指定位置的像素颜色
    def get_color(self, pos: tp.Coordinate) -> tp.Pixel:
        """ get the color of the pixel """
        return self.recog.color(pos[0], pos[1])

    # 根据给定的位置信息和比例获取坐标
    def get_pos(self, poly: tp.Location, x_rate: float = 0.5, y_rate: float = 0.5) -> tp.Coordinate:
        """ get the pos form tp.Location """
        if poly is None:
            raise RecognizeError('poly is empty')
        elif len(poly) == 4:
            # tp.Rectangle
            x = (poly[0][0] * (1 - x_rate) + poly[1][0] * (1 - x_rate) +
                 poly[2][0] * x_rate + poly[3][0] * x_rate) / 2
            y = (poly[0][1] * (1 - y_rate) + poly[3][1] * (1 - y_rate) +
                 poly[1][1] * y_rate + poly[2][1] * y_rate) / 2
        elif len(poly) == 2 and isinstance(poly[0], (list, tuple)):
            # tp.Scope
            x = poly[0][0] * (1 - x_rate) + poly[1][0] * x_rate
            y = poly[0][1] * (1 - y_rate) + poly[1][1] * y_rate
        else:
            # tp.Coordinate
            x, y = poly
        return (int(x), int(y))

    # 休眠指定时间间隔
    def sleep(self, interval: float = 1, rebuild: bool = True) -> None:
        """ sleeping for a interval """
        time.sleep(interval)
        self.recog.update(rebuild=rebuild)

    # 在指定区域输入文本
    def input(self, referent: str, input_area: tp.Scope, text: str = None) -> None:
        """ input text """
        logger.debug(f'input: {referent} {input_area}')
        self.device.tap(self.get_pos(input_area))
        time.sleep(0.5)
        if text is None:
            text = input(referent).strip()
        self.device.send_text(str(text))
        self.device.tap((0, 0))

    # 在图像中查找指定的目标
    def find(self, res: str, draw: bool = False, scope: tp.Scope = None, thres: int = None, judge: bool = True,
             strict: bool = False, score=0.0) -> tp.Scope:
        return self.recog.find(res, draw, scope, thres, judge, strict, score)
    # 在给定的多边形区域内进行点击操作
    def tap(self, poly: tp.Location, x_rate: float = 0.5, y_rate: float = 0.5, interval: float = 1,
            rebuild: bool = True) -> None:
        """ tap """
        # 获取点击位置
        pos = self.get_pos(poly, x_rate, y_rate)
        # 在设备上进行点击操作
        self.device.tap(pos)
        # 如果设置了间隔时间，进行休眠
        if interval > 0:
            self.sleep(interval, rebuild)

    # 在指定元素上进行点击操作
    def tap_element(self, element_name: str, x_rate: float = 0.5, y_rate: float = 0.5, interval: float = 1,
                    rebuild: bool = True,
                    draw: bool = False, scope: tp.Scope = None, judge: bool = True, detected: bool = False) -> bool:
        """ tap element """
        # 如果元素名为'nav_button'，则使用预定义的导航按钮元素
        if element_name == 'nav_button':
            element = self.recog.nav_button()
        else:
            # 否则根据元素名查找元素
            element = self.find(element_name, draw, scope, judge=judge)
        # 如果设置了检测标志并且未检测到元素，则返回 False
        if detected and element is None:
            return False
        # 在元素上进行点击操作
        self.tap(element, x_rate, y_rate, interval, rebuild)
        return True

    # 在屏幕上进行滑动操作
    def swipe(self, start: tp.Coordinate, movement: tp.Coordinate, duration: int = 100, interval: float = 1,
              rebuild: bool = True) -> None:
        """ swipe """
        # 计算滑动结束位置
        end = (start[0] + movement[0], start[1] + movement[1])
        # 在设备上进行滑动操作
        self.device.swipe(start, end, duration=duration)
        # 如果设置了间隔时间，进行休眠
        if interval > 0:
            self.sleep(interval, rebuild)

    # 在屏幕上进行滑动操作，不进行重建和重新捕获
    def swipe_only(self, start: tp.Coordinate, movement: tp.Coordinate, duration: int = 100,
                   interval: float = 1) -> None:
        """ swipe only, no rebuild and recapture """
        # 计算滑动结束位置
        end = (start[0] + movement[0], start[1] + movement[1])
        # 在设备上进行滑动操作
        self.device.swipe(start, end, duration=duration)
        # 如果设置了间隔时间，进行休眠
        if interval > 0:
            time.sleep(interval)

    # 在屏幕上进行滑动操作，使用点序列
    # def swipe_seq(self, points: list[tp.Coordinate], duration: int = 100, interval: float = 1, rebuild: bool = True) -> None:
    #     """ swipe with point sequence """
    #     # 在设备上进行滑动操作
    #     self.device.swipe(points, duration=duration)
    #     # 如果设置了间隔时间，进行休眠
    #     if interval > 0:
    #         self.sleep(interval, rebuild)
    # 定义一个方法，用于模拟滑动操作，接受起始坐标、移动序列、持续时间、间隔时间和重建标志作为参数
    def swipe_noinertia(self, start: tp.Coordinate, movement: tp.Coordinate, duration: int = 100, interval: float = 1,
                        rebuild: bool = False) -> None:
        """ swipe with no inertia (movement should be vertical) """
        # 初始化起始点
        points = [start]
        # 判断移动方向，计算移动距离
        if movement[0] == 0:
            dis = abs(movement[1])
            # 添加滑动路径的各个点
            points.append((start[0] + 100, start[1]))
            points.append((start[0] + 100, start[1] + movement[1]))
            points.append((start[0], start[1] + movement[1]))
        else:
            dis = abs(movement[0])
            # 添加滑动路径的各个点
            points.append((start[0], start[1] + 100))
            points.append((start[0] + movement[0], start[1] + 100))
            points.append((start[0] + movement[0], start[1]))
        # 调用设备对象的滑动扩展方法，传入滑动路径和持续时间
        self.device.swipe_ext(points, durations=[200, dis * duration // 100, 200])
        # 如果间隔时间大于0，则调用睡眠方法
        if interval > 0:
            self.sleep(interval, rebuild)

    # 定义一个方法，用于模拟返回按键事件，接受间隔时间和重建标志作为参数
    def back(self, interval: float = 1, rebuild: bool = True) -> None:
        """ send back keyevent """
        # 调用设备对象的发送按键事件方法，发送返回按键事件
        self.device.send_keyevent(KeyCode.KEYCODE_BACK)
        # 调用睡眠方法，传入间隔时间和重建标志
        self.sleep(interval, rebuild)

    # 定义一个方法，用于获取当前游戏场景
    def scene(self) -> int:
        """ get the current scene in the game """
        # 调用识别对象的获取场景方法，返回当前场景的编号
        return self.recog.get_scene()

    # 定义一个方法，用于获取当前基础设施场景
    def get_infra_scene(self) -> int:
        """ get the current scene in the infra """
        # 调用识别对象的获取基础设施场景方法，返回当前场景的编号
        return self.recog.get_infra_scene()

    # 定义一个方法，用于检查用户是否已登录
    def is_login(self):
        """ check if you are logged in """
        # 判断当前场景是否为登录相关场景，返回登录状态的布尔值
        return not (self.scene() // 100 == 1 or self.scene() // 100 == 99 or self.scene() == -1)
    # 获取导航栏状态，若存在则打开
    def get_navigation(self):
        # 设置重试次数
        retry_times = config.MAX_RETRYTIME
        # 循环尝试
        while retry_times:
            # 如果当前场景是导航栏，则返回True
            if self.scene() == Scene.NAVIGATION_BAR:
                return True
            # 如果未检测到导航栏按钮，则返回False
            elif not self.tap_element('nav_button', detected=True):
                return False
            # 重试次数减一
            retry_times -= 1

    # 返回到基建界面
    def back_to_infrastructure(self):
        # 返回到主页
        self.back_to_index()
        # 点击基建按钮
        self.tap_element('index_infrastructure')

    # 返回到理智兑换界面
    def back_to_reclamation_algorithm(self):
        # 更新识别信息
        self.recog.update()
        # 循环直到找到理智兑换按钮
        while self.find('index_terminal') is None:
            # 如果当前场景是未知，则退出游戏
            if self.scene() == Scene.UNKNOWN:
                self.device.exit('com.hypergryph.arknights')
            # 返回到主页
            self.back_to_index()
        # 记录日志
        logger.info('导航至生息演算')
        # 点击理智兑换按钮
        self.tap_element('index_terminal', 0.5)
        # 点击屏幕上的特定位置
        self.tap((self.recog.w * 0.2, self.recog.h * 0.8), interval=0.5)
    # 将当前页面导航至保全派驻页面
    def to_sss(self, sss_type, ec_type=3):
        # 更新页面识别信息
        self.recog.update()
        # 返回至首页
        self.back_to_index()
        # 点击进入终端页面
        self.tap_element('index_terminal', 0.5)
        # 点击进入保全派驻页面
        self.tap((self.recog.w * 0.7, self.recog.h * 0.95), interval=0.2)
        self.tap((self.recog.w * 0.85, self.recog.h * 0.5), interval=0.2)
        # 根据 sss_type 不同选择不同的操作
        if sss_type == 1:
            self.tap((self.recog.w * 0.2, self.recog.h * 0.3), interval=5)
        else:
            self.tap((self.recog.w * 0.4, self.recog.h * 0.6), interval=5)
        loop_count = 0
        ec_chosen_step = -99
        choose_team = False
        # 在未找到结束标志并且循环次数小于8的情况下执行循环
        while self.find('end_sss', score=0.8) is None and loop_count < 8:
            # 判断是否需要选择小队
            if loop_count == ec_chosen_step + 2 or self.find('sss_team_up') is not None:
                choose_team = True
                logger.info("选择小队")
            # 判断是否需要选择导能单元
            elif self.find('choose_ss_ec') is not None and not choose_team:
                if ec_type == 1:
                    self.tap((self.recog.w * 0.3, self.recog.h * 0.5), interval=0.2)
                elif ec_type == 2:
                    self.tap((self.recog.w * 0.5, self.recog.h * 0.5), interval=0.2)
                else:
                    self.tap((self.recog.w * 0.7, self.recog.h * 0.5), interval=0.2)
                ec_chosen_step = loop_count
                logger.info(f"选定导能单元:{ec_type + 1}")
            # 点击确认按钮，如果选择了小队则间隔时间为10，否则为0.2
            self.tap((self.recog.w * 0.95, self.recog.h * 0.95), interval=(0.2 if not choose_team else 10))
            # 更新页面识别信息
            self.recog.update()
            loop_count += 1
        # 如果循环次数达到8，则返回导航失败信息
        if loop_count == 8:
            return "保全派驻导航失败"
    # 需要等待的页面解决方法。触发超时重启会返回False
    def waiting_solver(self, scenes, wait_count=20, sleep_time=3):
        while wait_count > 0:
            # 等待指定时间
            self.sleep(sleep_time)
            # 如果当前场景不是指定场景，并且基础设施场景也不是指定场景，则返回True
            if self.scene() != scenes and self.get_infra_scene() != scenes:
                return True
            wait_count -= 1
        # 记录警告日志
        logger.warning("同一等待界面等待超时，重启方舟。")
        # 退出应用
        self.device.exit(self.package_name)
        time.sleep(3)
        self.device.check_current_focus()
        return False

    # 等待某个页面载入
    def wait_for_scene(self, scene, method, wait_count=10, sleep_time=1):
        while wait_count > 0:
            # 等待指定时间
            self.sleep(sleep_time)
            if method == "get_infra_scene":
                # 如果基础设施场景等于指定场景，则返回True
                if self.get_infra_scene() == scene:
                    return True
            elif method == "scene":
                # 如果当前场景等于指定场景，则返回True
                if self.scene() == scene:
                    return True
            wait_count -= 1
        # 抛出等待超时异常
        raise Exception("等待超时")

    # 邮件发送 EightyDollars
    def send_email(self, body='', subject='', subtype='plain', retry_times=3):
        # 如果邮件功能未开启，则记录日志并返回
        if 'mail_enable' in self.email_config.keys() and self.email_config['mail_enable'] == 0:
            logger.info('邮件功能未开启')
            return

        # 创建邮件对象
        msg = MIMEMultipart()
        msg.attach(MIMEText(body, subtype))
        msg['Subject'] = self.email_config['subject'] + subject
        msg['From'] = self.email_config['account']

        while retry_times > 0:
            try:
                # 连接到 SMTP 服务器
                s = smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=10.0)
                # 登录邮箱
                s.login(self.email_config['account'], self.email_config['pass_code'])
                # 开始发送邮件
                s.sendmail(self.email_config['account'], self.email_config['receipts'], msg.as_string())
                logger.info("邮件发送成功")
                break
            except Exception as e:
                # 记录发送失败日志
                logger.error("邮件发送失败")
                logger.exception(e)
                retry_times -= 1
                time.sleep(3)
```