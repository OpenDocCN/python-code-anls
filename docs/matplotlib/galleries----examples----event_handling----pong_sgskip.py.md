# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\pong_sgskip.py`

```py
"""
====
Pong
====

A Matplotlib based game of Pong illustrating one way to write interactive
animations that are easily ported to multiple backends.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

# 导入需要的库
import time  # 导入时间模块

import matplotlib.pyplot as plt  # 导入绘图库matplotlib
import numpy as np  # 导入数值计算库numpy
from numpy.random import randint, randn  # 导入随机数生成函数

from matplotlib.font_manager import FontProperties  # 导入字体管理器

instructions = """
Player A:       Player B:
  'e'      up     'i'
  'd'     down    'k'

press 't' -- close these instructions
            (animation will be much faster)
press 'a' -- add a puck
press 'A' -- remove a puck
press '1' -- slow down all pucks
press '2' -- speed up all pucks
press '3' -- slow down distractors
press '4' -- speed up distractors
press ' ' -- reset the first puck
press 'n' -- toggle distractors on/off
press 'g' -- toggle the game on/off

  """

# 球拍类
class Pad:
    def __init__(self, disp, x, y, type='l'):
        self.disp = disp  # 显示对象
        self.x = x  # X坐标
        self.y = y  # Y坐标
        self.w = .3  # 宽度
        self.score = 0  # 得分
        self.xoffset = 0.3  # X偏移量
        self.yoffset = 0.1  # Y偏移量
        if type == 'r':
            self.xoffset *= -1.0  # 右边球拍的X偏移量取反

        if type == 'l' or type == 'r':
            self.signx = -1.0  # X方向速度标志
            self.signy = 1.0  # Y方向速度标志
        else:
            self.signx = 1.0  # X方向速度标志
            self.signy = -1.0  # Y方向速度标志

    def contains(self, loc):
        return self.disp.get_bbox().contains(loc.x, loc.y)  # 检查位置是否在球拍内部


# 球类
class Puck:
    def __init__(self, disp, pad, field):
        self.vmax = .2  # 最大速度
        self.disp = disp  # 显示对象
        self.field = field  # 球场
        self._reset(pad)  # 重置球的初始状态

    def _reset(self, pad):
        self.x = pad.x + pad.xoffset  # X坐标
        if pad.y < 0:
            self.y = pad.y + pad.yoffset  # Y坐标
        else:
            self.y = pad.y - pad.yoffset  # Y坐标
        self.vx = pad.x - self.x  # X速度
        self.vy = pad.y + pad.w/2 - self.y  # Y速度
        self._speedlimit()  # 速度限制
        self._slower()  # 减速
        self._slower()  # 减速

    def update(self, pads):
        self.x += self.vx  # 更新X位置
        self.y += self.vy  # 更新Y位置
        for pad in pads:
            if pad.contains(self):  # 如果球在某个球拍内
                self.vx *= 1.2 * pad.signx  # X速度增加
                self.vy *= 1.2 * pad.signy  # Y速度增加
        fudge = .001  # 容错值
        # 可以用更清晰的方式...
        if self.x < fudge:
            pads[1].score += 1  # 球拍2得分加1
            self._reset(pads[0])  # 重置球的状态
            return True  # 返回True表示有得分
        if self.x > 7 - fudge:
            pads[0].score += 1  # 球拍1得分加1
            self._reset(pads[1])  # 重置球的状态
            return True  # 返回True表示有得分
        if self.y < -1 + fudge or self.y > 1 - fudge:
            self.vy *= -1.0  # Y速度取反
            # 添加一些随机性，使游戏更有趣
            self.vy -= (randn()/300.0 + 1/300.0) * np.sign(self.vy)  # 添加随机扰动
        self._speedlimit()  # 速度限制
        return False  # 返回False表示没有得分
    # 减慢速度方法，将水平和垂直速度分量分别除以5.0
    def _slower(self):
        self.vx /= 5.0
        self.vy /= 5.0

    # 加快速度方法，将水平和垂直速度分量分别乘以5.0
    def _faster(self):
        self.vx *= 5.0
        self.vy *= 5.0

    # 限制速度方法，若水平或垂直速度分量超过最大速度(self.vmax)，则将其限制为最大速度或负最大速度
    def _speedlimit(self):
        if self.vx > self.vmax:
            self.vx = self.vmax
        if self.vx < -self.vmax:
            self.vx = -self.vmax

        if self.vy > self.vmax:
            self.vy = self.vmax
        if self.vy < -self.vmax:
            self.vy = -self.vmax
# 定义一个名为 Game 的类
class Game:
    # 初始化方法，接受参数 ax，代表绘图区域
    def __init__(self, ax):
        # 设置初始线条属性
        self.ax = ax
        ax.xaxis.set_visible(False)  # 设置 X 轴不可见
        ax.set_xlim([0, 7])  # 设置 X 轴范围为 [0, 7]
        ax.yaxis.set_visible(False)  # 设置 Y 轴不可见
        ax.set_ylim([-1, 1])  # 设置 Y 轴范围为 [-1, 1]
        
        # 定义用于位置计算的变量
        pad_a_x = 0
        pad_b_x = .50
        pad_a_y = pad_b_y = .30
        pad_b_x += 6.3

        # 创建两个条形图，表示玩家 A 和玩家 B 的位置
        pA, = self.ax.barh(pad_a_y, .2,
                           height=.3, color='k', alpha=.5, edgecolor='b',
                           lw=2, label="Player B",
                           animated=True)
        pB, = self.ax.barh(pad_b_y, .2,
                           height=.3, left=pad_b_x, color='k', alpha=.5,
                           edgecolor='r', lw=2, label="Player A",
                           animated=True)

        # 创建多条线，用作分心物体
        self.x = np.arange(0, 2.22*np.pi, 0.01)
        self.line, = self.ax.plot(self.x, np.sin(self.x), "r",
                                  animated=True, lw=4)
        self.line2, = self.ax.plot(self.x, np.cos(self.x), "g",
                                   animated=True, lw=4)
        self.line3, = self.ax.plot(self.x, np.cos(self.x), "g",
                                   animated=True, lw=4)
        self.line4, = self.ax.plot(self.x, np.cos(self.x), "r",
                                   animated=True, lw=4)

        # 创建中心线
        self.centerline, = self.ax.plot([3.5, 3.5], [1, -1], 'k',
                                        alpha=.5, animated=True, lw=8)

        # 创建表示冰球的散点图
        self.puckdisp = self.ax.scatter([1], [1], label='_nolegend_',
                                        s=200, c='g',
                                        alpha=.9, animated=True)

        # 获取画布和背景
        self.canvas = self.ax.figure.canvas
        self.background = None
        self.cnt = 0
        self.distract = True
        self.res = 100.0
        self.on = False
        self.inst = True    # 显示起始时的说明
        # 创建两个 Pad 对象，表示两个玩家的控制板
        self.pads = [Pad(pA, pad_a_x, pad_a_y),
                     Pad(pB, pad_b_x, pad_b_y, 'r')]
        self.pucks = []  # 初始化冰球列表为空
        # 创建说明文本框
        self.i = self.ax.annotate(instructions, (.5, 0.5),
                                  name='monospace',
                                  verticalalignment='center',
                                  horizontalalignment='center',
                                  multialignment='left',
                                  xycoords='axes fraction',
                                  animated=False)
        # 绑定键盘按键事件处理函数
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
    # 获取绘图函数的引用
    draw_artist = self.ax.draw_artist
    
    # 如果背景为空，复制当前 Axes 区域的背景作为新的背景
    if self.background is None:
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    # 恢复清除画布时保存的背景
    self.canvas.restore_region(self.background)

    # 显示分散点（distractors）
    if self.distract:
        # 更新四条线的纵坐标，以正弦、余弦、正切的形式显示
        self.line.set_ydata(np.sin(self.x + self.cnt/self.res))
        self.line2.set_ydata(np.cos(self.x - self.cnt/self.res))
        self.line3.set_ydata(np.tan(self.x + self.cnt/self.res))
        self.line4.set_ydata(np.tan(self.x - self.cnt/self.res))
        # 逐一绘制四条线
        draw_artist(self.line)
        draw_artist(self.line2)
        draw_artist(self.line3)
        draw_artist(self.line4)

    # 绘制球拍和球
    if self.on:
        # 绘制中心线
        self.ax.draw_artist(self.centerline)
        
        # 遍历每个球拍，更新位置并绘制
        for pad in self.pads:
            pad.disp.set_y(pad.y)
            pad.disp.set_x(pad.x)
            self.ax.draw_artist(pad.disp)

        # 遍历每个球，更新位置并绘制
        for puck in self.pucks:
            # 如果球有更新，则检查是否有人得分
            if puck.update(self.pads):
                # 更新球拍得分并显示
                self.pads[0].disp.set_label(f"   {self.pads[0].score}")
                self.pads[1].disp.set_label(f"   {self.pads[1].score}")
                # 显示图例
                self.ax.legend(loc='center', framealpha=.2,
                               facecolor='0.5',
                               prop=FontProperties(size='xx-large',
                                                   weight='bold'))

                # 重置背景为None，以便下次重新绘制
                self.background = None
                # 强制刷新画布
                self.ax.figure.canvas.draw_idle()
                return
            
            # 更新球的位置并绘制
            puck.disp.set_offsets([[puck.x, puck.y]])
            self.ax.draw_artist(puck.disp)

    # 仅重新绘制Axes的矩形区域
    self.canvas.blit(self.ax.bbox)
    # 刷新事件，确保绘图响应
    self.canvas.flush_events()

    # 如果计数达到50000，输出警告信息并关闭绘图窗口
    if self.cnt == 50000:
        print("...and you've been playing for too long!!!")
        plt.close()

    # 计数增加
    self.cnt += 1
    # 处理键盘按下事件的方法
    def on_key_press(self, event):
        # 检查按下的键是否为 '3'，如果是，将结果乘以 5.0
        if event.key == '3':
            self.res *= 5.0
        # 检查按下的键是否为 '4'，如果是，将结果除以 5.0
        if event.key == '4':
            self.res /= 5.0

        # 检查按下的键是否为 'e'，如果是，使第一个挡板向上移动 0.1 单位
        if event.key == 'e':
            self.pads[0].y += .1
            # 如果挡板的位置超过了上边界 (1 - 0.3)，将其限制在上边界
            if self.pads[0].y > 1 - .3:
                self.pads[0].y = 1 - .3
        # 检查按下的键是否为 'd'，如果是，使第一个挡板向下移动 0.1 单位
        if event.key == 'd':
            self.pads[0].y -= .1
            # 如果挡板的位置超过了下边界 (-1)，将其限制在下边界
            if self.pads[0].y < -1:
                self.pads[0].y = -1

        # 检查按下的键是否为 'i'，如果是，使第二个挡板向上移动 0.1 单位
        if event.key == 'i':
            self.pads[1].y += .1
            # 如果挡板的位置超过了上边界 (1 - 0.3)，将其限制在上边界
            if self.pads[1].y > 1 - .3:
                self.pads[1].y = 1 - .3
        # 检查按下的键是否为 'k'，如果是，使第二个挡板向下移动 0.1 单位
        if event.key == 'k':
            self.pads[1].y -= .1
            # 如果挡板的位置超过了下边界 (-1)，将其限制在下边界
            if self.pads[1].y < -1:
                self.pads[1].y = -1

        # 检查按下的键是否为 'a'，如果是，向 pucks 列表中添加一个新的 Puck 对象
        if event.key == 'a':
            self.pucks.append(Puck(self.puckdisp,
                                   self.pads[randint(2)],
                                   self.ax.bbox))
        # 检查按下的键是否为 'A'，如果是且 pucks 列表不为空，移除最后一个 Puck 对象
        if event.key == 'A' and len(self.pucks):
            self.pucks.pop()
        # 检查按下的键是否为空格键 (' ')，如果是且 pucks 列表不为空，重置第一个 Puck 对象的位置
        if event.key == ' ' and len(self.pucks):
            self.pucks[0]._reset(self.pads[randint(2)])
        # 检查按下的键是否为 '1'，如果是，使所有 pucks 列表中的 Puck 对象减速
        if event.key == '1':
            for p in self.pucks:
                p._slower()
        # 检查按下的键是否为 '2'，如果是，使所有 pucks 列表中的 Puck 对象加速
        if event.key == '2':
            for p in self.pucks:
                p._faster()

        # 检查按下的键是否为 'n'，如果是，切换 self.distract 的布尔值
        if event.key == 'n':
            self.distract = not self.distract

        # 检查按下的键是否为 'g'，如果是，切换 self.on 的布尔值
        if event.key == 'g':
            self.on = not self.on
        # 检查按下的键是否为 't'，如果是，切换 self.inst 的布尔值，并根据 self.inst 的值显示或隐藏一些元素
        if event.key == 't':
            self.inst = not self.inst
            self.i.set_visible(not self.i.get_visible())
            self.background = None
            self.canvas.draw_idle()
        # 检查按下的键是否为 'q'，如果是，关闭当前的 matplotlib 窗口
        if event.key == 'q':
            plt.close()
# 创建一个新的图形和轴对象
fig, ax = plt.subplots()
# 获取图形的画布对象
canvas = ax.figure.canvas
# 创建一个游戏动画对象，并将其绘制在轴上
animation = Game(ax)

# 如果图形的默认键绑定已经存在，则取消默认键绑定
if fig.canvas.manager.key_press_handler_id is not None:
    canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

# 定义在重新绘制时重置 blitting 背景的函数
def on_redraw(event):
    animation.background = None

# 定义在动画开始时执行的函数
def start_anim(event):
    # 取消连接的事件处理器，确保只执行一次
    canvas.mpl_disconnect(start_anim.cid)

    # 添加一个回调函数来绘制动画
    start_anim.timer.add_callback(animation.draw)
    # 启动定时器
    start_anim.timer.start()
    # 连接绘制事件和重置背景函数
    canvas.mpl_connect('draw_event', on_redraw)

# 将绘制事件连接到开始动画函数，并记录连接的 ID
start_anim.cid = canvas.mpl_connect('draw_event', start_anim)
# 创建一个新的定时器对象，每隔1毫秒触发一次
start_anim.timer = animation.canvas.new_timer(interval=1)

# 记录开始时间
tstart = time.time()

# 显示图形界面
plt.show()
# 打印帧率信息，FPS 表示每秒帧数
print('FPS: %f' % (animation.cnt/(time.time() - tstart)))
```