# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\plot_camera.py`

```
# 导入pyglet的OpenGL接口
import pyglet.gl as pgl
# 导入旋转相关函数
from sympy.plotting.pygletplot.plot_rotation import get_spherical_rotatation
# 导入模型矩阵和屏幕模型转换相关函数
from sympy.plotting.pygletplot.util import get_model_matrix, model_to_screen, \
                                            screen_to_model, vec_subs

# 定义一个绘图相机类
class PlotCamera:

    # 设置相机的最小和最大距离
    min_dist = 0.05
    max_dist = 500.0

    # 设置正交投影模式下的最小和最大距离
    min_ortho_dist = 100.0
    max_ortho_dist = 10000.0

    # 默认的透视投影距离和正交投影距离
    _default_dist = 6.0
    _default_ortho_dist = 600.0

    # 预设的旋转角度，以键-值对形式存储
    rot_presets = {
        'xy': (0, 0, 0),
        'xz': (-90, 0, 0),
        'yz': (0, 90, 0),
        'perspective': (-45, 0, -45)
    }

    # 初始化方法，接收一个窗口和正交投影模式的标志
    def __init__(self, window, ortho=False):
        self.window = window
        self.axes = self.window.plot.axes  # 设置窗口的坐标轴
        self.ortho = ortho  # 是否使用正交投影模式
        self.reset()  # 调用重置方法初始化相机参数

    # 初始化旋转矩阵
    def init_rot_matrix(self):
        pgl.glPushMatrix()  # 压栈保存当前矩阵
        pgl.glLoadIdentity()  # 将当前矩阵设为单位矩阵
        self._rot = get_model_matrix()  # 获取模型矩阵
        pgl.glPopMatrix()  # 出栈恢复之前的矩阵状态

    # 根据预设的名称设置旋转角度
    def set_rot_preset(self, preset_name):
        self.init_rot_matrix()  # 初始化旋转矩阵
        if preset_name not in self.rot_presets:  # 如果预设名称不存在
            raise ValueError(
                "%s is not a valid rotation preset." % preset_name)  # 抛出值错误异常
        r = self.rot_presets[preset_name]  # 获取对应预设名称的旋转角度
        self.euler_rotate(r[0], 1, 0, 0)  # 绕x轴旋转
        self.euler_rotate(r[1], 0, 1, 0)  # 绕y轴旋转
        self.euler_rotate(r[2], 0, 0, 1)  # 绕z轴旋转

    # 重置相机参数
    def reset(self):
        self._dist = 0.0  # 设置距离初始值
        self._x, self._y = 0.0, 0.0  # 设置x和y方向偏移初始值
        self._rot = None  # 清空旋转矩阵
        if self.ortho:  # 如果使用正交投影模式
            self._dist = self._default_ortho_dist  # 设置距离为正交投影的默认距离
        else:
            self._dist = self._default_dist  # 否则设置距离为透视投影的默认距离
        self.init_rot_matrix()  # 初始化旋转矩阵

    # 在当前旋转矩阵基础上乘以新的旋转矩阵
    def mult_rot_matrix(self, rot):
        pgl.glPushMatrix()  # 压栈保存当前矩阵
        pgl.glLoadMatrixf(rot)  # 加载新的旋转矩阵
        pgl.glMultMatrixf(self._rot)  # 与当前旋转矩阵相乘
        self._rot = get_model_matrix()  # 获取新的模型矩阵
        pgl.glPopMatrix()  # 出栈恢复之前的矩阵状态

    # 设置投影矩阵
    def setup_projection(self):
        pgl.glMatrixMode(pgl.GL_PROJECTION)  # 设置当前矩阵为投影矩阵模式
        pgl.glLoadIdentity()  # 将当前矩阵设为单位矩阵
        if self.ortho:  # 如果使用正交投影模式
            # 使用伪正交投影
            pgl.gluPerspective(
                0.3, float(self.window.width)/float(self.window.height),
                self.min_ortho_dist - 0.01, self.max_ortho_dist + 0.01)
        else:
            # 使用透视投影
            pgl.gluPerspective(
                30.0, float(self.window.width)/float(self.window.height),
                self.min_dist - 0.01, self.max_dist + 0.01)
        pgl.glMatrixMode(pgl.GL_MODELVIEW)  # 将当前矩阵模式切换为模型视图矩阵

    # 获取缩放系数
    def _get_scale(self):
        return 1.0, 1.0, 1.0  # 返回默认的缩放系数

    # 应用所有的变换（平移、旋转、缩放）
    def apply_transformation(self):
        pgl.glLoadIdentity()  # 将当前矩阵设为单位矩阵
        pgl.glTranslatef(self._x, self._y, -self._dist)  # 平移相机位置
        if self._rot is not None:  # 如果有旋转矩阵
            pgl.glMultMatrixf(self._rot)  # 应用旋转矩阵
        pgl.glScalef(*self._get_scale())  # 缩放模型

    # 使用球面坐标系进行旋转
    def spherical_rotate(self, p1, p2, sensitivity=1.0):
        mat = get_spherical_rotatation(p1, p2, self.window.width,
                                       self.window.height, sensitivity)  # 获取球面旋转矩阵
        if mat is not None:
            self.mult_rot_matrix(mat)  # 应用球面旋转矩阵
    # 对象方法：根据给定的角度和旋转轴(x, y, z)，对当前模型进行欧拉旋转
    def euler_rotate(self, angle, x, y, z):
        # 保存当前的模型视图矩阵
        pgl.glPushMatrix()
        # 加载保存的旋转矩阵到当前模型视图矩阵
        pgl.glLoadMatrixf(self._rot)
        # 在当前模型视图矩阵的基础上继续旋转
        pgl.glRotatef(angle, x, y, z)
        # 获取更新后的模型矩阵并保存
        self._rot = get_model_matrix()
        # 恢复先前保存的模型视图矩阵
        pgl.glPopMatrix()

    # 对象方法：相对缩放视角，根据点击次数和灵敏度进行缩放操作
    def zoom_relative(self, clicks, sensitivity):

        # 如果是正交投影模式，增加距离的变化量
        if self.ortho:
            dist_d = clicks * sensitivity * 50.0
            min_dist = self.min_ortho_dist
            max_dist = self.max_ortho_dist
        else:
            # 非正交投影模式下的距离变化量
            dist_d = clicks * sensitivity
            min_dist = self.min_dist
            max_dist = self.max_dist

        # 计算新的距离值
        new_dist = (self._dist - dist_d)
        # 根据缩放方向和边界限制更新当前距离
        if (clicks < 0 and new_dist < max_dist) or new_dist > min_dist:
            self._dist = new_dist

    # 对象方法：根据鼠标移动的距离(dx, dy)，在屏幕上进行模型的平移
    def mouse_translate(self, x, y, dx, dy):
        # 保存当前的模型视图矩阵
        pgl.glPushMatrix()
        # 将当前模型视图矩阵重置为单位矩阵
        pgl.glLoadIdentity()
        # 在 z 轴方向上平移视角，以当前距离为基础
        pgl.glTranslatef(0, 0, -self._dist)
        # 获取模型在屏幕上的 z 坐标
        z = model_to_screen(0, 0, 0)[2]
        # 计算鼠标移动后的模型坐标变化
        d = vec_subs(screen_to_model(x, y, z), screen_to_model(x - dx, y - dy, z))
        # 恢复先前保存的模型视图矩阵
        pgl.glPopMatrix()
        # 根据计算得到的变化量更新模型在 x 和 y 轴上的位置
        self._x += d[0]
        self._y += d[1]
```