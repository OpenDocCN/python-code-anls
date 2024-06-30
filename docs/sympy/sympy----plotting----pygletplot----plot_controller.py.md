# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\plot_controller.py`

```
# 从 pyglet.window 模块导入 key 对象，用于处理键盘事件
# 从 pyglet.window.mouse 模块导入 LEFT、RIGHT、MIDDLE 常量，表示鼠标按键
# 从 sympy.plotting.pygletplot.util 模块导入 get_direction_vectors 和 get_basis_vectors 函数
from pyglet.window import key
from pyglet.window.mouse import LEFT, RIGHT, MIDDLE
from sympy.plotting.pygletplot.util import get_direction_vectors, get_basis_vectors

# 定义 PlotController 类，用于控制绘图相关的操作
class PlotController:

    # 普通模式下鼠标灵敏度设定为 4.0
    # 修改模式下鼠标灵敏度设定为 1.0
    normal_mouse_sensitivity = 4.0
    modified_mouse_sensitivity = 1.0

    # 普通模式下键盘灵敏度设定为 160.0
    # 修改模式下键盘灵敏度设定为 40.0
    normal_key_sensitivity = 160.0
    modified_key_sensitivity = 40.0

    # 键盘映射，将按键映射到相应的操作字符串
    keymap = {
        key.LEFT: 'left',          # 左箭头、A键、NUMPAD 4 键映射到 'left'
        key.A: 'left',
        key.NUM_4: 'left',

        key.RIGHT: 'right',        # 右箭头、D键、NUMPAD 6 键映射到 'right'
        key.D: 'right',
        key.NUM_6: 'right',

        key.UP: 'up',              # 上箭头、W键、NUMPAD 8 键映射到 'up'
        key.W: 'up',
        key.NUM_8: 'up',

        key.DOWN: 'down',          # 下箭头、S键、NUMPAD 2 键映射到 'down'
        key.S: 'down',
        key.NUM_2: 'down',

        key.Z: 'rotate_z_neg',     # Z键、NUMPAD 1 键映射到 'rotate_z_neg'
        key.NUM_1: 'rotate_z_neg',

        key.C: 'rotate_z_pos',     # C键、NUMPAD 3 键映射到 'rotate_z_pos'
        key.NUM_3: 'rotate_z_pos',

        key.Q: 'spin_left',        # Q键、NUMPAD 7 键映射到 'spin_left'
        key.NUM_7: 'spin_left',
        key.E: 'spin_right',       # E键、NUMPAD 9 键映射到 'spin_right'
        key.NUM_9: 'spin_right',

        key.X: 'reset_camera',     # X键、NUMPAD 5 键映射到 'reset_camera'
        key.NUM_5: 'reset_camera',

        key.NUM_ADD: 'zoom_in',    # NUMPAD + 键、PAGEUP键、R键映射到 'zoom_in'
        key.PAGEUP: 'zoom_in',
        key.R: 'zoom_in',

        key.NUM_SUBTRACT: 'zoom_out',   # NUMPAD - 键、PAGEDOWN键、F键映射到 'zoom_out'
        key.PAGEDOWN: 'zoom_out',
        key.F: 'zoom_out',

        key.RSHIFT: 'modify_sensitivity',   # RSHIFT键、LSHIFT键映射到 'modify_sensitivity'
        key.LSHIFT: 'modify_sensitivity',

        key.F1: 'rot_preset_xy',       # F1键映射到 'rot_preset_xy'
        key.F2: 'rot_preset_xz',       # F2键映射到 'rot_preset_xz'
        key.F3: 'rot_preset_yz',       # F3键映射到 'rot_preset_yz'
        key.F4: 'rot_preset_perspective',   # F4键映射到 'rot_preset_perspective'

        key.F5: 'toggle_axes',         # F5键映射到 'toggle_axes'
        key.F6: 'toggle_axe_colors',   # F6键映射到 'toggle_axe_colors'

        key.F8: 'save_image'           # F8键映射到 'save_image'
    }

    # 初始化方法，接受窗口对象和其他关键字参数
    def __init__(self, window, *, invert_mouse_zoom=False, **kwargs):
        self.invert_mouse_zoom = invert_mouse_zoom  # 是否反转鼠标缩放方向
        self.window = window      # 关联的窗口对象
        self.camera = window.camera   # 获取窗口的相机对象
        # 初始化操作字典，每个操作初始值为 False
        self.action = {
            'left': False,              # 左转操作
            'right': False,             # 右转操作
            'up': False,                # 上转操作
            'down': False,              # 下转操作
            'spin_left': False,         # 左旋操作
            'spin_right': False,        # 右旋操作
            'rotate_z_neg': False,      # 绕 Z 负方向旋转操作
            'rotate_z_pos': False,      # 绕 Z 正方向旋转操作
            'reset_camera': False,      # 重置相机操作
            'zoom_in': False,           # 放大操作
            'zoom_out': False,          # 缩小操作
            'modify_sensitivity': False,    # 修改灵敏度操作
            'rot_preset_xy': False,     # XY轴旋转预设操作
            'rot_preset_xz': False,     # XZ轴旋转预设操作
            'rot_preset_yz': False,     # YZ轴旋转预设操作
            'rot_preset_perspective': False,   # 透视旋转预设操作
            'toggle_axes': False,       # 切换坐标轴显示操作
            'toggle_axe_colors': False, # 切换坐标轴颜色操作
            'save_image': False         # 保存图片操作
        }
    # 更新函数，用于处理每帧的用户输入并更新相机状态
    def update(self, dt):
        # 初始化缩放变量为0
        z = 0
        # 如果用户要求缩小视野，则z减1
        if self.action['zoom_out']:
            z -= 1
        # 如果用户要求放大视野，则z加1
        if self.action['zoom_in']:
            z += 1
        # 如果z不为0，则按比例调整相机的缩放和敏感度
        if z != 0:
            self.camera.zoom_relative(z/10.0, self.get_key_sensitivity()/10.0)

        # 初始化移动变量(dx, dy, dz)为0
        dx, dy, dz = 0, 0, 0
        # 根据用户输入方向键的情况，更新dx, dy, dz
        if self.action['left']:
            dx -= 1
        if self.action['right']:
            dx += 1
        if self.action['up']:
            dy -= 1
        if self.action['down']:
            dy += 1
        if self.action['spin_left']:
            dz += 1
        if self.action['spin_right']:
            dz -= 1

        # 如果不是2D模式，则根据dx, dy, dz调用相机的欧拉角旋转方法
        if not self.is_2D():
            if dx != 0:
                self.camera.euler_rotate(dx*dt*self.get_key_sensitivity(),
                                         *(get_direction_vectors()[1]))
            if dy != 0:
                self.camera.euler_rotate(dy*dt*self.get_key_sensitivity(),
                                         *(get_direction_vectors()[0]))
            if dz != 0:
                self.camera.euler_rotate(dz*dt*self.get_key_sensitivity(),
                                         *(get_direction_vectors()[2]))
        else:
            # 如果是2D模式，则调用相机的鼠标平移方法
            self.camera.mouse_translate(0, 0, dx*dt*self.get_key_sensitivity(),
                                        -dy*dt*self.get_key_sensitivity())

        # 初始化绕z轴旋转变量为0
        rz = 0
        # 如果用户要求负向旋转z轴且不是2D模式，则rz减1
        if self.action['rotate_z_neg'] and not self.is_2D():
            rz -= 1
        # 如果用户要求正向旋转z轴且不是2D模式，则rz加1
        if self.action['rotate_z_pos'] and not self.is_2D():
            rz += 1
        # 如果rz不为0，则根据时间步长和敏感度调用相机的欧拉角旋转方法
        if rz != 0:
            self.camera.euler_rotate(rz*dt*self.get_key_sensitivity(),
                                     *(get_basis_vectors()[2]))

        # 如果用户要求重置相机，则调用相机的重置方法
        if self.action['reset_camera']:
            self.camera.reset()

        # 根据用户的预设旋转要求，调用相机的设置预设旋转方法
        if self.action['rot_preset_xy']:
            self.camera.set_rot_preset('xy')
        if self.action['rot_preset_xz']:
            self.camera.set_rot_preset('xz')
        if self.action['rot_preset_yz']:
            self.camera.set_rot_preset('yz')
        if self.action['rot_preset_perspective']:
            self.camera.set_rot_preset('perspective')

        # 如果用户要求切换坐标轴显示，则切换坐标轴的可见性
        if self.action['toggle_axes']:
            self.action['toggle_axes'] = False
            self.camera.axes.toggle_visible()

        # 如果用户要求切换坐标轴颜色，则切换坐标轴的颜色
        if self.action['toggle_axe_colors']:
            self.action['toggle_axe_colors'] = False
            self.camera.axes.toggle_colors()

        # 如果用户要求保存图像，则调用窗口绘图对象的保存图像方法
        if self.action['save_image']:
            self.action['save_image'] = False
            self.window.plot.saveimage()

        # 返回True，表示更新成功
        return True
    # 当按下键盘按键时触发的事件处理函数，记录对应动作为真
    def on_key_press(self, symbol, modifiers):
        # 如果按键符号在按键映射中
        if symbol in self.keymap:
            # 设置对应动作为真
            self.action[self.keymap[symbol]] = True

    # 当释放键盘按键时触发的事件处理函数，记录对应动作为假
    def on_key_release(self, symbol, modifiers):
        # 如果按键符号在按键映射中
        if symbol in self.keymap:
            # 设置对应动作为假
            self.action[self.keymap[symbol]] = False

    # 当拖动鼠标时触发的事件处理函数
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # 如果按下了左键
        if buttons & LEFT:
            # 如果是二维模式
            if self.is_2D():
                # 调用相机对象的鼠标平移方法
                self.camera.mouse_translate(x, y, dx, dy)
            else:
                # 调用相机对象的球面旋转方法
                self.camera.spherical_rotate((x - dx, y - dy), (x, y),
                                             self.get_mouse_sensitivity())
        # 如果按下了中键
        if buttons & MIDDLE:
            # 根据鼠标缩放反转设置调整相机的相对缩放
            self.camera.zoom_relative([1, -1][self.invert_mouse_zoom]*dy,
                                      self.get_mouse_sensitivity()/20.0)
        # 如果按下了右键
        if buttons & RIGHT:
            # 调用相机对象的鼠标平移方法
            self.camera.mouse_translate(x, y, dx, dy)

    # 当滚动鼠标滚轮时触发的事件处理函数，调整相机的缩放
    def on_mouse_scroll(self, x, y, dx, dy):
        # 根据鼠标滚动方向和鼠标灵敏度调整相机的相对缩放
        self.camera.zoom_relative([1, -1][self.invert_mouse_zoom]*dy,
                                  self.get_mouse_sensitivity())

    # 检查当前窗口是否处于二维模式
    def is_2D(self):
        # 获取窗口中所有绘图函数对象
        functions = self.window.plot._functions
        # 遍历每个绘图函数对象
        for i in functions:
            # 如果任何一个绘图函数对象的自变量数大于1或者因变量数大于2
            if len(functions[i].i_vars) > 1 or len(functions[i].d_vars) > 2:
                # 返回假，表示非二维模式
                return False
        # 如果所有绘图函数对象的自变量数都不大于1且因变量数都不大于2，则返回真，表示二维模式
        return True
```