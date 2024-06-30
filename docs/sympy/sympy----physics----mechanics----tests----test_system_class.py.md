# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_system_class.py`

```
import pytest  # 导入 pytest 测试框架

from sympy.core.symbol import symbols  # 导入 sympy 中的 symbols 符号变量定义模块
from sympy.core.sympify import sympify  # 导入 sympy 中的 sympify 符号表达式转换模块
from sympy.functions.elementary.trigonometric import cos, sin  # 导入 sympy 中的三角函数模块 cos 和 sin
from sympy.matrices.dense import eye, zeros  # 导入 sympy 中的矩阵模块，包括单位矩阵 eye 和零矩阵 zeros
from sympy.matrices.immutable import ImmutableMatrix  # 导入 sympy 中的不可变矩阵模块 ImmutableMatrix
from sympy.physics.mechanics import (  # 导入 sympy 中的力学模块
    Force, KanesMethod, LagrangesMethod, Particle, PinJoint, Point,
    PrismaticJoint, ReferenceFrame, RigidBody, Torque, TorqueActuator, System,
    dynamicsymbols)
from sympy.simplify.simplify import simplify  # 导入 sympy 中的化简模块 simplify
from sympy.solvers.solvers import solve  # 导入 sympy 中的求解模块 solve

t = dynamicsymbols._t  # type: ignore  # 定义时间符号 t，忽略类型检查
q = dynamicsymbols('q:6')  # type: ignore  # 定义广义坐标符号 q[0] 到 q[5]，忽略类型检查
qd = dynamicsymbols('q:6', 1)  # type: ignore  # 定义广义速度符号 qd[0] 到 qd[5]，忽略类型检查
u = dynamicsymbols('u:6')  # type: ignore  # 定义广义速度符号 u[0] 到 u[5]，忽略类型检查
ua = dynamicsymbols('ua:3')  # type: ignore  # 定义辅助速度符号 ua[0] 到 ua[2]，忽略类型检查


class TestSystemBase:
    @pytest.fixture()  # 定义 pytest 的测试 fixture
    def _empty_system_setup(self):
        self.system = System(ReferenceFrame('frame'), Point('fixed_point'))  # 创建系统对象，包括参考系和固定点

    def _empty_system_check(self, exclude=()):  # 定义检查空系统状态的方法，可以排除指定属性
        matrices = ('q_ind', 'q_dep', 'q', 'u_ind', 'u_dep', 'u', 'u_aux',
                    'kdes', 'holonomic_constraints', 'nonholonomic_constraints')
        tuples = ('loads', 'bodies', 'joints', 'actuators')
        for attr in matrices:
            if attr not in exclude:
                assert getattr(self.system, attr)[:] == []  # 断言系统属性为空列表
        for attr in tuples:
            if attr not in exclude:
                assert getattr(self.system, attr) == ()  # 断言系统元组属性为空元组
        if 'eom_method' not in exclude:
            assert self.system.eom_method is None  # 断言系统的运动方程求解方法为空

    def _create_filled_system(self, with_speeds=True):  # 创建填充了内容的系统
        self.system = System(ReferenceFrame('frame'), Point('fixed_point'))  # 创建系统对象，包括参考系和固定点
        u = dynamicsymbols('u:6') if with_speeds else qd  # 根据 with_speeds 决定使用 u 或 qd 作为广义速度
        self.bodies = symbols('rb1:5', cls=RigidBody)  # 创建 RigidBody 对象的符号列表
        self.joints = (
            PinJoint('J1', self.bodies[0], self.bodies[1], q[0], u[0]),  # 创建 PinJoint 连接
            PrismaticJoint('J2', self.bodies[1], self.bodies[2], q[1], u[1]),  # 创建 PrismaticJoint 连接
            PinJoint('J3', self.bodies[2], self.bodies[3], q[2], u[2])  # 创建 PinJoint 连接
        )
        self.system.add_joints(*self.joints)  # 添加关节到系统
        self.system.add_coordinates(q[3], independent=[False])  # 添加广义坐标到系统，标记为非独立
        self.system.add_speeds(u[3], independent=False)  # 添加广义速度到系统，标记为非独立
        if with_speeds:
            self.system.add_kdes(u[3] - qd[3])  # 添加速度关系式
            self.system.add_auxiliary_speeds(ua[0], ua[1])  # 添加辅助速度
        self.system.add_holonomic_constraints(q[2] - q[0] + q[1])  # 添加完整约束
        self.system.add_nonholonomic_constraints(u[3] - qd[1] + u[2])  # 添加非完整约束
        self.system.u_ind = u[:2]  # 设置独立广义速度
        self.system.u_dep = u[2:4]  # 设置非独立广义速度
        self.q_ind, self.q_dep = self.system.q_ind[:], self.system.q_dep[:]  # 复制系统的独立和非独立广义坐标
        self.u_ind, self.u_dep = self.system.u_ind[:], self.system.u_dep[:]  # 复制系统的独立和非独立广义速度
        self.kdes = self.system.kdes[:]  # 复制系统的速度关系式
        self.hc = self.system.holonomic_constraints[:]  # 复制系统的完整约束
        self.vc = self.system.velocity_constraints[:]  # 复制系统的速度约束
        self.nhc = self.system.nonholonomic_constraints[:]  # 复制系统的非完整约束

    @pytest.fixture()  # 定义填充了内容的系统的 pytest fixture
    def _filled_system_setup(self):
        self._create_filled_system(with_speeds=True)  # 调用创建填充了内容的系统的方法，包括广义速度
    # 定义一个 Pytest 的装置（fixture），用于设置一个没有速度信息的填充系统
    @pytest.fixture()
    def _filled_system_setup_no_speeds(self):
        # 调用内部方法创建一个填充系统，排除速度信息
        self._create_filled_system(with_speeds=False)
    
    # 定义一个内部方法，用于检查填充系统的各个部分是否符合预期
    def _filled_system_check(self, exclude=()):
        # 检查是否排除 'q_ind'，若未排除则验证系统的 q_ind 数组
        assert 'q_ind' in exclude or self.system.q_ind[:] == q[:3]
        # 检查是否排除 'q_dep'，若未排除则验证系统的 q_dep 数组
        assert 'q_dep' in exclude or self.system.q_dep[:] == [q[3]]
        # 检查是否排除 'q'，若未排除则验证系统的 q 数组
        assert 'q' in exclude or self.system.q[:] == q[:4]
        # 检查是否排除 'u_ind'，若未排除则验证系统的 u_ind 数组
        assert 'u_ind' in exclude or self.system.u_ind[:] == u[:2]
        # 检查是否排除 'u_dep'，若未排除则验证系统的 u_dep 数组
        assert 'u_dep' in exclude or self.system.u_dep[:] == u[2:4]
        # 检查是否排除 'u'，若未排除则验证系统的 u 数组
        assert 'u' in exclude or self.system.u[:] == u[:4]
        # 检查是否排除 'u_aux'，若未排除则验证系统的 u_aux 数组
        assert 'u_aux' in exclude or self.system.u_aux[:] == ua[:2]
        # 检查是否排除 'kdes'，若未排除则验证系统的 kdes 数组
        assert 'kdes' in exclude or self.system.kdes[:] == [
            ui - qdi for ui, qdi in zip(u[:4], qd[:4])]
        # 检查是否排除 'holonomic_constraints'，若未排除则验证系统的 holonomic_constraints 数组
        assert ('holonomic_constraints' in exclude or
                self.system.holonomic_constraints[:] == [q[2] - q[0] + q[1]])
        # 检查是否排除 'nonholonomic_constraints'，若未排除则验证系统的 nonholonomic_constraints 数组
        assert ('nonholonomic_constraints' in exclude or
                self.system.nonholonomic_constraints[:] == [u[3] - qd[1] + u[2]]
                )
        # 检查是否排除 'velocity_constraints'，若未排除则验证系统的 velocity_constraints 数组
        assert ('velocity_constraints' in exclude or
                self.system.velocity_constraints[:] == [
                    qd[2] - qd[0] + qd[1], u[3] - qd[1] + u[2]])
        # 检查是否排除 'bodies'，若未排除则验证系统的 bodies 元组
        assert ('bodies' in exclude or
                self.system.bodies == tuple(self.bodies))
        # 检查是否排除 'joints'，若未排除则验证系统的 joints 元组
        assert ('joints' in exclude or
                self.system.joints == tuple(self.joints))
    
    # 定义一个 Pytest 的装置（fixture），用于设置一个移动的点质量
    @pytest.fixture()
    def _moving_point_mass(self, _empty_system_setup):
        # 设置系统的 q_ind 数组为 q[0]
        self.system.q_ind = q[0]
        # 设置系统的 u_ind 数组为 u[0]
        self.system.u_ind = u[0]
        # 设置系统的 kdes 数组为 u[0] - q[0] 的时间导数
        self.system.kdes = u[0] - q[0].diff(t)
        # 创建一个粒子 'p'，其质量为符号 'm'
        p = Particle('p', mass=symbols('m'))
        # 向系统添加这个粒子
        self.system.add_bodies(p)
        # 设置粒子的质心位置，相对于系统的固定点，以 q[0] * self.system.x 为位置
        p.masscenter.set_pos(self.system.fixed_point, q[0] * self.system.x)
class TestSystem(TestSystemBase):
    # 测试空系统的方法
    def test_empty_system(self, _empty_system_setup):
        # 执行空系统检查
        self._empty_system_check()
        # 验证系统的有效性
        self.system.validate_system()

    # 测试已填充系统的方法
    def test_filled_system(self, _filled_system_setup):
        # 执行已填充系统检查
        self._filled_system_check()
        # 验证系统的有效性
        self.system.validate_system()

    # 初始化测试
    @pytest.mark.parametrize('frame', [None, ReferenceFrame('frame')])
    @pytest.mark.parametrize('fixed_point', [None, Point('fixed_point')])
    def test_init(self, frame, fixed_point):
        # 如果固定点和参考框架均为空，则创建一个默认的系统
        if fixed_point is None and frame is None:
            self.system = System()
        else:
            # 否则使用给定的固定点和参考框架创建系统
            self.system = System(frame, fixed_point)
        
        # 断言固定点的名称是否为 'inertial_point'，如果固定点为空
        if fixed_point is None:
            assert self.system.fixed_point.name == 'inertial_point'
        else:
            # 否则断言系统的固定点与给定的固定点相同
            assert self.system.fixed_point == fixed_point
        
        # 断言参考框架的名称是否为 'inertial_frame'，如果参考框架为空
        if frame is None:
            assert self.system.frame.name == 'inertial_frame'
        else:
            # 否则断言系统的参考框架与给定的参考框架相同
            assert self.system.frame == frame
        
        # 执行空系统检查
        self._empty_system_check()
        
        # 断言系统属性为不可变矩阵类型
        assert isinstance(self.system.q_ind, ImmutableMatrix)
        assert isinstance(self.system.q_dep, ImmutableMatrix)
        assert isinstance(self.system.q, ImmutableMatrix)
        assert isinstance(self.system.u_ind, ImmutableMatrix)
        assert isinstance(self.system.u_dep, ImmutableMatrix)
        assert isinstance(self.system.u, ImmutableMatrix)
        assert isinstance(self.system.kdes, ImmutableMatrix)
        assert isinstance(self.system.holonomic_constraints, ImmutableMatrix)
        assert isinstance(self.system.nonholonomic_constraints, ImmutableMatrix)

    # 从牛顿刚体创建系统的测试
    def test_from_newtonian_rigid_body(self):
        # 创建一个牛顿刚体
        rb = RigidBody('body')
        # 使用牛顿刚体创建系统
        self.system = System.from_newtonian(rb)
        # 断言系统的固定点与质心相同
        assert self.system.fixed_point == rb.masscenter
        # 断言系统的参考框架与牛顿刚体的参考框架相同
        assert self.system.frame == rb.frame
        # 执行空系统检查，排除 'bodies'
        self._empty_system_check(exclude=('bodies',))
        # 将牛顿刚体添加到系统的 'bodies' 属性中
        self.system.bodies = (rb,)

    # 从牛顿粒子创建系统的测试
    def test_from_newtonian_particle(self):
        # 创建一个牛顿粒子
        pt = Particle('particle')
        # 使用牛顿粒子创建系统应该引发 TypeError 异常
        with pytest.raises(TypeError):
            System.from_newtonian(pt)

    # 参数化测试
    @pytest.mark.parametrize('args, kwargs, exp_q_ind, exp_q_dep, exp_q', [
        # 第一组参数
        (q[:3], {}, q[:3], [], q[:3]),
        # 第二组参数
        (q[:3], {'independent': True}, q[:3], [], q[:3]),
        # 第三组参数
        (q[:3], {'independent': False}, [], q[:3], q[:3]),
        # 第四组参数
        (q[:3], {'independent': [True, False, True]}, [q[0], q[2]], [q[1]], [q[0], q[2], q[1]]),
    ])
    # 定义测试方法 test_coordinates，使用参数 _empty_system_setup, args, kwargs,
    # exp_q_ind, exp_q_dep, exp_q
    def test_coordinates(self, _empty_system_setup, args, kwargs,
                         exp_q_ind, exp_q_dep, exp_q):
        # 测试 add_coordinates 方法
        self.system.add_coordinates(*args, **kwargs)
        # 断言系统的独立坐标与预期相符
        assert self.system.q_ind[:] == exp_q_ind
        # 断言系统的依赖坐标与预期相符
        assert self.system.q_dep[:] == exp_q_dep
        # 断言系统的所有坐标与预期相符
        assert self.system.q[:] == exp_q
        # 调用私有方法 _empty_system_check，排除 'q_ind', 'q_dep', 'q' 参数
        self._empty_system_check(exclude=('q_ind', 'q_dep', 'q'))
        
        # 再次测试设置器方法 q_ind 和 q_dep
        self.system.q_ind = exp_q_ind
        self.system.q_dep = exp_q_dep
        # 断言系统的独立坐标与预期相符
        assert self.system.q_ind[:] == exp_q_ind
        # 断言系统的依赖坐标与预期相符
        assert self.system.q_dep[:] == exp_q_dep
        # 断言系统的所有坐标与预期相符
        assert self.system.q[:] == exp_q
        # 调用私有方法 _empty_system_check，排除 'q_ind', 'q_dep', 'q' 参数
        self._empty_system_check(exclude=('q_ind', 'q_dep', 'q'))

    # 使用 pytest 的 parametrize 装饰器定义多组参数化测试
    @pytest.mark.parametrize('func', ['add_coordinates', 'add_speeds'])
    @pytest.mark.parametrize('args, kwargs', [
        ((q[0], q[5]), {}),
        ((u[0], u[5]), {}),
        ((q[0],), {'independent': False}),
        ((u[0],), {'independent': False}),
        ((u[0], q[5]), {}),
        ((symbols('a'), q[5]), {}),
    ])
    # 定义测试方法 test_coordinates_speeds_invalid，使用参数 _filled_system_setup,
    # func, args, kwargs
    def test_coordinates_speeds_invalid(self, _filled_system_setup, func, args,
                                        kwargs):
        # 使用 pytest.raises 检查是否引发 ValueError 异常
        with pytest.raises(ValueError):
            # 调用系统对象的 func 方法，传入 args 和 kwargs 参数
            getattr(self.system, func)(*args, **kwargs)
        # 调用私有方法 _filled_system_check
        self._filled_system_check()

    # 使用 pytest 的 parametrize 装饰器定义多组参数化测试
    @pytest.mark.parametrize('args, kwargs, exp_u_ind, exp_u_dep, exp_u', [
        (u[:3], {}, u[:3], [], u[:3]),
        (u[:3], {'independent': True}, u[:3], [], u[:3]),
        (u[:3], {'independent': False}, [], u[:3], u[:3]),
        (u[:3], {'independent': [True, False, True]}, [u[0], u[2]], [u[1]],
         [u[0], u[2], u[1]]),
    ])
    # 定义测试方法 test_speeds，使用参数 _empty_system_setup, args, kwargs,
    # exp_u_ind, exp_u_dep, exp_u
    def test_speeds(self, _empty_system_setup, args, kwargs, exp_u_ind,
                    exp_u_dep, exp_u):
        # 测试 add_speeds 方法
        self.system.add_speeds(*args, **kwargs)
        # 断言系统的独立速度与预期相符
        assert self.system.u_ind[:] == exp_u_ind
        # 断言系统的依赖速度与预期相符
        assert self.system.u_dep[:] == exp_u_dep
        # 断言系统的所有速度与预期相符
        assert self.system.u[:] == exp_u
        # 调用私有方法 _empty_system_check，排除 'u_ind', 'u_dep', 'u' 参数
        self._empty_system_check(exclude=('u_ind', 'u_dep', 'u'))
        
        # 再次测试设置器方法 u_ind 和 u_dep
        self.system.u_ind = exp_u_ind
        self.system.u_dep = exp_u_dep
        # 断言系统的独立速度与预期相符
        assert self.system.u_ind[:] == exp_u_ind
        # 断言系统的依赖速度与预期相符
        assert self.system.u_dep[:] == exp_u_dep
        # 断言系统的所有速度与预期相符
        assert self.system.u[:] == exp_u
        # 调用私有方法 _empty_system_check，排除 'u_ind', 'u_dep', 'u' 参数
        self._empty_system_check(exclude=('u_ind', 'u_dep', 'u'))

    # 使用 pytest 的 parametrize 装饰器定义一组参数化测试
    @pytest.mark.parametrize('args, kwargs, exp_u_aux', [
        (ua[:3], {}, ua[:3]),
    ])
    # 定义测试方法 test_auxiliary_speeds，使用参数 _empty_system_setup, args, kwargs,
    # exp_u_aux
    def test_auxiliary_speeds(self, _empty_system_setup, args, kwargs,
                              exp_u_aux):
        # 测试 add_auxiliary_speeds 方法
        self.system.add_auxiliary_speeds(*args, **kwargs)
        # 断言系统的辅助速度与预期相符
        assert self.system.u_aux[:] == exp_u_aux
        # 调用私有方法 _empty_system_check，排除 'u_aux' 参数
        self._empty_system_check(exclude=('u_aux',))
        
        # 再次测试设置器方法 u_aux
        self.system.u_aux = exp_u_aux
        # 断言系统的辅助速度与预期相符
        assert self.system.u_aux[:] == exp_u_aux
        # 调用私有方法 _empty_system_check，排除 'u_aux' 参数
        self._empty_system_check(exclude=('u_aux',))
    # 使用 pytest 的参数化装饰器，为测试方法提供多组参数
    @pytest.mark.parametrize('args, kwargs', [
        # 使用第三个元素和第一个元素作为参数，以及空的关键字参数
        ((ua[2], q[0]), {}),
        # 使用第三个元素和第二个元素作为参数，以及空的关键字参数
        ((ua[2], u[1]), {}),
        # 使用第一个元素和第三个元素作为参数，以及空的关键字参数
        ((ua[0], ua[2]), {}),
        # 使用符号 'a' 和第三个元素作为参数，以及空的关键字参数
        ((symbols('a'), ua[2]), {}),
    ])
    # 测试添加辅助速度时出现 ValueError 异常的情况
    def test_auxiliary_invalid(self, _filled_system_setup, args, kwargs):
        # 断言调用系统方法 add_auxiliary_speeds(*args, **kwargs) 会引发 ValueError 异常
        with pytest.raises(ValueError):
            self.system.add_auxiliary_speeds(*args, **kwargs)
        # 执行系统状态检查
        self._filled_system_check()

    # 使用 pytest 的参数化装饰器，为测试方法提供多组参数
    @pytest.mark.parametrize('prop, add_func, args, kwargs', [
        # 测试重置后添加坐标 'q[0]' 的情况
        ('q_ind', 'add_coordinates', (q[0],), {}),
        # 测试重置后添加坐标 'q[3]' 的情况，并设置 'independent' 参数为 False
        ('q_dep', 'add_coordinates', (q[3],), {'independent': False}),
        # 测试重置后添加速度 'u[0]' 的情况
        ('u_ind', 'add_speeds', (u[0],), {}),
        # 测试重置后添加速度 'u[3]' 的情况，并设置 'independent' 参数为 False
        ('u_dep', 'add_speeds', (u[3],), {'independent': False}),
        # 测试重置后添加辅助速度 'ua[2]' 的情况
        ('u_aux', 'add_auxiliary_speeds', (ua[2],), {}),
        # 测试添加动力学微分方程 'qd[0] - u[0]' 的情况
        ('kdes', 'add_kdes', (qd[0] - u[0],), {}),
        # 测试添加完整约束 'q[0] - q[1]' 的情况
        ('holonomic_constraints', 'add_holonomic_constraints', (q[0] - q[1],), {}),
        # 测试添加非完整约束 'u[0] - u[1]' 的情况
        ('nonholonomic_constraints', 'add_nonholonomic_constraints', (u[0] - u[1],), {}),
        # 测试添加刚体 'RigidBody('body')' 的情况
        ('bodies', 'add_bodies', (RigidBody('body'),), {}),
        # 测试添加负载 'Force(Point('P'), ReferenceFrame('N').x)' 的情况
        ('loads', 'add_loads', (Force(Point('P'), ReferenceFrame('N').x),), {}),
        # 测试添加执行器 'TorqueActuator(symbols('T'), ReferenceFrame('N').x, ReferenceFrame('A'))' 的情况
        ('actuators', 'add_actuators', (TorqueActuator(symbols('T'), ReferenceFrame('N').x, ReferenceFrame('A')),), {}),
    ])
    # 测试重置后添加各种属性并进行断言
    def test_add_after_reset(self, _filled_system_setup, prop, add_func, args,
                             kwargs):
        # 设置 self.system 中指定属性为 ()
        setattr(self.system, prop, ())
        # 设置要排除的属性，包括 'prop', 'q', 'u'，如果是 'holonomic_constraints' 或 'nonholonomic_constraints'，还包括 'velocity_constraints'
        exclude = (prop, 'q', 'u')
        if prop in ('holonomic_constraints', 'nonholonomic_constraints'):
            exclude += ('velocity_constraints',)
        # 执行系统状态检查，传入要排除的属性
        self._filled_system_check(exclude=exclude)
        # 断言属性列表为空列表
        assert list(getattr(self.system, prop)[:]) == []
        # 调用 self.system 中的方法 add_func(*args, **kwargs)
        getattr(self.system, add_func)(*args, **kwargs)
        # 断言属性列表等于参数 args 的列表形式
        assert list(getattr(self.system, prop)[:]) == list(args)

    # 使用 pytest 的参数化装饰器，为测试方法提供多组参数
    @pytest.mark.parametrize('prop, add_func, value, error', [
        # 测试添加坐标 'symbols('a')' 时引发 ValueError 异常的情况
        ('q_ind', 'add_coordinates', symbols('a'), ValueError),
        # 测试添加坐标 'symbols('a')' 时引发 ValueError 异常的情况
        ('q_dep', 'add_coordinates', symbols('a'), ValueError),
        # 测试添加速度 'symbols('a')' 时引发 ValueError 异常的情况
        ('u_ind', 'add_speeds', symbols('a'), ValueError),
        # 测试添加速度 'symbols('a')' 时引发 ValueError 异常的情况
        ('u_dep', 'add_speeds', symbols('a'), ValueError),
        # 测试添加辅助速度 'symbols('a')' 时引发 ValueError 异常的情况
        ('u_aux', 'add_auxiliary_speeds', symbols('a'), ValueError),
        # 测试添加 kdes 时引发 TypeError 异常的情况
        ('kdes', 'add_kdes', 7, TypeError),
        # 测试添加 holonomic_constraints 时引发 TypeError 异常的情况
        ('holonomic_constraints', 'add_holonomic_constraints', 7, TypeError),
        # 测试添加 nonholonomic_constraints 时引发 TypeError 异常的情况
        ('nonholonomic_constraints', 'add_nonholonomic_constraints', 7, TypeError),
        # 测试添加 bodies 时引发 TypeError 异常的情况
        ('bodies', 'add_bodies', symbols('a'), TypeError),
        # 测试添加 loads 时引发 TypeError 异常的情况
        ('loads', 'add_loads', symbols('a'), TypeError),
        # 测试添加 actuators 时引发 TypeError 异常的情况
        ('actuators', 'add_actuators', symbols('a'), TypeError),
    ])
    # 测试添加属性时引发各种类型异常
    def test_type_error(self, _filled_system_setup, prop, add_func, value,
                        error):
        # 断言调用 getattr(self.system, add_func)(value) 会引发 error 类型的异常
        with pytest.raises(error):
            getattr(self.system, add_func)(value)
        # 断言设置 setattr(self.system, prop, value) 会引发 error 类型的异常
        with pytest.raises(error):
            setattr(self.system, prop, value)
        # 执行系统状态检查
        self._filled_system_check()
    @pytest.mark.parametrize('args, kwargs, exp_kdes', [
        ((), {}, [ui - qdi for ui, qdi in zip(u[:4], qd[:4])]),
        ((u[4] - qd[4], u[5] - qd[5]), {},
         [ui - qdi for ui, qdi in zip(u[:6], qd[:6])]),
    ])
    # 使用 pytest 的参数化装饰器，为 test_kdes 方法定义多组参数和期望结果
    def test_kdes(self, _filled_system_setup, args, kwargs, exp_kdes):
        # 调用系统方法 add_kdes，传入参数 args 和 kwargs
        self.system.add_kdes(*args, **kwargs)
        # 调用辅助方法 _filled_system_check，检查系统状态（排除 'kdes'）
        self._filled_system_check(exclude=('kdes',))
        # 断言系统的 kdes 属性与期望的 exp_kdes 相等
        assert self.system.kdes[:] == exp_kdes
        # 再次测试 kdes 的 setter 方法
        self.system.kdes = exp_kdes
        # 再次调用 _filled_system_check 方法，检查系统状态（排除 'kdes'）
        self._filled_system_check(exclude=('kdes',))
        # 断言系统的 kdes 属性与期望的 exp_kdes 相等
        assert self.system.kdes[:] == exp_kdes

    @pytest.mark.parametrize('args, kwargs', [
        ((u[0] - qd[0], u[4] - qd[4]), {}),
        ((-(u[0] - qd[0]), u[4] - qd[4]), {}),
        (([u[0] - u[0], u[4] - qd[4]]), {}),
    ])
    # 使用 pytest 的参数化装饰器，为 test_kdes_invalid 方法定义多组无效参数和期望抛出异常
    def test_kdes_invalid(self, _filled_system_setup, args, kwargs):
        # 使用 pytest 的断言方法检查是否抛出 ValueError 异常
        with pytest.raises(ValueError):
            self.system.add_kdes(*args, **kwargs)
        # 调用辅助方法 _filled_system_check，检查系统状态

    @pytest.mark.parametrize('args, kwargs, exp_con', [
        ((), {}, [q[2] - q[0] + q[1]]),
        ((q[4] - q[5], q[5] + q[3]), {},
         [q[2] - q[0] + q[1], q[4] - q[5], q[5] + q[3]]),
    ])
    # 使用 pytest 的参数化装饰器，为 test_holonomic_constraints 方法定义多组参数和期望结果
    def test_holonomic_constraints(self, _filled_system_setup, args, kwargs,
                                   exp_con):
        # 定义需要排除的属性名称
        exclude = ('holonomic_constraints', 'velocity_constraints')
        # 计算期望的 velocity_constraints
        exp_vel_con = [c.diff(t) for c in exp_con] + self.nhc
        # 调用系统方法 add_holonomic_constraints，传入参数 args 和 kwargs
        self.system.add_holonomic_constraints(*args, **kwargs)
        # 调用辅助方法 _filled_system_check，检查系统状态（排除 exclude 中的属性）
        self._filled_system_check(exclude=exclude)
        # 断言系统的 holonomic_constraints 属性与期望的 exp_con 相等
        assert self.system.holonomic_constraints[:] == exp_con
        # 断言系统的 velocity_constraints 属性与期望的 exp_vel_con 相等
        assert self.system.velocity_constraints[:] == exp_vel_con
        # 再次测试 holonomic_constraints 的 setter 方法
        self.system.holonomic_constraints = exp_con
        # 再次调用 _filled_system_check 方法，检查系统状态（排除 exclude 中的属性）
        self._filled_system_check(exclude=exclude)
        # 断言系统的 holonomic_constraints 属性与期望的 exp_con 相等
        assert self.system.holonomic_constraints[:] == exp_con
        # 断言系统的 velocity_constraints 属性与期望的 exp_vel_con 相等
        assert self.system.velocity_constraints[:] == exp_vel_con

    @pytest.mark.parametrize('args, kwargs', [
        ((q[2] - q[0] + q[1], q[4] - q[3]), {}),
        ((-(q[2] - q[0] + q[1]), q[4] - q[3]), {}),
        ((q[0] - q[0], q[4] - q[3]), {}),
    ])
    # 使用 pytest 的参数化装饰器，为 test_holonomic_constraints_invalid 方法定义多组无效参数和期望抛出异常
    def test_holonomic_constraints_invalid(self, _filled_system_setup, args,
                                           kwargs):
        # 使用 pytest 的断言方法检查是否抛出 ValueError 异常
        with pytest.raises(ValueError):
            self.system.add_holonomic_constraints(*args, **kwargs)
        # 调用辅助方法 _filled_system_check，检查系统状态
    # 定义测试方法，用于测试非完整约束
    def test_nonholonomic_constraints(self, _filled_system_setup, args, kwargs,
                                      exp_con):
        # 定义需要排除的属性
        exclude = ('nonholonomic_constraints', 'velocity_constraints')
        # 根据预期约束值和当前速度约束列表创建预期速度约束列表
        exp_vel_con = self.vc[:len(self.hc)] + exp_con
        # 测试添加非完整约束方法
        self.system.add_nonholonomic_constraints(*args, **kwargs)
        # 执行系统状态检查，排除指定的属性
        self._filled_system_check(exclude=exclude)
        # 断言系统的非完整约束列表与预期值相等
        assert self.system.nonholonomic_constraints[:] == exp_con
        # 断言系统的速度约束列表与预期速度约束列表相等
        assert self.system.velocity_constraints[:] == exp_vel_con
        # 再次测试设置非完整约束的 setter 方法
        self.system.nonholonomic_constraints = exp_con
        # 再次执行系统状态检查，排除指定的属性
        self._filled_system_check(exclude=exclude)
        # 再次断言系统的非完整约束列表与预期值相等
        assert self.system.nonholonomic_constraints[:] == exp_con
        # 再次断言系统的速度约束列表与预期速度约束列表相等
        assert self.system.velocity_constraints[:] == exp_vel_con

    # 参数化测试方法，测试非完整约束的无效输入
    @pytest.mark.parametrize('args, kwargs', [
        ((u[3] - qd[1] + u[2], u[4] - u[3]), {}),
        (-(u[3] - qd[1] + u[2]), u[4] - u[3], {}),
        ((u[0] - u[0], u[4] - u[3]), {}),
        (([u[0] - u[0], u[4] - u[3]]), {}),
    ])
    def test_nonholonomic_constraints_invalid(self, _filled_system_setup, args,
                                              kwargs):
        # 使用 pytest.raises 检测是否引发 ValueError 异常
        with pytest.raises(ValueError):
            self.system.add_nonholonomic_constraints(*args, **kwargs)
        # 执行系统状态检查
        self._filled_system_check()

    # 参数化测试方法，测试速度约束的覆盖行为
    @pytest.mark.parametrize('constraints, expected', [
        ([], []),
        (qd[2] - qd[0] + qd[1], [qd[2] - qd[0] + qd[1]]),
        ([qd[2] + qd[1], u[2] - u[1]], [qd[2] + qd[1], u[2] - u[1]]),
    ])
    def test_velocity_constraints_overwrite(self, _filled_system_setup,
                                            constraints, expected):
        # 设置系统的速度约束属性为给定的约束
        self.system.velocity_constraints = constraints
        # 执行系统状态检查，排除速度约束属性
        self._filled_system_check(exclude=('velocity_constraints',))
        # 断言系统的速度约束列表与预期列表相等
        assert self.system.velocity_constraints[:] == expected

    # 测试将速度约束重置为自动模式的方法
    def test_velocity_constraints_back_to_auto(self, _filled_system_setup):
        # 设置系统的速度约束属性为指定的约束
        self.system.velocity_constraints = qd[3] - qd[2]
        # 执行系统状态检查，排除速度约束属性
        self._filled_system_check(exclude=('velocity_constraints',))
        # 断言系统的速度约束列表与预期列表相等
        assert self.system.velocity_constraints[:] == [qd[3] - qd[2]]
        # 将系统的速度约束属性重置为 None
        self.system.velocity_constraints = None
        # 再次执行系统状态检查
        self._filled_system_check()
    # 测试添加刚体和质点到系统中的功能
    def test_bodies(self, _filled_system_setup):
        # 创建两个刚体和两个质点对象
        rb1, rb2 = RigidBody('rb1'), RigidBody('rb2')
        p1, p2 = Particle('p1'), Particle('p2')
        
        # 将刚体和质点添加到系统中
        self.system.add_bodies(rb1, p1)
        # 断言系统中的物体列表是否包含预期的刚体和质点
        assert self.system.bodies == (*self.bodies, rb1, p1)
        
        # 再次添加一个质点到系统中
        self.system.add_bodies(p2)
        # 断言系统中的物体列表是否包含预期的刚体、质点和新添加的质点
        assert self.system.bodies == (*self.bodies, rb1, p1, p2)
        
        # 将系统的物体列表清空
        self.system.bodies = []
        # 断言此时系统中的物体列表为空元组
        assert self.system.bodies == ()
        
        # 将系统的物体列表设为单个质点对象
        self.system.bodies = p2
        # 断言系统中的物体列表是否仅包含该质点对象
        assert self.system.bodies == (p2,)
        
        # 创建一个符号变量
        symb = symbols('symb')
        # 使用 pytest 检查尝试将符号变量添加到系统中是否引发 TypeError 异常
        pytest.raises(TypeError, lambda: self.system.add_bodies(symb))
        
        # 使用 pytest 检查尝试添加已经存在于系统中的质点是否引发 ValueError 异常
        pytest.raises(ValueError, lambda: self.system.add_bodies(p2))
        
        # 使用 pytest 检查尝试将多个物体一次性设置为系统物体列表是否引发 TypeError 异常
        with pytest.raises(TypeError):
            self.system.bodies = (rb1, rb2, p1, p2, symb)
        
        # 断言系统中的物体列表是否仅包含之前设置的单个质点对象
        assert self.system.bodies == (p2,)

    # 测试添加力和扭矩到系统中的功能
    def test_add_loads(self):
        # 创建一个空的系统对象
        system = System()
        # 创建两个参考系对象 N 和 A
        N, A = ReferenceFrame('N'), ReferenceFrame('A')
        # 创建一个在参考系 N 中的刚体对象 rb1
        rb1 = RigidBody('rb1', frame=N)
        # 创建一个不依赖于任何刚体的点对象 mc1
        mc1 = Point('mc1')
        # 创建一个依赖于点 mc1 的质点对象 p1
        p1 = Particle('p1', mc1)
        
        # 将扭矩和力添加到系统中
        system.add_loads(Torque(rb1, N.x), (mc1, A.x), Force(p1, A.x))
        # 断言系统中的载荷列表是否包含预期的扭矩和力
        assert system.loads == ((N, N.x), (mc1, A.x), (mc1, A.x))
        
        # 将系统的载荷列表设为单个力
        system.loads = [(A, A.x)]
        # 断言系统中的载荷列表是否仅包含该力
        assert system.loads == ((A, A.x),)
        
        # 使用 pytest 检查尝试添加不合法数量参数的载荷是否引发 ValueError 异常
        pytest.raises(ValueError, lambda: system.add_loads((N, N.x, N.y)))
        
        # 使用 pytest 检查尝试将多个物体一次性设置为系统载荷列表是否引发 TypeError 异常
        with pytest.raises(TypeError):
            system.loads = (N, N.x)
        
        # 断言系统中的载荷列表是否仅包含之前设置的单个力
        assert system.loads == ((A, A.x),)

    # 测试添加执行器到系统中的功能
    def test_add_actuators(self):
        # 创建一个空的系统对象
        system = System()
        # 创建两个参考系对象 N 和 A
        N, A = ReferenceFrame('N'), ReferenceFrame('A')
        # 创建一个在参考系 N 中的力矩执行器对象 act1
        act1 = TorqueActuator(symbols('T1'), N.x, N)
        # 创建一个在参考系 N 和 A 中的力矩执行器对象 act2
        act2 = TorqueActuator(symbols('T2'), N.y, N, A)
        
        # 将力矩执行器 act1 添加到系统中
        system.add_actuators(act1)
        # 断言系统中的执行器列表是否仅包含该力矩执行器 act1
        assert system.actuators == (act1,)
        # 断言系统中的载荷列表是否为空元组
        assert system.loads == ()
        
        # 将系统的执行器列表设为单个力矩执行器 act2
        system.actuators = (act2,)
        # 断言系统中的执行器列表是否仅包含该力矩执行器 act2
        assert system.actuators == (act2,)
    # 定义一个测试方法，用于测试添加关节功能
    def test_add_joints(self):
        # 定义动力符号变量 q1, q2, q3, q4 和速度符号变量 u1, u2, u3
        q1, q2, q3, q4, u1, u2, u3 = dynamicsymbols('q1:5 u1:4')
        # 定义 RigidBody 类的符号 rb1, rb2, rb3, rb4, rb5
        rb1, rb2, rb3, rb4, rb5 = symbols('rb1:6', cls=RigidBody)
        
        # 创建四个不同类型的关节对象 J1, J2, J3, J_lag
        J1 = PinJoint('J1', rb1, rb2, q1, u1)
        J2 = PrismaticJoint('J2', rb2, rb3, q2, u2)
        J3 = PinJoint('J3', rb3, rb4, q3, u3)
        J_lag = PinJoint('J_lag', rb4, rb5, q4, q4.diff(t))
        
        # 创建一个系统对象
        system = System()
        
        # 将关节 J1 添加到系统中，并断言系统的关节列表
        system.add_joints(J1)
        assert system.joints == (J1,)
        
        # 断言系统的物体列表和广义坐标指标
        assert system.bodies == (rb1, rb2)
        assert system.q_ind == ImmutableMatrix([q1])
        assert system.u_ind == ImmutableMatrix([u1])
        assert system.kdes == ImmutableMatrix([u1 - q1.diff(t)])
        
        # 向系统中添加一个物体和广义坐标，然后添加一个动力学方程
        system.add_bodies(rb4)
        system.add_coordinates(q3)
        system.add_kdes(u3 - q3.diff(t))
        
        # 将关节 J3 添加到系统中，并断言系统的关节列表
        system.add_joints(J3)
        assert system.joints == (J1, J3)
        
        # 再次断言系统的物体列表和广义坐标指标
        assert system.bodies == (rb1, rb2, rb4, rb3)
        assert system.q_ind == ImmutableMatrix([q1, q3])
        assert system.u_ind == ImmutableMatrix([u1, u3])
        assert system.kdes == ImmutableMatrix([u1 - q1.diff(t), u3 - q3.diff(t)])
        
        # 添加一个额外的动力学方程
        system.add_kdes(-(u2 - q2.diff(t)))
        
        # 将关节 J2 添加到系统中，并断言系统的关节列表
        system.add_joints(J2)
        assert system.joints == (J1, J3, J2)
        
        # 再次断言系统的物体列表、广义坐标指标和动力学方程
        assert system.bodies == (rb1, rb2, rb4, rb3)
        assert system.q_ind == ImmutableMatrix([q1, q3, q2])
        assert system.u_ind == ImmutableMatrix([u1, u3, u2])
        assert system.kdes == ImmutableMatrix([u1 - q1.diff(t), u3 - q3.diff(t),
                                               -(u2 - q2.diff(t))])
        
        # 将关节 J_lag 添加到系统中，并断言系统的关节列表
        system.add_joints(J_lag)
        assert system.joints == (J1, J3, J2, J_lag)
        
        # 最后断言系统的物体列表、广义坐标指标、动力学方程和约束指标列表为空
        assert system.bodies == (rb1, rb2, rb4, rb3, rb5)
        assert system.q_ind == ImmutableMatrix([q1, q3, q2, q4])
        assert system.u_ind == ImmutableMatrix([u1, u3, u2, q4.diff(t)])
        assert system.kdes == ImmutableMatrix([u1 - q1.diff(t), u3 - q3.diff(t),
                                               -(u2 - q2.diff(t))])
        assert system.q_dep[:] == []
        assert system.u_dep[:] == []
        
        # 使用 pytest.raises 检查添加重复关节和错误类型关节时是否会引发异常
        pytest.raises(ValueError, lambda: system.add_joints(J2))
        pytest.raises(TypeError, lambda: system.add_joints(rb1))

    # 使用 pytest.mark.parametrize 注入不同的参数化测试用例来测试 get_joint 方法
    @pytest.mark.parametrize('name, joint_index', [
        ('J1', 0),
        ('J2', 1),
        ('not_existing', None),
    ])
    # 定义测试方法 test_get_joint，测试获取关节的功能
    def test_get_joint(self, _filled_system_setup, name, joint_index):
        # 调用系统对象的 get_joint 方法获取指定名称的关节
        joint = self.system.get_joint(name)
        
        # 根据参数化传入的关节索引断言获取的关节对象是否符合预期
        if joint_index is None:
            assert joint is None
        else:
            assert joint == self.joints[joint_index]
    # 使用 pytest 的 parametrize 装饰器，定义多组测试参数，每组参数包括名称和 body_index
    @pytest.mark.parametrize('name, body_index', [
        ('rb1', 0),
        ('rb3', 2),
        ('not_existing', None),
    ])
    # 定义测试方法 test_get_body，接受参数 _filled_system_setup, name, body_index
    def test_get_body(self, _filled_system_setup, name, body_index):
        # 调用系统对象的 get_body 方法，根据名称 name 获取对应的 body
        body = self.system.get_body(name)
        # 如果 body_index 为 None，则断言返回的 body 也为 None
        if body_index is None:
            assert body is None
        # 否则断言返回的 body 等于预期的 self.bodies[body_index]
        else:
            assert body == self.bodies[body_index]
    
    # 使用 pytest 的 parametrize 装饰器，定义多组测试参数，每组参数包括 eom_method
    @pytest.mark.parametrize('eom_method', [KanesMethod, LagrangesMethod])
    # 定义测试方法 test_form_eoms_calls_subclass，接受参数 _moving_point_mass, eom_method
    def test_form_eoms_calls_subclass(self, _moving_point_mass, eom_method):
        # 定义一个名为 MyMethod 的子类，继承自 eom_method
        class MyMethod(eom_method):
            pass
    
        # 调用系统对象的 form_eoms 方法，设置 eom_method 为 MyMethod
        self.system.form_eoms(eom_method=MyMethod)
        # 断言系统对象的 eom_method 已经是 MyMethod 类的实例
        assert isinstance(self.system.eom_method, MyMethod)
    
    # 使用 pytest 的 parametrize 装饰器，定义多组测试参数，每组参数包括 kwargs 和 expected
    @pytest.mark.parametrize('kwargs, expected', [
        ({}, ImmutableMatrix([[-1, 0], [0, symbols('m')]])),
        ({'explicit_kinematics': True}, ImmutableMatrix([[1, 0],
                                                         [0, symbols('m')]])),
    ])
    # 定义测试方法 test_system_kane_form_eoms_kwargs，接受参数 _moving_point_mass, kwargs, expected
    def test_system_kane_form_eoms_kwargs(self, _moving_point_mass, kwargs, expected):
        # 调用系统对象的 form_eoms 方法，传入 kwargs 参数
        self.system.form_eoms(**kwargs)
        # 断言系统对象的 mass_matrix_full 属性等于预期的 expected
        assert self.system.mass_matrix_full == expected
    
    # 使用 pytest 的 parametrize 装饰器，定义多组测试参数，每组参数包括 kwargs, mm, gm
    @pytest.mark.parametrize('kwargs, mm, gm', [
        ({}, ImmutableMatrix([[1, 0], [0, symbols('m')]]),
         ImmutableMatrix([q[0].diff(t), 0])),
    ])
    # 定义测试方法 test_system_lagrange_form_eoms_kwargs，接受参数 _moving_point_mass, kwargs, mm, gm
    def test_system_lagrange_form_eoms_kwargs(self, _moving_point_mass, kwargs, mm, gm):
        # 调用系统对象的 form_eoms 方法，设置 eom_method 为 LagrangesMethod，传入 kwargs 参数
        self.system.form_eoms(eom_method=LagrangesMethod, **kwargs)
        # 断言系统对象的 mass_matrix_full 属性等于预期的 mm
        assert self.system.mass_matrix_full == mm
        # 断言系统对象的 forcing_full 属性等于预期的 gm
        assert self.system.forcing_full == gm
    
    # 使用 pytest 的 parametrize 装饰器，定义多组测试参数，每组参数包括 eom_method, kwargs, error
    @pytest.mark.parametrize('eom_method, kwargs, error', [
        (KanesMethod, {'non_existing_kwarg': 1}, TypeError),
        (LagrangesMethod, {'non_existing_kwarg': 1}, TypeError),
        (KanesMethod, {'bodies': []}, ValueError),
        (KanesMethod, {'kd_eqs': []}, ValueError),
        (LagrangesMethod, {'bodies': []}, ValueError),
        (LagrangesMethod, {'Lagrangian': 1}, ValueError),
    ])
    # 定义测试方法 test_form_eoms_kwargs_errors，接受参数 _empty_system_setup, eom_method, kwargs, error
    def test_form_eoms_kwargs_errors(self, _empty_system_setup, eom_method, kwargs, error):
        # 设置系统对象的 q_ind 属性为 q[0]
        self.system.q_ind = q[0]
        # 创建一个粒子对象 p，设置其质量为 symbols('m')
        p = Particle('p', mass=symbols('m'))
        # 向系统对象添加粒子 p
        self.system.add_bodies(p)
        # 设置粒子 p 的质心位置相对于系统的固定点
        p.masscenter.set_pos(self.system.fixed_point, q[0] * self.system.x)
        # 使用 pytest 的 raises 方法，断言调用系统对象的 form_eoms 方法会引发指定的 error 异常
        with pytest.raises(error):
            self.system.form_eoms(eom_method=eom_method, **kwargs)
# 定义一个测试类 TestValidateSystem，继承自 TestSystemBase 类
class TestValidateSystem(TestSystemBase):

    # 使用 pytest.mark.parametrize 装饰器定义参数化测试方法，参数包括有效方法、无效方法和速度设置
    @pytest.mark.parametrize('valid_method, invalid_method, with_speeds', [
        # 第一组参数：有效方法为 KanesMethod，无效方法为 LagrangesMethod，包含速度
        (KanesMethod, LagrangesMethod, True),
        # 第二组参数：有效方法为 LagrangesMethod，无效方法为 KanesMethod，不包含速度
        (LagrangesMethod, KanesMethod, False)
    ])
    # 定义测试方法 test_only_valid，接受 valid_method、invalid_method 和 with_speeds 参数
    def test_only_valid(self, valid_method, invalid_method, with_speeds):
        # 调用 _create_filled_system 方法创建填充后的系统，根据 with_speeds 参数设置速度
        self._create_filled_system(with_speeds=with_speeds)
        # 调用系统的 validate_system 方法，验证有效方法的系统设置
        self.system.validate_system(valid_method)
        # 使用 pytest.raises 检测是否抛出 ValueError 异常，验证无效方法的系统设置
        with pytest.raises(ValueError):
            self.system.validate_system(invalid_method)

    # 使用 pytest.mark.parametrize 装饰器定义参数化测试方法，参数包括方法和速度设置
    @pytest.mark.parametrize('method, with_speeds', [
        # 第一组参数：方法为 KanesMethod，包含速度
        (KanesMethod, True),
        # 第二组参数：方法为 LagrangesMethod，不包含速度
        (LagrangesMethod, False)
    ])
    # 定义测试方法 test_missing_joint_coordinate，接受 method 和 with_speeds 参数
    def test_missing_joint_coordinate(self, method, with_speeds):
        # 调用 _create_filled_system 方法创建填充后的系统，根据 with_speeds 参数设置速度
        self._create_filled_system(with_speeds=with_speeds)
        # 从系统中删除第一个广义坐标
        self.system.q_ind = self.q_ind[1:]
        # 从系统中删除最后一个广义速度
        self.system.u_ind = self.u_ind[:-1]
        # 从系统中删除最后一个广义动能方程
        self.system.kdes = self.kdes[:-1]
        # 使用 pytest.raises 检测是否抛出 ValueError 异常，验证系统设置
        pytest.raises(ValueError, lambda: self.system.validate_system(method))

    # 定义测试方法 test_missing_joint_speed，使用 _filled_system_setup 装饰器设置
    def test_missing_joint_speed(self, _filled_system_setup):
        # 从系统中删除最后一个广义坐标
        self.system.q_ind = self.q_ind[:-1]
        # 从系统中删除第一个广义速度
        self.system.u_ind = self.u_ind[1:]
        # 从系统中删除最后一个广义动能方程
        self.system.kdes = self.kdes[:-1]
        # 使用 pytest.raises 检测是否抛出 ValueError 异常，验证系统设置
        pytest.raises(ValueError, lambda: self.system.validate_system())

    # 定义测试方法 test_missing_joint_kdes，使用 _filled_system_setup 装饰器设置
    def test_missing_joint_kdes(self, _filled_system_setup):
        # 从系统中删除第一个广义动能方程
        self.system.kdes = self.kdes[1:]
        # 使用 pytest.raises 检测是否抛出 ValueError 异常，验证系统设置
        pytest.raises(ValueError, lambda: self.system.validate_system())

    # 定义测试方法 test_negative_joint_kdes，使用 _filled_system_setup 装饰器设置
    def test_negative_joint_kdes(self, _filled_system_setup):
        # 将第一个广义动能方程取反后放回系统
        self.system.kdes = [-self.kdes[0]] + self.kdes[1:]
        # 验证系统设置
        self.system.validate_system()

    # 使用 pytest.mark.parametrize 装饰器定义参数化测试方法，参数包括方法和速度设置
    @pytest.mark.parametrize('method, with_speeds', [
        # 第一组参数：方法为 KanesMethod，包含速度
        (KanesMethod, True),
        # 第二组参数：方法为 LagrangesMethod，不包含速度
        (LagrangesMethod, False)
    ])
    # 定义测试方法 test_missing_holonomic_constraint，接受 method 和 with_speeds 参数
    def test_missing_holonomic_constraint(self, method, with_speeds):
        # 调用 _create_filled_system 方法创建填充后的系统，根据 with_speeds 参数设置速度
        self._create_filled_system(with_speeds=with_speeds)
        # 清空系统的完整约束
        self.system.holonomic_constraints = []
        # 将非完整约束设置为系统的非完整约束加上一个关系式
        self.system.nonholonomic_constraints = self.nhc + [
            self.u_ind[1] - self.u_dep[0] + self.u_ind[0]]
        # 使用 pytest.raises 检测是否抛出 ValueError 异常，验证系统设置
        pytest.raises(ValueError, lambda: self.system.validate_system(method))
        # 清空系统的依赖广义坐标
        self.system.q_dep = []
        # 将系统的独立广义坐标设置为系统的独立广义坐标加上系统的依赖广义坐标
        self.system.q_ind = self.q_ind + self.q_dep
        # 验证系统设置
        self.system.validate_system(method)

    # 定义测试方法 test_missing_nonholonomic_constraint，使用 _filled_system_setup 装饰器设置
    def test_missing_nonholonomic_constraint(self, _filled_system_setup):
        # 清空系统的非完整约束
        self.system.nonholonomic_constraints = []
        # 使用 pytest.raises 检测是否抛出 ValueError 异常，验证系统设置
        pytest.raises(ValueError, lambda: self.system.validate_system())
        # 将系统的依赖广义速度设置为系统的依赖广义速度的第二个元素
        self.system.u_dep = self.u_dep[1]
        # 将系统的独立广义速度设置为系统的独立广义速度加上一个依赖广义速度的第一个元素
        self.system.u_ind = self.u_ind + [self.u_dep[0]]
        # 验证系统设置
        self.system.validate_system()
    # 测试坐标速度数量关系，测试更多速度超过坐标的情况
    self.system.u_ind = self.u_ind + [u[5]]
    # 更新控制速度与期望速度的列表，增加一个速度值
    self.system.kdes = self.kdes + [u[5] - qd[5]]
    # 验证系统的一致性，预期不会引发异常
    self.system.validate_system()
    
    # 测试坐标速度数量关系，测试更多坐标超过速度的情况
    self.system.q_ind = self.q_ind
    self.system.u_ind = self.u_ind[:-1]
    self.system.kdes = self.kdes[:-1]
    # 预期验证系统会引发值错误异常
    pytest.raises(ValueError, lambda: self.system.validate_system())

    # 测试不正确的 kdes 数量
    self.system.kdes = self.kdes[:-1]
    # 预期验证系统会引发值错误异常
    pytest.raises(ValueError, lambda: self.system.validate_system())
    # 更新 kdes 列表，增加一个新的期望速度值
    self.system.kdes = self.kdes + [u[2] + u[1] - qd[2]]
    # 再次预期验证系统会引发值错误异常
    pytest.raises(ValueError, lambda: self.system.validate_system())

    # 测试重复项验证，这个功能基本上不应该失败
    self.system.validate_system(check_duplicates=True)

    # 在拉格朗日方法下测试速度的设置
    self.system.u_ind = u[:len(self.u_ind)]
    # 预期验证系统会引发值错误异常
    with pytest.raises(ValueError):
        self.system.validate_system(LagrangesMethod)
    self.system.u_ind = []
    self.system.validate_system(LagrangesMethod)
    # 设置辅助速度，预期验证系统会引发值错误异常
    self.system.u_aux = ua
    with pytest.raises(ValueError):
        self.system.validate_system(LagrangesMethod)
    self.system.u_aux = []
    self.system.validate_system(LagrangesMethod)
    # 添加一个新的关节到系统中
    self.system.add_joints(
        PinJoint('Ju', RigidBody('rbu1'), RigidBody('rbu2')))
    self.system.u_ind = []
    # 预期验证系统会引发值错误异常
    with pytest.raises(ValueError):
        self.system.validate_system(LagrangesMethod)
    # 定义一个测试系统示例的类
    class TestSystemExamples:
        # 定义测试粒子在地面上滑动的情况，考虑摩擦力。假设施加的力为正且大于摩擦力。
        def test_box_on_ground(self):
            # 符号定义：重力加速度 g，质量 m，摩擦系数 mu
            g, m, mu = symbols('g m mu')
            # 动力学符号定义：广义坐标 q，广义速度 u，辅助速度 ua
            q, u, ua = dynamicsymbols('q u ua')
            # 动力学力定义：N（法向反作用力），F（切向力）
            N, F = dynamicsymbols('N F', positive=True)
            
            # 创建一个粒子 P，其质量为 m
            P = Particle("P", mass=m)
            # 创建一个动力学系统对象
            system = System()
            # 将粒子 P 添加到系统中
            system.add_bodies(P)
            # 设置粒子质心位置相对于固定点的位置，沿 x 轴方向
            P.masscenter.set_pos(system.fixed_point, q * system.x)
            # 设置粒子质心相对于系统框架的速度，分别沿 x 和 y 轴方向
            P.masscenter.set_vel(system.frame, u * system.x + ua * system.y)
            
            # 定义系统的广义坐标、广义速度和辅助速度
            system.q_ind, system.u_ind, system.u_aux = [q], [u], [ua]
            # 设置系统的运动方程描述，这里是 q 的时间导数等于 u
            system.kdes = [q.diff(t) - u]
            # 施加均匀重力加速度，沿负 y 轴方向
            system.apply_uniform_gravity(-g * system.y)
            # 添加系统的载荷：垂直于 P 的力 N，水平切向的力 F 以及摩擦力 mu * N
            system.add_loads(
                Force(P, N * system.y),
                Force(P, F * system.x - mu * N * system.x))
            # 验证系统的正确性
            system.validate_system()
            # 形成系统的运动方程
            system.form_eoms()
            
            # 测试其他输出
            # 定义不可变矩阵 Mk, gk, Md, gd 和 Mm, gm, aux_eqs 用于比较
            Mk = ImmutableMatrix([1])
            gk = ImmutableMatrix([u])
            Md = ImmutableMatrix([m])
            gd = ImmutableMatrix([F - mu * N])
            Mm = (Mk.row_join(zeros(1, 1))).col_join(zeros(1, 1).row_join(Md))
            gm = gk.col_join(gd)
            aux_eqs = ImmutableMatrix([N - m * g])
            
            # 断言验证质量矩阵、广义力、完整质量矩阵和完整广义力的计算结果是否为零矩阵
            assert simplify(system.mass_matrix - Md) == zeros(1, 1)
            assert simplify(system.forcing - gd) == zeros(1, 1)
            assert simplify(system.mass_matrix_full - Mm) == zeros(2, 2)
            assert simplify(system.forcing_full - gm) == zeros(2, 1)
            assert simplify(system.eom_method.auxiliary_eqs - aux_eqs) == zeros(1, 1)
```