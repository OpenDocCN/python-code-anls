# `D:\src\scipysrc\scipy\scipy\integrate\_ivp\tests\test_ivp.py`

```
# 导入 itertools 库中的 product 函数，用于生成迭代器的笛卡尔积
# 导入 numpy.testing 中的多个断言函数，用于进行数组测试和断言
# 导入 pytest 库及其 raises 函数，用于进行测试时的异常断言
# 导入 numpy 库，用于数值计算
# 导入 scipy.optimize._numdiff 库中的 group_columns 函数，用于对列进行分组
# 导入 scipy.integrate 库中的 solve_ivp 和不同的积分器（如 RK23、RK45 等）
# 导入 scipy.integrate 中的 OdeSolution 类，用于表示积分的解
# 导入 scipy.integrate._ivp.common 中的 num_jac 和 select_initial_step 函数
# 导入 scipy.integrate._ivp.base 中的 ConstantDenseOutput 类，用于常数稠密输出
# 导入 scipy.sparse 库中的 coo_matrix 和 csc_matrix 类，用于稀疏矩阵的表示

def fun_zero(t, y):
    # 返回与 y 形状相同的零数组作为函数值
    return np.zeros_like(y)


def fun_linear(t, y):
    # 返回与 y 形状相同的数组作为函数值
    return np.array([-y[0] - 5 * y[1], y[0] + y[1]])


def jac_linear():
    # 返回一个二维数组作为雅可比矩阵
    return np.array([[-1, -5], [1, 1]])


def sol_linear(t):
    # 返回一个二维数组作为解的函数值
    return np.vstack((-5 * np.sin(2 * t),
                      2 * np.cos(2 * t) + np.sin(2 * t)))


def fun_rational(t, y):
    # 返回一个数组作为有理函数的函数值
    return np.array([y[1] / t,
                     y[1] * (y[0] + 2 * y[1] - 1) / (t * (y[0] - 1))])


def fun_rational_vectorized(t, y):
    # 返回一个二维数组作为向量化有理函数的函数值
    return np.vstack((y[1] / t,
                      y[1] * (y[0] + 2 * y[1] - 1) / (t * (y[0] - 1))))


def jac_rational(t, y):
    # 返回一个二维数组作为有理函数的雅可比矩阵
    return np.array([
        [0, 1 / t],
        [-2 * y[1] ** 2 / (t * (y[0] - 1) ** 2),
         (y[0] + 4 * y[1] - 1) / (t * (y[0] - 1))]
    ])


def jac_rational_sparse(t, y):
    # 返回一个稀疏矩阵作为有理函数的稀疏雅可比矩阵
    return csc_matrix([
        [0, 1 / t],
        [-2 * y[1] ** 2 / (t * (y[0] - 1) ** 2),
         (y[0] + 4 * y[1] - 1) / (t * (y[0] - 1))]
    ])


def sol_rational(t):
    # 返回一个数组作为有理函数的解的函数值
    return np.asarray((t / (t + 10), 10 * t / (t + 10) ** 2))


def fun_medazko(t, y):
    # 根据时间 t 和状态向量 y 计算 Medazko 模型的右手边函数
    n = y.shape[0] // 2
    k = 100
    c = 4

    # 根据条件确定 phi 的值
    phi = 2 if t <= 5 else 0

    # 将 phi、0、y 和 y[-2] 水平堆叠成一个新的状态向量
    y = np.hstack((phi, 0, y, y[-2]))

    # 计算 alpha 和 beta 向量
    d = 1 / n
    j = np.arange(n) + 1
    alpha = 2 * (j * d - 1) ** 3 / c ** 2
    beta = (j * d - 1) ** 4 / c ** 2

    # 计算右手边函数 f
    j_2_p1 = 2 * j + 2
    j_2_m3 = 2 * j - 2
    j_2_m1 = 2 * j
    j_2 = 2 * j + 1

    f = np.empty(2 * n)
    f[::2] = (alpha * (y[j_2_p1] - y[j_2_m3]) / (2 * d) +
              beta * (y[j_2_m3] - 2 * y[j_2_m1] + y[j_2_p1]) / d ** 2 -
              k * y[j_2_m1] * y[j_2])
    f[1::2] = -k * y[j_2] * y[j_2_m1]

    return f


def medazko_sparsity(n):
    # 构建 Medazko 模型的稀疏雅可比矩阵
    cols = []
    rows = []

    i = np.arange(n) * 2

    cols.append(i[1:])
    rows.append(i[1:] - 2)

    cols.append(i)
    rows.append(i)

    cols.append(i)
    rows.append(i + 1)

    cols.append(i[:-1])
    rows.append(i[:-1] + 2)

    i = np.arange(n) * 2 + 1

    cols.append(i)
    rows.append(i)

    cols.append(i)
    rows.append(i - 1)

    cols = np.hstack(cols)
    rows = np.hstack(rows)

    return coo_matrix((np.ones_like(cols), (cols, rows)))


def fun_complex(t, y):
    # 返回复数 y 的负数作为函数值
    return -y


def jac_complex(t, y):
    # 返回单位矩阵乘以 -1 作为复杂函数的雅可比矩阵
    return -np.eye(y.shape[0])


def jac_complex_sparse(t, y):
    # 返回稀疏版本的复杂函数的雅可比矩阵
    return csc_matrix(jac_complex(t, y))


def sol_complex(t):
    # 返回一个复数解的函数值
    y = (0.5 + 1j) * np.exp(-t)
    return y.reshape((1, -1))


def fun_event_dense_output_LSODA(t, y):
    # 这个函数定义为空，未提供具体实现
    pass
    return y * (t - 2)



# 返回计算结果，结果是 y 乘以 (t - 2) 的值
# 定义一个事件函数，返回 t - 2
def jac_event_dense_output_LSODA(t, y):
    return t - 2


# 定义一个解析解函数，返回 exp(t^2 / 2 - 2 * t + log(0.05) - 6)
def sol_event_dense_output_LSODA(t):
    return np.exp(t ** 2 / 2 - 2 * t + np.log(0.05) - 6)


# 计算数值解与真实解之间的误差
def compute_error(y, y_true, rtol, atol):
    e = (y - y_true) / (atol + rtol * np.abs(y_true))
    return np.linalg.norm(e, axis=0) / np.sqrt(e.shape[0])


# 测试数值积分功能的函数
def test_integration():
    rtol = 1e-3  # 相对误差容限
    atol = 1e-6  # 绝对误差容限
    y0 = [1/3, 2/9]  # 初值条件

    # 对不同参数组合进行迭代
    for vectorized, method, t_span, jac in product(
            [False, True],  # 向量化与否
            ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA'],  # 积分方法
            [[5, 9], [5, 1]],  # 积分时间跨度
            [None, jac_rational, jac_rational_sparse]):  # 雅可比矩阵函数

        if vectorized:
            fun = fun_rational_vectorized  # 如果向量化，则使用向量化的函数
        else:
            fun = fun_rational  # 否则使用普通函数

        # 忽略警告信息
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       "The following arguments have no effect for a chosen "
                       "solver: `jac`")
            # 使用 solve_ivp 函数求解初值问题
            res = solve_ivp(fun, t_span, y0, rtol=rtol,
                            atol=atol, method=method, dense_output=True,
                            jac=jac, vectorized=vectorized)

        # 断言初始时间点正确
        assert_equal(res.t[0], t_span[0])
        # 断言没有事件发生
        assert_(res.t_events is None)
        assert_(res.y_events is None)
        # 断言求解成功
        assert_(res.success)
        # 断言状态为 0
        assert_equal(res.status, 0)

        # 针对不同方法，断言函数评估次数的限制
        if method == 'DOP853':
            # DOP853方法因为步长不够大，会多消耗函数评估次数
            assert_(res.nfev < 50)
        else:
            assert_(res.nfev < 40)

        # 针对不同方法，断言雅可比矩阵评估次数和线性求解次数的限制
        if method in ['RK23', 'RK45', 'DOP853', 'LSODA']:
            assert_equal(res.njev, 0)
            assert_equal(res.nlu, 0)
        else:
            assert_(0 < res.njev < 3)
            assert_(0 < res.nlu < 10)

        # 计算真实解
        y_true = sol_rational(res.t)
        # 计算数值解与真实解之间的误差
        e = compute_error(res.y, y_true, rtol, atol)
        assert_(np.all(e < 5))

        # 对解的插值进行测试
        tc = np.linspace(*t_span)
        yc_true = sol_rational(tc)
        yc = res.sol(tc)

        # 计算插值误差
        e = compute_error(yc, yc_true, rtol, atol)
        assert_(np.all(e < 5))

        # 在时间跨度中点处测试解的插值
        tc = (t_span[0] + t_span[-1]) / 2
        yc_true = sol_rational(tc)
        yc = res.sol(tc)

        # 计算中点处插值误差
        e = compute_error(yc, yc_true, rtol, atol)
        assert_(np.all(e < 5))

        # 断言数值解与解的插值结果在非常小的容限下接近
        assert_allclose(res.sol(res.t), res.y, rtol=1e-15, atol=1e-15)


# 复杂情形下的数值积分测试函数
def test_integration_complex():
    rtol = 1e-3  # 相对误差容限
    atol = 1e-6  # 绝对误差容限
    y0 = [0.5 + 1j]  # 复数初始条件
    t_span = [0, 1]  # 积分时间跨度
    tc = np.linspace(t_span[0], t_span[1])  # 在积分时间跨度内均匀取点
    # 使用 product 函数生成不同的 method 和 jac 组合，例如 ['RK23', 'RK45', 'DOP853', 'BDF'] 和 [None, jac_complex, jac_complex_sparse]
    for method, jac in product(['RK23', 'RK45', 'DOP853', 'BDF'],
                               [None, jac_complex, jac_complex_sparse]):
        # 使用 suppress_warnings 上下文管理器来忽略特定警告信息
        with suppress_warnings() as sup:
            # 设置过滤器以忽略特定的用户警告信息
            sup.filter(UserWarning,
                       "The following arguments have no effect for a chosen "
                       "solver: `jac`")
            # 调用 solve_ivp 函数求解微分方程组
            res = solve_ivp(fun_complex, t_span, y0, method=method,
                            dense_output=True, rtol=rtol, atol=atol, jac=jac)

        # 断言解的起始时间与给定时间段的起始时间相等
        assert_equal(res.t[0], t_span[0])
        # 断言没有事件发生
        assert_(res.t_events is None)
        assert_(res.y_events is None)
        # 断言求解成功
        assert_(res.success)
        # 断言求解状态为 0（成功）
        assert_equal(res.status, 0)

        # 如果使用 DOP853 方法，则断言函数评估次数小于 35
        if method == 'DOP853':
            assert res.nfev < 35
        else:
            # 否则断言函数评估次数小于 25
            assert res.nfev < 25

        # 如果使用 BDF 方法，则断言雅可比矩阵评估次数为 1，线性代数函数调用次数小于 6
        if method == 'BDF':
            assert_equal(res.njev, 1)
            assert res.nlu < 6
        else:
            # 否则断言雅可比矩阵评估次数为 0，线性代数函数调用次数为 0
            assert res.njev == 0
            assert res.nlu == 0

        # 计算真实解 y_true
        y_true = sol_complex(res.t)
        # 计算数值解与真实解之间的误差
        e = compute_error(res.y, y_true, rtol, atol)
        # 断言所有误差均小于 5
        assert np.all(e < 5)

        # 计算在时间点 tc 处的真实解 yc_true
        yc_true = sol_complex(tc)
        # 获得在时间点 tc 处的数值解 yc
        yc = res.sol(tc)
        # 计算数值解与真实解之间的误差
        e = compute_error(yc, yc_true, rtol, atol)
        # 断言所有误差均小于 5
        assert np.all(e < 5)
# 将该测试标记为一个失败慢速测试，其中失败会较慢返回结果。
@pytest.mark.fail_slow(5)
def test_integration_sparse_difference():
    # 设置初始变量
    n = 200
    # 时间跨度
    t_span = [0, 20]
    # 初始状态，创建一个长度为 2*n 的零数组，其中每隔一个值设置为 1
    y0 = np.zeros(2 * n)
    y0[1::2] = 1
    # 计算稀疏矩阵结构
    sparsity = medazko_sparsity(n)

    # 对于两种不同的数值积分方法进行循环测试
    for method in ['BDF', 'Radau']:
        # 使用数值积分求解微分方程组
        res = solve_ivp(fun_medazko, t_span, y0, method=method,
                        jac_sparsity=sparsity)

        # 断言开始时间点正确
        assert_equal(res.t[0], t_span[0])
        # 断言无事件发生
        assert_(res.t_events is None)
        assert_(res.y_events is None)
        # 断言求解成功
        assert_(res.success)
        # 断言状态为 0，即成功
        assert_equal(res.status, 0)

        # 对数值结果的特定值进行精确比较
        assert_allclose(res.y[78, -1], 0.233994e-3, rtol=1e-2)
        assert_allclose(res.y[79, -1], 0, atol=1e-3)
        assert_allclose(res.y[148, -1], 0.359561e-3, rtol=1e-2)
        assert_allclose(res.y[149, -1], 0, atol=1e-3)
        assert_allclose(res.y[198, -1], 0.117374129e-3, rtol=1e-2)
        assert_allclose(res.y[199, -1], 0.6190807e-5, atol=1e-3)
        assert_allclose(res.y[238, -1], 0, atol=1e-3)
        assert_allclose(res.y[239, -1], 0.9999997, rtol=1e-2)


# 对于积分过程中的常数雅可比矩阵进行测试
def test_integration_const_jac():
    # 设置相对误差和绝对误差
    rtol = 1e-3
    atol = 1e-6
    # 初始状态
    y0 = [0, 2]
    # 时间跨度
    t_span = [0, 2]
    # 计算线性系统雅可比矩阵
    J = jac_linear()
    # 转换成稀疏矩阵格式
    J_sparse = csc_matrix(J)

    # 对两种不同的积分方法和雅可比矩阵进行组合测试
    for method, jac in product(['Radau', 'BDF'], [J, J_sparse]):
        # 使用数值积分求解线性系统微分方程组
        res = solve_ivp(fun_linear, t_span, y0, rtol=rtol, atol=atol,
                        method=method, dense_output=True, jac=jac)
        # 断言开始时间点正确
        assert_equal(res.t[0], t_span[0])
        # 断言无事件发生
        assert_(res.t_events is None)
        assert_(res.y_events is None)
        # 断言求解成功
        assert_(res.success)
        # 断言状态为 0，即成功
        assert_equal(res.status, 0)

        # 断言函数评估次数少于 100 次
        assert_(res.nfev < 100)
        # 断言雅可比矩阵评估次数为 0
        assert_equal(res.njev, 0)
        # 断言 LU 分解次数在 0 和 15 之间
        assert_(0 < res.nlu < 15)

        # 计算数值解与真实解的误差
        y_true = sol_linear(res.t)
        e = compute_error(res.y, y_true, rtol, atol)
        assert_(np.all(e < 10))

        # 在整个时间跨度上计算数值解与真实解的误差
        tc = np.linspace(*t_span)
        yc_true = sol_linear(tc)
        yc = res.sol(tc)

        e = compute_error(yc, yc_true, rtol, atol)
        assert_(np.all(e < 15))

        # 断言数值解和插值解在非常小的误差范围内相等
        assert_allclose(res.sol(res.t), res.y, rtol=1e-14, atol=1e-14)


# 对于包含不同积分方法的刚性系统进行测试
@pytest.mark.slow
@pytest.mark.parametrize('method', ['Radau', 'BDF', 'LSODA'])
def test_integration_stiff(method):
    # 设置相对误差和绝对误差
    rtol = 1e-6
    atol = 1e-6
    # 初始状态
    y0 = [1e4, 0, 0]
    # 时间跨度
    tspan = [0, 1e8]

    # 定义 Roberton 刚性系统的微分方程
    def fun_robertson(t, state):
        x, y, z = state
        return [
            -0.04 * x + 1e4 * y * z,
            0.04 * x - 1e4 * y * z - 3e7 * y * y,
            3e7 * y * y,
        ]

    # 使用数值积分求解 Roberton 刚性系统的微分方程
    res = solve_ivp(fun_robertson, tspan, y0, rtol=rtol,
                    atol=atol, method=method)

    # 如果积分器未正确激活刚性模式，这些数字会大得多
    assert res.nfev < 5000
    assert res.njev < 200


# 测试事件处理机制
def test_events():
    # 定义第一个有理事件函数
    def event_rational_1(t, y):
        return y[0] - y[1] ** 0.7

    # 定义第二个有理事件函数
    def event_rational_2(t, y):
        return y[1] ** 0.6 - y[0]

    # 定义第三个有理事件函数，并标记为终止事件
    def event_rational_3(t, y):
        return t - 7.4

    event_rational_3.terminal = True

    # 测试向后方向的事件处理
    event_rational_1.direction = 0
    event_rational_2.direction = 0
# 定义一个求解谐振子运动方程的函数
def _get_harmonic_oscillator():
    # 定义谐振子运动方程 dy/dt = [y[1], -y[0]]
    def f(t, y):
        return [y[1], -y[0]]

    # 定义事件函数，当 y[0] == 0 时触发事件
    def event(t, y):
        return y[0]

    # 返回谐振子运动方程和事件函数
    return f, event


# 使用参数化测试进行测试，测试事件终止条件为整数的情况
@pytest.mark.parametrize('n_events', [3, 4])
def test_event_terminal_integer(n_events):
    # 获取谐振子运动方程和事件函数
    f, event = _get_harmonic_oscillator()
    # 设置事件对象的终止条件为 n_events
    event.terminal = n_events
    # 解决谐振子运动方程，记录事件发生的时间和状态
    res = solve_ivp(f, (0, 100), [1, 0], events=event)
    # 断言事件发生的次数与 n_events 相等
    assert len(res.t_events[0]) == n_events
    # 断言事件状态的数量与 n_events 相等
    assert len(res.y_events[0]) == n_events
    # 断言事件发生时 y[0] 的值接近 0
    assert_allclose(res.y_events[0][:, 0], 0, atol=1e-14)


# 测试事件对象终止条件为 None 的情况
def test_event_terminal_iv():
    # 获取谐振子运动方程和事件函数
    f, event = _get_harmonic_oscillator()
    args = (f, (0, 100), [1, 0])

    # 设置事件对象的终止条件为 None
    event.terminal = None
    # 解决谐振子运动方程，不记录任何事件发生的时间和状态
    res = solve_ivp(*args, events=event)
    # 重新设置事件对象的终止条件为 0
    event.terminal = 0
    # 解决谐振子运动方程，记录事件发生的时间和状态
    ref = solve_ivp(*args, events=event)
    # 断言两次求解的事件发生时间接近
    assert_allclose(res.t_events, ref.t_events)

    # 测试设置非法的终止条件，期望抛出 ValueError 异常
    message = "The `terminal` attribute..."
    event.terminal = -1
    with pytest.raises(ValueError, match=message):
        solve_ivp(*args, events=event)
    event.terminal = 3.5
    with pytest.raises(ValueError, match=message):
        solve_ivp(*args, events=event)


# 测试最大步长限制的情况
def test_max_step():
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    # 遍历不同的求解器和时间跨度
    for method in [RK23, RK45, DOP853, Radau, BDF, LSODA]:
        for t_span in ([5, 9], [5, 1]):
            # 解决有理函数方程，设定相对误差、绝对误差、最大步长等参数
            res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                            max_step=0.5, atol=atol, method=method,
                            dense_output=True)
            # 断言结果的起始时间与结束时间符合预期
            assert_equal(res.t[0], t_span[0])
            assert_equal(res.t[-1], t_span[-1])
            # 断言时间步长在指定的最大步长内
            assert_(np.all(np.abs(np.diff(res.t)) <= 0.5 + 1e-15))
            # 断言不记录任何事件发生
            assert_(res.t_events is None)
            # 断言求解成功
            assert_(res.success)
            # 断言求解状态为 0
            assert_equal(res.status, 0)

            # 计算解的真实值
            y_true = sol_rational(res.t)
            # 计算解的误差
            e = compute_error(res.y, y_true, rtol, atol)
            # 断言解的误差在一定范围内
            assert_(np.all(e < 5))

            # 对时间跨度内的连续点进行求解
            tc = np.linspace(*t_span)
            yc_true = sol_rational(tc)
            yc = res.sol(tc)

            # 计算解的误差
            e = compute_error(yc, yc_true, rtol, atol)
            # 断言解的误差在一定范围内
            assert_(np.all(e < 5))

            # 断言解函数在时间节点上与解值非常接近
            assert_allclose(res.sol(res.t), res.y, rtol=1e-15, atol=1e-15)

            # 测试设置非法的最大步长值，期望抛出 ValueError 异常
            assert_raises(ValueError, method, fun_rational, t_span[0], y0,
                          t_span[1], max_step=-1)

            # 对非 LSODA 求解器进行特定的错误测试
            if method is not LSODA:
                solver = method(fun_rational, t_span[0], y0, t_span[1],
                                rtol=rtol, atol=atol, max_step=1e-20)
                message = solver.step()
                message = solver.step()  # 第一步成功但第二步失败
                # 断言求解器状态为失败
                assert_equal(solver.status, 'failed')
                # 断言消息包含指定的错误信息
                assert_("step size is less" in message)
                # 断言抛出 RuntimeError 异常
                assert_raises(RuntimeError, solver.step)


# 测试首步长的情况
def test_first_step():
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    first_step = 0.1
    # 对于每个不同的数值积分方法，依次执行以下操作
    for method in [RK23, RK45, DOP853, Radau, BDF, LSODA]:
        # 对于每个给定的时间跨度范围，依次执行以下操作
        for t_span in ([5, 9], [5, 1]):
            # 使用数值积分方法解决常微分方程组，返回结果
            res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                            max_step=0.5, atol=atol, method=method,
                            dense_output=True, first_step=first_step)
    
            # 断言：确保数值积分结果的起始时间与指定的时间跨度起始值相等
            assert_equal(res.t[0], t_span[0])
            # 断言：确保数值积分结果的结束时间与指定的时间跨度结束值相等
            assert_equal(res.t[-1], t_span[-1])
            # 断言：确保第一步的大小与指定的起始时间的绝对差值接近5
            assert_allclose(first_step, np.abs(res.t[1] - 5))
            # 断言：确保没有触发任何事件
            assert_(res.t_events is None)
            # 断言：确保数值积分成功完成
            assert_(res.success)
            # 断言：确保数值积分的状态为0（成功）
            assert_equal(res.status, 0)
    
            # 计算真实解的值
            y_true = sol_rational(res.t)
            # 计算数值解与真实解的误差
            e = compute_error(res.y, y_true, rtol, atol)
            # 断言：确保误差在5以内
            assert_(np.all(e < 5))
    
            # 在整个时间跨度内生成均匀分布的时间点
            tc = np.linspace(*t_span)
            # 计算真实解在这些时间点上的值
            yc_true = sol_rational(tc)
            # 计算数值解在这些时间点上的值
            yc = res.sol(tc)
    
            # 计算数值解与真实解的误差
            e = compute_error(yc, yc_true, rtol, atol)
            # 断言：确保误差在5以内
            assert_(np.all(e < 5))
    
            # 断言：确保数值解与插值函数在离散时间点上的值非常接近
            assert_allclose(res.sol(res.t), res.y, rtol=1e-15, atol=1e-15)
    
            # 断言：确保传入非法参数时会引发 ValueError 异常
            assert_raises(ValueError, method, fun_rational, t_span[0], y0,
                          t_span[1], first_step=-1)
            assert_raises(ValueError, method, fun_rational, t_span[0], y0,
                          t_span[1], first_step=5)
def test_t_eval():
    rtol = 1e-3  # 相对误差容限
    atol = 1e-6  # 绝对误差容限
    y0 = [1/3, 2/9]  # 初值条件
    for t_span in ([5, 9], [5, 1]):  # 循环遍历时间跨度组合
        t_eval = np.linspace(t_span[0], t_span[1], 10)  # 生成均匀时间点
        res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                        t_eval=t_eval)  # 解微分方程并记录结果
        assert_equal(res.t, t_eval)  # 断言解的时间点与预期的时间点一致
        assert_(res.t_events is None)  # 断言没有事件发生
        assert_(res.success)  # 断言求解成功
        assert_equal(res.status, 0)  # 断言状态为0（成功）

        y_true = sol_rational(res.t)  # 计算真实解
        e = compute_error(res.y, y_true, rtol, atol)  # 计算误差
        assert_(np.all(e < 5))  # 断言所有误差均小于5

    t_eval = [5, 5.01, 7, 8, 8.01, 9]  # 非均匀时间点
    res = solve_ivp(fun_rational, [5, 9], y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)  # 解微分方程并记录结果
    assert_equal(res.t, t_eval)  # 断言解的时间点与预期的时间点一致
    assert_(res.t_events is None)  # 断言没有事件发生
    assert_(res.success)  # 断言求解成功
    assert_equal(res.status, 0)  # 断言状态为0（成功）

    y_true = sol_rational(res.t)  # 计算真实解
    e = compute_error(res.y, y_true, rtol, atol)  # 计算误差
    assert_(np.all(e < 5))  # 断言所有误差均小于5

    t_eval = [5, 4.99, 3, 1.5, 1.1, 1.01, 1]  # 非均匀时间点（倒序）
    res = solve_ivp(fun_rational, [5, 1], y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)  # 解微分方程并记录结果
    assert_equal(res.t, t_eval)  # 断言解的时间点与预期的时间点一致
    assert_(res.t_events is None)  # 断言没有事件发生
    assert_(res.success)  # 断言求解成功
    assert_equal(res.status, 0)  # 断言状态为0（成功）

    t_eval = [5.01, 7, 8, 8.01]  # 非均匀时间点（部分时间点）
    res = solve_ivp(fun_rational, [5, 9], y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)  # 解微分方程并记录结果
    assert_equal(res.t, t_eval)  # 断言解的时间点与预期的时间点一致
    assert_(res.t_events is None)  # 断言没有事件发生
    assert_(res.success)  # 断言求解成功
    assert_equal(res.status, 0)  # 断言状态为0（成功）

    y_true = sol_rational(res.t)  # 计算真实解
    e = compute_error(res.y, y_true, rtol, atol)  # 计算误差
    assert_(np.all(e < 5))  # 断言所有误差均小于5

    t_eval = [4.99, 3, 1.5, 1.1, 1.01]  # 非均匀时间点（倒序）
    res = solve_ivp(fun_rational, [5, 1], y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)  # 解微分方程并记录结果
    assert_equal(res.t, t_eval)  # 断言解的时间点与预期的时间点一致
    assert_(res.t_events is None)  # 断言没有事件发生
    assert_(res.success)  # 断言求解成功
    assert_equal(res.status, 0)  # 断言状态为0（成功）

    t_eval = [4, 6]  # 非法时间点（跨度大于t_span）
    assert_raises(ValueError, solve_ivp, fun_rational, [5, 9], y0,
                  rtol=rtol, atol=atol, t_eval=t_eval)


def test_t_eval_dense_output():
    rtol = 1e-3  # 相对误差容限
    atol = 1e-6  # 绝对误差容限
    y0 = [1/3, 2/9]  # 初值条件
    t_span = [5, 9]  # 时间跨度
    t_eval = np.linspace(t_span[0], t_span[1], 10)  # 生成均匀时间点
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)  # 解微分方程并记录结果
    res_d = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                      t_eval=t_eval, dense_output=True)  # 启用密集输出
    assert_equal(res.t, t_eval)  # 断言解的时间点与预期的时间点一致
    assert_(res.t_events is None)  # 断言没有事件发生
    assert_(res.success)  # 断言求解成功
    assert_equal(res.status, 0)  # 断言状态为0（成功）

    assert_equal(res.t, res_d.t)  # 断言解和密集输出的时间点一致
    assert_equal(res.y, res_d.y)  # 断言解和密集输出的解一致
    assert_(res_d.t_events is None)  # 断言密集输出没有事件发生
    assert_(res_d.success)  # 断言密集输出求解成功
    assert_equal(res_d.status, 0)  # 断言密集输出状态为0（成功）

    # 若解和密集输出的时间点和解一致，则仅测试一个情况下的值
    y_true = sol_rational(res.t)  # 计算真实解
    e = compute_error(res.y, y_true, rtol, atol)  # 计算误差
    assert_(np.all(e < 5))  # 断言所有误差均小于5


def test_t_eval_early_event():
    def early_event(t, y):
        return t - 7

    early_event.terminal = True  # 设置事件为终止事件

    rtol = 1e-3  # 相对误差容限
    atol = 1e-6  # 绝对误差容限
    y0 = [1/3, 2/9]  # 初值条件
    t_span = [5, 9]  # 时间跨度
    # 创建一个包含等间隔数字的 NumPy 数组，用于定义求解时间点
    t_eval = np.linspace(7.5, 9, 16)
    
    # 遍历不同的求解方法
    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        # 忽略特定的用户警告
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       "The following arguments have no effect for a chosen "
                       "solver: `jac`")
            # 使用 solve_ivp 函数求解常微分方程组
            res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                            method=method, t_eval=t_eval, events=early_event,
                            jac=jac_rational)
    
        # 断言求解成功
        assert res.success
        # 断言终止事件的消息
        assert res.message == 'A termination event occurred.'
        # 断言求解器的状态
        assert res.status == 1
        # 断言结果中没有时间点和解
        assert not res.t and not res.y
        # 断言终止事件数组的长度为 1
        assert len(res.t_events) == 1
        # 断言终止事件时间点数组的大小为 1
        assert res.t_events[0].size == 1
        # 断言终止事件的时间点为 7
        assert res.t_events[0][0] == 7
# 定义测试函数 test_event_dense_output_LSODA
def test_event_dense_output_LSODA():
    # 定义事件函数 event_lsoda，判断状态变量 y[0] 是否小于 2.02e-5
    def event_lsoda(t, y):
        return y[0] - 2.02e-5

    # 相对误差容限
    rtol = 1e-3
    # 绝对误差容限
    atol = 1e-6
    # 初始状态
    y0 = [0.05]
    # 时间区间
    t_span = [-2, 2]
    # 初始步长
    first_step = 1e-3
    # 解微分方程并获取结果
    res = solve_ivp(
        # 解微分方程的函数 fun_event_dense_output_LSODA
        fun_event_dense_output_LSODA,
        # 时间区间
        t_span,
        # 初始状态
        y0,
        # 求解方法 LSODA
        method="LSODA",
        # 输出稠密解
        dense_output=True,
        # 事件函数
        events=event_lsoda,
        # 初始步长
        first_step=first_step,
        # 最大步长
        max_step=1,
        # 相对误差容限
        rtol=rtol,
        # 绝对误差容限
        atol=atol,
        # 雅可比矩阵函数
        jac=jac_event_dense_output_LSODA,
    )

    # 断言首个时间点的结果是否与时间区间起始相等
    assert_equal(res.t[0], t_span[0])
    # 断言最后一个时间点的结果是否与时间区间结束相等
    assert_equal(res.t[-1], t_span[-1])
    # 断言首个时间点的步长是否与初始步长相近
    assert_allclose(first_step, np.abs(res.t[1] - t_span[0]))
    # 断言求解是否成功
    assert res.success
    # 断言状态是否为成功状态
    assert_equal(res.status, 0)

    # 获取事件的真实解
    y_true = sol_event_dense_output_LSODA(res.t)
    # 计算误差
    e = compute_error(res.y, y_true, rtol, atol)
    # 断言误差是否小于 5
    assert_array_less(e, 5)

    # 在时间区间内均匀采样
    tc = np.linspace(*t_span)
    # 获取时间区间内事件的真实解
    yc_true = sol_event_dense_output_LSODA(tc)
    # 获取数值解
    yc = res.sol(tc)
    # 计算误差
    e = compute_error(yc, yc_true, rtol, atol)
    # 断言误差是否小于 5
    assert_array_less(e, 5)

    # 断言数值解是否与稠密输出解相近
    assert_allclose(res.sol(res.t), res.y, rtol=1e-15, atol=1e-15)


# 定义测试函数 test_no_integration，测试不集成情况
def test_no_integration():
    # 遍历多种求解方法
    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        # 解微分方程，但不进行集成
        sol = solve_ivp(lambda t, y: -y, [4, 4], [2, 3],
                        method=method, dense_output=True)
        # 断言解在给定时间点的值是否与初始值相等
        assert_equal(sol.sol(4), [2, 3])
        # 断言解在给定时间序列的值是否与初始值序列相等
        assert_equal(sol.sol([4, 5, 6]), [[2, 2, 2], [3, 3, 3]])


# 定义测试函数 test_no_integration_class，测试不集成情况的类方法
def test_no_integration_class():
    # 遍历多种求解方法类
    for method in [RK23, RK45, DOP853, Radau, BDF, LSODA]:
        # 使用方法类初始化求解器，但不进行集成
        solver = method(lambda t, y: -y, 0.0, [10.0, 0.0], 0.0)
        # 执行一步集成
        solver.step()
        # 断言求解器状态是否为完成状态
        assert_equal(solver.status, 'finished')
        # 获取稠密输出解
        sol = solver.dense_output()
        # 断言稠密输出解在给定时间点的值是否与初始值相等
        assert_equal(sol(0.0), [10.0, 0.0])
        # 断言稠密输出解在给定时间序列的值是否与初始值序列相等
        assert_equal(sol([0, 1, 2]), [[10, 10, 10], [0, 0, 0]])

        # 使用方法类初始化求解器，不传入初始状态，执行一步集成
        solver = method(lambda t, y: -y, 0.0, [], np.inf)
        solver.step()
        # 断言求解器状态是否为完成状态
        assert_equal(solver.status, 'finished')
        # 获取稠密输出解
        sol = solver.dense_output()
        # 断言稠密输出解在给定时间点的值是否为空
        assert_equal(sol(100.0), [])
        # 断言稠密输出解在给定时间序列的值是否为空数组
        assert_equal(sol([0, 1, 2]), np.empty((0, 3)))


# 定义测试函数 test_empty，测试空状态下的情况
def test_empty():
    # 定义空状态下的函数
    def fun(t, y):
        return np.zeros((0,))

    # 初始状态为零数组
    y0 = np.zeros((0,))

    # 遍历多种求解方法
    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        # 解微分方程，但不进行集成
        sol = assert_no_warnings(solve_ivp, fun, [0, 10], y0,
                                 method=method, dense_output=True)
        # 断言解在给定时间点的值是否为零数组
        assert_equal(sol.sol(10), np.zeros((0,)))
        # 断言解在给定时间序列的值是否为零数组序列
        assert_equal(sol.sol([1, 2, 3]), np.zeros((0, 3)))

    # 遍历多种求解方法
    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        # 解微分方程，但不进行集成
        sol = assert_no_warnings(solve_ivp, fun, [0, np.inf], y0,
                                 method=method, dense_output=True)
        # 断言解在给定时间点的值是否为零数组
        assert_equal(sol.sol(10), np.zeros((0,)))
        # 断言解在给定时间序列的值是否为零数组序列
        assert_equal(sol.sol([1, 2, 3]), np.zeros((0, 3)))


# 定义测试函数 test_ConstantDenseOutput，测试常数稠密输出
def test_ConstantDenseOutput():
    # 初始化常数稠密输出对象
    sol = ConstantDenseOutput(0, 1, np.array([1, 2]))
    # 断言在给定时间点的值是否近似于预期常数数组
    assert_allclose(sol(1.5), [1, 2])
    # 断言在给定时间序列的值是否近似于预期常数数组序列
    assert_allclose(sol([1, 1.5, 2]), [[1, 1, 1], [2, 2, 2]])

    # 初始化空数组的常数稠密输出对象
    sol = ConstantDenseOutput(0, 1, np.array([]))
    # 对于输入参数为 1.5 的情况，验证 sol 函数返回一个空的 NumPy 数组是否接近于 np.empty(0)
    assert_allclose(sol(1.5), np.empty(0))
    
    # 对于输入参数为 [1, 1.5, 2] 的情况，验证 sol 函数返回一个空的 NumPy 数组是否接近于 np.empty((0, 3))
    assert_allclose(sol([1, 1.5, 2]), np.empty((0, 3)))
# 定义一个测试函数，用于测试不同的数值解法类的行为
def test_classes():
    # 初始条件
    y0 = [1 / 3, 2 / 9]
    # 遍历不同的数值解法类
    for cls in [RK23, RK45, DOP853, Radau, BDF, LSODA]:
        # 使用当前类创建求解器对象
        solver = cls(fun_rational, 5, y0, np.inf)
        
        # 断言解法器的属性值
        assert_equal(solver.n, 2)  # 断言状态维度为2
        assert_equal(solver.status, 'running')  # 断言状态为运行中
        assert_equal(solver.t_bound, np.inf)  # 断言时间上界为无穷大
        assert_equal(solver.direction, 1)  # 断言时间步进方向为正向
        assert_equal(solver.t, 5)  # 断言初始时间为5
        assert_equal(solver.y, y0)  # 断言初始状态与给定的y0相同
        assert_(solver.step_size is None)  # 断言步长为None
        
        # 根据类别不同，进行不同的断言
        if cls is not LSODA:
            assert_(solver.nfev > 0)  # 断言函数调用次数大于0
            assert_(solver.njev >= 0)  # 断言雅可比矩阵调用次数大于等于0
            assert_equal(solver.nlu, 0)  # 断言线性求解次数为0
        else:
            assert_equal(solver.nfev, 0)  # 断言函数调用次数为0
            assert_equal(solver.njev, 0)  # 断言雅可比矩阵调用次数为0
            assert_equal(solver.nlu, 0)  # 断言线性求解次数为0

        # 断言调用 dense_output 方法时会触发 RuntimeError
        assert_raises(RuntimeError, solver.dense_output)

        # 进行一步求解，并断言状态和消息
        message = solver.step()
        assert_equal(solver.status, 'running')  # 断言状态为运行中
        assert_equal(message, None)  # 断言消息为空
        assert_equal(solver.n, 2)  # 断言状态维度为2
        assert_equal(solver.t_bound, np.inf)  # 断言时间上界为无穷大
        assert_equal(solver.direction, 1)  # 断言时间步进方向为正向
        assert_(solver.t > 5)  # 断言当前时间大于5
        assert_(not np.all(np.equal(solver.y, y0)))  # 断言状态不再与初始状态y0完全相等
        assert_(solver.step_size > 0)  # 断言步长大于0
        assert_(solver.nfev > 0)  # 断言函数调用次数大于0
        assert_(solver.njev >= 0)  # 断言雅可比矩阵调用次数大于等于0
        assert_(solver.nlu >= 0)  # 断言线性求解次数大于等于0
        
        # 获取稠密输出函数 sol，并断言其与预期解在指定误差范围内一致
        sol = solver.dense_output()
        assert_allclose(sol(5), y0, rtol=1e-15, atol=0)


# 定义测试函数 test_OdeSolution
def test_OdeSolution():
    # 创建时间点数组 ts
    ts = np.array([0, 2, 5], dtype=float)
    # 创建 ConstantDenseOutput 对象 s1 和 s2
    s1 = ConstantDenseOutput(ts[0], ts[1], np.array([-1]))
    s2 = ConstantDenseOutput(ts[1], ts[2], np.array([1]))

    # 创建 OdeSolution 对象 sol
    sol = OdeSolution(ts, [s1, s2])

    # 断言在不同时间点求解的结果
    assert_equal(sol(-1), [-1])
    assert_equal(sol(1), [-1])
    assert_equal(sol(2), [-1])
    assert_equal(sol(3), [1])
    assert_equal(sol(5), [1])
    assert_equal(sol(6), [1])

    # 断言在给定时间点数组上求解的结果
    assert_equal(sol([0, 6, -2, 1.5, 4.5, 2.5, 5, 5.5, 2]),
                 np.array([[-1, 1, -1, -1, 1, 1, 1, 1, -1]]))

    # 创建新的时间点数组 ts
    ts = np.array([10, 4, -3])
    # 创建 ConstantDenseOutput 对象 s1 和 s2
    s1 = ConstantDenseOutput(ts[0], ts[1], np.array([-1]))
    s2 = ConstantDenseOutput(ts[1], ts[2], np.array([1]))

    # 创建新的 OdeSolution 对象 sol
    sol = OdeSolution(ts, [s1, s2])

    # 断言在不同时间点求解的结果
    assert_equal(sol(11), [-1])
    assert_equal(sol(10), [-1])
    assert_equal(sol(5), [-1])
    assert_equal(sol(4), [-1])
    assert_equal(sol(0), [1])
    assert_equal(sol(-3), [1])
    assert_equal(sol(-4), [1])

    # 断言在给定时间点数组上求解的结果
    assert_equal(sol([12, -5, 10, -3, 6, 1, 4]),
                 np.array([[-1, 1, -1, 1, -1, 1, -1]]))

    # 创建新的时间点数组 ts
    ts = np.array([1, 1])
    # 创建 ConstantDenseOutput 对象 s
    s = ConstantDenseOutput(1, 1, np.array([10]))
    # 创建新的 OdeSolution 对象 sol
    sol = OdeSolution(ts, [s])

    # 断言在不同时间点求解的结果
    assert_equal(sol(0), [10])
    assert_equal(sol(1), [10])
    assert_equal(sol(2), [10])

    # 断言在给定时间点数组上求解的结果
    assert_equal(sol([2, 1, 0]), np.array([[10, 10, 10]]))


# 定义测试函数 test_num_jac
def test_num_jac():
    # 定义测试函数 fun(t, y)
    def fun(t, y):
        return np.vstack([
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2
        ])
    # 定义雅可比矩阵计算函数，输入参数为时间 t 和状态向量 y，返回雅可比矩阵的 numpy 数组
    def jac(t, y):
        return np.array([
            # 雅可比矩阵的第一行
            [-0.04, 1e4 * y[2], 1e4 * y[1]],
            # 雅可比矩阵的第二行
            [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
            # 雅可比矩阵的第三行
            [0, 6e7 * y[1], 0]
        ])

    # 设置时间 t 的初始值
    t = 1
    # 设置状态向量 y 的初始值，为 numpy 数组
    y = np.array([1, 0, 0])
    # 调用 jac 函数，计算真实的雅可比矩阵 J_true
    J_true = jac(t, y)
    # 设置数值计算雅可比矩阵的阈值
    threshold = 1e-5
    # 调用函数 fun，返回的结果展平成一维数组 f
    f = fun(t, y).ravel()

    # 调用 num_jac 函数，计算数值估计的雅可比矩阵 J_num 和缩放因子 factor
    J_num, factor = num_jac(fun, t, y, f, threshold, None)
    # 使用 assert_allclose 函数断言数值计算的雅可比矩阵 J_num 与真实的 J_true 在指定的相对误差和绝对误差范围内相等
    assert_allclose(J_num, J_true, rtol=1e-5, atol=1e-5)

    # 再次调用 num_jac 函数，使用之前计算得到的缩放因子 factor，计算数值估计的雅可比矩阵 J_num
    J_num, factor = num_jac(fun, t, y, f, threshold, factor)
    # 再次使用 assert_allclose 函数断言数值计算的雅可比矩阵 J_num 与真实的 J_true 在指定的相对误差和绝对误差范围内相等
    assert_allclose(J_num, J_true, rtol=1e-5, atol=1e-5)
def test_args():

    # sys3 is actually two decoupled systems. (x, y) form a
    # linear oscillator, while z is a nonlinear first order
    # system with equilibria at z=0 and z=1. If k > 0, z=1
    # is stable and z=0 is unstable.
    
    # 定义一个非线性方程组，其中 (x, y) 是线性振荡器，
    # 而 z 是一个非线性一阶系统，在 z=0 和 z=1 处有平衡点。
    # 如果 k > 0，则 z=1 是稳定的，z=0 是不稳定的。
    def sys3(t, w, omega, k, zfinal):
        x, y, z = w
        return [-omega*y, omega*x, k*z*(1 - z)]

    # 定义 sys3 的雅可比矩阵函数
    def sys3_jac(t, w, omega, k, zfinal):
        x, y, z = w
        J = np.array([[0, -omega, 0],
                      [omega, 0, 0],
                      [0, 0, k*(1 - 2*z)]])
        return J

    # 定义 sys3 中 x0 的减小方向的事件函数
    def sys3_x0decreasing(t, w, omega, k, zfinal):
        x, y, z = w
        return x

    # 定义 sys3 中 y0 的增大方向的事件函数
    def sys3_y0increasing(t, w, omega, k, zfinal):
        x, y, z = w
        return y

    # 定义 sys3 中 z 终值的事件函数
    def sys3_zfinal(t, w, omega, k, zfinal):
        x, y, z = w
        return z - zfinal

    # 设置 sys3_x0decreasing 和 sys3_y0increasing 的事件标志
    sys3_x0decreasing.direction = -1
    sys3_y0increasing.direction = 1
    # 设置 sys3_zfinal 的终止事件标志
    sys3_zfinal.terminal = True

    omega = 2
    k = 4

    tfinal = 5
    zfinal = 0.99

    # Find z0 such that when z(0) = z0, z(tfinal) = zfinal.
    # The condition z(tfinal) = zfinal is the terminal event.
    # 计算 z0，使得当 z(0) = z0 时，满足 z(tfinal) = zfinal。
    # z(tfinal) = zfinal 是终止事件条件。
    z0 = np.exp(-k*tfinal)/((1 - zfinal)/zfinal + np.exp(-k*tfinal))

    w0 = [0, -1, z0]

    # Provide the jac argument and use the Radau method to ensure that the use
    # of the Jacobian function is exercised.
    # 提供雅可比参数并使用 Radau 方法以确保使用雅可比函数。
    # 设置仿真结束时间为 tfinal 的两倍
    tend = 2*tfinal
    # 使用 solve_ivp 函数求解微分方程系统 sys3，从时间 0 到 tend，初始条件为 w0
    # 同时定义事件触发条件为 sys3_x0decreasing, sys3_y0increasing, sys3_zfinal，并启用稠密输出
    # 其他参数包括 omega, k, zfinal，选择 Radau 方法求解，提供 sys3_jac 作为雅可比矩阵
    # 设置相对误差和绝对误差容限
    sol = solve_ivp(sys3, [0, tend], w0,
                    events=[sys3_x0decreasing, sys3_y0increasing, sys3_zfinal],
                    dense_output=True, args=(omega, k, zfinal),
                    method='Radau', jac=sys3_jac,
                    rtol=1e-10, atol=1e-13)

    # 检查是否在预期时间触发了预期事件
    x0events_t = sol.t_events[0]
    y0events_t = sol.t_events[1]
    zfinalevents_t = sol.t_events[2]
    # 使用 assert_allclose 检查 x0 事件时间是否接近预期值 [0.5*np.pi, 1.5*np.pi]
    assert_allclose(x0events_t, [0.5*np.pi, 1.5*np.pi])
    # 使用 assert_allclose 检查 y0 事件时间是否接近预期值 [0.25*np.pi, 1.25*np.pi]
    assert_allclose(y0events_t, [0.25*np.pi, 1.25*np.pi])
    # 使用 assert_allclose 检查 zfinal 事件时间是否接近预期值 [tfinal]
    assert_allclose(zfinalevents_t, [tfinal])

    # 检查解是否与已知的精确解一致
    t = np.linspace(0, zfinalevents_t[0], 250)
    w = sol.sol(t)
    # 使用 assert_allclose 检查 w[0] 是否接近预期的 np.sin(omega*t)
    assert_allclose(w[0], np.sin(omega*t), rtol=1e-9, atol=1e-12)
    # 使用 assert_allclose 检查 w[1] 是否接近预期的 -np.cos(omega*t)
    assert_allclose(w[1], -np.cos(omega*t), rtol=1e-9, atol=1e-12)
    # 使用 assert_allclose 检查 w[2] 是否接近预期值 1/(((1 - z0)/z0)*np.exp(-k*t) + 1)
    assert_allclose(w[2], 1/(((1 - z0)/z0)*np.exp(-k*t) + 1),
                    rtol=1e-9, atol=1e-12)

    # 检查事件发生时状态变量是否具有预期的值
    x0events = sol.sol(x0events_t)
    y0events = sol.sol(y0events_t)
    zfinalevents = sol.sol(zfinalevents_t)
    # 使用 assert_allclose 检查 x0events[0] 是否接近预期的 np.zeros_like(x0events[0])
    assert_allclose(x0events[0], np.zeros_like(x0events[0]), atol=5e-14)
    # 使用 assert_allclose 检查 x0events[1] 是否接近预期的 np.ones_like(x0events[1])
    assert_allclose(x0events[1], np.ones_like(x0events[1]))
    # 使用 assert_allclose 检查 y0events[0] 是否接近预期的 np.ones_like(y0events[0])
    assert_allclose(y0events[0], np.ones_like(y0events[0]))
    # 使用 assert_allclose 检查 y0events[1] 是否接近预期的 np.zeros_like(y0events[1])
    assert_allclose(y0events[1], np.zeros_like(y0events[1]), atol=5e-14)
    # 使用 assert_allclose 检查 zfinalevents[2] 是否接近预期值 [zfinal]
    assert_allclose(zfinalevents[2], [zfinal])
# 测试解决 `solve_ivp` 在处理类数组 `rtol` 时的 bug，参见 gh-15482
def test_array_rtol():
    
    # 定义测试函数 f(t, y)，返回 y[0] 和 y[1]
    def f(t, y):
        return y[0], y[1]

    # 当 `rtol` 是类数组时，不会出现警告（或错误）
    sol = solve_ivp(f, (0, 1), [1., 1.], rtol=[1e-1, 1e-1])
    # 计算误差 err1
    err1 = np.abs(np.linalg.norm(sol.y[:, -1] - np.exp(1)))

    # 当 `rtol` 中某个元素过小时会出现警告
    with pytest.warns(UserWarning, match="At least one element..."):
        sol = solve_ivp(f, (0, 1), [1., 1.], rtol=[1e-1, 1e-16])
        # 计算误差 err2
        err2 = np.abs(np.linalg.norm(sol.y[:, -1] - np.exp(1)))

    # 更紧的 `rtol` 会改善误差
    assert err2 < err1


# 使用参数化测试，测试不同的积分方法 method 对零初值问题的求解
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA'])
def test_integration_zero_rhs(method):
    # 调用 solve_ivp 解决 fun_zero 初始值问题
    result = solve_ivp(fun_zero, [0, 10], np.ones(3), method=method)
    # 断言求解成功
    assert_(result.success)
    # 断言状态为 0
    assert_equal(result.status, 0)
    # 断言结果数组接近于 1.0，相对误差为 1e-15
    assert_allclose(result.y, 1.0, rtol=1e-15)


# 测试 solve_ivp 对单个参数的处理
def test_args_single_value():
    # 定义带参数的函数 fun_with_arg(t, y, a)
    def fun_with_arg(t, y, a):
        return a * y

    # 预期抛出 TypeError 异常，匹配指定消息
    message = "Supplied 'args' cannot be unpacked."
    with pytest.raises(TypeError, match=message):
        solve_ivp(fun_with_arg, (0, 0.1), [1], args=-1)

    # 调用 solve_ivp 解决带参数的初值问题
    sol = solve_ivp(fun_with_arg, (0, 0.1), [1], args=(-1,))
    # 断言结果数组第一个元素接近于 exp(-0.1)
    assert_allclose(sol.y[0, -1], np.exp(-0.1))


# 参数化测试，测试初始状态的有限性
@pytest.mark.parametrize("f0_fill", [np.nan, np.inf])
def test_initial_state_finiteness(f0_fill):
    # 针对 gh-17846 的回归测试
    msg = "All components of the initial state `y0` must be finite."
    with pytest.raises(ValueError, match=msg):
        # 调用 solve_ivp 解决 fun_zero 初始值问题，初始状态有一个分量为 f0_fill
        solve_ivp(fun_zero, [0, 10], np.full(3, f0_fill))


# 参数化测试，测试在较小时间间隔内是否遵守时间界限
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_zero_interval(method):
    # 当积分上下限相同时的情况，积分结果应该与初始状态一致
    def f(t, y):
        return 2 * y
    # 解决 f[y(t)] = 2y(t) 在 t = 0.0 到 t = 0.0 的初值问题
    res = solve_ivp(f, (0.0, 0.0), np.array([1.0]), method=method)
    # 断言求解成功
    assert res.success
    # 断言结果数组最后一个元素接近于 1.0
    assert_allclose(res.y[0, -1], 1.0)


# 参数化测试，测试在较小时间间隔内是否遵守时间界限
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_tbound_respected_small_interval(method):
    """Regression test for gh-17341"""
    SMALL = 1e-4

    # f[y(t)] = 2y(t) 在 t ∈ [0,SMALL] 上，其他情况下未定义
    def f(t, y):
        if t > SMALL:
            raise ValueError("Function was evaluated outside interval")
        return 2 * y
    # 解决 f[y(t)] = 2y(t) 在 t = 0.0 到 t = SMALL 的初值问题
    res = solve_ivp(f, (0.0, SMALL), np.array([1]), method=method)
    # 断言求解成功
    assert res.success


# 参数化测试，测试在较大时间间隔内是否遵守时间界限
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_tbound_respected_larger_interval(method):
    """Regression test for gh-8848"""
    # 定义势函数 V(r)
    def V(r):
        return -11/r + 10 * r / (0.05 + r**2)
    # 定义一个函数 func，接受参数 t 和 p
    def func(t, p):
        # 检查 t 是否在指定区间之外，如果是则抛出 ValueError 异常
        if t < -17 or t > 2:
            raise ValueError("Function was evaluated outside interval")
        
        # 从参数 p 中提取 P 和 Q
        P = p[0]
        Q = p[1]
        
        # 计算 r，即 t 的指数函数
        r = np.exp(t)
        
        # 计算 dP/dr，即 P 对 r 的导数
        dPdr = r * Q
        
        # 计算 dQ/dr，即 Q 对 r 的导数
        dQdr = -2.0 * r * ((-0.2 - V(r)) * P + 1 / r * Q)
        
        # 返回包含 dP/dr 和 dQ/dr 的 NumPy 数组
        return np.array([dPdr, dQdr])

    # 使用 solve_ivp 函数求解微分方程组
    result = solve_ivp(func, 
                       (-17, 2),            # 指定求解的时间范围
                       y0=np.array([1, -11]),  # 初始条件，P(0)=1, Q(0)=-11
                       max_step=0.03,        # 最大步长
                       vectorized=False,     # 禁用向量化求解
                       t_eval=None,          # 返回所有内部步骤的 t 值
                       atol=1e-8,            # 绝对误差容限
                       rtol=1e-5)            # 相对误差容限

    # 断言求解是否成功，如果不成功会引发 AssertionError
    assert result.success
# 使用 pytest.mark.parametrize 装饰器为 test_tbound_respected_oscillator 函数参数化，参数为不同的数值积分方法
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_tbound_respected_oscillator(method):
    # 回归测试函数，用于检查 gh-9198 的问题
    "Regression test for gh-9198"
    
    # 定义反应函数 reactions_func，接受时间 t 和状态 y 作为参数
    def reactions_func(t, y):
        # 如果 t 大于 205，则抛出 ValueError 异常
        if (t > 205): 
            raise ValueError("Called outside interval")
        # 返回状态 y 的导数数组
        yprime = np.array([1.73307544e-02, 
                           6.49376470e-06, 
                           0.00000000e+00, 
                           0.00000000e+00])
        return yprime

    # 定义 run_sim2 函数，模拟时间 t_end 内的系统行为
    def run_sim2(t_end, n_timepoints=10, shortest_delay_line=10000000):
        # 初始状态数组
        init_state = np.array([134.08298555, 138.82348612, 100., 0.])
        t0 = 100.0
        t1 = 200.0
        # 调用 solve_ivp 函数求解微分方程
        return solve_ivp(reactions_func,
                         (t0, t1),
                         init_state.copy(), 
                         dense_output=True, 
                         max_step=t1 - t0)
    
    # 调用 run_sim2 函数执行模拟，并断言模拟成功
    result = run_sim2(1000, 100, 100)
    assert result.success


# 定义 test_initial_maxstep 函数，验证 select_initial_step 函数是否尊重 max_step 参数
def test_inital_maxstep():
    """Verify that select_inital_step respects max_step"""
    # 相对误差和绝对误差
    rtol = 1e-3
    atol = 1e-6
    # 初始状态数组
    y0 = np.array([1/3, 2/9])
    
    # 遍历时间段 (t0, t_bound) 的组合
    for (t0, t_bound) in ((5, 9), (5, 1)):
        # 遍历不同数值积分方法的误差估计阶数
        for method_order in [RK23.error_estimator_order,
                            RK45.error_estimator_order,
                            DOP853.error_estimator_order,
                            3,  # RADAU
                            1   # BDF
                            ]:
            # 调用 select_initial_step 函数，计算无最大步长限制的步长
            step_no_max = select_initial_step(fun_rational, t0, y0, t_bound,
                                            np.inf,
                                            fun_rational(t0,y0), 
                                            np.sign(t_bound - t0),
                                            method_order,
                                            rtol, atol)
            # 最大步长为无最大步长限制步长的一半
            max_step = step_no_max / 2
            # 再次调用 select_initial_step 函数，计算受最大步长限制的步长
            step_with_max = select_initial_step(fun_rational, t0, y0, t_bound,
                                            max_step,
                                            fun_rational(t0, y0),
                                            np.sign(t_bound - t0),
                                            method_order, 
                                            rtol, atol)
            # 断言最大步长与受限步长相等
            assert_equal(max_step, step_with_max)
```