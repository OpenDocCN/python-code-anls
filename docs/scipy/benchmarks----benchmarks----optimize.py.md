# `D:\src\scipysrc\scipy\benchmarks\benchmarks\optimize.py`

```
# 导入标准库和第三方库
import os  # 操作系统相关的功能
import time  # 时间相关的功能
import inspect  # 获取对象信息的功能
import json  # 处理 JSON 数据的功能
import traceback  # 获取异常堆栈信息的功能
from collections import defaultdict  # 默认字典

import numpy as np  # 数值计算库

# 导入本地模块和函数
from . import test_functions as funcs  # 导入 test_functions 模块
from . import go_benchmark_functions as gbf  # 导入 go_benchmark_functions 模块
from .common import Benchmark, is_xslow, safe_import  # 导入 common 模块中的类和函数
from .lsq_problems import extract_lsq_problems  # 导入 lsq_problems 模块中的函数

# 使用安全导入上下文管理器
with safe_import():
    # 导入 scipy.optimize 相关的模块和函数
    import scipy.optimize
    from scipy.optimize.optimize import rosen, rosen_der, rosen_hess
    from scipy.optimize import (leastsq, basinhopping, differential_evolution,
                                dual_annealing, shgo, direct)
    from scipy.optimize._minimize import MINIMIZE_METHODS
    # 导入 cutest 模块中的函数
    from .cutest.calfun import calfun
    from .cutest.dfoxs import dfoxs


class _BenchOptimizers(Benchmark):
    """优化器性能基准测试的框架

    Parameters
    ----------
    function_name : string
        函数名称
    fun : callable
        要优化的函数
    der : callable
        返回 fun 的导数（雅可比、梯度）的函数
    hess : callable
        返回 fun 的黑塞矩阵的函数
    minimizer_kwargs : kwargs
        传递给优化器的额外关键字参数，例如 tol, maxiter
    """

    def __init__(self, function_name, fun, der=None, hess=None,
                 **minimizer_kwargs):
        self.function_name = function_name
        self.fun = fun
        self.der = der
        self.hess = hess
        self.minimizer_kwargs = minimizer_kwargs
        if "tol" not in minimizer_kwargs:
            minimizer_kwargs["tol"] = 1e-4

        self.results = []

    @classmethod
    def from_funcobj(cls, function_name, function, **minimizer_kwargs):
        """从函数对象创建 _BenchOptimizers 的实例

        Parameters
        ----------
        function_name : string
            函数名称
        function : object
            函数对象
        minimizer_kwargs : kwargs
            传递给优化器的额外关键字参数

        Returns
        -------
        self : _BenchOptimizers
            创建的 _BenchOptimizers 实例
        """
        self = cls.__new__(cls)
        self.function_name = function_name

        self.function = function
        self.fun = function.fun
        if hasattr(function, 'der'):
            self.der = function.der

        self.bounds = function.bounds

        self.minimizer_kwargs = minimizer_kwargs
        self.results = []
        return self

    def reset(self):
        """重置测试结果列表"""
        self.results = []

    def energy_gradient(self, x):
        """计算函数在点 x 处的能量和梯度

        Parameters
        ----------
        x : array-like
            函数的参数向量

        Returns
        -------
        energy : float
            函数在点 x 处的能量
        gradient : array-like
            函数在点 x 处的梯度
        """
        return self.fun(x), self.function.der(x)

    def add_result(self, result, t, name):
        """向结果列表中添加一个结果

        Parameters
        ----------
        result : object
            要添加的结果对象
        t : float
            测试耗时
        name : string
            结果名称
        """
        result.time = t
        result.name = name
        if not hasattr(result, "njev"):
            result.njev = 0
        if not hasattr(result, "nhev"):
            result.nhev = 0
        self.results.append(result)
    # 打印当前结果列表的方法
    def print_results(self):
        """print the current list of results"""
        # 调用 average_results 方法获取结果列表并按照指定规则排序
        results = self.average_results()
        results = sorted(results, key=lambda x: (x.nfail, x.mean_time))
        # 如果结果列表为空则直接返回
        if not results:
            return
        # 打印分隔线和优化器基准信息
        print("")
        print("=========================================================")
        print("Optimizer benchmark: %s" % (self.function_name))
        # 打印维度信息和额外的参数设置
        print("dimensions: %d, extra kwargs: %s" %
              (results[0].ndim, str(self.minimizer_kwargs)))
        # 打印平均结果基于多少次起始配置
        print("averaged over %d starting configurations" % (results[0].ntrials))
        # 打印结果表头
        print("  Optimizer    nfail   nfev    njev    nhev    time")
        print("---------------------------------------------------------")
        # 遍历结果并打印每一行数据
        for res in results:
            print("%11s  | %4d  | %4d  | %4d  | %4d  | %.6g" %
                  (res.name, res.nfail, res.mean_nfev,
                   res.mean_njev, res.mean_nhev, res.mean_time))

    # 将结果按照优化器分组并计算平均值
    def average_results(self):
        """group the results by minimizer and average over the runs"""
        # 使用 defaultdict 将结果按照优化器名称分组
        grouped_results = defaultdict(list)
        for res in self.results:
            grouped_results[res.name].append(res)

        averaged_results = dict()
        # 遍历每个分组计算平均值并存储在 averaged_results 中
        for name, result_list in grouped_results.items():
            newres = scipy.optimize.OptimizeResult()
            newres.name = name
            newres.mean_nfev = np.mean([r.nfev for r in result_list])
            newres.mean_njev = np.mean([r.njev for r in result_list])
            newres.mean_nhev = np.mean([r.nhev for r in result_list])
            newres.mean_time = np.mean([r.time for r in result_list])
            funs = [r.fun for r in result_list]
            newres.max_obj = np.max(funs)
            newres.min_obj = np.min(funs)
            newres.mean_obj = np.mean(funs)

            newres.ntrials = len(result_list)
            newres.nfail = len([r for r in result_list if not r.success])
            newres.nsuccess = len([r for r in result_list if r.success])
            try:
                newres.ndim = len(result_list[0].x)
            except TypeError:
                newres.ndim = 1
            averaged_results[name] = newres
        return averaged_results

    # 用于 basinhopping 的接受测试方法
    def accept_test(self, x_new=None, *args, **kwargs):
        """
        Does the new candidate vector lie in between the bounds?

        Returns
        -------
        accept_test : bool
            The candidate vector lies in between the bounds
        """
        # 如果函数对象没有 xmin 属性，则始终接受新向量
        if not hasattr(self.function, "xmin"):
            return True
        # 如果新向量中任意元素小于 xmin，则拒绝
        if np.any(x_new < self.function.xmin):
            return False
        # 如果新向量中任意元素大于 xmax，则拒绝
        if np.any(x_new > self.function.xmax):
            return False
        # 否则接受新向量
        return True
    def run_basinhopping(self):
        """
        Do an optimization run for basinhopping
        """
        # 从self.minimizer_kwargs获取优化器参数
        kwargs = self.minimizer_kwargs
        # 如果self.fun具有"temperature"属性，则将"T"参数设置为self.fun.temperature
        if hasattr(self.fun, "temperature"):
            kwargs["T"] = self.function.temperature
        # 如果self.fun具有"stepsize"属性，则将"stepsize"参数设置为self.fun.stepsize
        if hasattr(self.fun, "stepsize"):
            kwargs["stepsize"] = self.function.stepsize

        # 设置minimizer_kwargs字典的优化方法为"L-BFGS-B"
        minimizer_kwargs = {"method": "L-BFGS-B"}

        # 获取初始向量
        x0 = self.function.initial_vector()

        # basinhopping - 不使用梯度
        minimizer_kwargs['jac'] = False
        # 将函数评估次数nfev设置为0
        self.function.nfev = 0

        # 记录起始时间
        t0 = time.time()

        # 进行basinhopping优化
        res = basinhopping(
            self.fun, x0, accept_test=self.accept_test,
            minimizer_kwargs=minimizer_kwargs,
            **kwargs)

        # 记录结束时间
        t1 = time.time()
        # 使用self.function.success检查优化结果的成功性，并将结果赋给res.success
        res.success = self.function.success(res.x)
        # 将函数评估次数nfev设置为优化函数的评估次数
        res.nfev = self.function.nfev
        # 将优化结果及耗时添加到结果集中，标记为'basinh.'
        self.add_result(res, t1 - t0, 'basinh.')

    def run_direct(self):
        """
        Do an optimization run for direct
        """
        # 将函数评估次数nfev设置为0
        self.function.nfev = 0

        # 记录起始时间
        t0 = time.time()

        # 进行direct优化
        res = direct(self.fun,
                     self.bounds)

        # 记录结束时间
        t1 = time.time()
        # 使用self.function.success检查优化结果的成功性，并将结果赋给res.success
        res.success = self.function.success(res.x)
        # 将函数评估次数nfev设置为优化函数的评估次数
        res.nfev = self.function.nfev
        # 将优化结果及耗时添加到结果集中，标记为'DIRECT'
        self.add_result(res, t1 - t0, 'DIRECT')

    def run_shgo(self):
        """
        Do an optimization run for shgo
        """
        # 将函数评估次数nfev设置为0
        self.function.nfev = 0

        # 记录起始时间
        t0 = time.time()

        # 进行shgo优化
        res = shgo(self.fun,
                   self.bounds)

        # 记录结束时间
        t1 = time.time()
        # 使用self.function.success检查优化结果的成功性，并将结果赋给res.success
        res.success = self.function.success(res.x)
        # 将函数评估次数nfev设置为优化函数的评估次数
        res.nfev = self.function.nfev
        # 将优化结果及耗时添加到结果集中，标记为'SHGO'
        self.add_result(res, t1 - t0, 'SHGO')

    def run_differentialevolution(self):
        """
        Do an optimization run for differential_evolution
        """
        # 将函数评估次数nfev设置为0
        self.function.nfev = 0

        # 记录起始时间
        t0 = time.time()

        # 进行differential_evolution优化，设置popsize为20
        res = differential_evolution(self.fun,
                                     self.bounds,
                                     popsize=20)

        # 记录结束时间
        t1 = time.time()
        # 使用self.function.success检查优化结果的成功性，并将结果赋给res.success
        res.success = self.function.success(res.x)
        # 将函数评估次数nfev设置为优化函数的评估次数
        res.nfev = self.function.nfev
        # 将优化结果及耗时添加到结果集中，标记为'DE'
        self.add_result(res, t1 - t0, 'DE')

    def run_dualannealing(self):
        """
        Do an optimization run for dual_annealing
        """
        # 将函数评估次数nfev设置为0
        self.function.nfev = 0

        # 记录起始时间
        t0 = time.time()

        # 进行dual_annealing优化
        res = dual_annealing(self.fun,
                             self.bounds)

        # 记录结束时间
        t1 = time.time()
        # 使用self.function.success检查优化结果的成功性，并将结果赋给res.success
        res.success = self.function.success(res.x)
        # 将函数评估次数nfev设置为优化函数的评估次数
        res.nfev = self.function.nfev
        # 将优化结果及耗时添加到结果集中，标记为'DA'
        self.add_result(res, t1 - t0, 'DA')
    # 对象方法，用于运行全局优化测试，支持多种优化方法
    def bench_run_global(self, numtrials=50, methods=None):
        """
        Run the optimization tests for the required minimizers.
        执行所需最小化器的优化测试。
        """

        if methods is None:
            methods = ['DE', 'basinh.', 'DA', 'DIRECT', 'SHGO']

        # 定义随机优化方法列表
        stochastic_methods = ['DE', 'basinh.', 'DA']

        # 定义每种方法对应的运行函数映射
        method_fun = {'DE': self.run_differentialevolution,
                      'basinh.': self.run_basinhopping,
                      'DA': self.run_dualannealing,
                      'DIRECT': self.run_direct,
                      'SHGO': self.run_shgo, }

        # 遍历所有指定的优化方法
        for m in methods:
            # 如果方法属于随机优化方法，则运行指定次数的试验
            if m in stochastic_methods:
                for i in range(numtrials):
                    method_fun[m]()
            else:
                # 否则直接运行该方法的优化
                method_fun[m]()

    # 对象方法，执行从 x0 起始的所有优化器的优化测试
    def bench_run(self, x0, methods=None, **minimizer_kwargs):
        """do an optimization test starting at x0 for all the optimizers"""
        kwargs = self.minimizer_kwargs

        if methods is None:
            methods = MINIMIZE_METHODS  # MINIMIZE_METHODS 是一个全局定义的优化方法列表

        # 只使用函数值的优化方法列表
        fonly_methods = ["COBYLA", 'COBYQA', 'Powell', 'nelder-mead',
                         'L-BFGS-B', 'BFGS', 'trust-constr', 'SLSQP']
        for method in fonly_methods:
            if method not in methods:
                continue
            t0 = time.time()
            # 执行优化，记录运行时间和结果
            res = scipy.optimize.minimize(self.fun, x0, method=method,
                                          **kwargs)
            t1 = time.time()
            self.add_result(res, t1-t0, method)

        # 支持梯度方法的优化器列表
        gradient_methods = ['L-BFGS-B', 'BFGS', 'CG', 'TNC', 'SLSQP',
                            'trust-constr']
        if self.der is not None:
            for method in gradient_methods:
                if method not in methods:
                    continue
                t0 = time.time()
                # 执行带有梯度信息的优化，记录运行时间和结果
                res = scipy.optimize.minimize(self.fun, x0, method=method,
                                              jac=self.der, **kwargs)
                t1 = time.time()
                self.add_result(res, t1-t0, method)

        # 支持海森矩阵方法的优化器列表
        hessian_methods = ["Newton-CG", 'dogleg', 'trust-ncg',
                           'trust-exact', 'trust-krylov', 'trust-constr']
        if self.hess is not None:
            for method in hessian_methods:
                if method not in methods:
                    continue
                t0 = time.time()
                # 执行带有海森矩阵信息的优化，记录运行时间和结果
                res = scipy.optimize.minimize(self.fun, x0, method=method,
                                              jac=self.der, hess=self.hess,
                                              **kwargs)
                t1 = time.time()
                self.add_result(res, t1-t0, method)
class BenchSmoothUnbounded(Benchmark):
    """Benchmark the optimizers with smooth, unbounded, functions"""
    # 定义参数列表，包括测试函数、求解器和结果类型
    params = [
        ['rosenbrock_slow', 'rosenbrock_nograd', 'rosenbrock', 'rosenbrock_tight',
         'simple_quadratic', 'asymmetric_quadratic',
         'sin_1d', 'booth', 'beale', 'LJ'],
        ["COBYLA", 'COBYQA', 'Powell', 'nelder-mead',
         'L-BFGS-B', 'BFGS', 'CG', 'TNC', 'SLSQP',
         "Newton-CG", 'dogleg', 'trust-ncg', 'trust-exact',
         'trust-krylov', 'trust-constr'],
        ["mean_nfev", "mean_time"]
    ]
    # 定义参数名称
    param_names = ["test function", "solver", "result type"]

    # 设置函数，根据给定的函数名、方法名和返回值计算结果
    def setup(self, func_name, method_name, ret_val):
        b = getattr(self, 'run_' + func_name)(methods=[method_name])
        r = b.average_results().get(method_name)
        if r is None:
            raise NotImplementedError()
        self.result = getattr(r, ret_val)

    # 跟踪所有结果
    def track_all(self, func_name, method_name, ret_val):
        return self.result

    # 对于 SlowRosen 函数，每次函数评估都有 50us 的延迟。通过与 rosenbrock_nograd 进行比较，
    # 可以了解最小化器内部使用的时间与函数评估所需时间的差异。
    def run_rosenbrock_slow(self, methods=None):
        s = funcs.SlowRosen()
        b = _BenchOptimizers("Rosenbrock function",
                             fun=s.fun)
        for i in range(10):
            b.bench_run(np.random.uniform(-3, 3, 3), methods=methods)
        return b

    # 查看如果需要使用数值微分，求解器的性能如何
    def run_rosenbrock_nograd(self, methods=None):
        b = _BenchOptimizers("Rosenbrock function",
                             fun=rosen)
        for i in range(10):
            b.bench_run(np.random.uniform(-3, 3, 3), methods=methods)
        return b

    def run_rosenbrock(self, methods=None):
        b = _BenchOptimizers("Rosenbrock function",
                             fun=rosen, der=rosen_der, hess=rosen_hess)
        for i in range(10):
            b.bench_run(np.random.uniform(-3, 3, 3), methods=methods)
        return b

    def run_rosenbrock_tight(self, methods=None):
        b = _BenchOptimizers("Rosenbrock function",
                             fun=rosen, der=rosen_der, hess=rosen_hess,
                             tol=1e-8)
        for i in range(10):
            b.bench_run(np.random.uniform(-3, 3, 3), methods=methods)
        return b

    def run_simple_quadratic(self, methods=None):
        s = funcs.SimpleQuadratic()
        #    print "checking gradient",
        #    scipy.optimize.check_grad(s.fun, s.der, np.array([1.1, -2.3]))
        b = _BenchOptimizers("simple quadratic function",
                             fun=s.fun, der=s.der, hess=s.hess)
        for i in range(10):
            b.bench_run(np.random.uniform(-2, 2, 3), methods=methods)
        return b
    # 定义一个方法，用于运行 AsymmetricQuadratic 类的优化函数
    def run_asymmetric_quadratic(self, methods=None):
        # 创建 AsymmetricQuadratic 类的实例对象 s
        s = funcs.AsymmetricQuadratic()
        # 创建 _BenchOptimizers 类的实例对象 b，用于测试优化函数
        b = _BenchOptimizers("function sum(x**2) + x[0]",
                             fun=s.fun, der=s.der, hess=s.hess)
        # 运行优化函数10次，每次使用随机生成的参数，记录结果
        for i in range(10):
            b.bench_run(np.random.uniform(-2, 2, 3), methods=methods)
        # 返回优化结果的实例对象 b
        return b

    # 定义一个方法，用于运行 1 维 sin 函数的优化
    def run_sin_1d(self, methods=None):
        # 定义 1 维 sin 函数
        def fun(x):
            return np.sin(x[0])

        # 定义 1 维 sin 函数的导数
        def der(x):
            return np.array([np.cos(x[0])])

        # 创建 _BenchOptimizers 类的实例对象 b，用于测试优化函数
        b = _BenchOptimizers("1d sin function",
                             fun=fun, der=der, hess=None)
        # 运行优化函数10次，每次使用随机生成的参数，记录结果
        for i in range(10):
            b.bench_run(np.random.uniform(-2, 2, 1), methods=methods)
        # 返回优化结果的实例对象 b
        return b

    # 定义一个方法，用于运行 Booth 函数的优化
    def run_booth(self, methods=None):
        # 创建 Booth 类的实例对象 s
        s = funcs.Booth()
        # 创建 _BenchOptimizers 类的实例对象 b，用于测试优化函数
        b = _BenchOptimizers("Booth's function",
                             fun=s.fun, der=s.der, hess=None)
        # 运行优化函数10次，每次使用随机生成的参数，记录结果
        for i in range(10):
            b.bench_run(np.random.uniform(0, 10, 2), methods=methods)
        # 返回优化结果的实例对象 b
        return b

    # 定义一个方法，用于运行 Beale 函数的优化
    def run_beale(self, methods=None):
        # 创建 Beale 类的实例对象 s
        s = funcs.Beale()
        # 创建 _BenchOptimizers 类的实例对象 b，用于测试优化函数
        b = _BenchOptimizers("Beale's function",
                             fun=s.fun, der=s.der, hess=None)
        # 运行优化函数10次，每次使用随机生成的参数，记录结果
        for i in range(10):
            b.bench_run(np.random.uniform(0, 10, 2), methods=methods)
        # 返回优化结果的实例对象 b
        return b

    # 定义一个方法，用于运行 LJ 函数的优化
    def run_LJ(self, methods=None):
        # 创建 LJ 类的实例对象 s
        s = funcs.LJ()
        # 计算原子数量
        natoms = 4
        # 创建 _BenchOptimizers 类的实例对象 b，用于测试优化函数
        b = _BenchOptimizers("%d atom Lennard Jones potential" % (natoms),
                             fun=s.fun, der=s.der, hess=None)
        # 运行优化函数10次，每次使用随机生成的参数，记录结果
        for i in range(10):
            b.bench_run(np.random.uniform(-2, 2, natoms*3), methods=methods)
        # 返回优化结果的实例对象 b
        return b
class BenchLeastSquares(Benchmark):
    """Class for benchmarking nonlinear least squares solvers."""

    # 从数据提取非线性最小二乘问题
    problems = extract_lsq_problems()

    # 参数设置：问题列表和结果类型列表
    params = [
        list(problems.keys()),  # 使用问题的键作为参数之一
        ["average time", "nfev", "success"]  # 结果类型包括平均时间、函数评估次数和成功率
    ]

    # 参数名称：指定参数的名称
    param_names = [
        "problem", "result type"
    ]

    # 跟踪所有结果的方法
    def track_all(self, problem_name, result_type):
        # 获取指定问题名称的问题实例
        problem = self.problems[problem_name]

        # 如果问题存在下界或上界，抛出未实现错误
        if problem.lb is not None or problem.ub is not None:
            raise NotImplementedError

        # 定义容差
        ftol = 1e-5

        # 如果结果类型为 'average time'
        if result_type == 'average time':
            n_runs = 10
            t0 = time.time()
            # 运行最小二乘求解器多次，并计算平均时间
            for _ in range(n_runs):
                leastsq(problem.fun, problem.x0, Dfun=problem.jac, ftol=ftol,
                        full_output=True)
            return (time.time() - t0) / n_runs

        # 否则运行一次最小二乘求解器，并返回相应的结果
        x, cov_x, info, message, ier = leastsq(
            problem.fun, problem.x0, Dfun=problem.jac,
            ftol=ftol, full_output=True
        )

        # 根据结果类型返回相应的信息
        if result_type == 'nfev':
            return info['nfev']  # 返回函数评估次数
        elif result_type == 'success':
            return int(problem.check_answer(x, ftol))  # 返回成功率（转换为整数）
        else:
            raise NotImplementedError


# `export SCIPY_XSLOW=1` to enable BenchGlobal.track_all
# `export SCIPY_GLOBAL_BENCH=AMGM,Adjiman,...` to run specific tests
# `export SCIPY_GLOBAL_BENCH_NUMTRIALS=10` to specify n_iterations, default 100
#
# 注意：运行可能需要数小时；中间输出可以在 benchmarks/global-bench-results.json 下找到


class BenchGlobal(Benchmark):
    """
    Benchmark the global optimizers using the go_benchmark_functions
    suite
    """
    
    # 超时设定为 300 秒
    timeout = 300

    # 获取所有在 go_benchmark_functions 套件中的全局优化器类，排除特定的类和问题类
    _functions = dict([
        item for item in inspect.getmembers(gbf, inspect.isclass)
        if (issubclass(item[1], gbf.Benchmark) and
            item[0] not in ('Benchmark') and
            not item[0].startswith('Problem'))
    ])

    # 如果不是 XSlow 模式，禁用所有函数
    if not is_xslow():
        _enabled_functions = []
    # 如果设置了 SCIPY_GLOBAL_BENCH 环境变量，则根据其指定的函数列表启用特定的函数
    elif 'SCIPY_GLOBAL_BENCH' in os.environ:
        _enabled_functions = [x.strip() for x in
                              os.environ['SCIPY_GLOBAL_BENCH'].split(',')]
    else:
        _enabled_functions = list(_functions.keys())  # 否则启用所有函数

    # 参数设置：函数列表、结果类型列表和求解器列表
    params = [
        list(_functions.keys()),  # 使用函数的键作为参数之一
        ["success%", "<nfev>", "average time"],  # 结果类型包括成功率、函数评估次数和平均时间
        ['DE', 'basinh.', 'DA', 'DIRECT', 'SHGO'],  # 求解器列表
    ]

    # 参数名称：指定参数的名称
    param_names = ["test function", "result type", "solver"]

    def __init__(self):
        # 检查是否启用 XSlow 模式
        self.enabled = is_xslow()

        # 尝试从环境变量中获取 SCIPY_GLOBAL_BENCH_NUMTRIALS 的值，否则默认为 100
        try:
            self.numtrials = int(os.environ['SCIPY_GLOBAL_BENCH_NUMTRIALS'])
        except (KeyError, ValueError):
            self.numtrials = 100

        # 定义结果输出文件的路径
        self.dump_fn = os.path.join(os.path.dirname(__file__),
                                    '..',
                                    'global-bench-results.json',)
        # 初始化结果字典
        self.results = {}
    # 初始化方法，用于设置基础参数和加载必要的数据
    def setup(self, name, ret_value, solver):
        # 如果指定的函数名不在启用函数列表中，则抛出未实现错误
        if name not in self._enabled_functions:
            raise NotImplementedError("skipped")

        # 加载 JSON 数据备份文件
        with open(self.dump_fn) as f:
            # 将文件内容解析为 Python 字典，并存储在实例变量中
            self.results = json.load(f)

    # 清理方法，在测试结束时调用，用于保存结果到文件
    def teardown(self, name, ret_value, solver):
        # 如果未启用测试，则直接返回
        if not self.enabled:
            return

        # 将结果字典以 JSON 格式写入到文件中，保存当前测试结果
        with open(self.dump_fn, 'w') as f:
            json.dump(self.results, f, indent=2, sort_keys=True)

    # 跟踪所有测试的方法，在每次测试后调用，记录测试结果并返回指定的度量值
    def track_all(self, name, ret_value, solver):
        # 如果指定的函数名和求解器已存在于结果中，则直接返回保存的度量值
        if name in self.results and solver in self.results[name]:
            av_results = self.results[name]
            if ret_value == 'success%':
                # 计算成功率百分比并返回
                return (100 * av_results[solver]['nsuccess']
                        / av_results[solver]['ntrials'])
            elif ret_value == '<nfev>':
                # 返回平均函数评估次数
                return av_results[solver]['mean_nfev']
            elif ret_value == 'average time':
                # 返回平均运行时间
                return av_results[solver]['mean_time']
            else:
                # 若度量值未知，则抛出数值错误
                raise ValueError()

        # 若未在结果中找到对应的函数名，则执行以下操作
        klass = self._functions[name]
        f = klass()
        try:
            # 使用函数对象创建基准优化器
            b = _BenchOptimizers.from_funcobj(name, f)
            with np.errstate(all='ignore'):
                # 运行全局基准测试，记录方法和试验次数
                b.bench_run_global(methods=[solver],
                                   numtrials=self.numtrials)

            # 获取平均测试结果
            av_results = b.average_results()

            # 如果函数名不在结果字典中，则添加新条目
            if name not in self.results:
                self.results[name] = {}
            # 将当前求解器的平均结果存储到结果字典中
            self.results[name][solver] = av_results[solver]

            # 根据请求的度量值返回对应的结果
            if ret_value == 'success%':
                return (100 * av_results[solver]['nsuccess']
                        / av_results[solver]['ntrials'])
            elif ret_value == '<nfev>':
                return av_results[solver]['mean_nfev']
            elif ret_value == 'average time':
                return av_results[solver]['mean_time']
            else:
                # 若度量值未知，则抛出数值错误
                raise ValueError()
        except Exception:
            # 捕获任何异常并打印异常信息，记录到结果中
            print("".join(traceback.format_exc()))
            self.results[name] = "".join(traceback.format_exc())

    # 设置缓存方法，在初始化时调用，用于创建空的 JSON 文件作为日志
    def setup_cache(self):
        # 如果未启用测试，则直接返回
        if not self.enabled:
            return

        # 创建初始的空 JSON 文件，用于后续记录结果
        with open(self.dump_fn, 'w') as f:
            json.dump({}, f, indent=2)
class BenchDFO(Benchmark):
    """
    Benchmark the optimizers with the CUTEST DFO benchmark of Moré and Wild.
    The original benchmark suite is available at
    https://github.com/POptUS/BenDFO
    """

    # 定义参数列表，包括问题编号、求解方法和结果类型
    params = [
        list(range(53)),  # adjust which problems to solve
        ["COBYLA", "COBYQA", "SLSQP", "Powell", "nelder-mead", "L-BFGS-B",
         "BFGS",
         "trust-constr"],  # note: methods must also be listed in bench_run
        ["mean_nfev", "min_obj"],  # defined in average_results
    ]
    # 参数名列表，与params对应，指定每个参数的含义
    param_names = ["DFO benchmark problem number", "solver", "result type"]

    # 设置方法，根据问题编号、方法名和返回值类型设置基准
    def setup(self, prob_number, method_name, ret_val):
        # 从文件加载 CUTEST DFO 的问题数据集
        probs = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                        "cutest", "dfo.txt"))
        # 获取特定问题编号的参数
        params = probs[prob_number]
        nprob = int(params[0])
        n = int(params[1])
        m = int(params[2])
        s = params[3]
        factor = 10 ** s

        # 定义函数 func，调用 calfun 函数计算问题的目标函数值
        def func(x):
            return calfun(x, m, nprob)

        # 初始化优化起始点 x0
        x0 = dfoxs(n, nprob, factor)
        
        # 调用 run_cutest 方法运行优化器，传入 func、x0 和其他参数
        b = getattr(self, "run_cutest")(
            func, x0, prob_number=prob_number, methods=[method_name]
        )
        
        # 计算给定方法的平均结果
        r = b.average_results().get(method_name)
        
        # 如果结果为空，则抛出未实现错误
        if r is None:
            raise NotImplementedError()
        
        # 将结果赋值给实例变量 self.result
        self.result = getattr(r, ret_val)

    # 跟踪所有结果的方法，返回已计算的结果
    def track_all(self, prob_number, method_name, ret_val):
        return self.result

    # 运行 CUTEST DFO 优化器的方法，传入目标函数 func、起始点 x0 和方法列表
    def run_cutest(self, func, x0, prob_number, methods=None):
        if methods is None:
            methods = MINIMIZE_METHODS
        # 创建 _BenchOptimizers 对象，设置优化函数和名称
        b = _BenchOptimizers(f"DFO benchmark problem {prob_number}", fun=func)
        # 调用 bench_run 方法执行基准测试运行，传入起始点 x0 和方法列表
        b.bench_run(x0, methods=methods)
        return b
```