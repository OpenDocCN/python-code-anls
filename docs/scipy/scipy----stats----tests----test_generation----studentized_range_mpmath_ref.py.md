# `D:\src\scipysrc\scipy\scipy\stats\tests\test_generation\studentized_range_mpmath_ref.py`

```
# To run this script, run
# `python studentized_range_mpmath_ref.py`
# in the "scipy/stats/tests/" directory
# 运行此脚本，请在 "scipy/stats/tests/" 目录下运行
# `python studentized_range_mpmath_ref.py`

# This script generates a JSON file "./data/studentized_range_mpmath_ref.json"
# that is used to compare the accuracy of `studentized_range` functions against
# precise (20 DOP) results generated using `mpmath`.
# 此脚本生成一个 JSON 文件 "./data/studentized_range_mpmath_ref.json"
# 用于比较 `studentized_range` 函数的准确性，与使用 `mpmath` 生成的精确（20位小数）结果进行比较。

# Equations in this file have been taken from
# https://en.wikipedia.org/wiki/Studentized_range_distribution
# and have been checked against the following reference:
# Lund, R. E., and J. R. Lund. "Algorithm AS 190: Probabilities and
# Upper Quantiles for the Studentized Range." Journal of the Royal
# Statistical Society. Series C (Applied Statistics), vol. 32, no. 2,
# 1983, pp. 204-210. JSTOR, www.jstor.org/stable/2347300. Accessed 18
# Feb. 2021.
# 本文件中的方程式来源于
# https://en.wikipedia.org/wiki/Studentized_range_distribution
# 并已验证通过以下参考文献：
# Lund, R. E., and J. R. Lund. "Algorithm AS 190: Probabilities and
# Upper Quantiles for the Studentized Range." Journal of the Royal
# Statistical Society. Series C (Applied Statistics), vol. 32, no. 2,
# 1983, pp. 204-210. JSTOR, www.jstor.org/stable/2347300. 访问日期：2021年2月18日。

# Note: I would have preferred to use pickle rather than JSON, but -
# due to security concerns - decided against it.
# 注意：我本来更愿意使用 pickle 而不是 JSON，但由于安全问题，我决定不这样做。

import itertools
from collections import namedtuple
import json
import time

import os
from multiprocessing import Pool, cpu_count

from mpmath import gamma, pi, sqrt, quad, inf, mpf, mp
from mpmath import npdf as phi
from mpmath import ncdf as Phi

results_filepath = "data/studentized_range_mpmath_ref.json"
num_pools = max(cpu_count() - 1, 1)

MPResult = namedtuple("MPResult", ["src_case", "mp_result"])

CdfCase = namedtuple("CdfCase",
                     ["q", "k", "v", "expected_atol", "expected_rtol"])

MomentCase = namedtuple("MomentCase",
                        ["m", "k", "v", "expected_atol", "expected_rtol"])

# Load previously generated JSON results, or init a new dict if none exist
# 加载先前生成的 JSON 结果，如果不存在则初始化一个新字典
if os.path.isfile(results_filepath):
    res_dict = json.load(open(results_filepath))
else:
    res_dict = dict()

# Frame out data structure. Store data with the function type as a top level
# key to allow future expansion
# 构建数据结构框架。将数据存储在以函数类型为顶级键的结构中，以支持未来扩展
res_dict["COMMENT"] = ("!!!!!! THIS FILE WAS AUTOGENERATED BY RUNNING "
                       "`python studentized_range_mpmath_ref.py` !!!!!!")
res_dict.setdefault("cdf_data", [])
res_dict.setdefault("pdf_data", [])
res_dict.setdefault("moment_data", [])

general_atol, general_rtol = 1e-11, 1e-11

mp.dps = 24

cp_q = [0.1, 1, 4, 10]
cp_k = [3, 10, 20]
cp_nu = [3, 10, 20, 50, 100, 120]

cdf_pdf_cases = [
    CdfCase(*case,
            general_atol,
            general_rtol)
    for case in
    itertools.product(cp_q, cp_k, cp_nu)
]

mom_atol, mom_rtol = 1e-9, 1e-9
# These are EXTREMELY slow - Multiple days each in worst case.
# 这些非常慢 - 最坏情况下每个可能需要多天时间。
moment_cases = [
    MomentCase(i, 3, 10, mom_atol, mom_rtol)
    for i in range(5)
]


def write_data():
    """Writes the current res_dict to the target JSON file"""
    # 将当前的 res_dict 写入目标 JSON 文件
    with open(results_filepath, mode="w") as f:
        json.dump(res_dict, f, indent=2)


def to_dict(named_tuple):
    """Converts a namedtuple to a dict"""
    # 将命名元组转换为字典
    return dict(named_tuple._asdict())


def mp_res_to_dict(mp_result):
    """Formats an MPResult namedtuple into a dict for JSON dumping"""
    # 将 MPResult 命名元组格式化为用于 JSON 导出的字典
    # 返回一个包含两个键值对的字典：
    # 键 "src_case"，对应值为将 mp_result.src_case 转换为字典格式的结果
    "src_case": to_dict(mp_result.src_case),

    # 键 "mp_result"，对应值为将 mp_result.mp_result 转换为浮点数格式的结果
    # 注意：由于 np assert 不能处理 mpf 类型，因此在此处接受精度损失。
    "mp_result": float(mp_result.mp_result)
# 计算学生化范围分布的累积分布函数（CDF）
def cdf_mp(q, k, nu):
    """Straightforward implementation of studentized range CDF"""
    q, k, nu = mpf(q), mpf(k), mpf(nu)  # 将输入参数转换为多精度浮点数对象

    def inner(s, z):
        return phi(z) * (Phi(z + q * s) - Phi(z)) ** (k - 1)  # 定义内部函数，计算内部积分部分

    def outer(s, z):
        return s ** (nu - 1) * phi(sqrt(nu) * s) * inner(s, z)  # 定义外部函数，计算外部积分部分

    def whole(s, z):
        return (sqrt(2 * pi) * k * nu ** (nu / 2)
                / (gamma(nu / 2) * 2 ** (nu / 2 - 1)) * outer(s, z))  # 计算整体函数的值

    # 使用高斯-勒让德积分法计算整体函数的积分值
    res = quad(whole, [0, inf], [-inf, inf],
               method="gauss-legendre", maxdegree=10)
    return res  # 返回计算得到的结果


# 计算学生化范围分布的概率密度函数（PDF）
def pdf_mp(q, k, nu):
    """Straightforward implementation of studentized range PDF"""
    q, k, nu = mpf(q), mpf(k), mpf(nu)  # 将输入参数转换为多精度浮点数对象

    def inner(s, z):
        return phi(z + q * s) * phi(z) * (Phi(z + q * s) - Phi(z)) ** (k - 2)  # 定义内部函数，计算内部积分部分

    def outer(s, z):
        return s ** nu * phi(sqrt(nu) * s) * inner(s, z)  # 定义外部函数，计算外部积分部分

    def whole(s, z):
        return (sqrt(2 * pi) * k * (k - 1) * nu ** (nu / 2)
                / (gamma(nu / 2) * 2 ** (nu / 2 - 1)) * outer(s, z))  # 计算整体函数的值

    # 使用高斯-勒让德积分法计算整体函数的积分值
    res = quad(whole, [0, inf], [-inf, inf],
               method="gauss-legendre", maxdegree=10)
    return res  # 返回计算得到的结果


# 计算学生化范围分布的矩
def moment_mp(m, k, nu):
    """Implementation of the studentized range moment"""
    m, k, nu = mpf(m), mpf(k), mpf(nu)  # 将输入参数转换为多精度浮点数对象

    def inner(q, s, z):
        return phi(z + q * s) * phi(z) * (Phi(z + q * s) - Phi(z)) ** (k - 2)  # 定义内部函数，计算内部积分部分

    def outer(q, s, z):
        return s ** nu * phi(sqrt(nu) * s) * inner(q, s, z)  # 定义外部函数，计算外部积分部分

    def pdf(q, s, z):
        return (sqrt(2 * pi) * k * (k - 1) * nu ** (nu / 2)
                / (gamma(nu / 2) * 2 ** (nu / 2 - 1)) * outer(q, s, z))  # 计算学生化范围分布的PDF

    def whole(q, s, z):
        return q ** m * pdf(q, s, z)  # 计算整体函数的值

    # 使用高斯-勒让德积分法计算整体函数的积分值
    res = quad(whole, [0, inf], [0, inf], [-inf, inf],
               method="gauss-legendre", maxdegree=10)
    return res  # 返回计算得到的结果


# 检查结果字典中是否存在与给定案例匹配的结果
def result_exists(set_key, case):
    """Searches the results dict for a result in the set that matches a case.
    Returns True if such a case exists."""
    if set_key not in res_dict:  # 检查结果字典中是否包含指定的键
        raise ValueError(f"{set_key} not present in data structure!")  # 抛出异常，指定键不存在

    case_dict = to_dict(case)  # 将案例转换为字典形式
    existing_res = list(filter(
        lambda res: res["src_case"] == case_dict,  # 使用lambda函数过滤符合条件的结果
        res_dict[set_key]))  # 从结果字典中筛选出符合条件的结果列表

    return len(existing_res) > 0  # 返回结果是否存在的布尔值


# 运行单个案例，并将计算结果插入到结果字典中
def run(case, run_lambda, set_key, index=0, total_cases=0):
    """Runs the single passed case, returning an mp dictionary and index"""
    t_start = time.perf_counter()  # 记录运行开始时间

    res = run_lambda(case)  # 调用指定的运行函数，获取计算结果

    print(f"Finished {index + 1}/{total_cases} in batch. "
          f"(Took {time.perf_counter() - t_start}s)")  # 打印完成信息和运行时间

    return index, set_key, mp_res_to_dict(MPResult(case, res))  # 返回索引、集合键和结果的字典形式


# 将计算结果写入到文件的回调函数
def write_result(res):
    """A callback for completed jobs. Inserts and writes a calculated result
     to file."""
    index, set_key, result_dict = res  # 解包结果元组

    # 将计算结果插入到结果字典中指定位置
    res_dict[set_key].insert(index, result_dict)

    write_data()  # 将更新后的数据写入文件


# 运行一组案例，并将结果写入文件
def run_cases(cases, run_lambda, set_key):
    """Runs an array of cases and writes to file"""
    # 从尚未生成结果的案例中生成需要运行的作业
    # 本函数未完全注释，但提供了运行一组案例的框架
    # 为每个测试用例生成作业参数列表，包括测试用例本身、运行lambda函数、设置关键字、索引和测试用例总数
    job_arg = [(case, run_lambda, set_key, index, len(cases))
               for index, case in enumerate(cases)
               if not result_exists(set_key, case)]

    # 打印出哪些测试用例的结果已经存在，无需重新计算
    print(f"{len(cases) - len(job_arg)}/{len(cases)} cases won't be "
          f"calculated because their results already exist.")

    # 创建一个空的作业列表和一个进程池对象，用于多进程执行
    jobs = []
    pool = Pool(num_pools)

    # 使用多进程运行所有作业
    for case in job_arg:
        # 将每个作业添加到进程池中，并指定回调函数为write_result
        jobs.append(pool.apply_async(run, args=case, callback=write_result))

    # 关闭进程池，等待所有进程执行完毕
    pool.close()
    pool.join()
# 主程序入口，用于执行各种测试用例并计时
def main():
    # 记录程序开始时间
    t_start = time.perf_counter()

    # 计算总共要处理的测试用例数量
    total_cases = 2 * len(cdf_pdf_cases) + len(moment_cases)
    # 打印总共要处理的测试用例数量信息
    print(f"Processing {total_cases} test cases")

    # 打印第一批测试用例信息，包括 PDF 类型的测试用例数量及其预计执行时间
    print(f"Running 1st batch ({len(cdf_pdf_cases)} PDF cases). "
          f"These take about 30s each.")
    # 执行 PDF 类型的测试用例
    run_cases(cdf_pdf_cases, run_pdf, "pdf_data")

    # 打印第二批测试用例信息，包括 CDF 类型的测试用例数量及其预计执行时间
    print(f"Running 2nd batch ({len(cdf_pdf_cases)} CDF cases). "
          f"These take about 30s each.")
    # 执行 CDF 类型的测试用例
    run_cases(cdf_pdf_cases, run_cdf, "cdf_data")

    # 打印第三批测试用例信息，包括 moment 类型的测试用例数量及其预计执行时间
    print(f"Running 3rd batch ({len(moment_cases)} moment cases). "
          f"These take about anywhere from a few hours to days each.")
    # 执行 moment 类型的测试用例
    run_cases(moment_cases, run_moment, "moment_data")

    # 打印整个测试数据生成所花费的时间
    print(f"Test data generated in {time.perf_counter() - t_start}s")


if __name__ == "__main__":
    # 调用主程序入口函数
    main()
```