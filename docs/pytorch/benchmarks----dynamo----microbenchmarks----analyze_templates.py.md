# `.\pytorch\benchmarks\dynamo\microbenchmarks\analyze_templates.py`

```py
"""
This script uses linear programming to analyze outputs of triton mm config tuning.
To generate output that can be fed into this script set the env varTORCHINDUCTOR_MM_LOGGING_FILE.

That file can be fed into this script to generate the minimizes total, weighted matmul time as a function of allowed templates.
"""
import json  # 导入处理 JSON 数据的模块
import click  # 导入用于创建命令行界面的模块
import pulp   # 导入用于线性规划的库


def parse_log_file(file_path):
    with open(file_path) as f:
        logs = json.load(f)  # 读取并解析 JSON 格式的日志文件

    occurrence_count = {}  # 存储每个形状出现的次数的字典
    benchmark_logs = {}    # 存储基准日志的字典

    # 解析日志文件
    for entry in logs:
        if "invoke" in entry:  # 如果日志条目包含 "invoke" 键
            shape = entry["invoke"]  # 提取出调用的形状
            if shape not in occurrence_count:
                occurrence_count[shape] = 0  # 若形状不在字典中，则初始化其出现次数为 0
            occurrence_count[shape] += 1  # 增加该形状的出现次数计数
        else:
            for shape, timings in entry.items():
                if shape not in benchmark_logs:
                    benchmark_logs[shape] = []  # 若形状不在基准日志字典中，则初始化其对应的时间列表为空
                benchmark_logs[shape].extend(timings)  # 将时间数据扩展到对应形状的时间列表中

    return occurrence_count, benchmark_logs  # 返回形状出现次数字典和基准日志字典


def optimize_templates(N, occurrence_count, benchmark_logs, verbose=False):
    # Set of all possible Triton templates keyed by their attributes
    triton_templates = set()  # 使用集合存储所有 Triton 模板的唯一属性组合
    for timings in benchmark_logs.values():
        for timing in timings:
            if timing["type"] == "triton":  # 如果时间记录的类型是 Triton
                triton_templates.add(
                    (
                        timing["BLOCK_M"],
                        timing["BLOCK_N"],
                        timing["BLOCK_K"],
                        timing["num_stages"],
                        timing["num_warps"],
                    )
                )  # 将 Triton 模板的属性组合添加到集合中

    # Print the initial data
    if verbose:
        print("Occurrence Count:", occurrence_count)  # 输出形状出现次数信息
        print("Triton Templates:", triton_templates)  # 输出 Triton 模板信息

    # Create a dictionary to store template selection variables
    template_vars = {
        template: pulp.LpVariable(f"Template_{template}", 0, 1, pulp.LpBinary)
        for template in triton_templates
    }  # 创建用于存储模板选择变量的字典，每个模板对应一个二进制线性规划变量

    # Variables to select specific timing option for each shape
    selection_vars = {
        (shape, "cublas"): pulp.LpVariable(
            f"Select_{shape}_cublas", 0, 1, pulp.LpBinary
        )
        for shape in occurrence_count
    }  # 创建用于每个形状选择特定计时选项的变量字典，包括使用 cuBLAS 的二进制变量

    for shape in occurrence_count:
        for template in triton_templates:
            selection_vars[(shape, template)] = pulp.LpVariable(
                f"Select_{shape}_{template}", 0, 1, pulp.LpBinary
            )  # 为每个形状和每个 Triton 模板创建二进制变量，用于选择特定的计时选项

    # Variables for the total time for each shape
    min_time_vars = pulp.LpVariable.dicts(
        "MinTime", occurrence_count.keys(), 0, None, pulp.LpContinuous
    )  # 创建用于每个形状的总时间的连续线性规划变量

    # Define the problem
    prob = pulp.LpProblem("MatrixMultiplicationOptimization", pulp.LpMinimize)  # 创建一个最小化问题实例

    # Objective: Minimize the weighted total time
    prob += pulp.lpSum(
        [occurrence_count[shape] * min_time_vars[shape] for shape in occurrence_count]
    )  # 定义目标函数，最小化加权总时间

    # Constraints to select exactly N templates
    # Add constraint: Sum of template variables must equal N
    prob += pulp.lpSum([template_vars[template] for template in triton_templates]) == N

    # Initialize an empty dictionary to store Triton options per shape for debugging
    triton_options_per_shape = {}

    # Iterate over each shape to impose constraints on total time
    for shape in occurrence_count:
        # Extract cuBLAS times for the current shape
        cublas_times = [
            timing["time"]
            for timing in benchmark_logs[shape]
            if timing["type"] == "cublas"
        ]
        # Find the minimum cuBLAS time for the current shape
        min_cublas_time = min(cublas_times)

        # Collect Triton options for the current shape
        triton_options = []
        for template in triton_templates:
            # Extract Triton times matching specific template configurations for the current shape
            triton_times = [
                timing["time"]
                for timing in benchmark_logs[shape]
                if timing["type"] == "triton"
                and (
                    timing["BLOCK_M"],
                    timing["BLOCK_N"],
                    timing["BLOCK_K"],
                    timing["num_stages"],
                    timing["num_warps"],
                )
                == template
            ]
            # If there are Triton times available, find the minimum and store the template
            if triton_times:
                min_triton_time = min(triton_times)
                triton_options.append((min_triton_time, template))

        # Save Triton options for the current shape in a dictionary
        triton_options_per_shape[shape] = triton_options

        # Ensure exactly one timing option is selected for each shape
        prob += (
            pulp.lpSum(
                [selection_vars[(shape, "cublas")]]
                + [
                    selection_vars[(shape, template)]
                    for triton_time, template in triton_options
                ]
            )
            == 1
        )

        # Ensure that min_time_vars[shape] matches the selected timing option
        prob += min_time_vars[shape] == (
            selection_vars[(shape, "cublas")] * min_cublas_time
            + pulp.lpSum(
                [
                    selection_vars[(shape, template)] * triton_time
                    for triton_time, template in triton_options
                ]
            )
        )

        # Ensure Triton templates can only be selected if they are within the allowed N templates
        for triton_time, template in triton_options:
            prob += selection_vars[(shape, template)] <= template_vars[template]

    # Print the constraints if verbose mode is enabled
    if verbose:
        print("Constraints:")
        for constraint in prob.constraints.values():
            print(constraint)

    # Solve the optimization problem with suppressed output
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Retrieve selected templates that have been assigned a value of 1
    selected_templates = [
        template
        for template in triton_templates
        if pulp.value(template_vars[template]) == 1
    ]

    # Calculate the total time by summing up minimized times for all shapes
    total_time = sum(
        pulp.value(min_time_vars[shape]) * occurrence_count[shape]
        for shape in occurrence_count
    )

    # Print the values of the decision variables after solving
    # 如果 verbose 参数为 True，则打印决策变量的取值
    if verbose:
        print("Decision Variable Values:")
        # 遍历优化问题的所有变量并打印其名称和取值
        for var in prob.variables():
            print(f"{var.name} = {var.varValue}")

    # 如果 verbose 参数为 True，则打印调试信息
    if verbose:
        # 遍历每个形状 (shape) 在 occurrence_count 字典中
        for shape in occurrence_count:
            print(f"Shape: {shape}")
            # 打印当前形状对应的最小时间 (min_time_vars[shape])
            print(f"  Min Time: {pulp.value(min_time_vars[shape])}")
            # 打印当前形状在优化问题中出现的次数 (occurrence_count[shape])
            print(f"  Occurrences: {occurrence_count[shape]}")
            # 打印使用 CuBLAS 的最小时间和选择的情况
            print(f"  Min CuBLAS Time: {min_cublas_time} Selected: {pulp.value(selection_vars[(shape, 'cublas')])}")
            # 遍历每个 Triton 模板及其时间，并打印选择情况
            for triton_time, template in triton_options_per_shape[shape]:
                print(f"  Triton Template: {template} Time: {triton_time} Selected: {pulp.value(selection_vars[(shape, template)])}")

    # 返回选择的模板和总时间
    return selected_templates, total_time
# 主函数，用于解析日志文件并优化模板
@click.command()
@click.argument("filename")  # 接受一个文件名作为参数
@click.option("--min-templates", default=0, help="最少模板数量")  # 最少模板数量选项，默认为0
@click.option("--max-templates", default=10, help="最多模板数量")  # 最多模板数量选项，默认为10
@click.option("--verbose", is_flag=True, help="启用详细输出")  # 是否启用详细输出的标志
def main(filename, min_templates, max_templates, verbose):
    # 解析日志文件，获取事件发生次数和基准日志数据
    occurrence_count, benchmark_logs = parse_log_file(filename)
    # 存储每个模板数量对应的总时间
    times = []
    # 遍历模板数量范围[min_templates, max_templates]
    for N in range(min_templates, max_templates + 1):
        # 优化模板选择，并计算总耗时
        selected_templates, total_time = optimize_templates(
            N, occurrence_count, benchmark_logs, verbose
        )
        # 打印当前模板数量 N
        print(f"N = {N}")
        # 打印所选模板列表
        print(f"Selected Templates: {selected_templates}")
        # 打印总加权时间
        print(f"Total Weighted Time: {total_time}")
        # 将总时间添加到 times 列表中
        times.append(total_time)
    # 打印所有模板数量对应的总时间列表
    print(times)


if __name__ == "__main__":
    main()
```