# `.\pytorch\benchmarks\distributed\rpc\parameter_server\metrics\ProcessedMetricsPrinter.py`

```
import statistics  # 导入 statistics 库，用于计算统计指标

import pandas as pd  # 导入 pandas 库，用于数据处理和分析
from tabulate import tabulate  # 导入 tabulate 库，用于以表格形式打印数据


class ProcessedMetricsPrinter:
    def print_data_frame(self, name, processed_metrics):
        # 打印给定名称的指标数据
        print(f"metrics for {name}")
        # 获取数据框
        data_frame = self.get_data_frame(processed_metrics)
        # 使用 tabulate 库以网格形式打印数据框
        print(
            tabulate(
                data_frame, showindex=False, headers=data_frame.columns, tablefmt="grid"
            )
        )

    def combine_processed_metrics(self, processed_metrics_list):
        r"""
        A method that merges the value arrays of the keys in the dictionary
        of processed metrics.

        Args:
            processed_metrics_list (list): a list containing dictionaries with
                recorded metrics as keys, and the values are lists of elapsed times.

        Returns::
            A merged dictionary that is created from the list of dictionaries passed
                into the method.

        Examples::
            >>> instance = ProcessedMetricsPrinter()
            >>> dict_1 = trainer1.get_processed_metrics()
            >>> dict_2 = trainer2.get_processed_metrics()
            >>> print(dict_1)
            {
                "forward_metric_type,forward_pass" : [.0429, .0888]
            }
            >>> print(dict_2)
            {
                "forward_metric_type,forward_pass" : [.0111, .0222]
            }
            >>> processed_metrics_list = [dict_1, dict_2]
            >>> result = instance.combine_processed_metrics(processed_metrics_list)
            >>> print(result)
            {
                "forward_metric_type,forward_pass" : [.0429, .0888, .0111, .0222]
            }
        """
        processed_metric_totals = {}
        # 遍历每个传入的处理后指标字典
        for processed_metrics in processed_metrics_list:
            # 遍历每个指标及其对应的值数组
            for metric_name, values in processed_metrics.items():
                # 如果指标名不在总计字典中，则添加空列表
                if metric_name not in processed_metric_totals:
                    processed_metric_totals[metric_name] = []
                # 将当前指标的值数组累加到总计字典中
                processed_metric_totals[metric_name] += values
        # 返回合并后的总计字典
        return processed_metric_totals

    def get_data_frame(self, processed_metrics):
        # 创建一个空的 pandas 数据框，列名为 name, min, max, mean, variance, stdev
        df = pd.DataFrame(columns=["name", "min", "max", "mean", "variance", "stdev"])
        # 对处理后的指标字典中的每个指标名进行排序处理
        for metric_name in sorted(processed_metrics.keys()):
            values = processed_metrics[metric_name]
            # 计算指标值的统计指标，并构建数据框的一行
            row = {
                "name": metric_name,
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "variance": statistics.variance(values),
                "stdev": statistics.stdev(values),
            }
            # 将当前行添加到数据框中，忽略索引
            df = df.append(row, ignore_index=True)
        # 返回构建好的数据框
        return df
    # 定义一个方法用于打印指标数据，接受名称和排名指标列表作为参数
    def print_metrics(self, name, rank_metrics_list):
        # 如果排名指标列表不为空，则执行以下操作
        if rank_metrics_list:
            # 初始化一个空列表用于存储指标数据
            metrics_list = []
            # 遍历排名指标列表中的每个排名和指标元组
            for rank, metric in rank_metrics_list:
                # 调用实例方法 print_data_frame，打印格式化后的名称和对应的指标数据
                self.print_data_frame(f"{name}={rank}", metric)
                # 将当前指标数据添加到 metrics_list 中
                metrics_list.append(metric)
            # 调用实例方法 combine_processed_metrics，将所有指标数据进行合并处理
            combined_metrics = self.combine_processed_metrics(metrics_list)
            # 再次调用 print_data_frame 方法，打印所有指标数据的汇总
            self.print_data_frame(f"all {name}", combined_metrics)

    # 定义一个方法用于将数据框保存到文件中，接受数据框和文件名作为参数
    def save_to_file(self, data_frame, file_name):
        # 构造完整的文件路径，文件名为 data_frames/{file_name}.csv
        file_name = f"data_frames/{file_name}.csv"
        # 调用数据框的 to_csv 方法，将数据框保存为 CSV 文件，使用 UTF-8 编码，不保存索引列
        data_frame.to_csv(file_name, encoding="utf-8", index=False)
```