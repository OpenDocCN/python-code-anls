# `.\yolov8\ultralytics\hub\google\__init__.py`

```py
# 导入所需的库和模块
import concurrent.futures  # 用于并发执行任务
import statistics  # 提供统计函数，如计算均值、中位数等
import time  # 提供时间相关的功能，如睡眠、计时等
from typing import List, Optional, Tuple  # 导入类型提示相关的模块

import requests  # 提供进行 HTTP 请求的功能


class GCPRegions:
    """
    A class for managing and analyzing Google Cloud Platform (GCP) regions.

    This class provides functionality to initialize, categorize, and analyze GCP regions based on their
    geographical location, tier classification, and network latency.

    Attributes:
        regions (Dict[str, Tuple[int, str, str]]): A dictionary of GCP regions with their tier, city, and country.

    Methods:
        tier1: Returns a list of tier 1 GCP regions.
        tier2: Returns a list of tier 2 GCP regions.
        lowest_latency: Determines the GCP region(s) with the lowest network latency.

    Examples:
        >>> from ultralytics.hub.google import GCPRegions
        >>> regions = GCPRegions()
        >>> lowest_latency_region = regions.lowest_latency(verbose=True, attempts=3)
        >>> print(f"Lowest latency region: {lowest_latency_region[0][0]}")
    """
    def __init__(self):
        """Initializes the GCPRegions class with predefined Google Cloud Platform regions and their details."""
        # 定义包含各个谷歌云平台地区及其详细信息的字典
        self.regions = {
            "asia-east1": (1, "Taiwan", "China"),
            "asia-east2": (2, "Hong Kong", "China"),
            "asia-northeast1": (1, "Tokyo", "Japan"),
            "asia-northeast2": (1, "Osaka", "Japan"),
            "asia-northeast3": (2, "Seoul", "South Korea"),
            "asia-south1": (2, "Mumbai", "India"),
            "asia-south2": (2, "Delhi", "India"),
            "asia-southeast1": (2, "Jurong West", "Singapore"),
            "asia-southeast2": (2, "Jakarta", "Indonesia"),
            "australia-southeast1": (2, "Sydney", "Australia"),
            "australia-southeast2": (2, "Melbourne", "Australia"),
            "europe-central2": (2, "Warsaw", "Poland"),
            "europe-north1": (1, "Hamina", "Finland"),
            "europe-southwest1": (1, "Madrid", "Spain"),
            "europe-west1": (1, "St. Ghislain", "Belgium"),
            "europe-west10": (2, "Berlin", "Germany"),
            "europe-west12": (2, "Turin", "Italy"),
            "europe-west2": (2, "London", "United Kingdom"),
            "europe-west3": (2, "Frankfurt", "Germany"),
            "europe-west4": (1, "Eemshaven", "Netherlands"),
            "europe-west6": (2, "Zurich", "Switzerland"),
            "europe-west8": (1, "Milan", "Italy"),
            "europe-west9": (1, "Paris", "France"),
            "me-central1": (2, "Doha", "Qatar"),
            "me-west1": (1, "Tel Aviv", "Israel"),
            "northamerica-northeast1": (2, "Montreal", "Canada"),
            "northamerica-northeast2": (2, "Toronto", "Canada"),
            "southamerica-east1": (2, "São Paulo", "Brazil"),
            "southamerica-west1": (2, "Santiago", "Chile"),
            "us-central1": (1, "Iowa", "United States"),
            "us-east1": (1, "South Carolina", "United States"),
            "us-east4": (1, "Northern Virginia", "United States"),
            "us-east5": (1, "Columbus", "United States"),
            "us-south1": (1, "Dallas", "United States"),
            "us-west1": (1, "Oregon", "United States"),
            "us-west2": (2, "Los Angeles", "United States"),
            "us-west3": (2, "Salt Lake City", "United States"),
            "us-west4": (2, "Las Vegas", "United States"),
        }

    def tier1(self) -> List[str]:
        """Returns a list of GCP regions classified as tier 1 based on predefined criteria."""
        # 返回符合预定义标准的属于第一层级的谷歌云平台地区列表
        return [region for region, info in self.regions.items() if info[0] == 1]

    def tier2(self) -> List[str]:
        """Returns a list of GCP regions classified as tier 2 based on predefined criteria."""
        # 返回符合预定义标准的属于第二层级的谷歌云平台地区列表
        return [region for region, info in self.regions.items() if info[0] == 2]

    @staticmethod
    def _ping_region(region: str, attempts: int = 1) -> Tuple[str, float, float, float, float]:
        """Pings a specified GCP region and returns latency statistics: mean, min, max, and standard deviation."""
        # 构建请求的 URL，使用指定的 GCP 地区
        url = f"https://{region}-docker.pkg.dev"
        # 存储每次请求的延迟时间
        latencies = []
        # 尝试多次请求
        for _ in range(attempts):
            try:
                # 记录请求开始时间
                start_time = time.time()
                # 发送 HEAD 请求到指定 URL，设置超时时间为 5 秒
                _ = requests.head(url, timeout=5)
                # 计算请求完成后的延迟时间（毫秒）
                latency = (time.time() - start_time) * 1000  # convert latency to milliseconds
                # 如果延迟时间不是无穷大，则添加到延迟时间列表中
                if latency != float("inf"):
                    latencies.append(latency)
            except requests.RequestException:
                pass
        # 如果未成功获取任何延迟数据，则返回无穷大的统计数据
        if not latencies:
            return region, float("inf"), float("inf"), float("inf"), float("inf")

        # 计算延迟时间的标准差，如果样本数大于1
        std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
        # 返回地区名称及其延迟统计数据：平均值、标准差、最小值、最大值
        return region, statistics.mean(latencies), std_dev, min(latencies), max(latencies)

    def lowest_latency(
        self,
        top: int = 1,
        verbose: bool = False,
        tier: Optional[int] = None,
        attempts: int = 1,
    # 返回一个列表，包含元组，每个元组代表 GCP 地区的延迟统计信息
    # 每个元组包含 (地区名, 平均延迟, 标准差, 最小延迟, 最大延迟)
    def lowest_latency(self, top: int, verbose: bool, tier: Optional[int], attempts: int) -> List[Tuple[str, float, float, float, float]]:
        """
        Determines the GCP regions with the lowest latency based on ping tests.

        Args:
            top (int): Number of top regions to return.
            verbose (bool): If True, prints detailed latency information for all tested regions.
            tier (int | None): Filter regions by tier (1 or 2). If None, all regions are tested.
            attempts (int): Number of ping attempts per region.

        Returns:
            (List[Tuple[str, float, float, float, float]]): List of tuples containing region information and
            latency statistics. Each tuple contains (region, mean_latency, std_dev, min_latency, max_latency).

        Examples:
            >>> regions = GCPRegions()
            >>> results = regions.lowest_latency(top=3, verbose=True, tier=1, attempts=2)
            >>> print(results[0][0])  # Print the name of the lowest latency region
        """
        # 如果 verbose 为 True，打印正在进行的 ping 测试信息
        if verbose:
            print(f"Testing GCP regions for latency (with {attempts} {'retry' if attempts == 1 else 'attempts'})...")

        # 根据 tier 条件过滤要测试的地区列表
        regions_to_test = [k for k, v in self.regions.items() if v[0] == tier] if tier else list(self.regions.keys())
        
        # 使用 ThreadPoolExecutor 并发执行 ping 测试
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            results = list(executor.map(lambda r: self._ping_region(r, attempts), regions_to_test))

        # 根据平均延迟对结果进行排序
        sorted_results = sorted(results, key=lambda x: x[1])

        # 如果 verbose 为 True，打印详细的延迟信息表格
        if verbose:
            print(f"{'Region':<25} {'Location':<35} {'Tier':<5} {'Latency (ms)'}")
            for region, mean, std, min_, max_ in sorted_results:
                tier, city, country = self.regions[region]
                location = f"{city}, {country}"
                if mean == float("inf"):
                    print(f"{region:<25} {location:<35} {tier:<5} {'Timeout'}")
                else:
                    print(f"{region:<25} {location:<35} {tier:<5} {mean:.0f} ± {std:.0f} ({min_:.0f} - {max_:.0f})")
            print(f"\nLowest latency region{'s' if top > 1 else ''}:")
            for region, mean, std, min_, max_ in sorted_results[:top]:
                tier, city, country = self.regions[region]
                location = f"{city}, {country}"
                print(f"{region} ({location}, {mean:.0f} ± {std:.0f} ms ({min_:.0f} - {max_:.0f}))")

        # 返回延迟最低的前 top 个地区的信息列表
        return sorted_results[:top]
# 如果脚本被直接执行（而不是被导入为模块），则执行以下代码
if __name__ == "__main__":
    # 创建一个 GCPRegions 的实例对象
    regions = GCPRegions()
    # 调用 lowest_latency 方法来获取最低延迟的地区列表
    # 参数解释：
    #   top=3: 获取延迟最低的前三个地区
    #   verbose=True: 打印详细信息，例如每次尝试的信息
    #   tier=1: 限定在第一层次的数据中进行选择
    #   attempts=3: 尝试获取数据的最大次数
    top_3_latency_tier1 = regions.lowest_latency(top=3, verbose=True, tier=1, attempts=3)
```