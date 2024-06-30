# `D:\src\scipysrc\seaborn\seaborn\colors\xkcd_rgb.py`

```
# 定义一个名为 find_missing 的函数，接收两个参数，一个是整数列表 nums1，另一个是整数列表 nums2
def find_missing(nums1, nums2):
    # 将 nums1 转换为集合，得到 nums1_set，用于快速查找元素
    nums1_set = set(nums1)
    # 将 nums2 转换为集合，得到 nums2_set，用于快速查找元素
    nums2_set = set(nums2)
    # 在 nums1_set 中查找不在 nums2_set 中的元素，得到结果列表
    missing = list(nums1_set - nums2_set)
    # 返回找到的缺失元素列表
    return missing
```