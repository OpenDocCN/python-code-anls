# `90_Tower\csharp\Models\Towers.cs`

```
using System;  // 导入 System 命名空间
using System.Collections;  // 导入 System.Collections 命名空间
using System.Collections.Generic;  // 导入 System.Collections.Generic 命名空间
using System.Linq;  // 导入 System.Linq 命名空间
using Tower.Resources;  // 导入 Tower.Resources 命名空间

namespace Tower.Models  // 定义 Tower.Models 命名空间
{
    internal class Towers : IEnumerable<(int, int, int)>  // Towers 类实现 IEnumerable 接口
    {
        private static int[] _availableDisks = new[] { 15, 13, 11, 9, 7, 5, 3 };  // 定义静态整型数组 _availableDisks

        private readonly Needle[] _needles = new[] { new Needle(), new Needle(), new Needle() };  // 定义只读 Needle 数组 _needles
        private readonly int _smallestDisk;  // 定义只读整型变量 _smallestDisk

        public Towers(int diskCount)  // Towers 类的构造函数，参数为 diskCount
        {
            foreach (int disk in _availableDisks.Take(diskCount))  // 遍历 _availableDisks 数组中的元素
            {
                this[1].TryPut(disk);  // 调用 this[1] 的 TryPut 方法，将 disk 放入指定位置
                _smallestDisk = disk;  # 将参数 disk 的值赋给 _smallestDisk 变量
            }
        }

        private Needle this[int i] => _needles[i-1];  # 索引器，根据索引 i 返回 _needles 列表中对应位置的元素

        public bool Finished => this[1].IsEmpty && this[2].IsEmpty;  # 属性，判断第一个和第二个元素是否为空，返回布尔值

        public bool TryFindDisk(int disk, out int needle, out string message)  # 方法，尝试查找磁盘
        {
            needle = default;  # 将 needle 变量的默认值赋给参数 needle
            message = default;  # 将 message 变量的默认值赋给参数 message

            if (disk < _smallestDisk)  # 如果参数 disk 的值小于 _smallestDisk 的值
            {
                message = Strings.DiskNotInPlay;  # 将指定字符串赋给 message 变量
                return false;  # 返回 false
            }

            for (needle = 1; needle <= 3; needle++)  # 循环，遍历 needle 变量从 1 到 3 的值
            {
                if (this[needle].Top == disk) { return true; }  // 如果当前针的顶部磁盘等于要移动的磁盘，则返回true
            }

            message = Strings.DiskUnavailable;  // 将消息设置为"磁盘不可用"
            return false;  // 返回false
        }

        public bool TryMoveDisk(int from, int to)
        {
            if (!this[from].TryGetTopDisk(out var disk))  // 如果针from的顶部没有磁盘，则抛出异常
            {
                throw new InvalidOperationException($"Needle {from} is empty");
            }

            if (this[to].TryPut(disk)) { return true; }  // 如果能够将磁盘放入针to，则返回true

            this[from].TryPut(disk);  // 尝试将磁盘放回针from
            return false;  // 返回false
        }
        public IEnumerator<(int, int, int)> GetEnumerator() => new TowersEnumerator(_needles);
        # 返回一个枚举器，用于遍历 Towers 对象中的元素

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        # 实现非泛型枚举器的显式接口成员，返回一个枚举器，用于遍历 Towers 对象中的元素

        private class TowersEnumerator : IEnumerator<(int, int, int)>
        {
            private readonly List<IEnumerator<int>> _enumerators;
            # 声明一个私有的 List 对象 _enumerators，用于存储整数类型的枚举器

            public TowersEnumerator(Needle[] needles)
            {
                _enumerators = needles.Select(n => n.GetEnumerator()).ToList();
                # 构造函数，将 Needle 数组中每个元素的枚举器存储到 _enumerators 中
            }

            public (int, int, int) Current =>
                (_enumerators[0].Current, _enumerators[1].Current, _enumerators[2].Current);
            # 获取当前位置的元素，返回一个包含三个整数的元组

            object IEnumerator.Current => Current;
            # 获取当前位置的元素

            public void Dispose() => _enumerators.ForEach(e => e.Dispose());
            # 释放 TowersEnumerator 对象中的资源
# 定义一个公共类，实现 IEnumerator 接口
public class MultiEnumerator<T> : IEnumerator<T>
{
    private readonly IEnumerator<T>[] _enumerators;  # 声明一个私有的泛型数组 _enumerators，用于存储多个枚举器

    # 构造函数，接受一个泛型数组参数
    public MultiEnumerator(params IEnumerator<T>[] enumerators)
    {
        _enumerators = enumerators;  # 将传入的枚举器数组赋值给私有的 _enumerators
    }

    # 实现 IEnumerator 接口的 Current 属性
    public T Current => _enumerators.Select(e => e.Current).First();

    # 实现 IEnumerator 接口的 Current 属性
    object IEnumerator.Current => Current;

    # 实现 IEnumerator 接口的 Dispose 方法
    public void Dispose() => _enumerators.ForEach(e => e.Dispose());

    # 实现 IEnumerator 接口的 MoveNext 方法
    public bool MoveNext() => _enumerators.All(e => e.MoveNext());

    # 实现 IEnumerator 接口的 Reset 方法
    public void Reset() => _enumerators.ForEach(e => e.Reset());
}
```