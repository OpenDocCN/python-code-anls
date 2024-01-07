# `basic-computer-games\90_Tower\csharp\Models\Towers.cs`

```

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Tower.Resources;

namespace Tower.Models
{
    internal class Towers : IEnumerable<(int, int, int)>
    {
        private static int[] _availableDisks = new[] { 15, 13, 11, 9, 7, 5, 3 };

        private readonly Needle[] _needles = new[] { new Needle(), new Needle(), new Needle() };
        private readonly int _smallestDisk;

        public Towers(int diskCount)
        {
            // 初始化塔的构造函数，根据给定的盘子数量放置盘子到第一个针上
            foreach (int disk in _availableDisks.Take(diskCount))
            {
                this[1].TryPut(disk);
                _smallestDisk = disk;
            }
        }

        // 获取指定索引的针
        private Needle this[int i] => _needles[i-1];

        // 判断游戏是否结束
        public bool Finished => this[1].IsEmpty && this[2].IsEmpty;

        // 尝试在针上找到指定大小的盘子
        public bool TryFindDisk(int disk, out int needle, out string message)
        {
            needle = default;
            message = default;

            if (disk < _smallestDisk)
            {
                message = Strings.DiskNotInPlay;
                return false;
            }

            for (needle = 1; needle <= 3; needle++)
            {
                if (this[needle].Top == disk) { return true; }
            }

            message = Strings.DiskUnavailable;
            return false;
        }

        // 尝试移动盘子
        public bool TryMoveDisk(int from, int to)
        {
            if (!this[from].TryGetTopDisk(out var disk))
            {
                throw new InvalidOperationException($"Needle {from} is empty");
            }

            if (this[to].TryPut(disk)) { return true; }

            this[from].TryPut(disk);
            return false;
        }

        // 实现 IEnumerable 接口的方法，用于迭代器
        public IEnumerator<(int, int, int)> GetEnumerator() => new TowersEnumerator(_needles);

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        // 内部类，用于实现迭代器
        private class TowersEnumerator : IEnumerator<(int, int, int)>
        {
            private readonly List<IEnumerator<int>> _enumerators;

            public TowersEnumerator(Needle[] needles)
            {
                _enumerators = needles.Select(n => n.GetEnumerator()).ToList();
            }

            public (int, int, int) Current =>
                (_enumerators[0].Current, _enumerators[1].Current, _enumerators[2].Current);

            object IEnumerator.Current => Current;

            public void Dispose() => _enumerators.ForEach(e => e.Dispose());

            public bool MoveNext() => _enumerators.All(e => e.MoveNext());

            public void Reset() => _enumerators.ForEach(e => e.Reset());
        }
    }
}

```