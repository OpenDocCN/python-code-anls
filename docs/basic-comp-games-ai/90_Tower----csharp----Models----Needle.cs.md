# `basic-computer-games\90_Tower\csharp\Models\Needle.cs`

```

// 引入命名空间
using System.Collections;
using System.Collections.Generic;
using System.Linq;

// 在 Tower.Models 命名空间下定义一个名为 Needle 的内部类，实现 IEnumerable 接口
namespace Tower.Models
{
    internal class Needle : IEnumerable<int>
    {
        // 使用栈来存储整数类型的数据
        private readonly Stack<int> _disks = new Stack<int>();

        // 判断栈是否为空的属性
        public bool IsEmpty => _disks.Count == 0;

        // 获取栈顶元素的属性
        public int Top => _disks.TryPeek(out var disk) ? disk : default;

        // 尝试将一个整数类型的数据放入栈中
        public bool TryPut(int disk)
        {
            if (_disks.Count == 0 || disk < _disks.Peek())
            {
                _disks.Push(disk);
                return true;
            }

            return false;
        }

        // 尝试从栈中取出栈顶元素
        public bool TryGetTopDisk(out int disk) => _disks.TryPop(out disk);

        // 实现 IEnumerable 接口的 GetEnumerator 方法
        public IEnumerator<int> GetEnumerator() =>
            // 生成一个包含指定元素的序列，然后连接到栈中的元素，返回迭代器
            Enumerable.Repeat(0, 7 - _disks.Count).Concat(_disks).GetEnumerator();

        // 实现 IEnumerable 接口的 GetEnumerator 方法
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}

```