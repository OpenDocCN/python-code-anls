# `basic-computer-games\90_Tower\csharp\Models\Needle.cs`

```
// 引入命名空间
using System.Collections;
using System.Collections.Generic;
using System.Linq;

// 在 Tower.Models 命名空间下定义 Needle 类
namespace Tower.Models
{
    // 定义 Needle 类实现 IEnumerable 接口
    internal class Needle : IEnumerable<int>
    {
        // 创建一个只读的整型栈 _disks
        private readonly Stack<int> _disks = new Stack<int>();

        // 判断栈是否为空的属性
        public bool IsEmpty => _disks.Count == 0;

        // 获取栈顶元素的属性
        public int Top => _disks.TryPeek(out var disk) ? disk : default;

        // 尝试将元素放入栈中的方法
        public bool TryPut(int disk)
        {
            // 如果栈为空或者待放入元素小于栈顶元素，则放入栈中并返回 true
            if (_disks.Count == 0 || disk < _disks.Peek())
            {
                _disks.Push(disk);
                return true;
            }
            // 否则返回 false
            return false;
        }

        // 尝试获取栈顶元素并弹出的方法
        public bool TryGetTopDisk(out int disk) => _disks.TryPop(out disk);

        // 实现 IEnumerable 接口的 GetEnumerator 方法
        public IEnumerator<int> GetEnumerator() =>
            // 返回一个迭代器，先生成指定数量的 0，再连接栈中的元素
            Enumerable.Repeat(0, 7 - _disks.Count).Concat(_disks).GetEnumerator();

        // 实现 IEnumerable 接口的非泛型 GetEnumerator 方法
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
```