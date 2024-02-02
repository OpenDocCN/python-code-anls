# `basic-computer-games\77_Salvo\csharp\Targetting\SearchPattern.cs`

```py
// 使用不可变数组存储搜索模式的偏移量
using System.Collections.Immutable;

namespace Salvo.Targetting
{
    // 定义搜索模式类
    internal class SearchPattern
    {
        // 使用不可变数组存储预定义的偏移量
        private static readonly ImmutableArray<Offset> _offsets =
            ImmutableArray.Create<Offset>(new(1, 1), new(-1, 1), new(1, -3), new(1, 1), new(0, 2), new(-1, 1));

        // 下一个偏移量的索引
        private int _nextIndex;

        // 尝试获取下一个偏移量
        internal bool TryGetOffset(out Offset offset)
        {
            offset = default;
            // 如果索引超出偏移量数组长度，则返回 false
            if (_nextIndex >= _offsets.Length) { return false; }
            
            // 获取下一个偏移量并增加索引
            offset = _offsets[_nextIndex++];
            return true;
        }

        // 重置偏移量数组的索引
        internal void Reset() => _nextIndex = 0;
    }
}
```