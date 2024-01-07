# `basic-computer-games\77_Salvo\csharp\Targetting\SearchPattern.cs`

```

// 使用不可变数组存储偏移量
using System.Collections.Immutable;

namespace Salvo.Targetting;

// 定义搜索模式类
internal class SearchPattern
{
    // 静态只读字段，存储偏移量数组
    private static readonly ImmutableArray<Offset> _offsets =
        ImmutableArray.Create<Offset>(new(1, 1), new(-1, 1), new(1, -3), new(1, 1), new(0, 2), new(-1, 1));

    // 下一个偏移量的索引
    private int _nextIndex;

    // 尝试获取下一个偏移量
    internal bool TryGetOffset(out Offset offset)
    {
        offset = default;
        // 如果下一个偏移量的索引超出数组长度，则返回false
        if (_nextIndex >= _offsets.Length) { return false; }
        
        // 获取下一个偏移量，并将索引加1
        offset = _offsets[_nextIndex++];
        return true;
    }

    // 重置偏移量索引
    internal void Reset() => _nextIndex = 0;
}

```