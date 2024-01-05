# `77_Salvo\csharp\Targetting\SearchPattern.cs`

```
using System.Collections.Immutable;  // 导入不可变集合的命名空间

namespace Salvo.Targetting;  // 声明命名空间 Salvo.Targetting

internal class SearchPattern  // 声明内部类 SearchPattern
{
    private static readonly ImmutableArray<Offset> _offsets =  // 声明不可变集合 _offsets，存储 Offset 类型的元素
        ImmutableArray.Create<Offset>(new(1, 1), new(-1, 1), new(1, -3), new(1, 1), new(0, 2), new(-1, 1));  // 初始化 _offsets

    private int _nextIndex;  // 声明私有整型变量 _nextIndex

    internal bool TryGetOffset(out Offset offset)  // 声明内部方法 TryGetOffset，返回布尔值，参数为输出参数 offset
    {
        offset = default;  // 将 offset 初始化为默认值
        if (_nextIndex >= _offsets.Length) { return false; }  // 如果 _nextIndex 大于等于 _offsets 的长度，则返回 false
        
        offset = _offsets[_nextIndex++];  // 将 _offsets 中的元素赋值给 offset，并将 _nextIndex 自增
        return true;  // 返回 true
    }
}
    # 重置 _nextIndex 变量的值为 0
    def reset(self):
        self._nextIndex = 0
```