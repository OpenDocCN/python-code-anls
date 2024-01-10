# `basic-computer-games\34_Digits\csharp\Matrix.cs`

```
// 命名空间为Digits，表示这个类在Digits命名空间下
namespace Digits;

// 定义一个内部类Matrix
internal class Matrix
{
    // 用于存储权重的私有只读字段
    private readonly int _weight;
    // 用于存储数值的私有只读二维数组
    private readonly int[,] _values;

    // 构造函数，接受宽度、权重和种子工厂函数作为参数
    public Matrix(int width, int weight, Func<int, int, int> seedFactory)
    {
        // 初始化权重
        _weight = weight;
        // 初始化数值数组
        _values = new int[width, 3];
        
        // 循环遍历数组，使用种子工厂函数填充数值数组
        for (int i = 0; i < width; i++)
        for (int j = 0; j < 3; j++)
        {
            _values[i, j] = seedFactory.Invoke(i, j);
        }

        // 设置索引为宽度减一
        Index = width - 1;
    }

    // 公共属性，用于获取和设置索引值
    public int Index { get; set; }

    // 公共方法，用于获取加权值
    public int GetWeightedValue(int row) => _weight * _values[Index, row];

    // 公共方法，用于增加数值并返回增加后的值
    public int IncrementValue(int row) => _values[Index, row]++;
}
```