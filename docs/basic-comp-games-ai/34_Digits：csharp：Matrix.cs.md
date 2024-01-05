# `34_Digits\csharp\Matrix.cs`

```
namespace Digits;

internal class Matrix
{
    private readonly int _weight; // 用于存储矩阵的权重
    private readonly int[,] _values; // 用于存储矩阵的值

    public Matrix(int width, int weight, Func<int, int, int> seedFactory)
    {
        _weight = weight; // 初始化矩阵的权重
        _values = new int[width, 3]; // 初始化矩阵的值为指定宽度和3列

        for (int i = 0; i < width; i++) // 循环遍历矩阵的宽度
            for (int j = 0; j < 3; j++) // 循环遍历矩阵的3列
            {
                _values[i, j] = seedFactory.Invoke(i, j); // 使用seedFactory函数生成矩阵的值
            }

        Index = width - 1; // 设置索引为宽度减一
    }
```
```csharp
# 定义一个公共属性 Index，用于存储和获取索引值
public int Index { get; set; }

# 定义一个方法 GetWeightedValue，用于返回加权值，参数为行数，使用 Index 属性作为索引，获取 _values 数组中的值并乘以 _weight
public int GetWeightedValue(int row) => _weight * _values[Index, row];

# 定义一个方法 IncrementValue，用于增加值，参数为行数，使用 Index 属性作为索引，获取 _values 数组中的值并增加 1
public int IncrementValue(int row) => _values[Index, row]++;
```