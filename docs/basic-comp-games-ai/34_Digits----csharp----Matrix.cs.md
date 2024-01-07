# `basic-computer-games\34_Digits\csharp\Matrix.cs`

```

// 命名空间为Digits，表示这个类在Digits命名空间下
namespace Digits;

// 定义一个叫Matrix的类，表示矩阵
internal class Matrix
{
    // 私有的只读整型变量_weight，表示权重
    private readonly int _weight;
    // 私有的只读二维整型数组_values，表示值
    private readonly int[,] _values;

    // 构造函数，接受宽度、权重和一个用于生成种子的函数
    public Matrix(int width, int weight, Func<int, int, int> seedFactory)
    {
        // 将传入的权重赋值给_weight
        _weight = weight;
        // 创建一个宽度为width，高度为3的二维数组，并赋值给_values
        _values = new int[width, 3];
        
        // 循环遍历矩阵的每个元素
        for (int i = 0; i < width; i++)
        for (int j = 0; j < 3; j++)
        {
            // 使用seedFactory函数生成种子，并赋值给_values数组
            _values[i, j] = seedFactory.Invoke(i, j);
        }

        // 将宽度减1的值赋给Index
        Index = width - 1;
    }

    // 公共的可读写属性Index，表示索引
    public int Index { get; set; }

    // 公共的方法GetWeightedValue，接受一个参数row，返回权重乘以_values[Index, row]的值
    public int GetWeightedValue(int row) => _weight * _values[Index, row];

    // 公共的方法IncrementValue，接受一个参数row，返回_values[Index, row]自增后的值
    public int IncrementValue(int row) => _values[Index, row]++;
}

```