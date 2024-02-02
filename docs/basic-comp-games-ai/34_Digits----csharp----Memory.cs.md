# `basic-computer-games\34_Digits\csharp\Memory.cs`

```py
namespace Digits;

public class Memory
{
    private readonly Matrix[] _matrices;  // 声明私有的 Matrix 数组 _matrices

    public Memory()
    {
        _matrices = new[]   // 初始化 _matrices 数组
        {
            new Matrix(27, 3, (_, _) => 1),  // 创建一个 27x3 的矩阵，每个元素的值都为 1
            new Matrix(9, 1, (i, j) => i == 4 * j ? 2 : 3),  // 创建一个 9x1 的矩阵，根据条件设置每个元素的值为 2 或 3
            new Matrix(3, 0, (_, _) => 9)  // 创建一个 3x0 的矩阵，所有元素的值都为 9
        };
    }

    public int GetWeightedSum(int row) => _matrices.Select(m => m.GetWeightedValue(row)).Sum();  // 计算指定行的加权和

    public void ObserveDigit(int digit)  // 观察数字
    {
        for (int i = 0; i < 3; i++)  // 循环三次
        {
            _matrices[i].IncrementValue(digit);  // 对每个矩阵的指定位置增加数字的值
        }

        _matrices[0].Index = _matrices[0].Index % 9 * 3 + digit;  // 更新第一个矩阵的索引
        _matrices[1].Index = _matrices[0].Index % 9;  // 更新第二个矩阵的索引
        _matrices[2].Index = digit;  // 更新第三个矩阵的索引
    }
}
```