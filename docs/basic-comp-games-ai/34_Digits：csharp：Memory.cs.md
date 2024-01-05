# `d:/src/tocomm/basic-computer-games\34_Digits\csharp\Memory.cs`

```
namespace Digits;  # 命名空间声明

public class Memory  # 定义名为 Memory 的公共类
{
    private readonly Matrix[] _matrices;  # 声明私有只读的 Matrix 数组 _matrices

    public Memory()  # Memory 类的构造函数
    {
        _matrices = new[]  # 初始化 _matrices 数组
        {
            new Matrix(27, 3, (_, _) => 1),  # 创建一个新的 Matrix 对象，传入参数 27, 3, 和一个函数
            new Matrix(9, 1, (i, j) => i == 4 * j ? 2 : 3),  # 创建一个新的 Matrix 对象，传入参数 9, 1, 和一个函数
            new Matrix(3, 0, (_, _) => 9)  # 创建一个新的 Matrix 对象，传入参数 3, 0, 和一个函数
        };
    }

    public int GetWeightedSum(int row) => _matrices.Select(m => m.GetWeightedValue(row)).Sum();  # 定义一个公共方法 GetWeightedSum，返回加权和

    public void ObserveDigit(int digit)  # 定义一个公共方法 ObserveDigit，传入参数 digit
    {
# 循环3次，对每个矩阵对象调用IncrementValue方法，传入参数digit
for (int i = 0; i < 3; i++)
{
    _matrices[i].IncrementValue(digit);
}

# 对第一个矩阵对象的Index属性进行计算赋值
_matrices[0].Index = _matrices[0].Index % 9 * 3 + digit;

# 对第二个矩阵对象的Index属性进行计算赋值
_matrices[1].Index = _matrices[0].Index % 9;

# 对第三个矩阵对象的Index属性赋值为digit
_matrices[2].Index = digit;
```