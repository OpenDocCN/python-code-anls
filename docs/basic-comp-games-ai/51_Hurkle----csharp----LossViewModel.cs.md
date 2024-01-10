# `basic-computer-games\51_Hurkle\csharp\LossViewModel.cs`

```
# 定义了一个名为hurkle的命名空间
namespace hurkle
{
    # 定义了一个内部类LossViewModel
    internal class LossViewModel
    {
        # 定义了一个公共属性MaxGuesses，用于获取最大猜测次数，且只能在初始化时设置
        public int MaxGuesses { get; init; }
        # 定义了一个公共属性HurkleLocation，用于获取Hurkle的位置，且只能在初始化时设置
        public GamePoint HurkleLocation { get; init; }
    }
}
```