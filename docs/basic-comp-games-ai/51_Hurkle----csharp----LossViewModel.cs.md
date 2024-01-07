# `basic-computer-games\51_Hurkle\csharp\LossViewModel.cs`

```

# 定义了一个名为hurkle的命名空间
namespace hurkle
{
    # 定义了一个名为LossViewModel的内部类
    internal class LossViewModel
    {
        # 定义了一个名为MaxGuesses的公共属性，只能在初始化时赋值
        public int MaxGuesses { get; init; }
        # 定义了一个名为HurkleLocation的公共属性，只能在初始化时赋值，类型为GamePoint
        public GamePoint HurkleLocation { get; init; }
    }
}

```