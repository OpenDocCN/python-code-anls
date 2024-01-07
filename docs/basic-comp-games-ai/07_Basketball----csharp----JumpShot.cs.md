# `basic-computer-games\07_Basketball\csharp\JumpShot.cs`

```

# 在 Basketball 命名空间下定义 JumpShot 类，继承自 Shot 类
namespace Basketball;

public class JumpShot : Shot
{
    # 创建 JumpShot 类的构造函数，调用基类 Shot 的构造函数并传入参数 "Jump shot"
    public JumpShot()
        : base("Jump shot")
    {
    }
}

```