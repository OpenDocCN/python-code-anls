# `basic-computer-games\07_Basketball\csharp\JumpShot.cs`

```
# 在 Basketball 命名空间下定义 JumpShot 类，继承自 Shot 类
namespace Basketball;

# 定义 JumpShot 类，继承自 Shot 类
public class JumpShot : Shot
{
    # 创建 JumpShot 类的构造函数
    public JumpShot()
        # 调用基类 Shot 的构造函数，传入参数 "Jump shot"
        : base("Jump shot")
    {
        # 构造函数体为空
    }
}
```