# `basic-computer-games\05_Bagels\csharp\BagelNumber.cs`

```py
# 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;

# 定义枚举类型，表示Bagel验证的结果
namespace BasicComputerGames.Bagels
{
    public enum BagelValidation
    {
        Valid,          # 表示验证通过
        WrongLength,    # 表示长度错误
        NotUnique,      # 表示不唯一
        NonDigit        # 表示包含非数字字符
    };
    # 定义BagelNumber类
    public class BagelNumber
    }
}
```