# `basic-computer-games\00_Common\dotnet\Games.Common\_InternalsVisibleTo.cs`

```
# 使用 System.Runtime.CompilerServices 命名空间
using System.Runtime.CompilerServices;

# 允许 Games.Common.Test 程序集访问当前程序集的内部成员
[assembly:InternalsVisibleTo("Games.Common.Test")]
```