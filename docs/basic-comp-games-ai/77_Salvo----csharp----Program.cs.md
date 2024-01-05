# `77_Salvo\csharp\Program.cs`

```
# 导入 System 模块
import System
# 导入 Games.Common.IO 模块
import Games.Common.IO
# 导入 Games.Common.Randomness 模块
import Games.Common.Randomness
# 导入 Salvo 模块
import Salvo
# 导入 Salvo.Ships 模块
import Salvo.Ships
# 导入 Salvo.Resources.Resource 模块中的所有内容
from Salvo.Resources.Resource import *

# 创建一个新的游戏对象，使用 ConsoleIO() 作为输入输出，使用 DataRandom() 作为随机数生成器，并开始游戏
Game(ConsoleIO(), DataRandom()).Play()
```