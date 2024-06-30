# `D:\src\scipysrc\seaborn\seaborn\__init__.py`

```
# 导入 seaborn 库的对象和函数
from .rcmod import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403
from .palettes import *  # noqa: F401,F403
from .relational import *  # noqa: F401,F403
from .regression import *  # noqa: F401,F403
from .categorical import *  # noqa: F401,F403
from .distributions import *  # noqa: F401,F403
from .matrix import *  # noqa: F401,F403
from .miscplot import *  # noqa: F401,F403
from .axisgrid import *  # noqa: F401,F403
from .widgets import *  # noqa: F401,F403
from .colors import xkcd_rgb, crayons  # noqa: F401
from . import cm  # noqa: F401

# 捕获原始的 matplotlib rcParams 参数
import matplotlib as mpl
# 复制当前的 rcParams 参数为原始参数备份
_orig_rc_params = mpl.rcParams.copy()

# 定义 seaborn 的版本号
__version__ = "0.14.0.dev0"
```