# `arknights-mower\arknights_mower\utils\device\minitouch\command.py`

```py
# 导入未来版本的注解特性
from __future__ import annotations

# 导入时间模块
import time

# 从指定路径导入日志模块中的 logger 对象
from ...log import logger

# 从当前目录下的 session 模块中导入 Session 类
from .session import Session

# 默认延迟时间
DEFAULT_DELAY = 0.05

# 定义 CommandBuilder 类
class CommandBuilder(object):
    """ Build command str for minitouch """

    # 初始化方法
    def __init__(self) -> None:
        # 初始化命令内容为空字符串
        self.content = ''
        # 初始化延迟时间为 0
        self.delay = 0

    # 添加命令内容的方法
    def append(self, new_content: str) -> None:
        # 将新内容追加到命令内容末尾，并换行
        self.content += new_content + '\n'

    # 提交命令的方法
    def commit(self) -> None:
        """ add minitouch command: 'c\n' """
        # 添加 minitouch 命令 'c'
        self.append('c')

    # 等待的方法
    def wait(self, ms: int) -> None:
        """ add minitouch command: 'w <ms>\n' """
        # 添加 minitouch 命令 'w <ms>'
        self.append(f'w {ms}')
        # 更新延迟时间
        self.delay += ms

    # 抬起的方法
    def up(self, contact_id: int) -> None:
        """ add minitouch command: 'u <contact_id>\n' """
        # 添加 minitouch 命令 'u <contact_id>'
        self.append(f'u {contact_id}')

    # 按下的方法
    def down(self, contact_id: int, x: int, y: int, pressure: int) -> None:
        """ add minitouch command: 'd <contact_id> <x> <y> <pressure>\n' """
        # 添加 minitouch 命令 'd <contact_id> <x> <y> <pressure>'
        self.append(f'd {contact_id} {x} {y} {pressure}')

    # 移动的方法
    def move(self, contact_id: int, x: int, y: int, pressure: int) -> None:
        """ add minitouch command: 'm <contact_id> <x> <y> <pressure>\n' """
        # 添加 minitouch 命令 'm <contact_id> <x> <y> <pressure>'
        self.append(f'm {contact_id} {x} {y} {pressure}')

    # 发布命令的方法
    def publish(self, session: Session):
        """ apply current commands to device """
        # 提交命令
        self.commit()
        # 记录调试信息
        logger.debug('send operation: %s' % self.content.replace('\n', '\\n'))
        # 发送命令到设备
        session.send(self.content)
        # 等待一段时间
        time.sleep(self.delay / 1000 + DEFAULT_DELAY)
        # 重置命令内容
        self.reset()

    # 重置命令内容的方法
    def reset(self):
        """ clear current commands """
        # 清空命令内容
        self.content = ''
```