# `90_Tower\python\tower.py`

```
    # 检查磁盘列表是否为空，返回布尔值
    return len(self.__disks) == 0

    def add(self, d: Disk) -> None:
    # 将磁盘对象添加到磁塔的磁盘列表中
        self.__disks.append(d)

    def move_top_to(self, t: 'Tower') -> None:
    # 从当前磁塔的磁盘列表中移除顶部磁盘，并将其添加到目标磁塔的磁盘列表中
        top = self.__disks.pop()
        t.add(top)

    def move_disks(self, n: int, destination: 'Tower', buffer: 'Tower') -> None:
    # 递归地将当前磁塔上的 n 个磁盘移动到目标磁塔，借助缓冲磁塔
        if n > 0:
            self.move_disks(n-1, buffer, destination)
            self.move_top_to(destination)
            buffer.move_disks(n-1, destination, self)

def main():
    n = 3
    towers = [Tower() for _ in range(3)]
    for i in range(n-1, -1, -1):
        towers[0].add(Disk(i))
    towers[0].move_disks(n, towers[2], towers[1])

if __name__ == "__main__":
    main()
        return len(self.__disks) == 0  # 检查栈是否为空，如果栈中元素个数为0，则返回True，否则返回False

    def top(self) -> Optional[Disk]:
        if self.empty():  # 如果栈为空
            return None  # 返回空值
        else:
            return self.__disks[-1]  # 返回栈顶元素

    def add(self, disk: Disk) -> None:
        if not self.empty():  # 如果栈不为空
            t = self.top()  # 获取栈顶元素
            assert t is not None  # 用于断言栈顶元素不为空，如果为空则抛出异常
            if disk.size() > t.size():  # 如果要添加的磁盘大小大于栈顶元素大小
                raise Exception(
                    "YOU CAN'T PLACE A LARGER DISK ON TOP OF A SMALLER ONE, IT MIGHT CRUSH IT!"
                )  # 抛出异常，不能将较大的磁盘放在较小的磁盘上
        self.__disks.append(disk)  # 将磁盘添加到栈中

    def pop(self) -> Disk:
        if self.empty():  # 如果栈为空
        raise Exception("empty pop")  # 如果尝试从空栈中弹出元素，则抛出异常
        return self.__disks.pop()  # 从栈中弹出一个元素并返回

    def print(self) -> None:  # 定义一个打印方法，不返回任何内容
        r = "Needle: [%s]" % (", ".join([str(x.size()) for x in self.__disks]))  # 格式化输出栈中元素的大小
        print(r)  # 打印结果


class Game:
    def __init__(self) -> None:  # 初始化方法
        # use fewer sizes to make debugging easier  # 使用较少的尺寸使得调试更容易
        # self.__sizes = [3, 5, 7]  # ,9,11,13,15]
        self.__sizes = [3, 5, 7, 9, 11, 13, 15]  # 初始化尺寸列表

        self.__sizes.sort()  # 对尺寸列表进行排序

        self.__towers = []  # 初始化一个空列表
        self.__moves = 0  # 初始化移动次数为0
        self.__towers = [Tower(), Tower(), Tower()]  # 初始化三个Tower对象的列表
        self.__sizes.reverse()  # 反转尺寸列表
        for size in self.__sizes:
            disk = Disk(size)  # 创建一个大小为size的磁盘对象
            self.__towers[0].add(disk)  # 将创建的磁盘对象添加到第一个塔上

    def winner(self) -> bool:
        return self.__towers[0].empty() and self.__towers[1].empty()  # 检查第一个和第二个塔是否都为空，返回布尔值

    def print(self) -> None:
        for t in self.__towers:
            t.print()  # 打印每个塔的状态

    def moves(self) -> int:
        return self.__moves  # 返回移动的次数

    def which_disk(self) -> int:
        w = int(input("WHICH DISK WOULD YOU LIKE TO MOVE\n"))  # 获取用户输入的要移动的磁盘大小
        if w in self.__sizes:  # 如果用户输入的磁盘大小在可移动的范围内
            return w  # 返回用户输入的磁盘大小
        raise Exception()  # 否则抛出异常
    def pick_disk(self) -> Optional[Tower]:  # 定义一个方法pick_disk，返回一个Tower对象或者None
        which = None  # 初始化which变量为None
        while which is None:  # 当which为None时循环执行以下代码
            try:  # 尝试执行以下代码
                which = self.which_disk()  # 调用which_disk方法，将返回值赋给which
            except Exception:  # 如果出现异常则执行以下代码
                print("ILLEGAL ENTRY... YOU MAY ONLY TYPE 3,5,7,9,11,13, OR 15.\n")  # 打印错误提示信息

        valids = [t for t in self.__towers if t.top() and t.top().size() == which]  # 通过列表推导式筛选出符合条件的Tower对象
        assert len(valids) in (0, 1)  # 断言valids列表长度为0或1
        if not valids:  # 如果valids为空
            print("THAT DISK IS BELOW ANOTHER ONE.  MAKE ANOTHER CHOICE.\n")  # 打印提示信息
            return None  # 返回None
        else:  # 否则
            assert valids[0].top().size() == which  # 断言valids列表第一个元素的顶部盘子大小等于which
            return valids[0]  # 返回valids列表的第一个元素

    def which_tower(self) -> Optional[Tower]:  # 定义一个方法which_tower，返回一个Tower对象或者None
        try:  # 尝试执行以下代码
            needle = int(input("PLACE DISK ON WHICH NEEDLE\n"))  # 获取用户输入的整数值
            tower = self.__towers[needle - 1]  # 从私有属性 __towers 中取出索引为 needle-1 的元素赋值给 tower
        except Exception:  # 捕获任何异常
            print(
                "I'LL ASSUME YOU HIT THE WRONG KEY THIS TIME.  BUT WATCH IT,\nI ONLY ALLOW ONE MISTAKE.\n"
            )  # 打印错误提示信息
            return None  # 返回 None
        else:  # 如果没有发生异常
            return tower  # 返回 tower

    def take_turn(self) -> None:  # 定义一个方法 take_turn，返回类型为 None
        from_tower = None  # 初始化 from_tower 为 None
        while from_tower is None:  # 当 from_tower 为 None 时循环
            from_tower = self.pick_disk()  # 调用 pick_disk 方法赋值给 from_tower

        to_tower = self.which_tower()  # 调用 which_tower 方法赋值给 to_tower
        if not to_tower:  # 如果 to_tower 为假值
            to_tower = self.which_tower()  # 再次调用 which_tower 方法赋值给 to_tower

        if not to_tower:  # 如果 to_tower 为假值
            print("I TRIED TO WARN YOU, BUT YOU WOULDN'T LISTEN.\nBYE BYE, BIG SHOT.\n")  # 打印警告信息
            sys.exit(0)  # 退出程序并返回状态码 0

        disk = from_tower.pop()  # 从 from_tower 中弹出一个元素并赋值给 disk
        try:
            to_tower.add(disk)  # 将 disk 添加到 to_tower 中
            self.__moves += 1  # 移动次数加 1
        except Exception as err:  # 捕获异常并赋值给 err
            print(err)  # 打印异常信息
            from_tower.add(disk)  # 将 disk 添加回 from_tower 中


def main() -> None:  # 主函数，不返回任何值
    print(
        """
    IN THIS PROGRAM, WE SHALL REFER TO DISKS BY NUMERICAL CODE.
    3 WILL REPRESENT THE SMALLEST DISK, 5 THE NEXT SIZE,
    7 THE NEXT, AND SO ON, UP TO 15.  IF YOU DO THE PUZZLE WITH
    2 DISKS, THEIR CODE NAMES WOULD BE 13 AND 15.  WITH 3 DISKS
    THE CODE NAMES WOULD BE 11, 13 AND 15, ETC.  THE NEEDLES
    ARE NUMBERED FROM LEFT TO RIGHT, 1 TO 3.  WE WILL
    START WITH THE DISKS ON NEEDLE 1, AND ATTEMPT TO MOVE THEM
    TO NEEDLE 3.
    # 从针1开始，尝试将盘子移动到针3上

    GOOD LUCK!
    # 祝你好运！

    """
    )
    # 以上是注释，表示开始游戏的提示信息

    game = Game()
    # 创建一个游戏实例

    while True:
        # 无限循环，直到游戏结束
        game.print()
        # 打印当前游戏状态

        game.take_turn()
        # 玩家进行一次移动

        if game.winner():
            # 如果游戏结束
            print(
                "CONGRATULATIONS!!\nYOU HAVE PERFORMED THE TASK IN %s MOVES.\n"
                % game.moves()
            )
            # 打印祝贺信息和完成游戏所用的步数
            while True:
                # 无限循环，直到玩家选择退出游戏
                # 提示用户输入是否再次尝试游戏
                yesno = input("TRY AGAIN (YES OR NO)\n")
                # 如果用户输入为YES，则重新开始游戏并跳出循环
                if yesno.upper() == "YES":
                    game = Game()
                    break
                # 如果用户输入为NO，则打印感谢信息并退出程序
                elif yesno.upper() == "NO":
                    print("THANKS FOR THE GAME!\n")
                    sys.exit(0)
                # 如果用户输入既不是YES也不是NO，则提示用户重新输入
                else:
                    print("'YES' OR 'NO' PLEASE\n")
        # 如果游戏移动次数超过128次，则打印提示信息并退出程序
        elif game.moves() > 128:
            print("SORRY, BUT I HAVE ORDERS TO STOP IF YOU MAKE MORE THAN 128 MOVES.")
            sys.exit(0)


if __name__ == "__main__":
    main()
```