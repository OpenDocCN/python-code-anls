# 02 Amazing

这个游戏会接收用户输入的长和宽，动态生成一个迷宫。

改进自 Frank Palazzolo 的版本。

## 导入

```py
import random
import os
from time import sleep
```

## 常量

```py
# 定义移动方向常量
GO_LEFT, GO_UP, GO_RIGHT, GO_DOWN = [0, 1, 2, 3]
# 定义连通方向常量
CONN_DOWN = 1
CONN_RIGHT = 2
```

## `get_width_length()`

```py
# 从用户输入获取迷宫的长和宽
def get_width_length():
    while True:
        try:
            width, length = input('What are your width and length?').split(',')
            width = int(width)
            length = int(length)
            if width != 1 and length != 1:
                return (width, length)
            print('Meaningless dimensions. Try again.')
        except ValueError:
            print('Meaningless dimensions. Try again.')
```

## `get_possible_dirs()`

```py
# 返回从一个格子的可移动方向
def get_possible_dirs(row, col, width, length, used):
    possible_dirs = [GO_LEFT,GO_UP,GO_RIGHT,GO_DOWN]
    # 如果不是最左边，并且左面一格没有访问
    # 那左边就是可以访问的，以此类推
    if col==0 or used[row][col-1]!=0:
        possible_dirs.remove(GO_LEFT)
    if row==0 or used[row-1][col]!=0:
        possible_dirs.remove(GO_UP)
    if col==width-1 or used[row][col+1]!=0: 
        possible_dirs.remove(GO_RIGHT)
    if row==length-1 or used[row+1][col]!=0:
        possible_dirs.remove(GO_DOWN)   
    return possible_dirs
```

## `get_next_used()`

```py
# 获取某个位置的下一个已访问格子
def get_next_used(row, col, used):
    width, length = len(used), len(used[0])
    while True:
        if col != width - 1:
            col += 1
        elif row != length - 1:
            row, col = row + 1, 0
        else:
            row, col = 0, 0
        if used[row][col] != 0:
            break
    return row, col
```

## `print_maze()`

```py
# 打印迷宫
def print_maze(walls, used, enter_col):
    # 每次打印之前先清屏
    os.system('cls')
    length, width = len(walls), len(walls[0])
    # 打印最上方的墙壁
    for col in range(width):
        # 如果是入口，就把它打开
        if col==enter_col:
            print('*  ',end='')
        else:
            print('*--',end='')
    print('*')
    for row in range(length):
        # 打印格子
        print('|',end='')
        for col in range(width):
            # 区分已访问和未访问
            cell_ch = '  ' if used[row][col] != 0 else '><'
            # 区分格子右边是否有墙壁
            if walls[row][col] | CONN_RIGHT == walls[row][col]:
                print(cell_ch + ' ',end='')
            else:
                print(cell_ch + '|',end='')
        print()
        # 打印格子下方的墙壁
        for col in range(width):
            # 区分格子下方是否有墙壁
            if walls[row][col] | CONN_DOWN == walls[row][col]:
                print('*  ',end='')
            else:
                print('*--',end='')
        print('*')
    # 短暂停留避免一闪而过
    sleep(0.1)
```

## `main()`

```py
# 程序的主要逻辑
def main():
    print(' '*28+'AMAZING PROGRAM')
    print(' '*15+'CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY')
    print()
    print()
    print()

    # 获取用户输入的长和宽
    width, length = get_width_length()
    # used 数组保存格子是否访问过
    # 0 表示未访问，其它数字表示该格子是第几个访问的
    used = [[0]*width for _ in range(length)]
    # wall 保存右边和下边是不是连通的
    # 0：不连通，1：下侧连通，2：右侧连通，3：右侧和下侧连通
    walls = [[0]*width for _ in range(length)]

    # 随机选取入口
    enter_col=random.randint(0,width-1)
    # 将起始位置定义为入口
    row,col=0,enter_col
    # 定义已访问格子数量
    count=1
    # 设置入口已访问
    used[row][col]=count
    # 每次格式发生变化都会打印迷宫，下同
    print_maze(walls, used, enter_col)

    # 在所有格子都已访问之前执行循环
    while count!=width*length:
        # 获取当然位置的可移动方向
        possible_dirs = get_possible_dirs(row, col, width, length, used)
        
        if len(possible_dirs)!=0:
            # 如果可以移动，那么随机选一个方向来移动
            # 并且把墙拆掉
            direction=random.choice(possible_dirs) 
            if direction==GO_LEFT:
                col=col-1
                walls[row][col] |= CONN_RIGHT
            elif direction==GO_UP:
                row=row-1 
                walls[row][col] |= CONN_DOWN
            elif direction==GO_RIGHT:
                walls[row][col] |= CONN_RIGHT    
                col=col+1
            elif direction==GO_DOWN:
                walls[row][col] |= CONN_DOWN
                row=row+1
            # 更新计数，设置已访问
            count=count+1
            used[row][col]=count
            print_maze(walls, used, enter_col)
        else:
            # 否则，选取下一个已访问的格子重复这个步骤
            # 因为外面已经检查了是否可移动，
            # 这里只检查已访问就可以了
            row, col = get_next_used(row, col, used)

    # 最后，随机选择出口
    col=random.randint(0,width-1)
    row=length-1
    walls[row][col]=walls[row][col]+1
    
    print_maze(walls, used, enter_col)
    

if __name__ == '__main__': main()
```
