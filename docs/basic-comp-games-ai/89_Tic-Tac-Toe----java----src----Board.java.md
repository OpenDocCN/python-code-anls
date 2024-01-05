# `89_Tic-Tac-Toe\java\src\Board.java`

```
        // 在指定位置上放置玩家的标记
        this.arr[position - 1] = player;
    }

    /**
     * 获取指定位置上的标记
     * @param position
     * @return
     */
    public char getArr(int position) {
        // 返回指定位置上的标记
        return this.arr[position - 1];
    }

    /**
     * 检查指定位置是否为空
     * @param position
     * @return
     */
    public boolean isPositionEmpty(int position) {
        // 检查指定位置上是否为空
        return this.arr[position - 1] == ' ';
    }

    /**
     * 打印当前棋盘状态
     */
    public void printBoard() {
        // 打印当前棋盘状态
        System.out.println(" " + this.arr[0] + " | " + this.arr[1] + " | " + this.arr[2]);
        System.out.println("---|---|---");
        System.out.println(" " + this.arr[3] + " | " + this.arr[4] + " | " + this.arr[5]);
        System.out.println("---|---|---");
        System.out.println(" " + this.arr[6] + " | " + this.arr[7] + " | " + this.arr[8]);
    }
}
        if (player == 'X') {  # 如果玩家是X
            this.arr[position-1] = 'X';  # 在数组中的指定位置放置X
        } else {  # 否则
            this.arr[position -1] = 'O';  # 在数组中的指定位置放置O
        }
    }

    public void printBoard() {  # 打印游戏板
        System.out.format("%-3c ! %-3c ! %-3c\n----+----+----\n%-3c ! %-3c ! %-3c\n----+----+----\n%-3c ! %-3c ! %-3c\n",
        this.arr[0], this.arr[1], this.arr[2], this.arr[3], this.arr[4], this.arr[5], this.arr[6], this.arr[7], this.arr[8]
        );  # 格式化打印游戏板
    }


    /**
     * @param x
     * @return the value of the char at a given position
     */
    public char getBoardValue(int x) {  # 获取指定位置的字符值
        return arr[x-1];  # 返回数组中指定位置的字符值
    }

    /**
     * 遍历游戏棋盘并检查是否获胜（水平、对角线、垂直）
     * @param player 玩家
     * @return 是否发生了获胜
     */
    public boolean checkWin(char player) {
        // 检查第一行是否有获胜
        if(this.arr[0] == player && this.arr[1] == player && this.arr[2] == player)
            return true;

        // 检查第二行是否有获胜
        if(this.arr[3] == player && this.arr[4] == player && this.arr[5] == player)
            return true;

        // 检查第三行是否有获胜
        if(this.arr[6] == player && this.arr[7] == player && this.arr[8] == player)
            return true;
        # 检查对角线上是否有相同的玩家标记，如果有则返回true
        if(this.arr[0] == player && this.arr[4] == player && this.arr[8] == player)
            return true;

        # 检查对角线上是否有相同的玩家标记，如果有则返回true
        if(this.arr[2] == player && this.arr[4] == player && this.arr[6] == player)
            return true;

        # 检查第一列上是否有相同的玩家标记，如果有则返回true
        if(this.arr[0] == player && this.arr[3] == player && this.arr[6] == player)
            return true;

        # 检查第二列上是否有相同的玩家标记，如果有则返回true
        if(this.arr[1] == player && this.arr[4] == player && this.arr[7] == player)
            return true;

        # 检查第三列上是否有相同的玩家标记，如果有则返回true
        if(this.arr[2] == player && this.arr[5] == player && this.arr[8] == player)
            return true;

        # 如果以上条件都不满足，则返回false
        return false;
    }
    # 检查是否为平局
    public boolean checkDraw() {
        # 如果X和O都没有获胜，则为平局
        if(this.checkWin('X') == false && this.checkWin('O') == false) {
                return true;
    }
    // 结束类定义

}

// 返回 false
return false;
}
// 重置棋盘
public void clear() {
    // 循环遍历数组，将每个元素设置为空格
    for (int x = 1; x <= 9; x++) {
        this.arr[x - 1] = ' ';
    }
}
// 结束方法定义
```