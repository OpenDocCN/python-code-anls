# `basic-computer-games\89_Tic-Tac-Toe\java\src\Board.java`

```

/**
 * @author Ollie Hensman-Crook
 */
// 定义名为 Board 的类
public class Board {
    // 声明私有字符数组 arr
    private char arr[];

    // 定义构造函数，初始化字符数组 arr
    public Board() {
        this.arr = new char[9];
        for (int x = 1; x <= 9; x++) {
            this.arr[x - 1] = ' ';
        }
    }

    /**
     * Place 'X' or 'O' on the board position passed
     * @param position
     * @param player
     */
    // 在指定位置放置 'X' 或 'O'
    public void setArr(int position, char player) {
        if (player == 'X') {
            this.arr[position-1] = 'X';
        } else {
            this.arr[position -1] = 'O';
        }
    }

    // 打印游戏板
    public void printBoard() {
        System.out.format("%-3c ! %-3c ! %-3c\n----+----+----\n%-3c ! %-3c ! %-3c\n----+----+----\n%-3c ! %-3c ! %-3c\n",
        this.arr[0], this.arr[1], this.arr[2], this.arr[3], this.arr[4], this.arr[5], this.arr[6], this.arr[7], this.arr[8]
        );
    }

    /**
     * @param x
     * @return the value of the char at a given position
     */
    // 返回指定位置的字符值
    public char getBoardValue(int x) {
        return arr[x-1];
    }

    /**
     * Go through the board and check for win (horizontal, diagonal, vertical)
     * @param player
     * @return whether a win has occured
     */
    // 检查是否有玩家获胜
    public boolean checkWin(char player) {
        // 检查水平方向
        if(this.arr[0] == player && this.arr[1] == player && this.arr[2] == player)
            return true;

        // 检查垂直方向
        if(this.arr[3] == player && this.arr[4] == player && this.arr[5] == player)
            return true;

        // 检查对角线方向
        if(this.arr[6] == player && this.arr[7] == player && this.arr[8] == player)
            return true;

        // 检查对角线方向
        if(this.arr[0] == player && this.arr[4] == player && this.arr[8] == player)
            return true;

        // 检查对角线方向
        if(this.arr[2] == player && this.arr[4] == player && this.arr[6] == player)
            return true;

        // 检查垂直方向
        if(this.arr[0] == player && this.arr[3] == player && this.arr[6] == player)
            return true;

        // 检查垂直方向
        if(this.arr[1] == player && this.arr[4] == player && this.arr[7] == player)
            return true;

        // 检查垂直方向
        if(this.arr[2] == player && this.arr[5] == player && this.arr[8] == player)
            return true;

        return false;
    }

    // 检查是否为平局
    public boolean checkDraw() {
        if(this.checkWin('X') == false && this.checkWin('O') == false) {
            if(this.getBoardValue(1) != ' ' && this.getBoardValue(2) != ' ' && this.getBoardValue(3) != ' ' && this.getBoardValue(4) != ' ' && this.getBoardValue(5) != ' ' && this.getBoardValue(6) != ' ' && this.getBoardValue(7) != ' ' && this.getBoardValue(8) != ' ' && this.getBoardValue(9) != ' ' ) {
                return true;
            }
        }
        return false;
    }

    // 重置游戏板
    public void clear() {
        for (int x = 1; x <= 9; x++) {
            this.arr[x - 1] = ' ';
        }
    }
}

```