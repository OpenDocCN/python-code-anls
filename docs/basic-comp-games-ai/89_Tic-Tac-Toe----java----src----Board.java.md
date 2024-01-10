# `basic-computer-games\89_Tic-Tac-Toe\java\src\Board.java`

```
/**
 * @author Ollie Hensman-Crook
 */
public class Board {
    private char arr[];

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
    public void setArr(int position, char player) {
        if (player == 'X') {
            this.arr[position-1] = 'X';
        } else {
            this.arr[position -1] = 'O';
        }
    }

    /**
     * Print the current state of the board
     */
    public void printBoard() {
        System.out.format("%-3c ! %-3c ! %-3c\n----+----+----\n%-3c ! %-3c ! %-3c\n----+----+----\n%-3c ! %-3c ! %-3c\n",
        this.arr[0], this.arr[1], this.arr[2], this.arr[3], this.arr[4], this.arr[5], this.arr[6], this.arr[7], this.arr[8]
        );
    }

    /**
     * Get the value of the char at a given position
     * @param x
     * @return the value of the char at a given position
     */
    public char getBoardValue(int x) {
        return arr[x-1];
    }

    /**
     * Go through the board and check for win (horizontal, diagonal, vertical)
     * @param player
     * @return whether a win has occurred
     */
}
    // 检查玩家是否获胜
    public boolean checkWin(char player) {
        // 检查第一行是否有三个相同的标记
        if(this.arr[0] == player && this.arr[1] == player && this.arr[2] == player)
            return true;

        // 检查第二行是否有三个相同的标记
        if(this.arr[3] == player && this.arr[4] == player && this.arr[5] == player)
            return true;

        // 检查第三行是否有三个相同的标记
        if(this.arr[6] == player && this.arr[7] == player && this.arr[8] == player)
            return true;

        // 检查从左上到右下的对角线是否有三个相同的标记
        if(this.arr[0] == player && this.arr[4] == player && this.arr[8] == player)
            return true;

        // 检查从右上到左下的对角线是否有三个相同的标记
        if(this.arr[2] == player && this.arr[4] == player && this.arr[6] == player)
            return true;

        // 检查第一列是否有三个相同的标记
        if(this.arr[0] == player && this.arr[3] == player && this.arr[6] == player)
            return true;

        // 检查第二列是否有三个相同的标记
        if(this.arr[1] == player && this.arr[4] == player && this.arr[7] == player)
            return true;

        // 检查第三列是否有三个相同的标记
        if(this.arr[2] == player && this.arr[5] == player && this.arr[8] == player)
            return true;

        return false;
    }
    
    // 检查是否为平局
    public boolean checkDraw() {
        // 如果X和O都没有获胜，并且棋盘上没有空格，则为平局
        if(this.checkWin('X') == false && this.checkWin('O') == false) {
            if(this.getBoardValue(1) != ' ' && this.getBoardValue(2) != ' ' && this.getBoardValue(3) != ' ' && this.getBoardValue(4) != ' ' && this.getBoardValue(5) != ' ' && this.getBoardValue(6) != ' ' && this.getBoardValue(7) != ' ' && this.getBoardValue(8) != ' ' && this.getBoardValue(9) != ' ' ) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * 重置棋盘
     */
    public void clear() {
        // 将棋盘上的每个位置重置为空格
        for (int x = 1; x <= 9; x++) {
            this.arr[x - 1] = ' ';
        }
    }
# 闭合前面的函数定义
```