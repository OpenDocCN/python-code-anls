# `basic-computer-games\08_Batnum\java\src\BatNum.java`

```
// 导入必要的类
import java.util.Arrays;
import java.util.Scanner;

/**
 * BatNum 游戏
 * <p>
 * 基于这里的基本 BatNum 游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/08%20Batnum/batnum.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 Basic 游戏的版本，而不引入新功能 - 没有添加额外的文本、错误检查等。
 */
public class BatNum {

    // 用于键盘输入
    private final Scanner kbScanner;

    // 当前游戏状态
    private GAME_STATE gameState;

    private int pileSize;

    // 游戏获胜的选项
    enum WIN_OPTION {
        TAKE_LAST,
        AVOID_LAST
    }

    // 追踪获胜者
    enum WINNER {
        COMPUTER,
        PLAYER
    }

    private WINNER winner;

    private WIN_OPTION winOption;

    private int minSelection;
    private int maxSelection;

    // 计算机用于最佳移动的范围
    private int rangeOfRemovals;

    public BatNum() {

        // 设置游戏状态为 STARTING
        gameState = GAME_STATE.STARTING;

        // 初始化键盘扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * 主游戏循环
     */
    }

    /**
     * 计算机的回合 - 即要移除的对象数量
     *
     * @param pileSizeLeft 当前大小
     * @return 要移除的对象数量
     */
    # 计算计算机应该移除的数量，以确保游戏的公平性
    private int calculateComputersTurn(int pileSizeLeft) {
        # 根据剩余的物品数量和移除范围计算计算机应该移除的数量
        int computersNumberToRemove = pileSizeLeft - rangeOfRemovals * (pileSizeLeft / rangeOfRemovals);
        # 如果计算机应该移除的数量小于最小可选数量，则设置为最小可选数量
        if (computersNumberToRemove < minSelection) {
            computersNumberToRemove = minSelection;
        }
        # 如果计算机应该移除的数量大于最大可选数量，则设置为最大可选数量
        if (computersNumberToRemove > maxSelection) {
            computersNumberToRemove = maxSelection;
        }

        # 返回计算机应该移除的数量
        return computersNumberToRemove;
    }

    # 打印游戏介绍信息
    private void intro() {
        # 打印游戏标题
        System.out.println(simulateTabs(33) + "BATNUM");
        # 打印游戏信息
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THIS PROGRAM IS A 'BATTLE OF NUMBERS' GAME, WHERE THE");
        # 打印游戏规则
        System.out.println("COMPUTER IS YOUR OPPONENT.");
        System.out.println();
        System.out.println("THE GAME STARTS WITH AN ASSUMED PILE OF OBJECTS. YOU");
        System.out.println("AND YOUR OPPONENT ALTERNATELY REMOVE OBJECTS FROM THE PILE.");
        System.out.println("WINNING IS DEFINED IN ADVANCE AS TAKING THE LAST OBJECT OR");
        System.out.println("NOT. YOU CAN ALSO SPECIFY SOME OTHER BEGINNING CONDITIONS.");
        System.out.println("DON'T USE ZERO, HOWEVER, IN PLAYING THE GAME.");
        System.out.println("ENTER A NEGATIVE NUMBER FOR NEW PILE SIZE TO STOP PLAYING.");
    }

    '''
     * 打印屏幕上的消息，然后从键盘接受输入。
     * 将输入转换为整数
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     '''
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }

    '''
     * 打印屏幕上的消息，然后从键盘接受输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    # 显示文本并获取用户输入
    private String displayTextAndGetInput(String text) {
        System.out.print(text);  # 打印文本
        return kbScanner.next();  # 获取用户输入
    }

    /**
     * 模拟旧的基本tab(xx)命令，将文本缩进xx个空格
     *
     * @param spaces 需要的空格数
     * @return 具有空格数的字符串
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];  # 创建一个包含指定空格数的字符数组
        Arrays.fill(spacesTemp, ' ');  # 用空格填充数组
        return new String(spacesTemp);  # 将字符数组转换为字符串
    }

    /**
     * 接受由逗号分隔的字符串，并返回第n个分隔值（从计数0开始）。
     *
     * @param text - 由逗号分隔的值的文本
     * @param pos  - 要返回值的位置
     * @return 值的int表示
     */
    private int getDelimitedValue(String text, int pos) {
        String[] tokens = text.split(",");  # 使用逗号分割文本
        return Integer.parseInt(tokens[pos]);  # 将指定位置的值转换为整数并返回
    }
# 闭合前面的函数定义
```