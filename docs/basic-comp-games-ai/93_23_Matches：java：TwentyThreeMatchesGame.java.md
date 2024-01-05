# `d:/src/tocomm/basic-computer-games\93_23_Matches\java\TwentyThreeMatchesGame.java`

```
/**
 * Game of 23 Matches
 * <p>
 * Based on the BASIC game of 23 Matches here
 * https://github.com/coding-horror/basic-computer-games/blob/main/93%2023%20Matches/23matches.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 * <p>
 * Converted from BASIC to Java by Darren Cardenas.
 */
public class TwentyThreeMatchesGame {

    public static void main(String[] args) {
        // 显示游戏介绍
        showIntro();
        // 创建游戏对象并开始游戏
        TwentyThreeMatches game = new TwentyThreeMatches();
        game.startGame();
    }

    // 显示游戏介绍
    private static void showIntro() {
*/
# 打印消息INTRO到控制台
System.out.println(Messages.INTRO);
# 结束类定义
}
# 结束类定义
}
```