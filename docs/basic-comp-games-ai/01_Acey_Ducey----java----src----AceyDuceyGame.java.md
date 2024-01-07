# `basic-computer-games\01_Acey_Ducey\java\src\AceyDuceyGame.java`

```

/**
 * This class is used to invoke the game.
 * 该类用于调用游戏。
 */
public class AceyDuceyGame {

    public static void main(String[] args) {

        boolean keepPlaying; // 声明一个布尔变量，用于判断是否继续游戏
        AceyDucey game = new AceyDucey(); // 创建 AceyDucey 游戏对象

        // Keep playing game until infinity or the player loses
        // 循环进行游戏，直到无限循环或玩家输掉游戏
        do {
            game.play(); // 调用游戏对象的 play 方法进行游戏
            System.out.println(); // 输出空行
            System.out.println(); // 输出空行
            System.out.println(); // 输出空行
            keepPlaying = game.playAgain(); // 调用游戏对象的 playAgain 方法判断是否继续游戏
        } while (keepPlaying); // 当 keepPlaying 为 true 时继续循环
    }
}

```