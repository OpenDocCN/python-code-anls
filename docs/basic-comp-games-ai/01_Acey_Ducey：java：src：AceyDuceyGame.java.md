# `01_Acey_Ducey\java\src\AceyDuceyGame.java`

```
/**
 * This class is used to invoke the game.
 * It contains the main method to start the game.
 */
public class AceyDuceyGame {

    public static void main(String[] args) {

        boolean keepPlaying; // Declare a boolean variable to control the game loop
        AceyDucey game = new AceyDucey(); // Create a new instance of the AceyDucey game

        // Keep playing game until infinity or the player loses
        do {
            game.play(); // Call the play method of the game to play a round
            System.out.println(); // Print an empty line for formatting
            System.out.println(); // Print an empty line for formatting
            System.out.println(); // Print an empty line for formatting
            keepPlaying = game.playAgain(); // Check if the player wants to play again
        } while (keepPlaying); // Continue the loop if the player wants to keep playing
    }
}
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```