# `basic-computer-games\01_Acey_Ducey\java\src\AceyDuceyGame.java`

```
/**
 * This class is used to invoke the game.
 *
 */
public class AceyDuceyGame {

    public static void main(String[] args) {

        // Declare a boolean variable to control the game loop
        boolean keepPlaying;
        // Create an instance of the AceyDucey game
        AceyDucey game = new AceyDucey();

        // Keep playing game until infinity or the player loses
        do {
            // Start the game
            game.play();
            // Print empty lines for formatting
            System.out.println();
            System.out.println();
            System.out.println();
            // Check if the player wants to play again
            keepPlaying = game.playAgain();
        } while (keepPlaying);
    }
}
```