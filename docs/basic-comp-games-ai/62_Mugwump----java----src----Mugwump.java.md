# `basic-computer-games\62_Mugwump\java\src\Mugwump.java`

```py
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Mugwump
 * <p>
 * Based on the Basic game of Mugwump here
 * https://github.com/coding-horror/basic-computer-games/blob/main/62%20Mugwump/mugwump.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */

public class Mugwump {

    public static final int NUMBER_OF_MUGWUMPS = 4;

    public static final int MAX_TURNS = 10;

    public static final int FOUND = -1;

    // Used for keyboard input
    private final Scanner kbScanner;

    private enum GAME_STATE {
        INIT,
        GAME_START,
        PLAY_TURN
    }

    // Current game state
    private GAME_STATE gameState;

    int[][] mugwumpLocations;

    int turn;

    public Mugwump() {
        kbScanner = new Scanner(System.in);
        gameState = GAME_STATE.INIT;
    }

    /**
     * Main game loop
     */
    }

    private void intro() {
        // 打印游戏介绍信息
        System.out.println(addSpaces(33) + "MUGWUMP");
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THE OBJECT OF THIS GAME IS TO FIND FOUR MUGWUMPS");
        System.out.println("HIDDEN ON A 10 BY 10 GRID.  HOMEBASE IS POSITION 0,0.");
        System.out.println("ANY GUESS YOU MAKE MUST BE TWO NUMBERS WITH EACH");
        System.out.println("NUMBER BETWEEN 0 AND 9, INCLUSIVE.  FIRST NUMBER");
        System.out.println("IS DISTANCE TO RIGHT OF HOMEBASE AND SECOND NUMBER");
        System.out.println("IS DISTANCE ABOVE HOMEBASE.");
        System.out.println();
        System.out.println("YOU GET 10 TRIES.  AFTER EACH TRY, I WILL TELL");
        System.out.println("YOU HOW FAR YOU ARE FROM EACH MUGWUMP.");
    }
    /**
     * Accepts a string delimited by comma's and returns the pos'th delimited
     * value (starting at count 0).
     *
     * @param text - text with values separated by comma's
     * @param pos  - which position to return a value for
     * @return the int representation of the value
     */
    private int getDelimitedValue(String text, int pos) {
        // Split the input text by comma and store the result in an array
        String[] tokens = text.split(",");
        // Parse the token at the specified position and return its integer representation
        return Integer.parseInt(tokens[pos]);
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private String displayTextAndGetInput(String text) {
        // Display the input text on the screen
        System.out.print(text);
        // Read the input from the keyboard and return it
        return kbScanner.nextLine();
    }

    /**
     * Return a string of x spaces
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String addSpaces(int spaces) {
        // Create an array of spaces with the specified length
        char[] spacesTemp = new char[spaces];
        // Fill the array with spaces
        Arrays.fill(spacesTemp, ' ');
        // Convert the array to a string and return it
        return new String(spacesTemp);
    }

    public static void main(String[] args) {

        // Create a new instance of Mugwump
        Mugwump mugwump = new Mugwump();
        // Call the play method of the Mugwump instance
        mugwump.play();
    }
# 闭合前面的函数定义
```