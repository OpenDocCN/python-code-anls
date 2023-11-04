# BasicComputerGames源码解析 80

# `84_Super_Star_Trek/python/superstartrekins.py`

这段代码是一个Python程序，它的目的是让用户输入一个字符串，然后判断输入的字符串是否以"N"为开头，如果是，则输出True，否则输出False。

具体来说，程序首先定义了一个名为`get_yes_no`的函数，该函数接收一个字符串参数`prompt`，并返回一个布尔值True或False。函数内部使用`input`函数获取用户输入的字符串，并将其转换为大写，然后比较字符串的第一位是否为"N"。

函数的作用是询问用户输入的字符串是否以"N"为开头，如果是，则返回True，否则返回False。用户输入的字符串必须包含"N"至少一次，函数才会返回True，否则返回False。


```
"""
SUPER STARTREK INSTRUCTIONS
MAR 5, 1978

Just the instructions for SUPERSTARTREK

Ported by Dave LeCompte
"""


def get_yes_no(prompt: str) -> bool:
    response = input(prompt).upper()
    return response[0] != "N"


```

这段代码定义了一个名为 `print_header` 的函数，它接受一个 `None` 类型的参数。

函数内部使用了一个 `for` 循环，循环变量为 `_`，循环条件为 `range(12)`，每次循环都会先输出一个空行，然后输出一行字符串 `"**"`，接着输出一行字符串 `"*"`。然后输出一行字符串 `"***"`。

接下来，代码输出了一行字符串 `"*美容王 Trainer Model *"`。

再往下，代码输出了两行字符串 `"***重温甲子行，差趣横生***"`。

下面是完整的 `print_header` 函数：

```
def print_header():
   for _ in range(12):
       print()
   t10 = " " * 10
   print(t10 + "*************************************")
   print(t10 + "*                                   *")
   print(t10 + "*                                   *")
   print(t10 + "*      * * SUPER STAR TREK * *      *")
   print(t10 + "*                                   *")
   print(t10 + "*                                   *")
   print(t10 + "*************************************")
   for _ in range(8):
       print()
```


```
def print_header() -> None:
    for _ in range(12):
        print()
    t10 = " " * 10
    print(t10 + "*************************************")
    print(t10 + "*                                   *")
    print(t10 + "*                                   *")
    print(t10 + "*      * * SUPER STAR TREK * *      *")
    print(t10 + "*                                   *")
    print(t10 + "*                                   *")
    print(t10 + "*************************************")
    for _ in range(8):
        print()


```

If the state of repair for all devices is negative, it means that the devices are temporarily damaged and cannot be used until they are repaired.

To display the state of repair for all devices, you can use the `print()` function with the format string `"Negative"`. This will display a message that indicates the state of repair for each device.

To display the state of repair for all devices in a more detailed format, you can use the `print()` function with the format string `"{\n    Negative )-{Information about devices}\n}"`. This will display a message that includes information about the state of repair for each device, such as the type of damage and the number of devices affected.

To allow the user to specify the library to use for the command, you can use the `COMMAND = LIBRARY-COMPUTER` format string. This will prompt the user to enter the name of the library they want to use, and then use that library for the command.

You can also use the `OPTION 0` format string to display the cumulative galactic record of the results of all previous short and long range sensor scans. This will display a summary of the sensor scans, including the number of scans and the distances from the enterprise.

To display the state of repair for all devices in a more detailed format, you can use the `OPTION 1` format string. This will display the number of kilクリンgoldings, stardates, and stargbases remaining in the game.

To display the direction and distance from the enterprise for all stargbases, you can use the `OPTION 2` format string. This will prompt the user to enter the direction and distance from the enterprise for each starghesis, and then use that information to calculate the direction and distance from the enterprise.

To display the state of repair for all devices in a more detailed format, you can use the `OPTION 3` format string. This will prompt the user to enter the name of the stargisphere they want to calculate the direction and distance for, and then use that information to calculate the direction and distance from the enterprise.

To display the state of repair for all devices in a more detailed format, you can use the `OPTION 4` format string. This will prompt the user to enter the coordinates for the direction and distance calculations, and then use that information to calculate the direction and distance from the enterprise.

To display the state of repair for all devices in a more detailed format, you can use the `OPTION 5` format string. This will prompt the user to enter the name of the major galactic region they want to print the names of, and then use that information to print the names of the十六 major galactic regions referred to in the game.


```
def print_instructions() -> None:
    # Back in the 70s, at this point, the user would be prompted to
    # turn on their (printing) TTY to capture the output to hard copy.

    print("      INSTRUCTIONS FOR 'SUPER STAR TREK'")
    print()
    print("1. WHEN YOU SEE \\COMMAND ?\\ PRINTED, ENTER ONE OF THE LEGAL")
    print("     COMMANDS (NAV,SRS,LRS,PHA,TOR,SHE,DAM,COM, OR XXX).")
    print("2. IF YOU SHOULD TYPE IN AN ILLEGAL COMMAND, YOU'LL GET A SHORT")
    print("     LIST OF THE LEGAL COMMANDS PRINTED OUT.")
    print("3. SOME COMMANDS REQUIRE YOU TO ENTER DATA (FOR EXAMPLE, THE")
    print("     'NAV' COMMAND COMES BACK WITH 'COURSE (1-9) ?'.)  IF YOU")
    print("     TYPE IN ILLEGAL DATA (LIKE NEGATIVE NUMBERS), THAN COMMAND")
    print("     WILL BE ABORTED")
    print()
    print("     THE GALAXY IS DIVIDED INTO AN 8 X 8 QUADRANT GRID,")
    print("AND EACH QUADRANT IS FURTHER DIVIDED INTO AN 8 X 8 SECTOR GRID.")
    print()
    print("     YOU WILL BE ASSIGNED A STARTING POINT SOMEWHERE IN THE")
    print("GALAXY TO BEGIN A TOUR OF DUTY AS COMANDER OF THE STARSHIP")
    print("\\ENTERPRISE\\; YOUR MISSION: TO SEEK AND DESTROY THE FLEET OF")
    print("KLINGON WARWHIPS WHICH ARE MENACING THE UNITED FEDERATION OF")
    print("PLANETS.")
    print()
    print("     YOU HAVE THE FOLLOWING COMMANDS AVAILABLE TO YOU AS CAPTAIN")
    print("OF THE STARSHIP ENTERPRISE:")
    print()
    print("\\NAV\\ COMMAND = WARP ENGINE CONTROL --")
    print("     COURSE IS IN A CIRCULAR NUMERICAL      4  3  2")
    print("     VECTOR ARRANGEMENT AS SHOWN             . . .")
    print("     INTEGER AND REAL VALUES MAY BE           ...")
    print("     USED.  (THUS COURSE 1.5 IS HALF-     5 ---*--- 1")
    print("     WAY BETWEEN 1 AND 2                      ...")
    print("                                             . . .")
    print("     VALUES MAY APPROACH 9.0, WHICH         6  7  8")
    print("     ITSELF IS EQUIVALENT TO 1.0")
    print("                                            COURSE")
    print("     ONE WARP FACTOR IS THE SIZE OF ")
    print("     ONE QUADTANT.  THEREFORE, TO GET")
    print("     FROM QUADRANT 6,5 TO 5,5, YOU WOULD")
    print("     USE COURSE 3, WARP FACTOR 1.")
    print()
    print("\\SRS\\ COMMAND = SHORT RANGE SENSOR SCAN")
    print("     SHOWS YOU A SCAN OF YOUR PRESENT QUADRANT.")
    print()
    print("     SYMBOLOGY ON YOUR SENSOR SCREEN IS AS FOLLOWS:")
    print("        <*> = YOUR STARSHIP'S POSITION")
    print("        +K+ = KLINGON BATTLE CRUISER")
    print("        >!< = FEDERATION STARBASE (REFUEL/REPAIR/RE-ARM HERE!)")
    print("         *  = STAR")
    print()
    print("     A CONDENSED 'STATUS REPORT' WILL ALSO BE PRESENTED.")
    print()
    print("\\LRS\\ COMMAND = LONG RANGE SENSOR SCAN")
    print("     SHOWS CONDITIONS IN SPACE FOR ONE QUADRANT ON EACH SIDE")
    print("     OF THE ENTERPRISE (WHICH IS IN THE MIDDLE OF THE SCAN)")
    print("     THE SCAN IS CODED IN THE FORM \\###\\, WHERE TH UNITS DIGIT")
    print("     IS THE NUMBER OF STARS, THE TENS DIGIT IS THE NUMBER OF")
    print("     STARBASES, AND THE HUNDRESDS DIGIT IS THE NUMBER OF")
    print("     KLINGONS.")
    print()
    print("     EXAMPLE - 207 = 2 KLINGONS, NO STARBASES, & 7 STARS.")
    print()
    print("\\PHA\\ COMMAND = PHASER CONTROL.")
    print("     ALLOWS YOU TO DESTROY THE KLINGON BATTLE CRUISERS BY ")
    print("     ZAPPING THEM WITH SUITABLY LARGE UNITS OF ENERGY TO")
    print("     DEPLETE THEIR SHIELD POWER.  (REMEMBER, KLINGONS HAVE")
    print("     PHASERS TOO!)")
    print()
    print("\\TOR\\ COMMAND = PHOTON TORPEDO CONTROL")
    print("     TORPEDO COURSE IS THE SAME AS USED IN WARP ENGINE CONTROL")
    print("     IF YOU HIT THE KLINGON VESSEL, HE IS DESTROYED AND")
    print("     CANNOT FIRE BACK AT YOU.  IF YOU MISS, YOU ARE SUBJECT TO")
    print("     HIS PHASER FIRE.  IN EITHER CASE, YOU ARE ALSO SUBJECT TO ")
    print("     THE PHASER FIRE OF ALL OTHER KLINGONS IN THE QUADRANT.")
    print()
    print("     THE LIBRARY-COMPUTER (\\COM\\ COMMAND) HAS AN OPTION TO ")
    print("     COMPUTE TORPEDO TRAJECTORY FOR YOU (OPTION 2)")
    print()
    print("\\SHE\\ COMMAND = SHIELD CONTROL")
    print("     DEFINES THE NUMBER OF ENERGY UNITS TO BE ASSIGNED TO THE")
    print("     SHIELDS.  ENERGY IS TAKEN FROM TOTAL SHIP'S ENERGY.  NOTE")
    print("     THAN THE STATUS DISPLAY TOTAL ENERGY INCLUDES SHIELD ENERGY")
    print()
    print("\\DAM\\ COMMAND = DAMMAGE CONTROL REPORT")
    print("     GIVES THE STATE OF REPAIR OF ALL DEVICES.  WHERE A NEGATIVE")
    print("     'STATE OF REPAIR' SHOWS THAT THE DEVICE IS TEMPORARILY")
    print("     DAMAGED.")
    print()
    print("\\COM\\ COMMAND = LIBRARY-COMPUTER")
    print("     THE LIBRARY-COMPUTER CONTAINS SIX OPTIONS:")
    print("     OPTION 0 = CUMULATIVE GALACTIC RECORD")
    print("        THIS OPTION SHOWES COMPUTER MEMORY OF THE RESULTS OF ALL")
    print("        PREVIOUS SHORT AND LONG RANGE SENSOR SCANS")
    print("     OPTION 1 = STATUS REPORT")
    print("        THIS OPTION SHOWS THE NUMBER OF KLINGONS, STARDATES,")
    print("        AND STARBASES REMAINING IN THE GAME.")
    print("     OPTION 2 = PHOTON TORPEDO DATA")
    print("        WHICH GIVES DIRECTIONS AND DISTANCE FROM THE ENTERPRISE")
    print("        TO ALL KLINGONS IN YOUR QUADRANT")
    print("     OPTION 3 = STARBASE NAV DATA")
    print("        THIS OPTION GIVES DIRECTION AND DISTANCE TO ANY ")
    print("        STARBASE WITHIN YOUR QUADRANT")
    print("     OPTION 4 = DIRECTION/DISTANCE CALCULATOR")
    print("        THIS OPTION ALLOWS YOU TO ENTER COORDINATES FOR")
    print("        DIRECTION/DISTANCE CALCULATIONS")
    print("     OPTION 5 = GALACTIC /REGION NAME/ MAP")
    print("        THIS OPTION PRINTS THE NAMES OF THE SIXTEEN MAJOR ")
    print("        GALACTIC REGIONS REFERRED TO IN THE GAME.")


```

这段代码是一个Python程序，名为"main"。程序导出了一个名为"None"的类，这意味着这是一个不会修改任何外部对象的类。

程序中的两个函数分别名为"print_header"和"print_instructions"，但它们在函数内部没有定义任何函数体，因此这两个函数不会输出任何内容。

程序中的一个名为"if __name__ == "__main__":`if`语句用于检查当前目录是否保存了一个名为"main.py"的Python文件。如果是，则程序将开始执行。

如果 __name__ == "__main__":`if` 语句为真，那么程序将执行 main 函数。main 函数将先调用 "print_header" 函数，然后调用 "print_instructions" 函数。

"print_header" 函数将在屏幕上输出 "DO YOU NEED INSTRUCTIONS (Y/N)? " 这条消息，然后等待用户输入一个 Y 或者 N。

"print_instructions" 函数在 "print_header" 函数完成后输出 "Welcome to the programming course!" 这条消息。


```
def main() -> None:
    print_header()
    if not get_yes_no("DO YOU NEED INSTRUCTIONS (Y/N)? "):
        return
    print_instructions()


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


# Super Star Trek - Rust version

Explanation of modules:

- [main.rs](./src/main.rs) - creates the galaxy (generation functions are in model.rs as impl methods) then loops listening for commands. after each command checks for victory or defeat condtions.
- [model.rs](./src/model.rs) - all the structs and enums that represent the galaxy. key methods in here (as impl methods) are generation functions on galaxy and quadrant, and various comparison methods on the 'Pos' tuple type.
- [commands.rs](./src/commands.rs) - most of the code that implements instructions given by the player (some code logic is in the model impls, and some in view.rs if its view only).
- [view.rs](./src/view.rs) - all text printed to the output, mostly called by command.rs (like view::bad_nav for example). also contains the prompts printed to the user (e.g. view::prompts::COMMAND).
- [input.rs](./src/input.rs) - utility methods for getting input from the user, including logic for parsing numbers, repeating prompts until a correct value is provided etc.

Basically the user is asked for the next command, this runs a function that usually checks if the command system is working, and if so will gather additional input (see next note for a slight change here), then either the model is read and info printed, or its mutated in some way (e.g. firing a torpedo, which reduces the torpedo count on the enterprise and can destroy klingons and star bases; finally the klingons fire back and can destroy the enterprise). Finally the win/lose conditions are checked before the loop repeats.

## Changes from the original

I have tried to keep it as close as possible. Notable changes are:

- commands can be given with parameters in line. e.g. while 'nav' will ask for course and then warp speed in the original, here you can *optionally* also do this as one line, e.g. `nav 1 0.1` to move one sector east. I'm sorry - it was driving me insane in its original form (which is still sorted, as is partial application e.g. nav 1 to preset direction and then provide speed).
- text is mostly not uppercase, as text was in the basic version. this would be easy to change however as all text is in view.rs, but I chose not to.
- the navigation system (plotting direction, paths and collision detection) is as close as I could make it to the basic version (by using other language conversions as specification sources) but I suspect is not perfect. seems to work well enough however.

Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Synonym

A synonym of a word is another word (in the English language) which has the same, or very nearly the same, meaning. This program tests your knowledge of synonyms of a few common words.

The computer chooses a word and asks you for a synonym. The computer then tells you whether you’re right or wrong. If you can’t think of a synonym, type “HELP” which causes a synonym to be printed.

You may put in words of your choice in the data statements. The number following DATA in Statement 500 is the total number of data statements. In each data statement, the first number is the number of words in that statement.

Can you think of a way to make this into a more general kind of CAI program for any subject?

Walt Koetke of Lexington High School, Massachusetts created this program.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=164)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=179)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

 - Each time the player asks for HELP, one of the synonyms is shown
   and discarded. There is no protection against the player using up
   all of the help.

 - The player can ask for HELP and then submit that answer. Is it
   meant to be a clue, or just giving a correct answer to the player?


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `85_Synonym/csharp/Synonym.cs`

This is a class that defines a program. The class defines a method called `AskForSynonyms`, which prompts the user to input a word and displays a list of synonyms. It also defines a method called `PlayTheGame`, which displays the game flow.

The `AskForSynonyms` method first generates a random index for a word in the list of words, displays the synonym prompt to the user, and then reads the user's response. It then checks if the response is successful and adds extra line space for formatting.

The `PlayTheGame` method generates a random list of words, displays the game flow, and calls the `AskForSynonyms` method to prompt the user for synonyms. It then displays the intro, the list of words, the user's response, and the outro.

The `Main` method is the entry point of the program. It instantiates the `Synonym` class and calls the `PlayTheGame` method.


```
﻿using System.Text;

namespace Synonym
{
    class Synonym
    {
        Random rand = new Random();

        // Initialize list of corrent responses
        private string[] Affirmations = { "Right", "Correct", "Fine", "Good!", "Check" };

        // Initialize list of words and their synonyms
        private string[][] Words =
        {
                new string[] {"first", "start", "beginning", "onset", "initial"},
                new string[] {"similar", "alike", "same", "like", "resembling"},
                new string[] {"model", "pattern", "prototype", "standard", "criterion"},
                new string[] {"small", "insignificant", "little", "tiny", "minute"},
                new string[] {"stop", "halt", "stay", "arrest", "check", "standstill"},
                new string[] {"house", "dwelling", "residence", "domicile", "lodging", "habitation"},
                new string[] {"pit", "hole", "hollow", "well", "gulf", "chasm", "abyss"},
                new string[] {"push", "shove", "thrust", "prod", "poke", "butt", "press"},
                new string[] {"red", "rouge", "scarlet", "crimson", "flame", "ruby"},
                new string[] {"pain", "suffering", "hurt", "misery", "distress", "ache", "discomfort"}
         };

        private void DisplayIntro()
        {
            Console.WriteLine("");
            Console.WriteLine("SYNONYM".PadLeft(23));
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine("");
            Console.WriteLine("A synonym of a word means another word in the English");
            Console.WriteLine("language which has the same or very nearly the same meaning.");
            Console.WriteLine("I choose a word -- you type a synonym.");
            Console.WriteLine("If you can't think of a synonym, type the word 'help'");
            Console.WriteLine("and I will tell you a synonym.");
            Console.WriteLine("");
        }

        private void DisplayOutro()
        {
            Console.WriteLine("Synonym drill completed.");
        }

        private void RandomizeTheList()
        {
            // Randomize the list of Words to pick from
            int[] Order = new int[Words.Length];
            foreach (int i in Order)
            {
                Order[i] = rand.Next();
            }
            Array.Sort(Order, Words);
        }

        private string GetAnAffirmation()
        {
            return Affirmations[rand.Next(Affirmations.Length)];
        }

        private bool CheckTheResponse(string WordName, int WordIndex, string LineInput, string[] WordList)
        {
            if (LineInput.Equals("help"))
            {
                // Choose a random correct synonym response that doesn't equal the current word given
                int HelpIndex = rand.Next(WordList.Length);
                while (HelpIndex == WordIndex)
                {
                    HelpIndex = rand.Next(0, WordList.Length);
                }
                Console.WriteLine("**** A synonym of {0} is {1}.", WordName, WordList[HelpIndex]);

                return false;
            }
            else
            {
                // Check to see if the response is one of the listed synonyms and not the current word prompt
                if (WordList.Contains(LineInput) && LineInput != WordName)
                {
                    // Randomly display one of the five correct answer exclamations
                    Console.WriteLine(GetAnAffirmation());

                    return true;
                }
                else
                {
                    // Incorrect response.  Try again.
                    Console.WriteLine("     Try again.".PadLeft(5));

                    return false;
                }
            }
        }

        private string PromptForSynonym(string WordName)
        {
            Console.Write("     What is a synonym of {0}? ", WordName);
            string LineInput = Console.ReadLine().Trim().ToLower();

            return LineInput;
        }

        private void AskForSynonyms()
        {
            Random rand = new Random();

            // Loop through the now randomized list of Words and display a random word from each to prompt for a synonym
            foreach (string[] WordList in Words)
            {
                int WordIndex = rand.Next(WordList.Length);  // random word position in the current list of words
                string WordName = WordList[WordIndex];       // what is that actual word
                bool Success = false;

                while (!Success)
                {
                    // Ask for the synonym of the current word
                    string LineInput = PromptForSynonym(WordName);

                    // Check the response
                    Success = CheckTheResponse(WordName, WordIndex, LineInput, WordList);

                    // Add extra line space for formatting
                    Console.WriteLine("");
                }
            }
        }

        public void PlayTheGame()
        {
            RandomizeTheList();

            DisplayIntro();

            AskForSynonyms();

            DisplayOutro();
        }
    }
    class Program
    {
        static void Main(string[] args)
        {

            new Synonym().PlayTheGame();

        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `85_Synonym/java/src/Synonym.java`

Here are some suggestions for improving the Java code:

1. It is a good practice to use a try-catch block in all sections of the code that may throw an exception. This will help in organizing the code and making it easier to maintain.

2. The code is very tightly coupled to the Keyboard input. It would be better to separate the Input and Output logic. This will make the code more maintainable and testable.

3. The displayTextAndGetInput() method should take a String parameter instead of a String and a int parameter. This will make the method more flexible.

4. The code should use a more sophisticated naming conventions such as camelCaseNamingConvention instead of snake\_caseNamingConvention. This will make the code more maintainable and easy to understand.

5. The code should use more descriptive and meaningful variable names.

6. The code should have more in-depth documentation.

7. It is a good practice to remove any unnecessary code or variable.

8. The code should have a more sophisticated error handling mechanism.

9. The code should have a more robust design.

10. It is a good practice to write test cases for all the functions. This will help in ensuring that the code is working as expected and will make the code more maintainable.


```
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Synonym
 * <p>
 * Based on the Basic game of Synonym here
 * https://github.com/coding-horror/basic-computer-games/blob/main/85%20Synonym/synonym.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Synonym {

    public static final String[] RANDOM_ANSWERS = {"RIGHT", "CORRECT", "FINE", "GOOD!", "CHECK"};

    // Used for keyboard input
    private final Scanner kbScanner;

    // List of words and synonyms
    private final ArrayList<SynonymList> synonyms;

    private enum GAME_STATE {
        INIT,
        PLAY,
        GAME_OVER
    }

    // Current game state
    private GAME_STATE gameState;

    private int currentQuestion;

    public Synonym() {

        kbScanner = new Scanner(System.in);
        synonyms = new ArrayList<>();

        gameState = GAME_STATE.INIT;
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                case INIT:
                    intro();
                    currentQuestion = 0;

                    // Load data
                    synonyms.add(new SynonymList("FIRST", new String[]{"START", "BEGINNING", "ONSET", "INITIAL"}));
                    synonyms.add(new SynonymList("SIMILAR", new String[]{"SAME", "LIKE", "RESEMBLING"}));
                    synonyms.add(new SynonymList("MODEL", new String[]{"PATTERN", "PROTOTYPE", "STANDARD", "CRITERION"}));
                    synonyms.add(new SynonymList("SMALL", new String[]{"INSIGNIFICANT", "LITTLE", "TINY", "MINUTE"}));
                    synonyms.add(new SynonymList("STOP", new String[]{"HALT", "STAY", "ARREST", "CHECK", "STANDSTILL"}));
                    synonyms.add(new SynonymList("HOUSE", new String[]{"DWELLING", "RESIDENCE", "DOMICILE", "LODGING", "HABITATION"}));
                    synonyms.add(new SynonymList("PIT", new String[]{"HOLE", "HOLLOW", "WELL", "GULF", "CHASM", "ABYSS"}));
                    synonyms.add(new SynonymList("PUSH", new String[]{"SHOVE", "THRUST", "PROD", "POKE", "BUTT", "PRESS"}));
                    synonyms.add(new SynonymList("RED", new String[]{"ROUGE", "SCARLET", "CRIMSON", "FLAME", "RUBY"}));
                    synonyms.add(new SynonymList("PAIN", new String[]{"SUFFERING", "HURT", "MISERY", "DISTRESS", "ACHE", "DISCOMFORT"}));

                    gameState = GAME_STATE.PLAY;
                    break;

                case PLAY:

                    // Get the word and synonyms to ask a question about
                    SynonymList synonym = synonyms.get(currentQuestion);
                    String getAnswer = displayTextAndGetInput("     WHAT IS A SYNONYM OF " + synonym.getWord() + " ? ");

                    // HELP is used to give a random synonym for the current word
                    if (getAnswer.equals("HELP")) {
                        int randomSynonym = (int) (Math.random() * synonym.size());
                        System.out.println("**** A SYNONYM OF " + synonym.getWord() + " IS " + synonym.getSynonyms()[randomSynonym] + ".");
                    } else {
                        // Check if the entered word is in the synonym list
                        if (synonym.exists(getAnswer)) {
                            // If it is, give a random "correct" response
                            System.out.println(RANDOM_ANSWERS[(int) (Math.random() * RANDOM_ANSWERS.length)]);
                            currentQuestion++;
                            // Have we reached the final word/synonyms on file?
                            if (currentQuestion == synonyms.size()) {
                                // We have so end game.
                                System.out.println("SYNONYM DRILL COMPLETED.");
                                gameState = GAME_STATE.GAME_OVER;
                            }
                        } else {
                            // Word does not exist in the synonym list
                            System.out.println("TRY AGAIN.");
                        }
                    }
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    private void intro() {
        System.out.println(simulateTabs(33) + "SYNONYM");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("A SYNONYM OF A WORD MEANS ANOTHER WORD IN THE ENGLISH");
        System.out.println("LANGUAGE WHICH HAS THE SAME OR VERY NEARLY THE SAME");
        System.out.println(" MEANING.");
        System.out.println("I CHOOSE A WORD -- YOU TYPE A SYNONYM.");
        System.out.println("IF YOU CAN'T THINK OF A SYNONYM, TYPE THE WORD 'HELP'");
        System.out.println("AND I WILL TELL YOU A SYNONYM.");
        System.out.println();
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to uppercase.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next().toUpperCase();
    }

    /**
     * Simulate the old basic tab(xx) command which indented text by xx spaces.
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }
}

```

# `85_Synonym/java/src/SynonymGame.java`

该代码创建了一个名为SynonymGame的Java类。在这个类中，有两个静态方法，分别是play()和synonym.play()。play()方法的具体作用如下：

1. 创建一个名为Synonym的类对象synonym，并将其赋值给变量synonym。
2.调用synonym对象中的play()方法，该方法的作用不明确，没有提供任何输出或执行任何操作。

因此，该代码的作用是创建一个SynonymGame对象，并调用其play()方法，但play()方法本身没有具体的作用，需要根据上下文来确定。


```
public class SynonymGame {
    public static void main(String[] args) {
        Synonym synonym = new Synonym();
        synonym.play();
    }
}

```

# `85_Synonym/java/src/SynonymList.java`

这段代码定义了一个名为 `SynonymList` 的类，用于存储一个单词及其同义词列表。同义词存储在一个名为 `ArrayList` 的对象中，该对象实现了 `contains` 方法以检查单词是否存在于同义词列表中。

`exists` 方法使用 `anyMatch` 方法来检查单词是否存在于同义词列表中。如果存在，则返回 `true`，否则返回 `false`。

`getWord` 方法返回存储在 `this` 对象中的单词。

`size` 方法返回同义词列表中的单词数量。

`getSynonyms` 方法返回同义词列表中的所有单词，以字符串数组格式返回。


```
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Stores a word and a list of synonyms for that word
 */
public class SynonymList {

    private final String word;

    private final ArrayList<String> synonyms;

    public SynonymList(String word, String[] synonyms) {
        this.word = word;
        this.synonyms = new ArrayList<>(Arrays.asList(synonyms));
    }

    /**
     * Check if the word passed to this method exists in the list of synonyms
     * N.B. Case insensitive
     *
     * @param word word to search for
     * @return true if found, otherwise false
     */
    public boolean exists(String word) {
        return synonyms.stream().anyMatch(str -> str.equalsIgnoreCase(word));
    }

    public String getWord() {
        return word;
    }

    public int size() {
        return synonyms.size();
    }

    /**
     * Returns all synonyms for this word in string array format
     *
     * @return
     */
    public String[] getSynonyms() {
        // Parameter to toArray method determines type of the resultant array
        return synonyms.toArray(new String[0]);
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `85_Synonym/javascript/synonym.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在文档中创建一个输出元素（在代码中是通过 `document.getElementById` 获取输出元素的 ID，输出到该元素中），并将一个字符串作为参数传递给该函数。这个字符串会被追加到预先创建好的输出元素中，字符串中的每个字符都会被将其转换为 `<br>` 标签，以便在输出时正确地换行。

`input` 函数的作用是获取用户输入的字符串，它会提示用户输入一个字符串，并将其存储在变量 `input_str` 中。该函数通过添加一个输入框（`document.createElement("INPUT")`）和一个空格（`document.getElementById("output").appendChild(document.createTextNode(" "))`），将用户输入的字符串存储到输入框中。函数会在用户点击输入框时获取用户的输入，并将其存储在 `input_str` 变量中。然后，函数会将用户输入的字符串追加到预先创建好的输出元素中，并输出一个换行符（`print(" ");`）。最后，函数会将 `input_str` 中的字符串打印出来，并将其存储为输入框中的值。


```
// SYNONYM
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

```

这是一段 JavaScript 代码，定义了一个名为 "tab" 的函数，用于将结果字符串中的空白字符（ space）替换为指定的单词。

函数接收一个参数 "space"，代表要替换的空格数量。函数内部使用 while 循环，每次循环代表替换一个空格。在循环变量 "space" 递减时，将替换到的单词添加到字符串中。

代码中还定义了两个变量：ra 和 la。ra 数组包含了 6 个单词，分别是 "RIGHT"、"CORRECT"、"FINE"、"GOOD!"、"CHECK" 和 "SPACE"。la 数组包含了 7 个单词，与ra数组中的单词有相同的长度，但是只包含了与 "SPACE" 变量相同位置的单词。tried 数组记录了已经尝试过的单词，以防止重复。

该函数的作用是替换 "SPACE" 变量中的每个空格，生成一个与输入单词相同长度的结果字符串。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var ra = [, "RIGHT", "CORRECT", "FINE", "GOOD!", "CHECK"];
var la = [];
var tried = [];

var synonym = [[5,"FIRST","START","BEGINNING","ONSET","INITIAL"],
               [5,"SIMILAR","ALIKE","SAME","LIKE","RESEMBLING"],
               [5,"MODEL","PATTERN","PROTOTYPE","STANDARD","CRITERION"],
               [5,"SMALL","INSIGNIFICANT","LITTLE","TINY","MINUTE"],
               [6,"STOP","HALT","STAY","ARREST","CHECK","STANDSTILL"],
               [6,"HOUSE","DWELLING","RESIDENCE","DOMICILE","LODGING","HABITATION"],
               [7,"PIT","HOLE","HOLLOW","WELL","GULF","CHASM","ABYSS"],
               [7,"PUSH","SHOVE","THRUST","PROD","POKE","BUTT","PRESS"],
               [6,"RED","ROUGE","SCARLET","CRIMSON","FLAME","RUBY"],
               [7,"PAIN","SUFFERING","HURT","MISERY","DISTRESS","ACHE","DISCOMFORT"]
               ];

```

This is a JavaScript program that will drill down a list of synonyms, given a single word, until it finds a valid synonym or exhausts the list. The program will then output the synonyms found for the input word.

The program starts by setting a variable `c` to 0 and a variable `synonym` to an empty list.

It then enters a while loop that runs until the user enters "HELP". Inside the loop, the program will ask the user to input a synonym for the given word.

If the user enters "HELP", the program will exit and start a new one. Otherwise, the program will loop through the synonyms in the `synonym` list. For each synonym, the program will check if it finds a match in the `synonym` list.

If it finds a match, the program will replace the first word with the last word in the synonym and continue to the next iteration of the loop. If it does not find a match, the program will print a message and continue to the next iteration.

The program will repeat this process until the user finds a valid synonym or exhausts the `synonym` list. Once the loop is complete, the program will print a message and exit.

Overall, the program is designed to be a simple and effective tool for finding synonyms for a given word.


```
// Main program
async function main()
{
    print(tab(33) + "SYNONYM\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    for (c = 0; c <= synonym.length; c++)
        tried[c] = false;
    print("A SYNONYM OF A WORD MEANS ANOTHER WORD IN THE ENGLISH\n");
    print("LANGUAGE WHICH HAS THE SAME OR VERY NEARLY THE SAME");
    print(" MEANING.\n");
    print("I CHOOSE A WORD -- YOU TYPE A SYNONYM.\n");
    print("IF YOU CAN'T THINK OF A SYNONYM, TYPE THE WORD 'HELP'\n");
    print("AND I WILL TELL YOU A SYNONYM.\n");
    print("\n");
    c = 0;
    while (c < synonym.length) {
        c++;
        do {
            n1 = Math.floor(Math.random() * synonym.length + 1);
        } while (tried[n1]) ;
        tried[n1] = true;
        n2 = synonym[n1][0];    // Length of synonym list
        // This array keeps a list of words not shown
        for (j = 1; j <= n2; j++)
            la[j] = j;
        la[0] = n2;
        g = 1;  // Always show first word
        print("\n");
        la[g] = la[la[0]];  // Replace first word with last word
        la[0] = n2 - 1; // Reduce size of list by one.
        print("\n");
        while (1) {
            print("     WHAT IS A SYNONYM OF " + synonym[n1][g]);
            str = await input();
            if (str == "HELP") {
                g1 = Math.floor(Math.random() * la[0] + 1);
                print("**** A SYNONYM OF " + synonym[n1][g] + " IS " + synonym[n1][la[g1]] + ".\n");
                print("\n");
                la[g1] = la[la[0]];
                la[0]--;
                continue;
            }
            for (k = 1; k <= n2; k++) {
                if (g == k)
                    continue;
                if (str == synonym[n1][k])
                    break;
            }
            if (k > n2) {
                print("     TRY AGAIN.\n");
            } else {
                print(synonym[n1][Math.floor(Math.random() * 5 + 1)] + "\n");
                break;
            }
        }
    }
    print("\n");
    print("SYNONYM DRILL COMPLETED.\n");
}

```

这是经典的 "Hello, World!" 程序，用于在 C 语言环境中打印出 "Hello, World!" 这个字符串。

在 C 语言中，`main()` 函数是程序的入口点，当程序运行时，首先执行的就是这个函数。因此，`main()` 函数可以被视为程序的 "门面"，用来向程序外部表明程序的功能和作用。

当程序运行时，首先会读取输入文件中的输入数据，如果输入文件不存在，则会提示用户输入数据。然后，程序会根据输入的数据执行不同的操作，这些操作就是程序的功能。

例如，如果程序是一个文本编辑器，用户可以通过它来打开、编辑和保存文本文件。程序需要读取用户输入的一些信息，比如文本文件的内容、保存的位置、用户名、密码等等。

`main()` 函数是程序的核心，负责程序的输入、输出、读取、执行等操作。它是程序的一扇门，让程序和用户之间建立了一个联系，让程序知道了如何接收用户的输入，如何与用户进行交互。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)

I used List::Util to do all the heavy work to show that perl can handle all the various
array functions.  It would be interesting to see a version that handled all of this
manually as there ended up being very little code left in this program.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `85_Synonym/python/synonym.py`

这段代码是一个Python程序，它的目的是提供一场词汇测试。程序包括以下两个主要部分：

1. 导入random模块，用于从标准随机数生成器中获取随机数。
2. 定义了一个名为PAGE_WIDTH的变量，其值为64，表示每行最多可以输出64个字符。
3. 定义了一个名为print_centered的函数，它接收一个字符串参数msg，然后将其居中显示。
4. 在print_centered函数内部，使用spaces变量来获取需要填充的空间，然后使用print函数将msg字符串填充到空间中。最后，使用spaces和msg的组合来获取居中后的字符串，并将其打印出来。
5. 在主程序部分，使用print_centered函数打印出"Vocabulary quiz"这个测试标题。

总之，这段代码的主要目的是提供一个简单的词汇测试，程序会随机选择一些单词，然后让用户判断这些单词是否正确，测试结束后将输出测试结果。


```
"""
SYNONYM

Vocabulary quiz

Ported by Dave LeCompte
"""

import random

PAGE_WIDTH = 64


def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


```

这两函数主要作用是输出一个带标题的文本，并对文本进行水平居中以及增加行距。

第一个函数 `print_header` 接收一个字符串参数 `title`，并使用 `print_centered` 函数对其进行居中处理。接着在已经居中处理过的文本上，再次使用 `print_centered` 函数输出 "CREATIVE COMPUTING MORRISTOWN, NEW JERSEY"，然后分别输出四个制表符，使得文本在水平方向上产生四个空行。

第二个函数 `print_instructions` 同样接收一个字符串参数 `instructions`，然后对其中的每一行进行居中处理。接着输出 "A SYNONYM OF A WORD MEANS ANOTHER WORD IN THE ENGLISH"，"LANGUAGE WHICH HAS THE SAME OR VERY NEARLY THE SAME MEANING"，"I CHOOSE A WORD -- YOU TYPE A SYNONYM"，然后是两个制表符，最后一个字符串结束。


```
def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print()
    print()
    print()


def print_instructions() -> None:
    print("A SYNONYM OF A WORD MEANS ANOTHER WORD IN THE ENGLISH")
    print("LANGUAGE WHICH HAS THE SAME OR VERY NEARLY THE SAME MEANING.")
    print("I CHOOSE A WORD -- YOU TYPE A SYNONYM.")
    print("IF YOU CAN'T THINK OF A SYNONYM, TYPE THE WORD 'HELP'")
    print("AND I WILL TELL YOU A SYNONYM.")
    print()


```

以上代码定义了两个列表right_words和synonym_words。right_words是一个包含六个单词的列表，它们都是正确的或适当的表达语。synonym_words是一个包含六个单词的列表，它们是类似于right_words中单词的近义词或替换词。这两个列表都是通过对单词进行语义相似性比较来生成的。


```
right_words = ["RIGHT", "CORRECT", "FINE", "GOOD!", "CHECK"]

synonym_words = [
    ["FIRST", "START", "BEGINNING", "ONSET", "INITIAL"],
    ["SIMILAR", "ALIKE", "SAME", "LIKE", "RESEMBLING"],
    ["MODEL", "PATTERN", "PROTOTYPE", "STANDARD", "CRITERION"],
    ["SMALL", "INSIGNIFICANT", "LITTLE", "TINY", "MINUTE"],
    ["STOP", "HALT", "STAY", "ARREST", "CHECK", "STANDSTILL"],
    ["HOUSE", "DWELLING", "RESIDENCE", "DOMICILE", "LODGING", "HABITATION"],
    ["PIT", "HOLE", "HOLLOW", "WELL", "GULF", "CHASM", "ABYSS"],
    ["PUSH", "SHOVE", "THRUST", "PROD", "POKE", "BUTT", "PRESS"],
    ["RED", "ROUGE", "SCARLET", "CRIMSON", "FLAME", "RUBY"],
    ["PAIN", "SUFFERING", "HURT", "MISERY", "DISTRESS", "ACHE", "DISCOMFORT"],
]


```

这两函数是在英语中用于从一系列同义词中选择一个单词的函数。

print_right函数的作用是打印出从right_words列表中随机选择的单词。

ask_question函数的作用是询问用户一个question_number，它对应的是同义词列表中的单词。这个函数从words列表中选择一个单词，然后从clues列表中选择一个或多个与base_word同义的单词。如果用户选择的是“HELP”，那么程序就不会询问用户选择同义词，而是直接从words列表中选择一个单词并返回。


```
def print_right() -> None:
    print(random.choice(right_words))


def ask_question(question_number: int) -> None:
    words = synonym_words[question_number]
    clues = words[:]
    base_word = clues.pop(0)

    while True:
        question = f"     WHAT IS A SYNONYM OF {base_word}? "
        response = input(question).upper()

        if response == "HELP":
            clue = random.choice(clues)
            print(f"**** A SYNONYM OF {base_word} IS {clue}.")
            print()

            # remove the clue from available clues
            clues.remove(clue)
            continue

        if (response != base_word) and (response in words):
            print_right()
            return


```

这段代码定义了一个名为 "finish" 的函数，它接受一个空函数类型参数 "None"。

接下来，定义了一个名为 "main" 的函数，它也接受一个空函数类型参数 "None"。

在 "main" 函数中，首先打印一个包含 "SYNONYM DRILL COMPLETED." 的消息，然后使用一个 print_header() 函数打印一些关于用户输入的问题数量和问题的说明。

接下来，使用一个列表推导式获取用户输入的问题列表，其中使用 range() 函数获取输入数字的索引，然后使用随机移位函数 shuffle() 对其进行随机化。

在循环中，使用 ask_question() 函数来向用户询问每个问题。

最后，在 print_header() 函数中，打印 "SYNONYM"。


```
def finish() -> None:
    print()
    print("SYNONYM DRILL COMPLETED.")


def main() -> None:
    print_header("SYNONYM")
    print_instructions()

    num_questions = len(synonym_words)
    word_indices = list(range(num_questions))
    random.shuffle(word_indices)

    for word_number in word_indices:
        ask_question(word_number)

    finish()


```

这段代码是一个Python程序中的一个if语句，其作用是在程序运行时判断是否是作为主程序运行，如果是，那么程序会执行if语句内部的语句。

"__name__"是一个Python魔术字符，用于判断当前程序是否是作为主程序运行。如果程序是作为主程序运行，那么 "__name__"将等于 "__main__"，否则将等于 "__config__"。

if __name__ == "__main__":
   main()

这段代码的作用是判断当前程序是否是作为主程序运行，如果是，那么程序会执行if语句内部的语句，其中main()是一个函数，可能是用于执行一些操作。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by [Jadi](https://github.com/jadijadi)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Target

In this program, you are firing a weapon from a spaceship in 3-dimensional space. Your ship, the Starship Enterprise, is located at the origin (0,0,0) of a set of x,y,z coordinates. You will be told the approximate location of the target in 3-dimensional rectangular coordinates, the approximate angular deviation from the x and z axes in both radians and degrees, and the approximate distance to the target.

Given this information, you then proceed to shoot at the target. A shot within 20 kilometers of the target destroys it. After each shot, you are given information as to the position of the explosion of your shot and a somewhat improved estimate of the location of the target. Fortunately, this is just practice and the target doesn’t shoot back. After you have attained proficiency, you ought to be able to destroy a target in 3 or 4 shots. However, attaining proficiency might take a while!

The author is H. David Crockett of Fort Worth, Texas.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=165)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=180)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html


#### Porting Notes

(please note any difficulties or challenges in porting here)


# `86_Target/csharp/Angle.cs`

这段代码定义了一个名为Target的命名空间，其中包含一个内部类Angle。Angle类包含一个const类型的成员变量PI，以及一个private类型的成员变量DegreesPerRadian，它们都被定义为float类型。

在Angle类中，还包含一个private类型的成员变量_radians，它被定义为float类型。在这个成员变量中，可以用来计算radians，即将角度转换为弧度的值。

接下来，定义了两个Angle的静态方法，InDegrees和InRotations。这两个方法接受一个float类型的参数，分别将角度转换为角度制和旋转角度。

另外，还定义了一个Angle的静态类型的成员变量，Angle.可以用来访问Angle的静态类型的成员变量。

最后，还定义了一个operator float(Angle angle)，用来将Angle类型的角度转换为float类型的值。


```
namespace Target
{
    internal class Angle
    {
        // Use same precision for constants as original code
        private const float PI = 3.14159f;
        private const float DegreesPerRadian = 57.296f;

        private readonly float _radians;

        private Angle(float radians) => _radians = radians;

        public static Angle InDegrees(float degrees) => new (degrees / DegreesPerRadian);
        public static Angle InRotations(float rotations) => new (2 * PI * rotations);

        public static implicit operator float(Angle angle) => angle._radians;
    }
}

```

# `86_Target/csharp/Explosion.cs`

这段代码定义了一个名为"Explosion"的类，其内部包含一个名为"Explosion"的内部类。

在"Explosion"的内部类中，定义了一个名为"FromTarget"的属性，其类型为"Offset"，用于存储目标位置的偏移量。

定义了一个名为"DistanceToTarget"的属性，其类型为"float"，用于存储目标位置与物体的最远距离。

定义了一个名为"GetBearing"的函数，用于获取目标位置与法线之间的角度。

定义了一个名为"IsHit"的函数，用于判断物体是否与物体相碰，其中判断条件是物体距离目标位置是否小于或等于20。

定义了一个名为"IsTooClose"的函数，用于判断物体是否过于靠近目标位置，其中判断条件是物体距离目标位置是否小于或等于20。

最后，在"Explosion"的内部类中，定义了一个名为"Explosion"的函数，用于创建一个新的"Explosion"对象，需要提供目标位置和目标偏移量。函数内部将这些属性的值从构造函数中读取。


```
namespace Target
{
    internal class Explosion
    {
        private readonly Point _position;

        public Explosion(Point position, Offset targetOffset)
        {
            _position = position;
            FromTarget = targetOffset;
            DistanceToTarget = targetOffset.Distance;
        }

        public Point Position => _position;
        public Offset FromTarget { get; }
        public float DistanceToTarget { get; }
        public string GetBearing() => _position.GetBearing();

        public bool IsHit => DistanceToTarget <= 20;
        public bool IsTooClose => _position.Distance < 20;
    }
}

```

# `86_Target/csharp/FiringRange.cs`

这段代码定义了一个名为 "FiringRange" 的类，它包含一个内部类 "FiringRange"，以及一个私有变量 "IRandom"。这个类继承自 "Randomness" 类，它应该已经在 "using Games.Common.Randomness" 中导入。

在 "FiringRange" 类的 "FiringRange" 构造函数中，我们创建了一个 "IRandom" 类的实例，并将其存储在 "FiringRange" 类的实例中。

在 "FiringRange" 类的 "NextTarget" 方法中，我们使用 "IRandom.NextPosition" 方法获取一个随机位置，并将其存储在 "FiringRange" 类的实例中的 "NextTarget" 变量中。

在 "FiringRange" 类的 "Fire" 方法中，我们使用一个随机角度 "angleFromX" 和 "angleFromZ"，以及一个距离 "distance"。我们使用这两个角度和距离计算出一个新的爆炸位置，并将其存储在 "FiringRange" 类的实例中的 "explosionPosition" 变量中。我们还使用 "IRandom.NextPosition" 方法获取一个与 "FiringRange" 类的实例中的 "NextTarget" 变量互补的新位置，并将其存储在 "explosionPosition" 变量中。

因此，这个代码定义了一个用于生成随机爆炸的 "FiringRange" 类，它可以被用来在游戏地图中生成新的爆炸。


```
using Games.Common.Randomness;

namespace Target
{
    internal class FiringRange
    {
        private readonly IRandom _random;
        private Point _targetPosition;

        public FiringRange(IRandom random)
        {
            _random = random;
        }

        public Point NextTarget() =>  _targetPosition = _random.NextPosition();

        public Explosion Fire(Angle angleFromX, Angle angleFromZ, float distance)
        {
            var explosionPosition = new Point(angleFromX, angleFromZ, distance);
            var targetOffset = explosionPosition - _targetPosition;
            return new (explosionPosition, targetOffset);
        }
    }
}

```

# `86_Target/csharp/Game.cs`

This is a C# class written in Unity that defines a reporting system for shots fired by the player.

The class has several methods such as ReportMiss, ReportHit, and ReportAngle, which are used to report the effects of shots fired, including missed shots, hits, and angles of shots, respectively.

The class also defines a method called ReportFiringRange, which is responsible for calculating the distance a bullet should travel to achieve a certain firing range and is used by the ReportHit method.

Additionally, the class has a method called GetOffsetText, which is used to generate text for positive and negative shot reports.

Overall, this class is designed to provide a comprehensive reporting system for shots fired in a game.


```
using System;
using Games.Common.IO;

namespace Target
{
    internal class Game
    {
        private readonly IReadWrite _io;
        private readonly FiringRange _firingRange;
        private int _shotCount;

        public Game(IReadWrite io, FiringRange firingRange)
        {
            _io = io;
            _firingRange = firingRange;
        }

        public void Play()
        {
            _shotCount = 0;
            var target = _firingRange.NextTarget();
            _io.WriteLine(target.GetBearing());
            _io.WriteLine($"Target sighted: approximate coordinates:  {target}");

            while (true)
            {
                _io.WriteLine($"     Estimated distance: {target.EstimateDistance()}");
                _io.WriteLine();

                var explosion = Shoot();

                if (explosion.IsTooClose)
                {
                    _io.WriteLine("You blew yourself up!!");
                    return;
                }

                _io.WriteLine(explosion.GetBearing());

                if (explosion.IsHit)
                {
                    ReportHit(explosion.DistanceToTarget);
                    return;
                }

                ReportMiss(explosion);
            }
        }

        private Explosion Shoot()
        {
            var (xDeviation, zDeviation, distance) = _io.Read3Numbers(
                "Input angle deviation from X, angle deviation from Z, distance");
            _shotCount++;
            _io.WriteLine();

            return _firingRange.Fire(Angle.InDegrees(xDeviation), Angle.InDegrees(zDeviation), distance);
        }

        private void ReportHit(float distance)
        {
            _io.WriteLine();
            _io.WriteLine($" * * * HIT * * *   Target is non-functional");
            _io.WriteLine();
            _io.WriteLine($"Distance of explosion from target was {distance} kilometers.");
            _io.WriteLine();
            _io.WriteLine($"Mission accomplished in {_shotCount} shots.");
        }

        private void ReportMiss(Explosion explosion)
        {
            ReportMiss(explosion.FromTarget);
            _io.WriteLine($"Approx position of explosion:  {explosion.Position}");
            _io.WriteLine($"     Distance from target = {explosion.DistanceToTarget}");
            _io.WriteLine();
            _io.WriteLine();
            _io.WriteLine();
        }

        private void ReportMiss(Offset targetOffset)
        {
            ReportMiss(targetOffset.DeltaX, "in front of", "behind");
            ReportMiss(targetOffset.DeltaY, "to left of", "to right of");
            ReportMiss(targetOffset.DeltaZ, "above", "below");
        }

        private void ReportMiss(float delta, string positiveText, string negativeText) =>
            _io.WriteLine(delta >= 0 ? GetOffsetText(positiveText, delta) : GetOffsetText(negativeText, -delta));

        private static string GetOffsetText(string text, float distance) => $"Shot {text} target {distance} kilometers.";
    }
}

```

# `86_Target/csharp/Offset.cs`



这段代码定义了一个名为`Offset`的类，其作用是为了解决三角形拓扑问题。在大多数情况下，当我们想要在一个三角形中计算两个点之间的距离时，我们会遇到一个问题，即如果我们只知道两个点之间的向量，则无法计算出它们之间的实际距离。为了解决这个问题，我们可以使用Offset类。

Offset类包含一个`float`类型的变量`Distance`，它用于存储两个点之间的距离。此外，它还包含三个`float`类型的变量`DeltaX`,`DeltaY`,`DeltaZ`，它们分别用于存储两个点之间的垂直，水平和距离之差。

Offset类有一个构造函数，用于初始化上述三个变量。构造函数的参数为三个`float`类型的参数`deltaX`,`deltaY`,`deltaZ`，它们用于计算两个点之间的垂直，水平和距离之差。

Offset类的`public`字段`Offset`用于访问上述四个变量，而`private`字段`Distance`用于存储两个点之间的距离。

总结起来，Offset类用于在三角形拓扑问题中计算两个点之间的距离，并存储这些距离。


```
using System;

namespace Target
{
    internal class Offset
    {
        public Offset(float deltaX, float deltaY, float deltaZ)
        {
            DeltaX = deltaX;
            DeltaY = deltaY;
            DeltaZ = deltaZ;

            Distance = (float)Math.Sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ + deltaZ);
        }

        public float DeltaX { get; }
        public float DeltaY { get; }
        public float DeltaZ { get; }
        public float Distance { get; }
    }
}

```

# `86_Target/csharp/Point.cs`



这段代码定义了一个名为 Point 的内部类，用于计算从 X 轴和 Z 轴到指定点之间的距离。Point 类包含一个角度变量 angleFromX 和 angleFromZ，以及从距离、到 X 轴和从 Z 轴的距离变量 distance。

Point 类有一个内部方法 Distance，它计算从起点 (0,0,0) 到终点 (distance, distance, 0) 的距离，并返回它。

此外，Point 类还有一个方法 EstimateDistance，它使用另一组参数精度(precision) 来计算从点到指定位置的估计距离。此方法的实现与 Distance 方法的实现非常相似，只是输出结果中多了一些占位符(angleFromX, angleFromZ, distance)。

最后，Point 类定义了一个名为 operator - 的元组类型 operator，用于在两个点之间计算它们之间的向量差。


```
using System;

namespace Target
{
    internal class Point
    {
        private readonly float _angleFromX;
        private readonly float _angleFromZ;

        private readonly float _x;
        private readonly float _y;
        private readonly float _z;

        private int _estimateCount;

        public Point(Angle angleFromX, Angle angleFromZ, float distance)
        {
            _angleFromX = angleFromX;
            _angleFromZ = angleFromZ;
            Distance = distance;

            _x = distance * (float)Math.Sin(_angleFromZ) * (float)Math.Cos(_angleFromX);
            _y = distance * (float)Math.Sin(_angleFromZ) * (float)Math.Sin(_angleFromX);
            _z = distance * (float)Math.Cos(_angleFromZ);
        }

        public float Distance { get; }

        public float EstimateDistance() =>
            ++_estimateCount switch
            {
                1 => EstimateDistance(20),
                2 => EstimateDistance(10),
                3 => EstimateDistance(5),
                4 => EstimateDistance(1),
                _ => Distance
            };

        public float EstimateDistance(int precision) => (float)Math.Floor(Distance / precision) * precision;

        public string GetBearing() => $"Radians from X axis = {_angleFromX}   from Z axis = {_angleFromZ}";

        public override string ToString() => $"X= {_x}   Y = {_y}   Z= {_z}";

        public static Offset operator -(Point p1, Point p2) => new (p1._x - p2._x, p1._y - p2._y, p1._z - p2._z);
    }
}

```

# `86_Target/csharp/Program.cs`

这段代码是一个用于在Windows平台上玩 targeting game的应用程序。它包括以下主要部分：

1. 引入必要的命名空间和类：System、System.Reflection、Games.Common、Games. common.IO、Games. common.Randomness。
2. 定义一个Target类，该类将作为应用程序的主类。
3. 创建一个ConsoleIO类，用于与游戏进行交互，并使用RandomNumberGenerator类创建一个随机数生成器。
4.创建一个Game类，该类继承自Games. common.Randomness，用于管理游戏过程。
5.创建一个Program类，该类继承自System.Application。
6.编写一个Main方法，是应用程序的入口点。
7.在Main方法的内部，创建一个ConsoleIO实例，Game实例和一个FiringRange实例。
8.调用FiringRange实例的Play方法，并传递一个IO实例，一个表示是否再次玩游戏的函数和一个字符串，表示游戏已经准备好。
9.在Play方法中，显示游戏标题和游戏说明，然后进入一个无限循环，调用game.Play方法，每5秒钟重复一次，并在每次调用后输出游戏当前状态的信息。
10.在DisplayTitleAndInstructions方法中，从指定的文件中读取游戏说明的文本内容，并将其写入控制台。


```
﻿using System;
using System.Reflection;
using Games.Common.IO;
using Games.Common.Randomness;

namespace Target
{
    class Program
    {
        static void Main()
        {
            var io = new ConsoleIO();
            var game = new Game(io, new FiringRange(new RandomNumberGenerator()));

            Play(game, io, () => true);
        }

        public static void Play(Game game, TextIO io, Func<bool> playAgain)
        {
            DisplayTitleAndInstructions(io);

            while (playAgain())
            {
                game.Play();

                io.WriteLine();
                io.WriteLine();
                io.WriteLine();
                io.WriteLine();
                io.WriteLine();
                io.WriteLine("Next target...");
                io.WriteLine();
            }
        }

        private static void DisplayTitleAndInstructions(TextIO io)
        {
            using var stream = Assembly.GetExecutingAssembly()
                .GetManifestResourceStream("Target.Strings.TitleAndInstructions.txt");
            io.Write(stream);
        }
    }
}

```