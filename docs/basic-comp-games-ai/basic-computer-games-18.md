# BasicComputerGames源码解析 18

# `10_Blackjack/csharp/Player.cs`

这段代码定义了一个名为“Blackjack”的命名空间，其中定义了一个名为“Player”的类。这个类表示一个玩家，每个玩家的ID都是从0到9999999999，姓名是该ID加1的字符串，手上有两个分别名为“Hand”和“SecondHand”的手。该类还定义了与比赛相关的变量，如当前轮到的下注数、赢得的分数、以及总赢得的分数。


```
namespace Blackjack
{
    public class Player
    {
        public Player(int index)
        {
            Index = index;
            Name = (index + 1).ToString();
            Hand = new Hand();
            SecondHand = new Hand();
        }

        public int Index { get; private set; }

        public string Name { get; private set; }

        public Hand Hand { get; private set; }

        public Hand SecondHand { get; private set;}

        public int RoundBet { get; set; }

        public int RoundWinnings { get; set; }

        public int TotalWinnings { get; set; }
    }
}

```

# `10_Blackjack/csharp/Program.cs`

这段代码是一个C#程序，它提供了一个简单的21游戏的控制台应用程序。这个程序的作用是帮助玩家输入游戏中的每一轮的赌注，然后根据玩家的输入来决定是否继续游戏。

具体来说，这个程序首先会询问玩家是否需要说明书或者是否想从控制台获取输入。如果玩家选择不需要说明书，那么程序会提示玩家输入人数，然后创建一个具有指定人数的游戏实例，并开始玩这个游戏。

在游戏的过程中，程序会逐轮向所有玩家询问他们的下一步行动。如果某个玩家想要继续游戏，他需要输入“S”来选择下一步行动。如果某个玩家想要结束游戏，他需要输入“H”来输入结束标志。如果某个玩家在游戏过程中选择了“/”符号，他需要输入“s”来结束当前的上下文。

总之，这个程序的主要作用是提供一个简单易用的21游戏控制台应用程序，以便玩家可以方便地输入游戏中的每一轮赌注，并与程序进行交互。


```
﻿using System;

namespace Blackjack
{
    static class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("{0}BLACK JACK", new string(' ', 31));
            Console.WriteLine("{0}CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY", new string(' ', 15));
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();

            OfferInstructions();

            var numberOfPlayers = Prompt.ForInteger("Number of players?", 1, 6);
            var game = new Game(numberOfPlayers);
            game.PlayGame();
        }

        private static void OfferInstructions()
        {
            if (!Prompt.ForYesNo("Do you want instructions?"))
                return;

            Console.WriteLine("This is the game of 21. As many as 7 players may play the");
            Console.WriteLine("game. On each deal, bets will be asked for, and the");
            Console.WriteLine("players' bets should be typed in. The cards will then be");
            Console.WriteLine("dealt, and each player in turn plays his hand. The");
            Console.WriteLine("first response should be either 'D', indicating that the");
            Console.WriteLine("player is doubling down, 'S', indicating that he is");
            Console.WriteLine("standing, 'H', indicating he wants another card, or '/',");
            Console.WriteLine("indicating that he wants to split his cards. After the");
            Console.WriteLine("initial response, all further responses should be 's' or");
            Console.WriteLine("'H', unless the cards were split, in which case doubling");
            Console.WriteLine("down is again permitted. In order to collect for");
            Console.WriteLine("Blackjack, the initial response should be 'S'.");
        }
    }
}

```

# `10_Blackjack/csharp/Prompt.cs`

This is a class called `Prompt` that contains functions for generating human-like input from the user.

The `ForYesNo` function prompts the user to enter "y" or "n" and returns a boolean value indicating the user's response.

The `ForInteger` function prompts the user to enter a number between the specified minimum and maximum values ( inclusive ) and returns the entered number.

The `ForCommandCharacter` function prompts the user to enter a single character from a list of allowed characters and returns the character as a string.

It also has a `WriteNotUnderstood` function which will print "Sorry, I didn't understand." if the user does not respond to the prompt.


```
using System;

namespace Blackjack
{
    public static class Prompt
    {
        public static bool ForYesNo(string prompt)
        {
            while(true)
            {
                Console.Write("{0} ", prompt);
                var input = Console.ReadLine();
                if (input.StartsWith("y", StringComparison.InvariantCultureIgnoreCase))
                    return true;
                if (input.StartsWith("n", StringComparison.InvariantCultureIgnoreCase))
                    return false;
                WriteNotUnderstood();
            }
        }

        public static int ForInteger(string prompt, int minimum = 1, int maximum = int.MaxValue)
        {
            while (true)
            {
                Console.Write("{0} ", prompt);
                if (!int.TryParse(Console.ReadLine(), out var number))
                    WriteNotUnderstood();
                else if (number < minimum || number > maximum)
                    Console.WriteLine("Sorry, I need a number between {0} and {1}.", minimum, maximum);
                else
                    return number;
            }
        }

        public static string ForCommandCharacter(string prompt, string allowedCharacters)
        {
            while (true)
            {
                Console.Write("{0} ", prompt);
                var input = Console.ReadLine();
                if (input.Length > 0)
                {
                    var character = input.Substring(0, 1);
                    var characterIndex = allowedCharacters.IndexOf(character, StringComparison.InvariantCultureIgnoreCase);
                    if (characterIndex != -1)
                        return allowedCharacters.Substring(characterIndex, 1);
                }

                Console.WriteLine("Type one of {0} please", String.Join(", ", allowedCharacters.ToCharArray()));
            }
        }

        private static void WriteNotUnderstood()
        {
            Console.WriteLine("Sorry, I didn't understand.");
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `10_Blackjack/java/src/Blackjack.java`

这段代码是一个在终端上玩扑克牌的游戏。它的主要目的是展示如何使用Java编写更可维护的代码。具体来说，这段代码包括以下几个主要部分：

1. 导入输入流和输出流对象：
```java
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
```
2. 创建两个`Reader`和两个`Writer`对象：
```java
import java.util.Collections;
```
3. 创建一个包含20个扑克牌的列表：
```java
List<Card> cards = new ArrayList<Card>();
for (int i = 0; i < 20; i++) {
   cards.add(new Card());
}
```
4. 使用`Collections.shuffle()`方法随机洗牌：
```java
Collections.shuffle(cards);
```
5. 使用`inputStreamToReader()`和`outputStreamToWriter()`方法将输入流和输出流对象转换为字符串：
```java
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
```
6. 遍历列表并输出：
```java
for (Card card : cards) {
   System.out.println(card.toString());
}
```
7. 创建一个`Reader`对象并读取：
```java
import java.io.Reader;
```
8. 创建一个`Writer`对象并写入：
```java
import java.io.Writer;
```
9. 将所有部分组合起来，并输出。
```java
public class Blackjack {
   public static void main(String[] args) {
       String output = " thus " + "it " + "is " + "a " + "card " + "游戏 " + "of " + "blackjack " + ".";
       System.out.println(output);
   }
}
```
10. 输出Javadoc：
```java
/**
* Plays a game of blackjack on the terminal. Looking at the code,
* the reader might conclude that this implementation is "over engineered." We use
* many techniques and patterns developed for much larger code bases to create
* more maintainable code, which may not be as relevant for a simple game
* of Blackjack. To wit, the rules and requirements are not likely to ever change
* so there is not so much value making the code flexible.
* 
* Nevertheless, this is meant to be an example that the reader can learn good
* Java coding techniques from. Furthermore, many of the "over-engineering"
* tactics are as much about testability as they are about maintainability.
* 
* Another "unnecessary" aspect of this codebase is good Javadoc. Again,
* this is meant to be educational, but another often overlooked benefit is that most
* IDEs will display Javadoc in "autocomplete" suggestions. This is
* remarkably helpful when using a class as a quick reminder of what you coded
* earlier. * 
* Furthermore, this code is backwards. In most cases, the
* `main` method should be the starting point, not the end.
*/
```
这段代码的主要目的是展示如何编写更可维护的Java代码。它包括使用`Collections.shuffle()`和`inputStreamToReader()`，`outputStreamToWriter()`方法将输入流和输出流对象转换为字符串。此外，它还通过遍历列表并输出，创建了一个`Reader`对象并读取，创建了一个`Writer`对象并写入，最后组合所有部分并输出。


```
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.util.Collections;

/**
 * Plays a game of blackjack on the terminal. Looking at the code, the reader
 * might conclude that this implementation is "over engineered." We use many
 * techniques and patterns developed for much larger code bases to create more
 * maintainable code, which may not be as relevant for a simple game of
 * Blackjack. To wit, the rules and requirements are not likely to ever change
 * so there is not so much value making the code flexible.
 * 
 * Nevertheless, this is meant to be an example that the reader can learn good
 * Java coding techniques from. Furthermore, many of the "over-engineering"
 * tactics are as much about testability as they are about maintainability.
 * Imagine trying to manually test infrequent scenarios like Blackjack,
 * insurance, or splitting without any ability to automate a specific scenario
 * and the value of unit testing becomes immediately apparent.
 * 
 * Another "unnecessary" aspect of this codebase is good Javadoc. Again, this is
 * meant to be educational, but another often overlooked benefit is that most
 * IDEs will display Javadoc in "autocomplete" suggestions. This is remarkably
 * helpful when using a class as a quick reminder of what you coded earlier.
 * This is true even if no one ever publishes or reads the HTML output of the
 * javadoc.
 * 
 */
```

该代码定义了一个名为"Blackjack"的类，其目的是在Java应用程序的主函数中执行一个和非交互性的测试。

在主函数中，使用try-catch-finally块，尝试从标准输入(System.in)中读取输入数据。然后，使用新创建的"OutputStreamWriter"将数据写入标准输出(System.out)。

接下来，使用自己定义的"shuffle"算法来洗牌。算法的核心思想是使用Collections.shuffle()方法对整张牌的列表进行随机重排，然后返回排好的一套牌。

接下来，定义了一个名为"Deck"的类，其使用自己定义的"shuffle"算法来洗牌。在"Deck"类中，使用一个变量-card数组来实现洗牌，然后在循环中使用userIo.println()方法输出"RESHUFFLING"，然后使用Collections.shuffle()方法重新洗牌，最后返回排好的一套牌。

接下来，定义了一个名为"Game"的类，其具有以下方法：

- run()方法：使用新创建的Deck对象，和用户输入的代码流一起执行游戏。首先，使用in对象从标准输入中读取输入数据，并将其存储在userIo对象中。然后，定义一个isDisc循枚举类型，并使用该类型来判断是否为打牌模式。如果是打牌模式，就执行game的run()方法，否则执行game的run1()方法。在run()方法中，使用userIo对象从标准输入中读取输入数据，并将其存储在card数组中。然后，执行定义的"shuffle"算法，并使用userIo对象将数组中的数据存储到标准输出中。最后，使用userIo对象将洗牌后的card数组存储到Game对象中，并继续执行游戏。

最后，在try-catch-finally块中，处理应用程序可能抛出的异常，并使用System.out.println()方法输出错误消息，然后使用System.exit()方法结束应用程序。


```
public class Blackjack {
	public static void main(String[] args) {
		// Intuitively it might seem like the main program logic should be right
		// here in 'main' and that we should just use System.in and System.out
		// directly whenever we need them.  However, notice that System.out and
		// System.in are just an OutputStream and InputStream respectively. By
		// allowing alternate streams to be specified to Game at runtime, we can
		// write non-interactive tests of the code. See UserIoTest as an
		// example.
		// Likewise, by allowing an alternative "shuffle" algorithm, test code
		// can provide a deterministic card ordering.
		try (Reader in = new InputStreamReader(System.in)) {
			Writer out = new OutputStreamWriter(System.out);
			UserIo userIo = new UserIo(in, out);
			Deck deck = new Deck(cards -> {
				userIo.println("RESHUFFLING");
			    Collections.shuffle(cards);
			    return cards;
			});
			Game game = new Game(deck, userIo);
			game.run();
		} catch (Exception e) {
			// This allows us to elegantly handle CTRL+D / CTRL+Z by throwing an exception.
			System.out.println(e.getMessage());
			System.exit(1);
		}
	}
}

```

# `10_Blackjack/java/src/Card.java`

This is a Java class called `Card.` It has two instance variables, `value` and `suit`. `value` is of type `int` and `suit` is of type `Suit`. `Suit` has four possible values, `HEARTS`, `DIAMONDS`, `SPADES`, and `CLUBS`.

The `toString()` method is used for debugging purposes to print the card value in a more readable format. It returns the string "AN [x]", where `x` is the value of the card, or "A [X]", where `x` is the value of the card.

The `toProseString()` method returns the string "AN [x]" when the value of the card is either 1 or 8, and "A [X]" otherwise.

The class has an `@link` annotation with the `/` before it, which creates a reference to the `toString()` method.


```
/**
 * This is an example of an "record" class in Java. That's just a fancy way
 * of saying the properties (value and suit) can't change after the object has
 * been created (it has no 'setter' methods and the properties are implicitly 'final'). 
 *
 * Immutability often makes it easier to reason about code logic and avoid
 * certain classes of bugs.
 *
 * Since it would never make sense for a card to change in the middle of a game,
 * this is a good candidate for immutability.
 */
record Card(int value, Suit suit) {

	public enum Suit {
		HEARTS, DIAMONDS, SPADES, CLUBS;
	}

	public Card {
        if(value < 1 || value > 13) {
            throw new IllegalArgumentException("Invalid card value " + value);
        }
        if(suit == null) {
            throw new IllegalArgumentException("Card suit must be non-null");
        }
	}

    public String toString() {
        StringBuilder result = new StringBuilder(2); 
        if(value == 1) {
            result.append("A");
        } else if(value < 11) {
            result.append(value);
        } else if(value == 11) {
            result.append('J');
        } else if(value == 12) {
            result.append('Q');
        } else if(value == 13) {
            result.append('K');
        }
        // Uncomment to include the suit in output. Useful for debugging, but
        // doesn't match the original BASIC behavior.
        // result.append(suit.name().charAt(0));
        return result.toString();
    }

    /**
     * Returns the value of {@link #toString()} preceded by either "AN " or "A " depending on which is gramatically correct.
     * 
     * @return "AN [x]" when [x] is "an" ace or "an" 8, and "A [X]" otherwise.
     */
    public String toProseString() {
		if(value == 1 || value == 8) {
            return "AN " + toString();
        } else {
            return "A " + toString();
        }
    }

}
```

# `10_Blackjack/java/src/Deck.java`

This code is a Java class that represents a deck of cards with 2 decks. It contains a method for reshuffling the cards, a method for dealing a card, and a method for getting the number and cards in the deck. The class also has a constructor that takes a function for shuffling the cards and a constructor that initializes the deck with 2 copies of a standard 52 card deck.


```
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;

public class Deck {

    private LinkedList<Card> cards;
    private Function<LinkedList<Card>, LinkedList<Card>> shuffleAlgorithm;
    
    /**
     * Initialize the game deck with the given number of standard decks.
     * e.g. if you want to play with 2 decks, then {@code new Decks(2)} will
     * initialize 'cards' with 2 copies of a standard 52 card deck.
     * 
     * @param shuffleAlgorithm A function that takes the initial sorted card
     * list and returns a shuffled list ready to deal.
     * 
     */
    public Deck(Function<LinkedList<Card>, LinkedList<Card>> shuffleAlgorithm) {
        this.shuffleAlgorithm = shuffleAlgorithm;
    }

    /**
     * Deals one card from the deck, removing it from this object's state. If
     * the deck is empty, it will be reshuffled before dealing a new card.
     * 
     * @return The card that was dealt.
     */
    public Card deal() {
        if(cards == null || cards.isEmpty()) {
            reshuffle();
        }
        return cards.pollFirst();
    }

    /**
     * Shuffle the cards in this deck using the shuffleAlgorithm.
     */
    public void reshuffle() {
        LinkedList<Card> newCards = new LinkedList<>();
        for(Card.Suit suit : Card.Suit.values()) {
            for(int value = 1; value < 14; value++) {
                newCards.add(new Card(value, suit));
            }
        }
        this.cards = this.shuffleAlgorithm.apply(newCards);
    }

    /**
     * Get the number of cards in this deck.
     * @return The number of cards in this deck. For example, 52 for a single deck.
     */
    public int size() {
        return cards.size();
    }

    /**
     * Returns the cards in this deck.
     * @return An immutable view of the cards in this deck.
     */
    public List<Card> getCards() {
        // The returned list is immutable because we don't want other code messing with the deck.
        return Collections.unmodifiableList(cards);
    }
}

```

# `10_Blackjack/java/src/Game.java`

This appears to be a Java class that implements the logic for a card game where players and a dealer compete. It uses a `ScoringUtils` class to determine the outcome of the game, and it does not seem to use any specific deck of cards.

The class has several methods, including `dealerResult`, `scoreHand`, `totalBet`, and `userIo`, which are used to determine the outcome of the game, calculate the total bet, and display the result to the user, respectively.

The class also has a `betsAreValid` method, which checks that all bets are between 0 (exclusive) and 500 (inclusive). Fractional bets are valid, according to the documentation.

Overall, it seems like this class is designed to simulate a simplified card game where players and a dealer compete, and it uses a variety of methods to determine the outcome of the game and calculate the total bet.


```
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.text.DecimalFormat;

/**
 * This is the primary class that runs the game itself.
 */
public class Game {
    
    private Deck deck;
    private UserIo userIo;

    public Game(Deck deck, UserIo userIo) {
        this.deck = deck;
        this.userIo = userIo;
    }

	/**
	 * Run the game, running rounds until ended with CTRL+D/CTRL+Z or CTRL+C
	 */
    public void run() {
		userIo.println("BLACK JACK", 31);
		userIo.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n", 15);
		if(userIo.promptBoolean("DO YOU WANT INSTRUCTIONS")){
			userIo.println("THIS IS THE GAME OF 21. AS MANY AS 7 PLAYERS MAY PLAY THE");
			userIo.println("GAME. ON EACH DEAL, BETS WILL BE ASKED FOR, AND THE");
			userIo.println("PLAYERS' BETS SHOULD BE TYPED IN. THE CARDS WILL THEN BE");
			userIo.println("DEALT, AND EACH PLAYER IN TURN PLAYS HIS HAND. THE");
			userIo.println("FIRST RESPONSE SHOULD BE EITHER 'D', INDICATING THAT THE");
			userIo.println("PLAYER IS DOUBLING DOWN, 'S', INDICATING THAT HE IS");
			userIo.println("STANDING, 'H', INDICATING HE WANTS ANOTHER CARD, OR '/',");
			userIo.println("INDICATING THAT HE WANTS TO SPLIT HIS CARDS. AFTER THE");
			userIo.println("INITIAL RESPONSE, ALL FURTHER RESPONSES SHOULD BE 'S' OR");
			userIo.println("'H', UNLESS THE CARDS WERE SPLIT, IN WHICH CASE DOUBLING");
			userIo.println("DOWN IS AGAIN PERMITTED. IN ORDER TO COLLECT FOR");
			userIo.println("BLACKJACK, THE INITIAL RESPONSE SHOULD BE 'S'.");
		}

		int nPlayers = 0;
		while(nPlayers < 1 || nPlayers > 7) {
			nPlayers = userIo.promptInt("NUMBER OF PLAYERS");
		}

		deck.reshuffle();

		Player dealer = new Player(0); //Dealer is Player 0
		
		List<Player> players = new ArrayList<>();
		for(int i = 0; i < nPlayers; i++) {
			players.add(new Player(i + 1));
		}

		while(true) {
			while(!betsAreValid(players)){
				userIo.println("BETS:");
				for(int i = 0; i < nPlayers; i++) {
					double bet = userIo.promptDouble("#" + (i + 1)); // 1st player is "Player 1" not "Player 0"
					players.get(i).setCurrentBet(bet);
				}
			}

			// It doesn't *really* matter whether we deal two cards at once to each player
			// or one card to each and then a second card to each, but this technically
			// mimics the way a deal works in real life.
			for(int i = 0; i < 2; i++){
				for(Player player : players){
					player.dealCard(deck.deal());
				}
				dealer.dealCard(deck.deal());
			}

			printInitialDeal(players, dealer);

			if(dealer.getHand().get(0).value() == 1) {
				collectInsurance(players);
			}

			if(ScoringUtils.scoreHand(dealer.getHand()) == 21) {
				userIo.println("DEALER HAS " + dealer.getHand().get(1).toProseString() + " IN THE HOLE");
				userIo.println("FOR BLACKJACK");
			} else {
				Card dealerFirstCard = dealer.getHand().get(0);
				if(dealerFirstCard.value() == 1 || dealerFirstCard.value() > 9) {
					userIo.println("");
					userIo.println("NO DEALER BLACKJACK.");
				} // else dealer blackjack is imposible
				for(Player player : players){
					play(player);
				}

				if(shouldPlayDealer(players)){
					playDealer(dealer);
				} else {
					userIo.println("DEALER HAD " + dealer.getHand().get(1).toProseString() + " CONCEALED.");
				}
			}

			evaluateRound(players, dealer);
		} 
    }

	protected void collectInsurance(Iterable<Player> players) {
		boolean isInsurance = userIo.promptBoolean("ANY INSURANCE");
		if(isInsurance) {
			userIo.println("INSURANCE BETS");
			for(Player player : players) {
				while(true) {
					double insuranceBet = userIo.promptDouble("# " + player.getPlayerNumber() + " ");
					// 0 indicates no insurance for that player.
					if(insuranceBet >= 0 && insuranceBet <= (player.getCurrentBet() / 2)) {
						player.setInsuranceBet(insuranceBet);
						break;
					}
				}
			}
		}
	}

	/**
	 * Print the cards for each player and the up card for the dealer.
	 * Prints the initial deal in the following format:
	 *		
	 *	PLAYER 1     2    DEALER
     *         7    10     4   
     *         2     A   
	 */
	private void printInitialDeal(List<Player> players, Player dealer) {
	
        StringBuilder output = new StringBuilder(); 
		output.append("PLAYERS ");
		for (Player player : players) {
			output.append(player.getPlayerNumber() + "\t");
		}
		output.append("DEALER\n");
		//Loop through two rows of cards		
        for (int j = 0; j < 2; j++) {
			output.append("\t");
			for (Player player : players) {
				output.append(player.getHand().get(j).toString()).append("\t");
			}
			if(j == 0 ){
				output.append(dealer.getHand().get(j).toString());
			}
			output.append("\n");
		}
		userIo.print(output.toString());
	}

	/**
	 * Plays the players turn. Prompts the user to hit (H), stay (S), or if
	 * appropriate, split (/) or double down (D), and then performs those
	 * actions. On a hit, prints "RECEIVED A  [x]  HIT? "
	 * 
	 * @param player
	 */
	protected void play(Player player) {
		play(player, 1);
	}

	private void play(Player player, int handNumber) {
		String action;
		if(player.isSplit()){
			action = userIo.prompt("HAND " + handNumber);
		} else {
			action = userIo.prompt("PLAYER " + player.getPlayerNumber() + " ");
		}
		while(true){
			if(action.equalsIgnoreCase("H")){ // HIT
				Card c = deck.deal();
				player.dealCard(c, handNumber);
				if(ScoringUtils.scoreHand(player.getHand(handNumber)) > 21){
					userIo.println("RECEIVED " + c.toProseString() + "  ...BUSTED");
					break;
				}
				action = userIo.prompt("RECEIVED " + c.toProseString() + " HIT");
			} else if(action.equalsIgnoreCase("S")){ // STAY
				break;
			} else if(action.equalsIgnoreCase("D") && player.canDoubleDown(handNumber)) { // DOUBLE DOWN
				Card c = deck.deal();
				player.doubleDown(c, handNumber);
				if(ScoringUtils.scoreHand(player.getHand(handNumber)) > 21){
					userIo.println("RECEIVED " + c.toProseString() + "  ...BUSTED");
					break;
				}
				userIo.println("RECEIVED " + c.toProseString());
				break;
			} else if(action.equalsIgnoreCase("/")) { // SPLIT
				if(player.isSplit()) {
					// The original basic code printed different output
					// if a player tries to split twice vs if they try to split
					// a non-pair hand.
					action = userIo.prompt("TYPE H, S OR D, PLEASE");
				} else if(player.canSplit()) {
					player.split();
					Card card = deck.deal();
					player.dealCard(card, 1);
					userIo.println("FIRST HAND RECEIVES " + card.toProseString());
					card = deck.deal();
					player.dealCard(card, 2);
					userIo.println("SECOND HAND RECEIVES " + card.toProseString());					
					if(player.getHand().get(0).value() > 1){ //Can't play after splitting aces
						play(player, 1);
						play(player, 2);
					}
					return; // Don't fall out of the while loop and print another total
				} else {
					userIo.println("SPLITTING NOT ALLOWED");
					action = userIo.prompt("PLAYER " + player.getPlayerNumber() + " ");
				}
			} else {
				if(player.getHand(handNumber).size() == 2) {
					action = userIo.prompt("TYPE H,S,D, OR /, PLEASE");
				} else {
					action = userIo.prompt("TYPE H, OR S, PLEASE");
				}
			}
		}
		int total = ScoringUtils.scoreHand(player.getHand(handNumber));
		if(total == 21) {
			userIo.println("BLACKJACK");
		} else {
			userIo.println("TOTAL IS " + total);
		}
	}

	/**
	 * Check the Dealer's hand should be played out. If every player has either busted or won with natural Blackjack,
	 * the Dealer doesn't need to play.
	 * 
	 * @param players
	 * @return boolean whether the dealer should play
	 */
	protected boolean shouldPlayDealer(List<Player> players){
		for(Player player : players){
			int score = ScoringUtils.scoreHand(player.getHand());
			if(score < 21 || (score == 21 && player.getHand().size() > 2)){
				return true;
			}
			if(player.isSplit()){				
				int splitScore = ScoringUtils.scoreHand(player.getHand(2));
				if(splitScore < 21 || (splitScore == 21 && player.getHand(2).size() > 2)){
					return true;
				}
			}
		}
		return false;
	}	

	/**
	 * Play the dealer's hand. The dealer draws until they have >=17 or busts. Prints each draw as in the following example:
	 * 
	 * DEALER HAS A  5 CONCEALED FOR A TOTAL OF 11 
	 * DRAWS 10   ---TOTAL IS 21
	 *  
	 * @param dealerHand
	 */
	protected void playDealer(Player dealer) {
		int score = ScoringUtils.scoreHand(dealer.getHand());
		userIo.println("DEALER HAS " + dealer.getHand().get(1).toProseString() + " CONCEALED FOR A TOTAL OF " + score);

		if(score < 17){
			userIo.print("DRAWS");
		}
		while(score < 17) {
			Card dealtCard = deck.deal();
			dealer.dealCard(dealtCard);
			score = ScoringUtils.scoreHand(dealer.getHand());
			userIo.print("  " + String.format("%-4s", dealtCard.toString()));
		}
		
		if(score > 21) {
			userIo.println("...BUSTED\n");
		} else {
			userIo.println("---TOTAL IS " + score + "\n");
		}
	}

	/**
	 * Evaluates the result of the round, prints the results, and updates player/dealer totals.
	 * 
	 *	PLAYER 1 LOSES   100 TOTAL=-100 
	 *	PLAYER 2  WINS   150 TOTAL= 150
	 *	DEALER'S TOTAL= 200
	  *
	 * @param players
	 * @param dealerHand
	 */
	protected void evaluateRound(List<Player> players, Player dealer) {
		DecimalFormat formatter = new DecimalFormat("0.#"); //Removes trailing zeros
		for(Player player : players){
			int result = ScoringUtils.compareHands(player.getHand(), dealer.getHand());
			double totalBet = 0;
			if(result > 0) {
				totalBet += player.getCurrentBet();
			} else if(result < 0){
				totalBet -= player.getCurrentBet();
			}
			if(player.isSplit()) {
				int splitResult = ScoringUtils.compareHands(player.getHand(2), dealer.getHand());
				if(splitResult > 0){
					totalBet += player.getSplitBet();
				} else if(splitResult < 0){
					totalBet -= player.getSplitBet();
				} 
			}
			if(player.getInsuranceBet() != 0){
				int dealerResult = ScoringUtils.scoreHand(dealer.getHand());
				if(dealerResult == 21 && dealer.getHand().size() == 2){
					totalBet += (player.getInsuranceBet() * 2);
				} else {
					totalBet -= player.getInsuranceBet();
				}
			}
			
			userIo.print("PLAYER " + player.getPlayerNumber());
			if(totalBet < 0) {
				userIo.print(" LOSES " + String.format("%6s", formatter.format(Math.abs(totalBet)))); 
			} else if(totalBet > 0) {
				userIo.print("  WINS " + String.format("%6s", formatter.format(totalBet))); 
			} else {
				userIo.print(" PUSHES      ");
			}
			player.recordRound(totalBet);
			dealer.recordRound(totalBet * (-1));
			userIo.println(" TOTAL= " + formatter.format(player.getTotal()));
			player.resetHand();
		}
		userIo.println("DEALER'S TOTAL= " + formatter.format(dealer.getTotal()) + "\n");
		dealer.resetHand();
	}

	/**
	 * Validates that all bets are between 0 (exclusive) and 500 (inclusive). Fractional bets are valid.
	 * 
	 * @param players The players with their current bet set.
	 * @return true if all bets are valid, false otherwise.
	 */
	public boolean betsAreValid(Collection<Player> players) {
		return players.stream()
			.map(Player::getCurrentBet)
			.allMatch(bet -> bet > 0 && bet <= 500);
	}
}

```

# `10_Blackjack/java/src/Player.java`

This appears to be a Java class that represents a game of cards where the user can place a bet and the game to a specific hand. The class has methods for doubling down on the hand, resetting the hand, and getting the hand.

The `canDoubleDown` method takes an integer `handNumber` and checks whether the hand is valid for doubling down. If the hand is valid, it returns `true`.

The `doubleDown` method takes a `Card` object `card` and an integer `handNumber`, and deals the specified card to the hand and doubles the bet for the hand. It does this by first setting the current bet to the current bet multiplied by 2 if the hand is a singleton, or setting the splitBet to the splitBet multiplied by 2 if the hand is a doubleton.

The `resetHand` method resets the hand to an empty list and the splitHand to null.

The `getHand` method returns the specified hand. If the hand number is 1 it returns the hand without the split, if it's 2 it returns the split hand.

The `getHand` method appears to use the `Collections.unmodifiableList` method to get the hand, this is a modifierList that keeps the order and add, remove or modify the elements.


```
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * Represents a player and data related to them (number, bets, cards).
 */
public class Player {

    private int playerNumber;     // e.g. playerNumber = 1 means "this is Player 1"
    private double currentBet;
    private double insuranceBet; // 0 when the player has not made an insurance bet (either it does not apply or they chose not to)
    private double splitBet; // 0 whenever the hand is not split
    private double total;
    private LinkedList<Card> hand;
    private LinkedList<Card> splitHand; // null whenever the hand is not split

    /**
    * Represents a player in the game with cards, bets, total and a playerNumber. 
    */
    public Player(int playerNumber) {
        this.playerNumber = playerNumber;
        currentBet = 0;
        insuranceBet = 0;
        splitBet = 0;
        total = 0;
        hand = new LinkedList<>();
        splitHand = null;
    }

    public int getPlayerNumber() {
        return this.playerNumber;
    }
    
    public double getCurrentBet() {
        return this.currentBet;
    }

    public void setCurrentBet(double currentBet) {
        this.currentBet = currentBet;
    }

    public double getSplitBet() {
        return splitBet;
    }

    public double getInsuranceBet() {
        return insuranceBet;
    }

    public void setInsuranceBet(double insuranceBet) {
        this.insuranceBet = insuranceBet;
    }

    /**
    * RecordRound adds input paramater 'totalBet' to 'total' and then 
    * sets 'currentBet', 'splitBet', and 'insuranceBet' to zero
    */
    public void recordRound(double totalBet) {
        this.total = this.total + totalBet;
        this.currentBet = 0;
        this.splitBet = 0;
        this.insuranceBet = 0;
    }

    /**
     * Returns the total of all bets won/lost.
     * @return Total value
     */
    public double getTotal() {
        return this.total;
    }

    /**
     * Add the given card to the players main hand.
     * 
     * @param card The card to add.
     */
    public void dealCard(Card card) {
        dealCard(card, 1);
    }
    
    /**
     * Adds the given card to the players hand or split hand depending on the handNumber.
     * 
     * @param card The card to add
     * @param handNumber 1 for the "first" hand and 2 for the "second" hand in a split hand scenario.
     */
    public void dealCard(Card card, int handNumber) {
        if(handNumber == 1) {
            hand.add(card);
        } else if (handNumber == 2) {
            splitHand.add(card);
        } else {
            throw new IllegalArgumentException("Invalid hand number " + handNumber);
        }
    }

    /**
     * Determines whether the player is eligible to split.
     * @return True if the player has not already split, and their hand is a pair. False otherwise.
     */
    public boolean canSplit() {
        if(isSplit()) {
            // Can't split twice
            return false;
        } else {
            boolean isPair = this.hand.get(0).value() == this.hand.get(1).value();
            return isPair;
        }
    }

    /**
     * Determines whether the player has already split their hand.
     * @return false if splitHand is null, true otherwise.
     */
    public boolean isSplit() {
        return this.splitHand != null;
    }

    /**
     * Removes first card from hand to add it to new split hand
     */
    public void split() {
        this.splitBet = this.currentBet;
        this.splitHand = new LinkedList<>();
        splitHand.add(hand.pop());
    }

    /**
     * Determines whether the player can double down.
     * 
     * @param handNumber
     * @return
     */
    public boolean canDoubleDown(int handNumber) {
        if(handNumber == 1){
            return this.hand.size() == 2;
        } else if(handNumber == 2){
            return this.splitHand.size() == 2;
        } else {
            throw new IllegalArgumentException("Invalid hand number " + handNumber);
        }
    }

    /**
     * Doubles down on the given hand. Specifically, this method doubles the bet for the given hand and deals the given card.
     * 
     * @param card The card to deal
     * @param handNumber The hand to deal to and double the bet for
     */
    public void doubleDown(Card card, int handNumber) {
        if(handNumber == 1){
            this.currentBet = this.currentBet * 2;
        } else if(handNumber == 2){
            this.splitBet = this.splitBet * 2;
        } else {
            throw new IllegalArgumentException("Invalid hand number " + handNumber);
        }
        this.dealCard(card, handNumber);
    }

    /**
     * Resets the hand to an empty list and the splitHand to null.
     */
    public void resetHand() {
        this.hand = new LinkedList<>();
        this.splitHand = null;
    }

    public List<Card> getHand() {
        return getHand(1);
    }

    /**
     * Returns the given hand
     * @param handNumber 1 for the "first" of a split hand (or the main hand when there is no split) or 2 for the "second" hand of a split hand.
     * @return The hand specified by handNumber
     */
    public List<Card> getHand(int handNumber) {
        if(handNumber == 1){
            return Collections.unmodifiableList(this.hand);
        } else if(handNumber == 2){
            return Collections.unmodifiableList(this.splitHand);
        } else {
            throw new IllegalArgumentException("Invalid hand number " + handNumber);
        }
    }

}
```

# `10_Blackjack/java/src/ScoringUtils.java`

This is a Java class that represents a game of blackjack where two players take turns playing cards and trying to get a total higher than the dealer's total. The class implements the `Comparable` interface, which allows for the comparison of the hands to be expressed as positive or negative integers rather than just their order.

The `scoreHand` method takes a list of cards and returns the score of the hand by adding up the value of each card. The `compareHands` method compares the scores of the two hands and returns a negative integer if the first hand is less than or equal to the second, or a positive integer if the first hand is greater than the second.

The `scoreHand` method takes a list of cards and returns the score of the hand by adding up the value of each card. The implementation is as follows:
```
public int scoreHand(List<Card> hand) {
   int score = 0;
   for (Card card : hand) {
       score += card.getValue();
   }
   return score;
}
```
The `compareHands` method takes two lists of cards and returns a negative integer if the first hand is less than or equal to the second, or a positive integer if the first hand is greater than the second. The implementation is as follows:
```
public static final int compareHands(List<Card> handA, List<Card> handB) {
   int scoreA = scoreHand(handA);
   int scoreB = scoreHand(handB);
   if (scoreA == 21 && scoreB == 21) {
       if (handA.size() == 2 && handB.size() != 2) {
           return 1; // Hand A wins with a natural blackjack
       } else if (handA.size() != 2 && handB.size() == 2) {
           return -1; // Hand B wins with a natural blackjack
       } else {
           return 0; // Tie
       }
   } else if (scoreA > 21 || scoreB > 21) {
       if (scoreA > 21 && scoreB > 21) {
           return 0; // Tie, both bust
       } else if (scoreB > 21) {
           return 1; // A wins, B busted
       } else {
           return -1; // B wins, A busted
       }
   } else {
       return Integer.compare(scoreA, scoreB);
   }
}
```
Note that the natural blackjack is represented by the value 21, so if a hand has a score of 21, it is considered to have a natural blackjack, regardless of the order in which it was received.


```
import java.util.List;

public final class ScoringUtils {

	/**
	 * Calculates the value of a hand. When the hand contains aces, it will
	 * count one of them as 11 if that does not result in a bust.
	 * 
	 * @param hand the hand to evaluate
	 * @return The numeric value of a hand. A value over 21 indicates a bust.
	 */
	public static final int scoreHand(List<Card> hand) {
		int nAces = (int) hand.stream().filter(c -> c.value() == 1).count();
		int value = hand.stream()
				.mapToInt(Card::value)
				.filter(v -> v != 1) // start without aces
				.map(v -> v > 10 ? 10 : v) // all face cards are worth 10. The 'expr ? a : b' syntax is called the
											// 'ternary operator'
				.sum();
		value += nAces; // start by treating all aces as 1
		if (nAces > 0 && value <= 11) {
			value += 10; // We can use one of the aces to an 11
			// You can never use more than one ace as 11, since that would be 22 and a bust.
		}
		return value;
	}

	/**
	 * Compares two hands accounting for natural blackjacks and busting using the
	 * java.lang.Comparable convention of returning positive or negative integers
	 * 
	 * @param handA hand to compare
	 * @param handB other hand to compare
	 * @return a negative integer, zero, or a positive integer as handA is less
	 *         than, equal to, or greater than handB.
	 */
	public static final int compareHands(List<Card> handA, List<Card> handB) {
		int scoreA = scoreHand(handA);
		int scoreB = scoreHand(handB);
		if (scoreA == 21 && scoreB == 21) {
			if (handA.size() == 2 && handB.size() != 2) {
				return 1; // Hand A wins with a natural blackjack
			} else if (handA.size() != 2 && handB.size() == 2) {
				return -1; // Hand B wins with a natural blackjack
			} else {
				return 0; // Tie
			}
		} else if (scoreA > 21 || scoreB > 21) {
			if (scoreA > 21 && scoreB > 21) {
				return 0; // Tie, both bust
			} else if (scoreB > 21) {
				return 1; // A wins, B busted
			} else {
				return -1; // B wins, A busted
			}
		} else {
			return Integer.compare(scoreA, scoreB);
		}
	}

}

```

# `10_Blackjack/java/src/UserIo.java`

这段代码定义了一个名为 `PrintInputAndOutput` 的类。这个类的主要作用是读取用户输入的数据，将其打印到屏幕上，并将用户输入的数据从屏幕上读取回来。

具体来说，这个类需要两个参数，一个是要读取的数据来源，通常是一个 `Reader`，另一个是要写入的数据目的地，通常是一个 `Writer`。读取的数据源可以是字符串、文件或者其他可读取的资源，而写入的数据可以是字符串、文件或者其他可写入的资源。

在 `main` 函数中，这个类创建了一个 `Reader` 对象来读取用户输入的数据，创建了一个 `Writer` 对象来将用户输出的数据写入到屏幕上。通常情况下，这两个对象都是 `System.in` 和 `System.out`，但也可以是其他的名字，例如 `FileReader` 和 `FileWriter`。

当用户提交输入数据后，这个类会先尝试从 `System.in` 读取数据。如果从 `System.in` 读取数据失败，它将抛出一个 `EOFException` 异常。如果从 `System.in` 读取数据成功，这个类会将读取的数据打印到屏幕上，并从 `System.out` 写入数据。

当用户提交输出数据后，这个类会先尝试从 `System.out` 读取数据。如果从 `System.out` 读取数据失败，它将抛出一个 `EOFException` 异常。如果从 `System.out` 读取数据成功，这个类会将读取的数据打印到屏幕上，并从 `System.in` 写入数据。

这个类的实现比较简单，主要的作用是读取和写入用户输入的数据，方便测试和实际应用中需要进行数据交互的情况。


```
import java.io.BufferedReader;
import java.io.EOFException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.util.stream.IntStream;

/**
 * This class is responsible for printing output to the screen and reading input
 * from the user. It must be initialized with a reader to get input data from
 * and a writer to send output to. Typically these will wrap System.in and
 * System.out respectively, but can be a StringReader and StringWriter when
 * running in test code.
 */
```

This is a Java class that provides a simple way to prompt the user for input.

The class has two methods for prompting the user for an integer and double. The integer prompt is similar to the one described in the question, while the double prompt is as follows:

"The prompt to display to the user is followed by a question mark and a space.

As in Vintage Basic, an error will be generated and the user will be re-prompted if the input is non-numeric."

The readLine() method is used to read input from the user. It returns the input as a string. The code uses try-catch blocks to catch any exceptions that may occur when parsing the input as a number or double. If an exception is caught, the code will print an error message and prompt the user to try again.

Overall, this class provides a simple way to prompt the user for input, but it may not be suitable for very complex use cases.


```
public class UserIo {

    private BufferedReader in;
    private PrintWriter out;

	/**
	 * Initializes the UserIo with the given reader/writer. The reader will be
	 * wrapped in a BufferedReader and so should <i>not</i> be a BufferedReader
	 * already (to avoid double buffering).
	 * 
	 * @param in Typically an InputStreamReader wrapping System.in or a StringReader
	 * @param out Typically an OuputStreamWriter wrapping System.out or a StringWriter
	 */
    public UserIo(Reader in, Writer out) {
        this.in = new BufferedReader(in);
        this.out = new PrintWriter(out, true);
    }

	/**
	 * Print the line of text to output including a trailing linebreak.
	 * 
	 * @param text the text to print
	 */
	public void println(String text) {
		out.println(text);
	}

	/**
	 * Print the given text left padded with spaces.
	 * 
	 * @param text The text to print
	 * @param leftPad The number of spaces to pad with.
	 */
	public void println(String text, int leftPad) {
		IntStream.range(0, leftPad).forEach((i) -> out.print(' '));
		out.println(text);
	}

	/**
	 * Print the given text <i>without</i> a trailing linebreak.
	 * 
	 * @param text The text to print.
	 */
	public void print(String text) {
		out.print(text);
		out.flush();
	}

	/**
	 * Reads a line of text from input.
	 * 
	 * @return The line entered into input.
	 * @throws UncheckedIOException if the line is null (CTRL+D or CTRL+Z was pressed)
	 */
	private String readLine() {
		try {
			String line = in.readLine();
			if(line == null) {
				throw new UncheckedIOException("!END OF INPUT", new EOFException());
			}
			return line;
		} catch (IOException e) {
			throw new UncheckedIOException(e);
		}
	}

	/**
	 * Prompt the user via input.
	 * 
	 * @param prompt The text to display as a prompt. A question mark and space will be added to the end, so if prompt = "EXAMPLE" then the user will see "EXAMPLE? ".
	 * @return The line read from input.
	 */
	public String prompt(String prompt) {
		print(prompt + "? ");
		return readLine();
	}

	/**
	 * Prompts the user for a "Yes" or "No" answer.
	 * @param prompt The prompt to display to the user on STDOUT.
	 * @return false if the user enters a value beginning with "N" or "n"; true otherwise.
	 */
	public boolean promptBoolean(String prompt) {
		print(prompt + "? ");

		String input = readLine();

		if(input.toLowerCase().startsWith("n")) {
			return false;
		} else {
			return true;
		}
	}

	/**
	 * Prompts the user for an integer.  As in Vintage Basic, "the optional
	 * prompt string is followed by a question mark and a space." and if the
	 * input is non-numeric, "an error will be generated and the user will be
	 * re-prompted.""
	 *
	 * @param prompt The prompt to display to the user.
	 * @return the number given by the user.
	 */
	public int promptInt(String prompt) {
		print(prompt + "? ");

		while(true) {
			String input = readLine();
			try {
				return Integer.parseInt(input);
			} catch(NumberFormatException e) {
				// Input was not numeric.
				println("!NUMBER EXPECTED - RETRY INPUT LINE");
				print("? ");
				continue;
			}
		}
	}

	/**
	 * Prompts the user for a double.  As in Vintage Basic, "the optional
	 * prompt string is followed by a question mark and a space." and if the
	 * input is non-numeric, "an error will be generated and the user will be
	 * re-prompted.""
	 *
	 * @param prompt The prompt to display to the user.
	 * @return the number given by the user.
	 */
	public double promptDouble(String prompt) {
		print(prompt + "? ");

		while(true) {
			String input = readLine();
			try {
				return Double.parseDouble(input);
			} catch(NumberFormatException e) {
				// Input was not numeric.
				println("!NUMBER EXPECTED - RETRY INPUT LINE");
				print("? ");
				continue;
			}
		}
	}
}

```

# `10_Blackjack/java/test/DeckTest.java`

这段代码是一个名为 "DeckTest" 的测试类，旨在测试 "Deck" 类的功能。

在测试代码中，首先定义了一个名为 "testInit" 的方法。在这个方法中，使用创建了一个 "Deck" 对象，然后对其进行了 "reshuffle" 操作。

接下来，使用 "nCards" 和 "nSuits" 两个变量来存储 "Deck" 对象中所有卡片的数量。然后，使用 "suit" 和 "value" 两个方法来获取每张卡片的牌面和价值，并使用 "count" 方法来统计每种牌面的数量。

最后，使用 "assertAll" 方法来验证 "Deck" 对象中所有卡片的数量、牌面数量和价值都符合预期。如果验证失败，则会输出详细的错误信息。


```
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertAll;
import org.junit.jupiter.api.Test;

public class DeckTest {

    @Test
    void testInit() {
        // When
        Deck deck = new Deck((cards) -> cards);
        deck.reshuffle();

        // Then
        long nCards = deck.size();
        long nSuits = deck.getCards().stream()
                .map(card -> card.suit())
                .distinct()
                .count();
        long nValues = deck.getCards().stream()
                .map(card -> card.value())
                .distinct()
                .count();

        assertAll("deck",
            () -> assertEquals(52, nCards, "Expected 52 cards in a deck, but got " + nCards),
            () -> assertEquals(4, nSuits, "Expected 4 suits, but got " + nSuits),
            () -> assertEquals(13, nValues, "Expected 13 values, but got " + nValues)
        );
        
    }

}

```

# `10_Blackjack/java/test/GameTest.java`

该代码是一个JUnit测试类，用于测试"MyString"类的功能。具体来说，该代码以下的注释可以提供以下有关该测试的作用的说明：

1. "import org.junit.jupiter.api.*"：这是JUnit测试的导入部分，用于导入JUnit和"MyString"类的其他声明。

2. "import org.junit.jupiter.api.AfterEach; import org.junit.jupiter.api.BeforeEach; import org.junit.jupiter.api.DisplayName;"：这是用于在测试结束后和测试开始时执行的代码。包括设置测试显示名称、导入需要测试的类以及导入JUnit和其他声明。

3. "import org.junit.jupiter.api.AfterEach; import org.junit.jupiter.api.BeforeEach;"：这是两个无参数的测试方法，用于在测试结束后和测试开始时执行。

4. "import org.junit.jupiter.api.DisplayName;"：这是用于设置测试的显示名称的测试方法。

5. "import java.io.EOFException; import java.io.StringReader; import java.io.StringWriter;"：这是导入需要测试的类的代码。

6. "import java.util.concurrent.*;"：这是导入Java并发编程中的流库的代码。

7. "import static org.junit.jupiter.api.Assertions.assertAll; import static org.junit.jupiter.api.Assertions.assertEquals; import static org.junit.jupiter.api.Assertions.assertFalse; import static org.junit.jupiter.api.Assertions.assertThrows;"：这是几个测试方法的导入，这些方法是通过"assertEquals()"、"assertThrows()"和"assertAll()"实现的。

8. "import static org.junit.jupiter.api.Assertions.assertTrue;"：这是用于测试方法"assertTrue()"的导入。

9. "public class MyStringTest {"：这是测试类的声明。

10. "import org.junit.jupiter.api.Test;"：这是JUnit测试的声明。

11. "public class MyStringTest implements org.junit.jupiter.api.Test {"：这是测试类的声明，它实现了JUnit测试的接口。

12. "{Test""：这是测试类的实现，它包含了所有需要测试的代码。

13. "public void testMyString() throws EOFException { try (MyString myString = new MyString()) {"：这是测试方法的实现，它使用了一个"try"块来捕获EOFException异常并创建一个MyString对象。

14. "System.out.println(myString.getHelloWorld());"：这是测试方法的一个分支，用于打印MyString类的一个"getHelloWorld()"方法的方法的返回值。

15. "} catch (EOFException e) {"：这是捕获EOFException异常的代码块。

16. "} finally {"：这是捕获EOFException异常的代码块。

17. "}"：这是测试方法的结束标记。


```
import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.EOFException;
import java.io.StringReader;
import java.io.StringWriter;
```

This is a Java class that tests the `shouldPlayDealer()` and `playDealer()` methods of the `Game` class.

The `shouldPlayDealer()` method tests whether the game should play dealer when the player does not have a blackjack. It does this by randomly drawing one card and then drawing another card if the player's hand is still less than the dealer's hand. It then compares the result of the game's decision with the expected result.

The `playDealer()` method tests whether the game should play dealer when the player's hand is less than the dealer's hand. It does this by randomly drawing one card and then drawing another card if the player's hand is still more than the dealer's hand. It then compares the result of the game's decision with the expected result.

The `out` variable is an output stream that is used to print the result of the game's decision. In the `shouldPlayDealer()` method, it contains the string "DRAWS". In the `playDealer()` method, it contains the string "---TOTAL IS".


```
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class GameTest {

    private StringReader in;
    private StringWriter out;
    private Game game;

    private StringBuilder playerActions;
    private LinkedList<Card> cards;

    @BeforeEach
    public void resetIo() {
        in = null;
        out = null;
        game = null;
        playerActions = new StringBuilder();
        cards = new LinkedList<>();
    }

    private void playerGets(int value, Card.Suit suit) {
        cards.add(new Card(value, suit));
    }

    private void playerSays(String action) {
        playerActions.append(action).append(System.lineSeparator());
    }

    private void initGame() {
        System.out.printf("Running game with input: %s\tand cards: %s\n",playerActions.toString(), cards);
        in = new StringReader(playerActions.toString());
        out = new StringWriter();
        UserIo userIo = new UserIo(in, out);
        Deck deck = new Deck((c) -> cards);
        game = new Game(deck, userIo);
    }

    @AfterEach
    private void printOutput() {
        System.out.println(out.toString());
    }

    @Test
    public void shouldQuitOnCtrlD() {
        // Given
        playerSays("\u2404"); // U+2404 is "End of Transmission" sent by CTRL+D (or CTRL+Z on Windows)
        initGame();

        // When
        Exception e = assertThrows(UncheckedIOException.class, game::run);

        // Then
        assertTrue(e.getCause() instanceof EOFException);
        assertEquals("!END OF INPUT", e.getMessage());
    }

    @Test
    @DisplayName("collectInsurance() should not prompt on N")
    public void collectInsuranceNo(){
        // Given
        List<Player> players = Collections.singletonList(new Player(1));
        playerSays("N");
        initGame();

        // When
        game.collectInsurance(players);

        // Then
        assertAll(
            () -> assertTrue(out.toString().contains("ANY INSURANCE")),
            () -> assertFalse(out.toString().contains("INSURANCE BETS"))
        );
    }

    @Test
    @DisplayName("collectInsurance() should collect on Y")
    public void collectInsuranceYes(){
        // Given
        List<Player> players = Collections.singletonList(new Player(1));
        players.get(0).setCurrentBet(100);
        playerSays("Y");
        playerSays("50");
        initGame();

        // When
        game.collectInsurance(players);

        // Then
        assertAll(
            () -> assertTrue(out.toString().contains("ANY INSURANCE")),
            () -> assertTrue(out.toString().contains("INSURANCE BETS")),
            () -> assertEquals(50, players.get(0).getInsuranceBet())
        );
    }

    @Test
    @DisplayName("collectInsurance() should not allow more than 50% of current bet")
    public void collectInsuranceYesTooMuch(){
        // Given
        List<Player> players = Collections.singletonList(new Player(1));
        players.get(0).setCurrentBet(100);
        playerSays("Y");
        playerSays("51");
        playerSays("50");
        initGame();

        // When
        game.collectInsurance(players);

        // Then
        assertAll(
            () -> assertEquals(50, players.get(0).getInsuranceBet()),
            () -> assertTrue(out.toString().contains("# 1 ? # 1 ?"))
        );
    }

    @Test
    @DisplayName("collectInsurance() should not allow negative bets")
    public void collectInsuranceYesNegative(){
        // Given
        List<Player> players = Collections.singletonList(new Player(1));
        players.get(0).setCurrentBet(100);
        playerSays("Y");
        playerSays("-1");
        playerSays("1");
        initGame();

        // When
        game.collectInsurance(players);

        // Then
        assertAll(
            () -> assertEquals(1, players.get(0).getInsuranceBet()),
            () -> assertTrue(out.toString().contains("# 1 ? # 1 ?"))
        );
    }

    @Test
    @DisplayName("collectInsurance() should prompt all players")
    public void collectInsuranceYesTwoPlayers(){
        // Given
        List<Player> players = Arrays.asList(
            new Player(1),
            new Player(2)
        );
        players.get(0).setCurrentBet(100);
        players.get(1).setCurrentBet(100);

        playerSays("Y");
        playerSays("50");
        playerSays("25");
        initGame();

        // When
        game.collectInsurance(players);

        // Then
        assertAll(
            () -> assertEquals(50, players.get(0).getInsuranceBet()),
            () -> assertEquals(25, players.get(1).getInsuranceBet()),
            () -> assertTrue(out.toString().contains("# 1 ? # 2 ?"))
        );
    }

    @Test
    @DisplayName("play() should end on STAY")
    public void playEndOnStay(){
        // Given
        Player player = new Player(1);
        player.dealCard(new Card(3, Card.Suit.CLUBS));
        player.dealCard(new Card(2, Card.Suit.SPADES));
        playerSays("S"); // "I also like to live dangerously."
        initGame();

        // When
        game.play(player);

        // Then
        assertTrue(out.toString().startsWith("PLAYER 1 ? TOTAL IS 5"));
    }

    @Test
    @DisplayName("play() should allow HIT until BUST")
    public void playHitUntilBust() {
        // Given
        Player player = new Player(1);
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(10, Card.Suit.SPADES));

        playerSays("H");
        playerGets(1, Card.Suit.SPADES); // 20
        playerSays("H");
        playerGets(1, Card.Suit.HEARTS); // 21
        playerSays("H");
        playerGets(1, Card.Suit.CLUBS); // 22 - D'oh!
        initGame();

        // When
        game.play(player);

        // Then
        assertTrue(out.toString().contains("BUSTED"));
    }

    @Test
    @DisplayName("Should allow double down on initial turn")
    public void playDoubleDown(){
        // Given
        Player player = new Player(1);
        player.setCurrentBet(100);
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(4, Card.Suit.SPADES));

        playerSays("D");
        playerGets(7, Card.Suit.SPADES);
        initGame();

        // When
        game.play(player);

        // Then
        assertTrue(player.getCurrentBet() == 200);
        assertTrue(player.getHand().size() == 3);
    }

    @Test
    @DisplayName("Should NOT allow double down after initial deal")
    public void playDoubleDownLate(){
        // Given
        Player player = new Player(1);
        player.setCurrentBet(100);
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(2, Card.Suit.SPADES));

        playerSays("H");
        playerGets(7, Card.Suit.SPADES);
        playerSays("D");
        playerSays("S");
        initGame();

        // When
        game.play(player);

        // Then
        assertTrue(out.toString().contains("TYPE H, OR S, PLEASE"));
    }

    @Test
    @DisplayName("play() should end on STAY after split")
    public void playSplitEndOnStay(){
        // Given
        Player player = new Player(1);
        player.dealCard(new Card(1, Card.Suit.CLUBS));
        player.dealCard(new Card(1, Card.Suit.SPADES));

        playerSays("/");
        playerGets(2, Card.Suit.SPADES); // First hand
        playerSays("S");
        playerGets(2, Card.Suit.SPADES); // Second hand
        playerSays("S");
        initGame();

        // When
        game.play(player);

        // Then
        assertTrue(out.toString().contains("FIRST HAND RECEIVES"));
        assertTrue(out.toString().contains("SECOND HAND RECEIVES"));
    }

    @Test
    @DisplayName("play() should allow HIT until BUST after split")
    public void playSplitHitUntilBust() {
        // Given
        Player player = new Player(1);
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(10, Card.Suit.SPADES));

        playerSays("/");
        playerGets(12, Card.Suit.SPADES); // First hand has 20
        playerSays("H");
        playerGets(12, Card.Suit.HEARTS); // First hand busted
        playerGets(10, Card.Suit.HEARTS); // Second hand gets a 10
        playerSays("S");
        initGame();

        // When
        game.play(player);

        // Then
        assertTrue(out.toString().contains("BUSTED"));
    }

    @Test
    @DisplayName("play() should allow HIT on split hand until BUST")
    public void playSplitHitUntilBustHand2() {
        // Given
        Player player = new Player(1);
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(10, Card.Suit.SPADES));

        playerSays("/");
        playerGets(1, Card.Suit.CLUBS); // First hand is 21
        playerSays("S");
        playerGets(12, Card.Suit.SPADES); // Second hand is 20
        playerSays("H");
        playerGets(12, Card.Suit.HEARTS); // Busted
        playerSays("H");
        initGame();

        // When
        game.play(player);

        // Then
        assertTrue(out.toString().contains("BUSTED"));
    }

    @Test
    @DisplayName("play() should allow double down on split hands")
    public void playSplitDoubleDown(){
        // Given
        Player player = new Player(1);
        player.setCurrentBet(100);
        player.dealCard(new Card(9, Card.Suit.HEARTS));
        player.dealCard(new Card(9, Card.Suit.SPADES));

        playerSays("/");
        playerGets(5, Card.Suit.DIAMONDS); // First hand is 14
        playerSays("D");
        playerGets(6, Card.Suit.HEARTS); // First hand is 20
        playerGets(7, Card.Suit.CLUBS); // Second hand is 16
        playerSays("D");
        playerGets(4, Card.Suit.CLUBS); // Second hand is 20
        initGame();

        // When
        game.play(player);

        // Then
        assertAll(
            () -> assertEquals(200, player.getCurrentBet(), "Current bet should be doubled"),
            () -> assertEquals(200, player.getSplitBet(), "Split bet should be doubled"),
            () -> assertEquals(3, player.getHand(1).size(), "First hand should have exactly three cards"),
            () -> assertEquals(3, player.getHand(2).size(), "Second hand should have exactly three cards")
        );
    }

    @Test
    @DisplayName("play() should NOT allow re-splitting first split hand")
    public void playSplitTwice(){
        // Given
        Player player = new Player(1);
        player.setCurrentBet(100);
        player.dealCard(new Card(2, Card.Suit.HEARTS));
        player.dealCard(new Card(2, Card.Suit.SPADES));

        playerSays("/");
        playerGets(13, Card.Suit.CLUBS); // First hand
        playerSays("/"); // Not allowed
        playerSays("S");
        playerGets(13, Card.Suit.SPADES); // Second hand
        playerSays("S");
        initGame();

        // When
        game.play(player);

        // Then
        assertTrue(out.toString().contains("TYPE H, S OR D, PLEASE"));
    }

    @Test
    @DisplayName("play() should NOT allow re-splitting second split hand")
    public void playSplitTwiceHand2(){
        // Given
        Player player = new Player(1);
        player.setCurrentBet(100);
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(10, Card.Suit.SPADES));

        playerSays("/");
        playerGets(13, Card.Suit.CLUBS); // First hand
        playerSays("S");
        playerGets(13, Card.Suit.SPADES); // Second hand
        playerSays("/"); // Not allowed
        playerSays("S");
        initGame();

        // When
        game.play(player);

        // Then
        assertTrue(out.toString().contains("TYPE H, S OR D, PLEASE"));
    }

    @Test
    @DisplayName("evaluateRound() should total both hands when split")
    public void evaluateRoundWithSplitHands(){
        // Given
        Player dealer = new Player(0); //Dealer
        dealer.dealCard(new Card(1, Card.Suit.HEARTS));
        dealer.dealCard(new Card(1, Card.Suit.SPADES));

        Player player = new Player(1);
        player.recordRound(200);//Set starting total
        player.setCurrentBet(50);
        player.dealCard(new Card(1, Card.Suit.HEARTS));
        player.dealCard(new Card(1, Card.Suit.SPADES));
        
        playerSays("/");
        playerGets(13, Card.Suit.CLUBS); // First hand
        playerSays("S");
        playerGets(13, Card.Suit.SPADES); // Second hand
        playerSays("S");
        initGame();

        // When
        game.play(player);
        game.evaluateRound(Arrays.asList(player), dealer);

        // Then
        assertAll(
            () -> assertTrue(out.toString().contains("PLAYER 1  WINS    100 TOTAL= 300")),
            () -> assertTrue(out.toString().contains("DEALER'S TOTAL= -100"))
        );
    }

    @Test
    @DisplayName("evaluateRound() should total add twice insurance bet")
    public void evaluateRoundWithInsurance(){
        // Given
        Player dealer = new Player(0); //Dealer
        dealer.dealCard(new Card(10, Card.Suit.HEARTS));
        dealer.dealCard(new Card(1, Card.Suit.SPADES));

        Player player = new Player(1);
        player.setCurrentBet(50);
        player.setInsuranceBet(10);
        player.dealCard(new Card(2, Card.Suit.HEARTS));
        player.dealCard(new Card(1, Card.Suit.SPADES));
        initGame();

        // When
        game.evaluateRound(Arrays.asList(player), dealer);

        // Then
        // Loses current bet (50) and wins 2*10 for total -30
        assertAll(
            () -> assertTrue(out.toString().contains("PLAYER 1 LOSES     30 TOTAL= -30")),
            () -> assertTrue(out.toString().contains("DEALER'S TOTAL= 30"))
        );
    }

    @Test
    @DisplayName("evaluateRound() should push with no total change")
    public void evaluateRoundWithPush(){
        // Given
        Player dealer = new Player(0);
        dealer.dealCard(new Card(10, Card.Suit.HEARTS));
        dealer.dealCard(new Card(8, Card.Suit.SPADES)); 

        Player player = new Player(1);
        player.setCurrentBet(10);
        player.dealCard(new Card(9, Card.Suit.HEARTS));
        player.dealCard(new Card(9, Card.Suit.SPADES));
        initGame();

        // When (Dealer and Player both have 19)
        game.evaluateRound(Arrays.asList(player), dealer);

        // Then        
        assertAll(
            () -> assertTrue(out.toString().contains("PLAYER 1 PUSHES       TOTAL= 0")),
            () -> assertTrue(out.toString().contains("DEALER'S TOTAL= 0"))
        );
    }

    @Test
    @DisplayName("shouldPlayDealer() return false when players bust")
    public void shouldPlayDealerBust(){
        // Given
        Player player = new Player(1);
        player.dealCard(new Card(10, Card.Suit.SPADES));
        player.dealCard(new Card(10, Card.Suit.SPADES));
        player.split();
        player.dealCard(new Card(5, Card.Suit.SPADES));
        player.dealCard(new Card(8, Card.Suit.SPADES));//First hand Busted

        player.dealCard(new Card(5, Card.Suit.SPADES),2);
        player.dealCard(new Card(8, Card.Suit.SPADES),2);//Second hand Busted

        Player playerTwo = new Player(2);
        playerTwo.dealCard(new Card(7, Card.Suit.HEARTS));
        playerTwo.dealCard(new Card(8, Card.Suit.HEARTS));
        playerTwo.dealCard(new Card(9, Card.Suit.HEARTS));
        initGame();

        // When 
        boolean result = game.shouldPlayDealer(Arrays.asList(player,playerTwo));

        // Then        
        assertFalse(result);
    }

    @Test
    @DisplayName("shouldPlayDealer() return false when players bust")
    public void ShouldPlayer(){
        // Given
        Player player = new Player(1);
        player.dealCard(new Card(10, Card.Suit.SPADES));
        player.dealCard(new Card(10, Card.Suit.SPADES));
        player.split();
        player.dealCard(new Card(5, Card.Suit.SPADES));
        player.dealCard(new Card(8, Card.Suit.SPADES));//First hand Busted

        player.dealCard(new Card(5, Card.Suit.SPADES),2);
        player.dealCard(new Card(8, Card.Suit.SPADES),2);//Second hand Busted

        Player playerTwo = new Player(2);
        playerTwo.dealCard(new Card(7, Card.Suit.HEARTS));
        playerTwo.dealCard(new Card(8, Card.Suit.HEARTS));
        playerTwo.dealCard(new Card(9, Card.Suit.HEARTS));
        initGame();

        // When 
        boolean result = game.shouldPlayDealer(Arrays.asList(player,playerTwo));

        // Then        
        assertFalse(result);
    }

    @Test
    @DisplayName("shouldPlayDealer() return true when player has non-natural blackjack")
    public void shouldPlayDealerNonNaturalBlackjack(){
        // Given
        Player player = new Player(1);
        player.dealCard(new Card(5, Card.Suit.SPADES));
        player.dealCard(new Card(6, Card.Suit.DIAMONDS));
        player.dealCard(new Card(10, Card.Suit.SPADES));

        initGame();

        // When 
        boolean result = game.shouldPlayDealer(Arrays.asList(player));

        // Then        
        assertTrue(result);
    }

    @Test
    @DisplayName("shouldPlayDealer() return true when player doesn't have blackjack")
    public void shouldPlayDealerNonBlackjack(){
        // Given
        Player player = new Player(1);
        player.dealCard(new Card(10, Card.Suit.SPADES));
        player.dealCard(new Card(6, Card.Suit.DIAMONDS));
        initGame();

        // When 
        boolean result = game.shouldPlayDealer(Arrays.asList(player));

        // Then        
        assertTrue(result);
    }


    @Test
    @DisplayName("playDealer() should DRAW on less than 17 intial deal")
    public void playDealerLessThanSeventeen(){
        // Given
        Player dealer = new Player(0);
        dealer.dealCard(new Card(10, Card.Suit.SPADES));
        dealer.dealCard(new Card(6, Card.Suit.DIAMONDS));
        playerGets(11, Card.Suit.DIAMONDS);
        initGame();

        // When 
       game.playDealer(dealer);

        // Then        
        assertTrue(out.toString().contains("DRAWS"));
        assertTrue(out.toString().contains("BUSTED"));
    }

    @Test
    @DisplayName("playDealer() should stay on more than 17 intial deal")
    public void playDealerMoreThanSeventeen(){
        // Given
        Player dealer = new Player(0);
        dealer.dealCard(new Card(10, Card.Suit.SPADES));
        dealer.dealCard(new Card(8, Card.Suit.DIAMONDS));
        initGame();

        // When 
       game.playDealer(dealer);

        // Then        
        assertFalse(out.toString().contains("DRAWS"));
        assertFalse(out.toString().contains("BUSTED"));
        assertTrue(out.toString().contains("---TOTAL IS"));
    }

}

```