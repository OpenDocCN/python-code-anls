# BasicComputerGames源码解析 41

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


## Porting notes

Variables:

* E: score_limit
* H(2): Scores
* T(2): Team toggle
* T: team who currently possesses the ball
* L: Offset
* P: Who has the ball
* K: yards
* R: Runback current team in yards
* P$(20): Actions (see data.json)

Functions:

* `P$(I)`: Access index `I` of the `P` array
* ABS: abs (absolute value)
* RND(1): random()
* GOSUB: Execute a function - will jump back to this
* GOTO: Just jump

Patterns:

* `T=T(T)`: Toggle the team who currently has the ball


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Fur Trader

You are the leader of a French fur trading expedition in 1776 leaving the Ontario area to sell furs and get supplies for the next year. You have a choice of three forts at which you may trade. The cost of supplies and the amount you recieve for your furs will depend upon the fort you choose. You also specify what types of furs that you have to trade.

The game goes on and on until you elect to trade no longer.

Author of the program is Dan Bachor, University of Calgary, Alberta, Canada.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=69)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=84)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- The value of some furs are not changed from the previous fort when you select fort 2 or 3.  As a result, you will get a different value for your firs depending on whether you have previously visited a different fort.  (All fur values are set when you visit Fort 1.)

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `38_Fur_Trader/c/furtrader.c`

这段代码是一个 C 语言程序，它是由 krt@krt.com.au 开发的，目的是将 furtrader.bas 文件中的内容 port 到 ANSI C (C99) 标准中。它使用了 GCC 编译器，并在编译时使用了 `-Werror` 和 `-Wall` 选项来警告可能的错误。

具体来说，这段代码的作用是将 furtrader.bas 文件中的所有内容（包括数据和注释）保存到一个名为 furtrader 的可执行文件中。该程序的主要目的是在不同的编译器中提供一个统一的 API，以便在未来的开发中更容易地使用相同的代码。


```

/*
 * Ported from furtrader.bas to ANSI C (C99) by krt@krt.com.au
 *
 * compile with:
 *    gcc -g -Wall -Werror furtrader.c -o furtrader
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/* Constants */
```

这段代码定义了一系列宏，用于定义和命名不同的毛皮动物类型。

宏定义了四个名为FUR_TYPE_COUNT的常量，值为4。

定义了一个名为FUR_MINK的常量，值为0。

定义了一个名为FUR_BEAVER的常量，值为1。

定义了一个名为FUR_ERMINE的常量，值为2。

定义了一个名为FUR_FOX的常量，值为3。

定义了一个名为MAX_FURS的常量，值为190。

定义了一个名为FUR_NAMES的数组，用于存储毛皮动物类型的名称。该数组长度为FUR_TYPE_COUNT，即4。

定义了一个名为FORT_TYPE_COUNT的常量，值为3。

定义了一个名为FORT_MONTREAL的常量，值为1。

定义了一个名为FORT_QUEBEC的常量，值为2。

定义了一个名为FORT_NEWYORK的常量，值为3。

定义了一个名为FORT_NAMES的数组，用于存储FORT_MONTREAL,FORT_QUEBEC,FORT_NEWYORK三个城市的名称。该数组长度为FORT_TYPE_COUNT，即3。

最后，定义了一系列常量，用于定义FUR_NAMES数组中的元素个数。


```
#define FUR_TYPE_COUNT    4
#define FUR_MINK          0
#define FUR_BEAVER        1
#define FUR_ERMINE        2
#define FUR_FOX           3
#define MAX_FURS        190
const char *FUR_NAMES[FUR_TYPE_COUNT] = { "MINK", "BEAVER", "ERMINE", "FOX" };

#define FORT_TYPE_COUNT 3
#define FORT_MONTREAL   1
#define FORT_QUEBEC     2
#define FORT_NEWYORK    3
const char *FORT_NAMES[FORT_TYPE_COUNT] = { "HOCHELAGA (MONTREAL)", "STADACONA (QUEBEC)", "NEW YORK" };



```

这两段代码定义了两个函数：printAtColumn和print。printAtColumn函数接受一个整数column和一个字符串words，并输出这些单词在column列中的位置。print函数接受一个字符串words，并输出这些单词。

printAtColumn函数首先输出一个空行，然后遍历words中的每个单词，将其输出并输出一个空格。接着，将输出结果与words中的最后一个单词'\n'连接并输出。这样，在output列中，每个单词都会输出在自己的位置，并在该列中的下一行输出一个空格。

print函数与printAtColumn函数类似，但输出的结果中没有'\n'这个特殊字符。print函数首先输出一个空行，然后遍历words中的每个单词，将其输出并输出一个空格。接着，将输出结果与words中的最后一个单词'\n'连接并输出。这样，在output列中，每个单词都会输出在自己的位置，并在该列中的下一行输出一个空格。


```
/* Print the words at the specified column */
void printAtColumn( int column, const char *words )
{
    int i;
    for ( i=0; i<column; i++ )
        printf( " " );
    printf( "%s\n", words );
}

/* trivial function to output a line with a \n */
void print( const char *words )
{
    printf( "%s\n", words );
}

```

这段代码是一个名为 `showIntroduction` 的函数，它的作用是输出一条介绍性的消息给玩家。

具体来说，这段代码会输出以下内容：

"YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN 1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET SUPPLIES FOR THE NEXT YEAR. YOU HAVE A CHOICE OF THREE FORTS AT WHICH YOU MAY TRADE. THE COST OF SUPPLIES AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND ON THE FORT THAT YOU CHOOSE."

然后，它会让玩家从三个选项中选择一个 fort，并根据所选 fort 输出两个值：

"THE FORT THAT YOU CHOOSE" 和 "THE AMOUNT YOU RECEIVE FOR YOUR FURS" 分别取决于所选 fort。


```
/* Show the player the introductory message */
void showIntroduction()
{
    print( "YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN " );
    print( "1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET" );
    print( "SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE" );
    print( "FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES" );
    print( "AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND" );
    print( "ON THE FORT THAT YOU CHOOSE." );
    print( "" );
}


/*
 * Prompt the user for input.
 * When input is given, try to conver it to an integer
 * return the integer converted value or 0 on error
 */
```

该代码的作用是获取用户输入的数字，并将其存储在整数变量result中。在代码中，首先创建了一个字符数组buffer，用于存储用户输入的字符串。然后定义了一个指向字符结束标志的指针endstr，用于在字符串结束时进行读取。接下来，使用while循环，不断 prompt用户输入字符串，并将其存储到buffer中。然后使用strtol函数将buffer中的字符串转换为整数，并将其存储在result中。在转换过程中，使用endstr指针检查是否已经读取到字符串的结尾，如果没有，说明字符串转为了数字，可以安全地将其转换为整数并返回。如果endstr和buffer指向的是同一个字符串，说明读取失败，返回-1。


```
int getNumericInput()
{
    int  result = -1;
    char buffer[64];   /* somewhere to store user input */
    char *endstr;

    while ( result == -1 )
    {
        printf( ">> " );                                 /* prompt the user */
        fgets( buffer, sizeof( buffer ), stdin );        /* read from the console into the buffer */
        result = (int)strtol( buffer, &endstr, 10 );     /* only simple error checking */

        if ( endstr == buffer )                          /* was the string -> integer ok? */
            result = -1;
    }

    return result;
}


```

这段代码的作用是 prompt the user for a Yes/No input and convert the input to a single upper-case letter. It does this by first initializing a variable `result` to '!', and then storing the user's input in a variable `buffer`.

It then enters a while loop that continues until the user enters either 'Y' or 'N', which is checked in the inside of the loop. Inside the loop, the code prints "ANSWER YES OR NO" to the console, and then reads the user's input into the `buffer` variable.

After that, the code checks whether the `buffer` variable contains the letter 'Y' or 'N' and converts the input to the corresponding upper-case letter in the `result` variable.

Finally, the `result` variable is returned as the output of the function.


```
/*
 * Prompt the user for YES/NO input.
 * When input is given, try to work out if it's YES, Yes, yes, Y, etc.
 * And convert to a single upper-case letter
 * Returns a character of 'Y' or 'N'.
 */
char getYesOrNo()
{
    char result = '!';
    char buffer[64];   /* somewhere to store user input */

    while ( !( result == 'Y' || result == 'N' ) )       /* While the answer was not Yes or No */
    {
        print( "ANSWER YES OR NO" );
        printf( ">> " );

        fgets( buffer, sizeof( buffer ), stdin );            /* read from the console into the buffer */
        if ( buffer[0] == 'Y' || buffer[0] == 'y' )
            result = 'Y';
        else if ( buffer[0] == 'N' || buffer[0] == 'n' )
            result = 'N';
    }

    return result;
}



```

这段代码是一个用于向玩家显示Fort的不同选择，并获取他们的选择码。如果玩家的输入有效(1, 2, 3)，则返回该选择码，否则一直 prompting the user。

具体来说，代码首先定义了一个变量 result，用于保存玩家的输入。然后在一个 while 循环中，每次循环都会显示一些信息，包括Fort的选择列表和提示玩家可以选择1、2或3中的任意一个数字。每次循环都会从玩家获取一个整数类型的输入，并将其存储在 result 变量中。如果循环变量中存储的选择码等于 1、2 或 3，则程序返回该选择码并退出循环。否则，程序将继续循环，继续提示玩家输入并获取新的输入。


```
/*
 * Show the player the choices of Fort, get their input, if the
 * input is a valid choice (1,2,3) return it, otherwise keep
 * prompting the user.
 */
int getFortChoice()
{
    int result = 0;

    while ( result == 0 )
    {
        print( "" );
        print( "YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2," );
        print( "OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)" );
        print( "AND IS UNDER THE PROTECTION OF THE FRENCH ARMY." );
        print( "FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE" );
        print( "PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST" );
        print( "MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS." );
        print( "FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL." );
        print( "YOU MUST CROSS THROUGH IROQUOIS LAND." );
        print( "ANSWER 1, 2, OR 3." );

        result = getNumericInput();   /* get input from the player */
    }

    return result;
}


```

这段代码是一个名为 `showFortComment` 的函数，它用于输出一个用于指定堡垒（Fort）的描述。堡垒可以是蒙特利号（Fort Montreal）、魁北克号（Fort Quebec）或者是新泽西号（Fort New York）。

函数首先打印一个空字符串，然后根据选择的堡垒类型输出相应的描述。如果选择的堡垒不存在的，函数会输出一个错误消息并退出。


```
/*
 * Print the description for the fort
 */
void showFortComment( int which_fort )
{
    print( "" );
    if ( which_fort == FORT_MONTREAL )
    {
        print( "YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT" );
        print( "IS FAR FROM ANY SEAPORT.  THE VALUE" );
        print( "YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST" );
        print( "OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK." );
    }
    else if ( which_fort == FORT_QUEBEC )
    {
        print( "YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION," );
        print( "HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN" );
        print( "THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE" );
        print( "FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE." );
    }
    else if ( which_fort == FORT_NEWYORK )
    {
        print( "YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT" );
        print( "FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE" );
        print( "FOR YOUR FURS.  THE COST OF YOUR SUPPLIES" );
        print( "WILL BE LOWER THAN AT ALL THE OTHER FORTS." );
    }
    else
    {
        printf( "Internal error #1, fort %d does not exist\n", which_fort );
        exit( 1 );  /* you have a bug */
    }
    print( "" );
}


```

这段代码的作用是提示玩家购买毛皮，并接受数字输入，如果输入的值不正确，则重新提示。它要求玩家选择想要的毛皮类型，包括水貂、狐狸、骡子和紫貂。

具体来说，代码首先输出毛皮类型列表，然后提示玩家输入想要的毛皮类型数量。接着，代码循环输出每个毛皮类型的数量，直到玩家输入正确的数量为止。循环结束后，玩家可以继续购买其他毛皮。


```
/*
 * Prompt the player for how many of each fur type they want.
 * Accept numeric inputs, re-prompting on incorrect input values
 */
void getFursPurchase( int *furs )
{
    int i;

    printf( "YOUR %d FURS ARE DISTRIBUTED AMONG THE FOLLOWING\n", FUR_TYPE_COUNT );
    print( "KINDS OF PELTS: MINK, BEAVER, ERMINE AND FOX." );
    print( "" );

    for ( i=0; i<FUR_TYPE_COUNT; i++ )
    {
        printf( "HOW MANY %s DO YOU HAVE\n", FUR_NAMES[i] );
        furs[i] = getNumericInput();
    }
}


```

这段代码有两个函数，第一个函数名为 "zeroInventory"，作用是清空玩家的物品库存，具体实现方式是遍历玩家所有的物品，将每个物品的数量置为零。第二个函数名同第一个函数名，但作用是计算玩家物品库存的总数，具体实现方式是将所有物品数量加一，并返回该值。

在这段注释中，作者对代码进行了说明，指出该函数的作用是清空玩家物品库存，并提供了两个函数实现该目的。其中，第一个函数使用了数组下标，遍历了所有的物品，并将每个物品的数量置为零；第二个函数则使用了循环，计算了所有物品的数量，并返回该值。


```
/*
 * (Re)Set the player's inventory to zero
 */
void zeroInventory( int *player_fur_count )
{
    int i;
    for ( i=0; i<FUR_TYPE_COUNT; i++ )
    {
        player_fur_count[i] = 0;
    }
}


/*
 * Tally the player's inventory
 */
```

这段代码定义了一个名为`sumInventory`的函数，其参数为一个整型指针变量`player_fur_count`，该变量存储了玩家所拥有的动物毛发的数量。函数返回玩家毛发总数。

具体来说，这段代码实现了一个循环，该循环遍历了`FUR_TYPE_COUNT`类型变量所包含的`player_fur_count`变量。在循环体内，将当前遍历到的`player_fur_count`值加到了一个名为`result`的整型变量中。

最终，函数返回了`result`，即玩家毛发总数。


```
int sumInventory( int *player_fur_count )
{
    int result = 0;
    int i;
    for ( i=0; i<FUR_TYPE_COUNT; i++ )
    {
        result += player_fur_count[i];
    }

    return result;
}


/*
 * Return a random number between a & b
 * Ref: https://stackoverflow.com/a/686376/1730895
 */
```

这段代码定义了一个名为float randomAB(float a, float b)的函数，用于生成0到1之间的随机浮点数。

接着定义了一个名为float randFloat()的函数，这个函数调用randomAB函数生成0到1之间的随机浮点数。

接下来定义了三个枚举类型STATE_STARTING、STATE_CHOOSING_FORT、STATE_TRAVELLING，它们分别代表游戏主循环的三个状态，分别为开始、选择攀爬、行走。

最后在程序开始时，将初始值设为STATE_STARTING，然后进入游戏主循环，每经过一定时间步之后，根据玩家的选择状态执行相应的函数。比如开始攀爬时，调用randFloat()函数生成一个0到1之间的随机浮点数，然后更新玩家当前的攀爬高度。


```
float randomAB(float a, float b)
{
    return ((b - a) * ((float)rand() / (float)RAND_MAX)) + a;
}
/* Random floating point number between 0 and 1 */
float randFloat()
{
    return randomAB( 0, 1 );
}


/* States to allow switching in main game-loop */
#define STATE_STARTING      1
#define STATE_CHOOSING_FORT 2
#define STATE_TRAVELLING    3
```

This appears to be a game of trade where the player is given a list of furs (beaver, fox, ermine, and mink) that they can trade at a New York fort. The player must trade these furs for other furs, and the value of each trade is calculated based on the player's current funds. The player can continue trading until they choose to exit the game.


```
#define STATE_TRADING       4

int main( void )
{
    /* variables for storing player's status */
    float player_funds = 0;                              /* no money */
    int   player_furs[FUR_TYPE_COUNT]  = { 0, 0, 0, 0 }; /* no furs */

    /* player input holders */
    char  yes_or_no;
    int   event_picker;
    int   which_fort;

    /* what part of the game is in play */
    int   game_state = STATE_STARTING;

    /* commodity prices */
    float mink_price   = -1;
    float beaver_price = -1;
    float ermine_price = -1;
    float fox_price    = -1;  /* sometimes this takes the "last" price (probably this was a bug) */

    float mink_value;
    float beaver_value;
    float ermine_value;
    float fox_value;      /* for calculating sales results */


    srand( time( NULL ) );  /* seed the random number generator */

    printAtColumn( 31, "FUR TRADER" );
    printAtColumn( 15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY" );
    printAtColumn( 15, "(Ported to ANSI-C Oct 2012 krt@krt.com.au)" );
    print( "\n\n\n" );

    /* Loop forever until the player asks to quit */
    while ( 1 )
    {
        if ( game_state == STATE_STARTING )
        {
            showIntroduction();

            player_funds = 600;            /* Initial player start money */
            zeroInventory( player_furs );  /* Player fur inventory */

            print( "DO YOU WISH TO TRADE FURS?" );
            yes_or_no = getYesOrNo();
            if ( yes_or_no == 'N' )
                exit( 0 );                 /* STOP */
            game_state = STATE_TRADING;
        }

        else if ( game_state == STATE_TRADING )
        {
            print( "" );
            printf( "YOU HAVE $ %1.2f IN SAVINGS\n", player_funds );
            printf( "AND %d FURS TO BEGIN THE EXPEDITION\n", MAX_FURS );
            getFursPurchase( player_furs );

            if ( sumInventory( player_furs ) > MAX_FURS )
            {
                print( "" );
                print( "YOU MAY NOT HAVE THAT MANY FURS." );
                print( "DO NOT TRY TO CHEAT.  I CAN ADD." );
                print( "YOU MUST START AGAIN." );
                print( "" );
                game_state = STATE_STARTING;   /* T/N: Wow, harsh. */
            }
            else
            {
                game_state = STATE_CHOOSING_FORT;
            }
        }

        else if ( game_state == STATE_CHOOSING_FORT )
        {
            which_fort = getFortChoice();
            showFortComment( which_fort );
            print( "DO YOU WANT TO TRADE AT ANOTHER FORT?" );
            yes_or_no = getYesOrNo();
            if ( yes_or_no == 'N' )
                game_state = STATE_TRAVELLING;
        }

        else if ( game_state == STATE_TRAVELLING )
        {
            print( "" );
            if ( which_fort == FORT_MONTREAL )
            {
                mink_price   = ( ( 0.2 * randFloat() + 0.70 ) * 100 + 0.5 ) / 100;
                ermine_price = ( ( 0.2 * randFloat() + 0.65 ) * 100 + 0.5 ) / 100;
                beaver_price = ( ( 0.2 * randFloat() + 0.75 ) * 100 + 0.5 ) / 100;
                fox_price    = ( ( 0.2 * randFloat() + 0.80 ) * 100 + 0.5 ) / 100;

                print( "SUPPLIES AT FORT HOCHELAGA COST $150.00." );
                print( "YOUR TRAVEL EXPENSES TO HOCHELAGA WERE $10.00." );
                player_funds -= 160;
            }

            else if ( which_fort == FORT_QUEBEC )
            {
                mink_price   = ( ( 0.30 * randFloat() + 0.85 ) * 100 + 0.5 ) / 100;
                ermine_price = ( ( 0.15 * randFloat() + 0.80 ) * 100 + 0.5 ) / 100;
                beaver_price = ( ( 0.20 * randFloat() + 0.90 ) * 100 + 0.5 ) / 100;
                fox_price    = ( ( 0.25 * randFloat() + 1.10 ) * 100 + 0.5 ) / 100;
                event_picker = ( 10 * randFloat() ) + 1;

                if ( event_picker <= 2 )
                {
                    print( "YOUR BEAVER WERE TOO HEAVY TO CARRY ACROSS" );
                    print( "THE PORTAGE.  YOU HAD TO LEAVE THE PELTS, BUT FOUND" );
                    print( "THEM STOLEN WHEN YOU RETURNED." );
                    player_furs[ FUR_BEAVER ] = 0;
                }
                else if ( event_picker <= 6 )
                {
                    print( "YOU ARRIVED SAFELY AT FORT STADACONA." );
                }
                else if ( event_picker <= 8 )
                {
                    print( "YOUR CANOE UPSET IN THE LACHINE RAPIDS.  YOU" );
                    print( "LOST ALL YOUR FURS." );
                    zeroInventory( player_furs );
                }
                else if ( event_picker <= 10 )
                {
                    print( "YOUR FOX PELTS WERE NOT CURED PROPERLY." );
                    print( "NO ONE WILL BUY THEM." );
                    player_furs[ FUR_FOX ] = 0;
                }
                else
                {
                    printf( "Internal Error #3, Out-of-bounds event_picker %d\n", event_picker );
                    exit( 1 );  /* you have a bug */
                }

                print( "" );
                print( "SUPPLIES AT FORT STADACONA COST $125.00." );
                print( "YOUR TRAVEL EXPENSES TO STADACONA WERE $15.00." );
                player_funds -= 140;
            }

            else if ( which_fort == FORT_NEWYORK )
            {
                mink_price   = ( ( 0.15 * randFloat() + 1.05 ) * 100 + 0.5 ) / 100;
                ermine_price = ( ( 0.15 * randFloat() + 0.95 ) * 100 + 0.5 ) / 100;
                beaver_price = ( ( 0.25 * randFloat() + 1.00 ) * 100 + 0.5 ) / 100;
                if ( fox_price < 0 )
                {
                    /* Original Bug?  There is no Fox price generated for New York,
                       it will use any previous "D1" price.
                       So if there was no previous value, make one up */
                    fox_price = ( ( 0.25 * randFloat() + 1.05 ) * 100 + 0.5 ) / 100; /* not in orginal code */
                }
                event_picker = ( 10 * randFloat() ) + 1;

                if ( event_picker <= 2 )
                {
                    print( "YOU WERE ATTACKED BY A PARTY OF IROQUOIS." );
                    print( "ALL PEOPLE IN YOUR TRADING GROUP WERE" );
                    print( "KILLED.  THIS ENDS THE GAME." );
                    exit( 0 );
                }
                else if ( event_picker <= 6 )
                {
                    print( "YOU WERE LUCKY.  YOU ARRIVED SAFELY" );
                    print( "AT FORT NEW YORK." );
                }
                else if ( event_picker <= 8 )
                {
                    print( "YOU NARROWLY ESCAPED AN IROQUOIS RAIDING PARTY." );
                    print( "HOWEVER, YOU HAD TO LEAVE ALL YOUR FURS BEHIND." );
                    zeroInventory( player_furs );
                }
                else if ( event_picker <= 10 )
                {
                    mink_price /= 2;
                    fox_price  /= 2;
                    print( "YOUR MINK AND BEAVER WERE DAMAGED ON YOUR TRIP." );
                    print( "YOU RECEIVE ONLY HALF THE CURRENT PRICE FOR THESE FURS." );
                }
                else
                {
                    print( "Internal Error #4, Out-of-bounds event_picker %d\n" );
                    exit( 1 );  /* you have a bug */
                }

                print( "" );
                print( "SUPPLIES AT NEW YORK COST $85.00." );
                print( "YOUR TRAVEL EXPENSES TO NEW YORK WERE $25.00." );
                player_funds -= 105;
            }

            else
            {
                printf( "Internal error #2, fort %d does not exist\n", which_fort );
                exit( 1 );  /* you have a bug */
            }

            /* Calculate sales */
            beaver_value = beaver_price * player_furs[ FUR_BEAVER ];
            fox_value    = fox_price    * player_furs[ FUR_FOX ];
            ermine_value = ermine_price * player_furs[ FUR_ERMINE ];
            mink_value   = mink_price   * player_furs[ FUR_MINK ];

            print( "" );
            printf( "YOUR BEAVER SOLD FOR $%6.2f\n", beaver_value );
            printf( "YOUR FOX SOLD FOR    $%6.2f\n", fox_value );
            printf( "YOUR ERMINE SOLD FOR $%6.2f\n", ermine_value );
            printf( "YOUR MINK SOLD FOR   $%6.2f\n", mink_value );

            player_funds += beaver_value + fox_value + ermine_value + mink_value;

            print( "" );
            printf( "YOU NOW HAVE $ %1.2f INCLUDING YOUR PREVIOUS SAVINGS\n", player_funds );

            print( "" );
            print( "DO YOU WANT TO TRADE FURS NEXT YEAR?" );
            yes_or_no = getYesOrNo();
            if ( yes_or_no == 'N' )
                exit( 0 );             /* STOP */
            else
                game_state = STATE_TRADING;

        }
    }

    return 0; /* exit OK */
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [ANSI-C](https://en.wikipedia.org/wiki/ANSI_C)

##### Translator Notes:
I tried to preserve as much of the original layout and flow of the code
as possible.  However I did use enumerated types for the Fort numbers
and Fur types.  I think this was certainly a change for the better, and
makes the code much easier to read.

I also tried to minimise the use of pointers, and stuck with old-school
C formatting, because you never know how old the compiler is.

Interestingly the code seems to have a bug around the prices of Fox Furs.
The commodity-rate for these is stored in the variable `D1`, however some
paths through the code do not set this price.  So there was a chance of
using this uninitialised, or whatever the previous loop set.  I don't
think this was the original authors intent.  So I preserved the original flow
of the code (using the previous `D1` value), but also catching the
uninitialised path, and assigning a "best guess" value.

krt@krt.com.au 2020-10-10


# `38_Fur_Trader/csharp/Game.cs`

This is a C# class that appears to be a simple command-line game where players can buy furs to train their animals. The game has a few different commands for interacting with the player and their animals, as well as a method for counting the player's furs.

The class has a `GetSelectedFur()` method that prompts the player to select the fur they want to buy, and a `GetFurCount()` method that returns the player's current fur count. The class also has a `AnimalPrefs` class that appears to store the player's preferred animals for their animals to care about, and a `FurTrade()` class that handles the actual trade of fur between the player and the store.

It is important to note that this is a very basic implementation and lacks many features and optimizations that you would expect from a proper game.


```
﻿using System;

namespace FurTrader
{
    public class Game
    {
        /// <summary>
        /// random number generator; no seed to be faithful to original implementation
        /// </summary>
        private Random Rnd { get; } = new Random();

        /// <summary>
        /// Generate a price for pelts based off a factor and baseline value
        /// </summary>
        /// <param name="factor">Multiplier for the price</param>
        /// <param name="baseline">The baseline price</param>
        /// <returns>A randomised price for pelts</returns>
        internal double RandomPriceGenerator(double factor, double baseline)
        {
            var price = (Convert.ToInt32((factor * Rnd.NextDouble() + baseline) * 100d) + 5) / 100d;
            return price;
        }

        /// <summary>
        /// Main game loop function. This will play the game endlessly until the player chooses to quit or a GameOver event occurs
        /// </summary>
        /// <remarks>
        /// General structure followed from Adam Dawes (@AdamDawes575) implementation of Acey Ducey.");
        /// </remarks>
        internal void GameLoop()
        {
            // display instructions to the player
            DisplayIntroText();

            var state = new GameState();

            // loop for each turn until the player decides not to continue (or has a Game Over event)
            while ((!state.GameOver) && ContinueGame())
            {
                // clear display at start of each turn
                Console.Clear();

                // play the next turn; pass game state for details and updates from the turn
                PlayTurn(state);
            }

            // end screen; show some statistics to the player
            Console.Clear();
            Console.WriteLine("Thanks for playing!");
            Console.WriteLine("");
            Console.WriteLine($"Total Expeditions: {state.ExpeditionCount}");
            Console.WriteLine($"Final Amount:      {state.Savings.ToString("c")}");
        }

        /// <summary>
        /// Display instructions on how to play the game and wait for the player to press a key.
        /// </summary>
        private void DisplayIntroText()
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Fur Trader.");
            Console.WriteLine("Creating Computing, Morristown, New Jersey.");
            Console.WriteLine("");

            Console.ForegroundColor = ConsoleColor.DarkGreen;
            Console.WriteLine("Originally published in 1978 in the book 'Basic Computer Games' by David Ahl.");
            Console.WriteLine("");

            Console.ForegroundColor = ConsoleColor.Gray;
            Console.WriteLine("You are the leader of a French fur trading expedition in 1776 leaving the Lake Ontario area to sell furs and get supplies for the next year.");
            Console.WriteLine("");
            Console.WriteLine("You have a choice of three forts at which you may trade. The cost of supplies and the amount you receive for your furs will depend on the fort that you choose.");
            Console.WriteLine("");

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Press any key start the game.");
            Console.ReadKey(true);

        }

        /// <summary>
        /// Prompt the player to try again, and wait for them to press Y or N.
        /// </summary>
        /// <returns>Returns true if the player wants to try again, false if they have finished playing.</returns>
        private bool ContinueGame()
        {
            Console.WriteLine("");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("Do you wish to trade furs? ");
            Console.Write("Answer (Y)es or (N)o ");
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("> ");

            char pressedKey;
            // Keep looping until we get a recognised input
            do
            {
                // Read a key, don't display it on screen
                ConsoleKeyInfo key = Console.ReadKey(true);
                // Convert to upper-case so we don't need to care about capitalisation
                pressedKey = Char.ToUpper(key.KeyChar);
                // Is this a key we recognise? If not, keep looping
            } while (pressedKey != 'Y' && pressedKey != 'N');

            // Display the result on the screen
            Console.WriteLine(pressedKey);

            // Return true if the player pressed 'Y', false for anything else.
            return (pressedKey == 'Y');
        }

        /// <summary>
        /// Play a turn
        /// </summary>
        /// <param name="state">The current game state</param>
        private void PlayTurn(GameState state)
        {
            state.UnasignedFurCount = 190;      /// start with 190 furs each turn

            // provide current status to user
            Console.WriteLine(new string('_', 70));
            Console.WriteLine("");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("");
            Console.WriteLine($"You have {state.Savings.ToString("c")} savings and {state.UnasignedFurCount} furs to begin the expedition.");
            Console.WriteLine("");
            Console.WriteLine($"Your {state.UnasignedFurCount} furs are distributed among the following kinds of pelts: Mink, Beaver, Ermine, and Fox");
            Console.WriteLine("");

            // get input on number of pelts
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("How many Mink pelts do you have? ");
            state.MinkPelts = GetPelts(state.UnasignedFurCount);
            state.UnasignedFurCount -= state.MinkPelts;
            Console.WriteLine("");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($"You have {state.UnasignedFurCount} furs remaining for distribution");
            Console.Write("How many Beaver pelts do you have? ");
            state.BeaverPelts = GetPelts(state.UnasignedFurCount);
            state.UnasignedFurCount -= state.BeaverPelts;
            Console.WriteLine("");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($"You have {state.UnasignedFurCount} furs remaining for distribution");
            Console.Write("How many Ermine pelts do you have? ");
            state.ErminePelts = GetPelts(state.UnasignedFurCount);
            state.UnasignedFurCount -= state.ErminePelts;
            Console.WriteLine("");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($"You have {state.UnasignedFurCount} furs remaining for distribution");
            Console.Write("How many Fox pelts do you have? ");
            state.FoxPelts = GetPelts(state.UnasignedFurCount);
            state.UnasignedFurCount -= state.FoxPelts;

            // get input on which fort to trade with; user gets an opportunity to evaluate and re-select fort after selection until user confirms selection
            do
            {
                Console.ForegroundColor = ConsoleColor.White;
                Console.WriteLine("");
                Console.WriteLine("Do you want to trade your furs at Fort 1, Fort 2, or Fort 3");
                Console.WriteLine("Fort 1 is Fort Hochelaga (Montreal) and is under the protection of the French army.");
                Console.WriteLine("Fort 2 is Fort Stadacona (Quebec) and is under the protection of the French army. However, you must make a portage and cross the Lachine rapids.");
                Console.WriteLine("Fort 3 is Fort New York and is under Dutch control. You must cross through Iroquois land.");
                Console.WriteLine("");
                state.SelectedFort = GetSelectedFort();

                DisplaySelectedFortInformation(state.SelectedFort);

            } while (TradeAtAnotherFort());

            // process the travel to the fort
            ProcessExpeditionOutcome(state);

            // display results of expedition (savings change) to the user
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("You now have ");
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write($"{state.Savings.ToString("c")}");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine(" including your previous savings.");

            // update the turn count now that another turn has been played
            state.ExpeditionCount += 1;
        }

        /// <summary>
        /// Method to show the expedition costs to the player with some standard formatting
        /// </summary>
        /// <param name="fortname">The name of the fort traded with</param>
        /// <param name="supplies">The cost of the supplies at the fort</param>
        /// <param name="expenses">The travel expenses for the expedition</param>
        internal void DisplayCosts(string fortname, double supplies, double expenses)
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"Supplies at {fortname} cost".PadLeft(55));
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"{supplies.ToString("c").PadLeft(10)}");
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"Your travel expenses to {fortname} were".PadLeft(55));
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"{expenses.ToString("c").PadLeft(10)}");
            Console.ForegroundColor = ConsoleColor.White;
        }

        /// <summary>
        /// Process the results of the expedition
        /// </summary>
        /// <param name="state">the game state</param>
        private void ProcessExpeditionOutcome(GameState state)
        {
            var beaverPrice = RandomPriceGenerator(0.25d, 1.00d);
            var foxPrice =    RandomPriceGenerator(0.2d , 0.80d);
            var erminePrice = RandomPriceGenerator(0.15d, 0.95d);
            var minkPrice =   RandomPriceGenerator(0.2d , 0.70d);

            var fortname = String.Empty;
            var suppliesCost = 0.0d;
            var travelExpenses = 0.0d;

            // create a random value 1 to 10 for the different outcomes at each fort
            var p = ((int)(10 * Rnd.NextDouble())) + 1;
            Console.WriteLine("");

            switch (state.SelectedFort)
            {
                case 1:     // outcome for expedition to Fort Hochelaga
                    beaverPrice = RandomPriceGenerator(0.2d, 0.75d);
                    foxPrice =    RandomPriceGenerator(0.2d, 0.80d);
                    erminePrice = RandomPriceGenerator(0.2d, 0.65d);
                    minkPrice =   RandomPriceGenerator(0.2d, 0.70d);
                    fortname = "Fort Hochelaga";
                    suppliesCost = 150.0d;
                    travelExpenses = 10.0d;
                    break;
                case 2:     // outcome for expedition to Fort Stadacona
                    beaverPrice = RandomPriceGenerator(0.2d , 0.90d);
                    foxPrice =    RandomPriceGenerator(0.2d , 0.80d);
                    erminePrice = RandomPriceGenerator(0.15d, 0.80d);
                    minkPrice =   RandomPriceGenerator(0.3d , 0.85d);
                    fortname = "Fort Stadacona";
                    suppliesCost = 125.0d;
                    travelExpenses = 15.0d;
                    if (p <= 2)
                    {
                        state.BeaverPelts = 0;
                        Console.WriteLine("Your beaver were to heavy to carry across the portage.");
                        Console.WriteLine("You had to leave the pelts but found them stolen when you returned");
                    }
                    else if (p <= 6)
                    {
                        Console.WriteLine("You arrived safely at Fort Stadacona");
                    }
                    else if (p <= 8)
                    {
                        state.BeaverPelts = 0;
                        state.FoxPelts = 0;
                        state.ErminePelts = 0;
                        state.MinkPelts = 0;
                        Console.WriteLine("Your canoe upset in the Lachine Rapids.");
                        Console.WriteLine("Your lost all your furs");
                    }
                    else if (p <= 10)
                    {
                        state.FoxPelts = 0;
                        Console.WriteLine("Your fox pelts were not cured properly.");
                        Console.WriteLine("No one will buy them.");
                    }
                    else
                    {
                        throw new Exception($"Unexpected Outcome p = {p}");
                    }
                    break;
                case 3:     // outcome for expedition to Fort New York
                    beaverPrice = RandomPriceGenerator(0.2d , 1.00d);
                    foxPrice =    RandomPriceGenerator(0.25d, 1.10d);
                    erminePrice = RandomPriceGenerator(0.2d , 0.95d);
                    minkPrice =   RandomPriceGenerator(0.15d, 1.05d);
                    fortname = "Fort New York";
                    suppliesCost = 80.0d;
                    travelExpenses = 25.0d;
                    if (p <= 2)
                    {
                        state.BeaverPelts = 0;
                        state.FoxPelts = 0;
                        state.ErminePelts = 0;
                        state.MinkPelts = 0;
                        suppliesCost = 0.0d;
                        travelExpenses = 0.0d;
                        Console.WriteLine("You were attacked by a party of Iroquois.");
                        Console.WriteLine("All people in your trading group were killed.");
                        Console.WriteLine("This ends the game (press any key).");
                        Console.ReadKey(true);
                        state.GameOver = true;
                    }
                    else if (p <= 6)
                    {
                        Console.WriteLine("You were lucky. You arrived safely at Fort New York.");
                    }
                    else if (p <= 8)
                    {
                        state.BeaverPelts = 0;
                        state.FoxPelts = 0;
                        state.ErminePelts = 0;
                        state.MinkPelts = 0;
                        Console.WriteLine("You narrowly escaped an Iroquois raiding party.");
                        Console.WriteLine("However, you had to leave all your furs behind.");
                    }
                    else if (p <= 10)
                    {
                        beaverPrice = beaverPrice / 2;
                        minkPrice = minkPrice / 2;
                        Console.WriteLine("Your mink and beaver were damaged on your trip.");
                        Console.WriteLine("You receive only half the current price for these furs.");
                    }
                    else
                    {
                        throw new Exception($"Unexpected Outcome p = {p}");
                    }
                    break;
                default:
                    break;
            }

            var beaverSale = state.BeaverPelts * beaverPrice;
            var foxSale = state.FoxPelts * foxPrice;
            var ermineSale = state.ErminePelts * erminePrice;
            var minkSale = state.MinkPelts * minkPrice;
            var profit = beaverSale + foxSale + ermineSale + minkSale - suppliesCost - travelExpenses;
            state.Savings += profit;

            Console.WriteLine("");
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"Your {state.BeaverPelts.ToString().PadLeft(3, ' ')} beaver pelts sold @ {beaverPrice.ToString("c")} per pelt for a total");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"{beaverSale.ToString("c").PadLeft(10)}");
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"Your {state.FoxPelts.ToString().PadLeft(3, ' ')} fox    pelts sold @ {foxPrice.ToString("c")} per pelt for a total");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"{foxSale.ToString("c").PadLeft(10)}");
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"Your {state.ErminePelts.ToString().PadLeft(3, ' ')} ermine pelts sold @ {erminePrice.ToString("c")} per pelt for a total");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"{ermineSale.ToString("c").PadLeft(10)}");
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"Your {state.MinkPelts.ToString().PadLeft(3, ' ')} mink   pelts sold @ {minkPrice.ToString("c")} per pelt for a total");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"{minkSale.ToString("c").PadLeft(10)}");
            Console.WriteLine("");
            DisplayCosts(fortname, suppliesCost, travelExpenses);
            Console.WriteLine("");
            Console.Write($"Profit / Loss".PadLeft(55));
            Console.ForegroundColor = profit >= 0.0d ? ConsoleColor.Green : ConsoleColor.Red;
            Console.WriteLine($"{profit.ToString("c").PadLeft(10)}");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("");
        }

        private void DisplaySelectedFortInformation(int selectedFort)
        {
            Console.WriteLine("");
            Console.ForegroundColor = ConsoleColor.White;

            switch (selectedFort)
            {
                case 1:    // selected fort details for Fort Hochelaga
                    Console.WriteLine("You have chosen the easiest route.");
                    Console.WriteLine("However, the fort is far from any seaport.");
                    Console.WriteLine("The value you receive for your furs will be low.");
                    Console.WriteLine("The cost of supplies will be higher than at Forts Stadacona or New York");
                    break;
                case 2:    // selected fort details for Fort Stadacona
                    Console.WriteLine("You have chosen a hard route.");
                    Console.WriteLine("It is, in comparsion, harder than the route to Hochelaga but easier than the route to New York.");
                    Console.WriteLine("You will receive an average value for your furs.");
                    Console.WriteLine("The cost of your supplies will be average.");
                    break;
                case 3:    // selected fort details for Fort New York
                    Console.WriteLine("You have chosen the most difficult route.");
                    Console.WriteLine("At Fort New York you will receive the higest value for your furs.");
                    Console.WriteLine("The cost of your supplies will be lower than at all the other forts.");
                    break;
                default:
                    break;
            }
        }

        private bool TradeAtAnotherFort()
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("");
            Console.WriteLine("Do you want to trade at another fort?");
            Console.Write("Answer (Y)es or (N)o ");
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("> ");

            char pressedKey;
            // Keep looping until we get a recognised input
            do
            {
                // Read a key, don't display it on screen
                ConsoleKeyInfo key = Console.ReadKey(true);
                // Convert to upper-case so we don't need to care about capitalisation
                pressedKey = Char.ToUpper(key.KeyChar);
                // Is this a key we recognise? If not, keep looping
            } while (pressedKey != 'Y' && pressedKey != 'N');

            // Display the result on the screen
            Console.WriteLine(pressedKey);

            // Return true if the player pressed 'Y', false for anything else.
            return (pressedKey == 'Y');
        }

        /// <summary>
        /// Get an amount of pelts from the user
        /// </summary>
        /// <param name="currentMoney">The total pelts available</param>
        /// <returns>Returns the amount the player selects</returns>
        private int GetPelts(int furCount)
        {
            int peltCount;

            // loop until the user enters a valid value
            do
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("> ");
                string input = Console.ReadLine();

                // parse user information to check if it is a valid entry
                if (!int.TryParse(input, out peltCount))
                {
                    // invalid entry; message back to user
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Sorry, I didn't understand. Please enter the number of pelts.");

                    // continue looping
                    continue;
                }

                // check if plet amount is more than the available pelts
                if (peltCount > furCount)
                {
                    // too many pelts selected
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"You may not have that many furs. Do not try to cheat. I can add.");

                    // continue looping
                    continue;
                }

                // valid pelt amount entered
                break;
            } while (true);

            // return pelt count to the user
            return peltCount;
        }

        /// <summary>
        /// Prompt the user for their selected fort
        /// </summary>
        /// <returns>returns the fort the user has selected</returns>
        private int GetSelectedFort()
        {
            int selectedFort;

            // loop until the user enters a valid value
            do
            {
                Console.ForegroundColor = ConsoleColor.White;
                Console.Write("Answer 1, 2, or 3 ");
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("> ");
                string input = Console.ReadLine();

                // is the user entry valid
                if (!int.TryParse(input, out selectedFort))
                {
                    // no, invalid data
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Sorry, I didn't understand. Please answer 1, 2, or 3.");

                    // continue looping
                    continue;
                }

                // is the selected fort an option (one, two or three only)
                if (selectedFort != 1 && selectedFort != 2 && selectedFort != 3)
                {
                    // no, invalid for selected
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"Please answer 1, 2, or 3.");

                    // continue looping
                    continue;
                }

                // valid fort selected, stop looping
                break;
            } while (true);

            // return the players selected fort
            return selectedFort;
        }
    }
}

```

# `38_Fur_Trader/csharp/GameState.cs`



这段代码是一个名为GameState的内部类，用于保存游戏中的状态信息。这个类定义了游戏是否结束、游戏中未消费的毛皮数量、已消耗的毛皮数量、已捕获的狐狸数量、当前选派的 Fort 值、以及其他的游戏状态信息。

下面是这个类的详细解释：

- GameOver：一个布尔值，表示游戏是否已经结束。
- Savings：玩家当前游戏中未消费的毛皮数量。
- ExpeditionCount：玩家当前已经探索过的狐狸数量。
- UnasignedFurCount：玩家当前未消费的狐狸毛皮数量。
- Pelts：一个4维数组，用于保存每个狐狸的毛皮数量。
- MinkPelts：狐狸毛皮的第一件，用于标记狐狸是否已经被捕获。
- BeaverPelts：狐狸毛皮的第二个，用于标记狐狸是否已经被捕获。
- ErminePelts：狐狸毛皮的第三件，用于标记狐狸是否已经被捕获。
- FoxPelts：狐狸毛皮的第四件，用于标记狐狸是否已经被捕获。
- SelectedFort：玩家当前选派的狐狸数量。
- StartTurn：一个方法，用于在每次游戏开始时重置游戏状态。
- StartTurn()的参数包括游戏状态的一些变量，如已消费的毛皮数量、已探索过的狐狸数量等等，用于在每次游戏开始时将这些变量清零，使游戏重新开始。

这个类提供了一个方便的方式来管理游戏中的状态信息，以便在每次游戏开始时都能够从头开始，并且允许玩家在游戏中保留他们的游戏进度。


```
﻿using System;
using System.Collections.Generic;
using System.Text;

namespace FurTrader
{
    internal class GameState
    {
        internal bool GameOver { get; set; }

        internal double Savings { get; set; }

        internal int ExpeditionCount { get; set; }

        internal int UnasignedFurCount { get; set; }

        internal int[] Pelts { get; private set; }

        internal int MinkPelts { get { return this.Pelts[0]; } set { this.Pelts[0] = value; } }
        internal int BeaverPelts { get { return this.Pelts[1]; } set { this.Pelts[1] = value; } }
        internal int ErminePelts { get { return this.Pelts[2]; } set { this.Pelts[2] = value; } }
        internal int FoxPelts { get { return this.Pelts[3]; } set { this.Pelts[3] = value; } }

        internal int SelectedFort { get; set; }

        internal GameState()
        {
            this.Savings = 600;
            this.ExpeditionCount = 0;
            this.UnasignedFurCount = 190;
            this.Pelts = new int[4];
            this.SelectedFort = 0;
        }

        internal void StartTurn()
        {
            this.SelectedFort = 0;              // reset to a default value
            this.UnasignedFurCount = 190;       // each turn starts with 190 furs
            this.Pelts = new int[4];            // reset pelts on each turn
        }
    }
}

```

# `38_Fur_Trader/csharp/Program.cs`

这段代码是一个C#程序，定义了一个名为“Program”的类，以及一个名为“Main”的函数。函数内部创建了一个名为“game”的变量，并将其赋值为一个名为“Game”的类实例。接下来，game类的一个名为“GameLoop”的函数被调用，这个函数将启动一个无限循环，在每次循环中游戏将继续运行，直到玩家选择退出为止。


```
﻿using System;

namespace FurTrader
{
    public class Program
    {
        /// <summary>
        /// This function will be called automatically when the application begins
        /// </summary>
        /// <param name="args"></param>
        public static void Main(string[] args)
        {
            // Create an instance of our main Game class
            var game = new Game();

            // Call its GameLoop function. This will play the game endlessly in a loop until the player chooses to quit.
            game.GameLoop();
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `38_Fur_Trader/java/src/FurTrader.java`

This code appears to be a Java class that simulates a game of "BibleScrabble", where players must insert correct punctuation within a given number of spaces to score points.

The `displayTextAndGetNumber()` method prints a message to the screen and accepts a player input from the keyboard. It converts the input to an integer and returns the result.

The `displayTextAndGetInput()` method prints a message to the screen and accepts a player input from the keyboard. It converts the input to a string and returns it, along with the number of spaces required to insert the text.

The `simulateTabs()` method simulates the "Tab" (xx) command to indent text by a number of spaces.

The `yesEntered()` method checks whether a player entered "Y" or "YES" in response to a question.

The `stringIsAnyValue()` method checks whether a given string is equal to one of a variable number of values. It uses the `Arrays.stream()` method to iterate over the values and returns true if any comparison was found in one of the variable number of strings passed.


```
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Fur Trader
 * <p>
 * Based on the Basic game of Fur Trader here
 * https://github.com/coding-horror/basic-computer-games/blob/main/38%20Fur%20Trader/furtrader.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class FurTrader {

    public static final double START_SAVINGS_AMOUNT = 600.0;
    public static final int STARTING_FURS = 190;

    public static final int FORT_HOCHELAGA_MONTREAL = 1;
    public static final int FORT_STADACONA_QUEBEC = 2;
    public static final int FORT_NEW_YORK = 3;

    public static final String MINK = "MINK";
    public static final int MINK_ENTRY = 0;
    public static final String BEAVER = "BEAVER";
    public static final int BEAVER_ENTRY = 1;
    public static final String ERMINE = "ERMINE";
    public static final int ERMINE_ENTRY = 2;
    public static final String FOX = "FOX";
    public static final int FOX_ENTRY = 3;

    // Used for keyboard input
    private final Scanner kbScanner;

    private enum GAME_STATE {
        STARTUP,
        INIT,
        TRADE_AT_FORT,
        TRADE_SUMMARY,
        TRADE_AGAIN,
        GAME_OVER
    }

    // Current game state
    private GAME_STATE gameState;

    private double savings;
    private double minkPrice;
    private double beaverPrice;
    private double erminePrice;
    private double foxPrice;

    private ArrayList<Pelt> pelts;

    private boolean playedOnce;

    public FurTrader() {
        kbScanner = new Scanner(System.in);
        gameState = GAME_STATE.INIT;
        playedOnce = false;
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                case INIT:
                    savings = START_SAVINGS_AMOUNT;

                    // Only display initial game heading once
                    if (!playedOnce) {
                        playedOnce = true;
                        gameStartupMessage();
                    }

                    intro();
                    if (yesEntered(displayTextAndGetInput("DO YOU WISH TO TRADE FURS? "))) {
                        System.out.println("YOU HAVE $" + formatNumber(savings) + " SAVINGS.");
                        System.out.println("AND " + STARTING_FURS + " FURS TO BEGIN THE EXPEDITION.");

                        // Create a new array of Pelts.
                        pelts = initPelts();
                        gameState = GAME_STATE.STARTUP;
                    } else {
                        gameState = GAME_STATE.GAME_OVER;
                    }

                    break;

                case STARTUP:

                    // Reset count of pelts (all types)
                    resetPelts();

                    // This is where we will go to after processing all pelts.
                    gameState = GAME_STATE.TRADE_AT_FORT;

                    int totalPelts = 0;
                    // Cycle through all types of pelts
                    for (int i = 0; i < pelts.size(); i++) {
                        Pelt pelt = pelts.get(i);
                        int number = getPeltCount(pelt.getName());
                        totalPelts += number;
                        if (totalPelts > STARTING_FURS) {
                            System.out.println("YOU MAY NOT HAVE THAT MANY FURS.");
                            System.out.println("DO NOT TRY TO CHEAT.  I CAN ADD.");
                            System.out.println("YOU MUST START AGAIN.");
                            System.out.println();
                            // Restart the game
                            gameState = GAME_STATE.INIT;
                            break;
                        } else {
                            // update count entered by player and save back to ArrayList.
                            pelt.setPeltCount(number);
                            pelts.set(i, pelt);
                            // Its possible for the player to put all their pelt allocation
                            // into one or more different pelts.  They don't have to use all four types.
                            // If we have an exact count of pelts matching the MAX
                            // don't bother continuing to ask for more.
                            if (totalPelts == STARTING_FURS) {
                                break;
                            }
                        }
                    }

                    // Only move onto the trading part of the game if the player didn't add too many pelts
                    if (gameState != GAME_STATE.STARTUP) {
                        // Set ermine and beaver default prices here, depending on where you trade these
                        // defaults will either be used or overwritten with other values.
                        // check out the tradeAt??? methods for more info.
                        erminePrice = ((.15 * Math.random() + .95) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
                        beaverPrice = ((.25 * Math.random() + 1.00) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
                        System.out.println();
                    }
                    break;

                case TRADE_AT_FORT:

                    extendedTradingInfo();
                    int answer = displayTextAndGetNumber("ANSWER 1, 2, OR 3. ");

                    System.out.println();

                    // Now show the details of the fort they are about to trade
                    // and give the player the chance to NOT proceed.
                    // A "No" or false means they do not want to change to another fort
                    if (!confirmFort(answer)) {
                        switch (answer) {
                            case 1:
                                tradeAtFortHochelagaMontreal();
                                gameState = GAME_STATE.TRADE_SUMMARY;
                                break;

                            case 2:
                                tradeAtFortStadaconaQuebec();
                                gameState = GAME_STATE.TRADE_SUMMARY;
                                break;
                            case 3:
                                // Did the player and party all die?
                                if (!tradeAtFortNewYork()) {
                                    gameState = GAME_STATE.GAME_OVER;
                                } else {
                                    gameState = GAME_STATE.TRADE_SUMMARY;
                                }
                                break;
                        }

                        break;
                    }

                case TRADE_SUMMARY:

                    System.out.println();
                    double beaverTotal = beaverPrice * pelts.get(BEAVER_ENTRY).getNumber();
                    System.out.print("YOUR BEAVER SOLD FOR $ " + formatNumber(beaverTotal));

                    double foxTotal = foxPrice * pelts.get(FOX_ENTRY).getNumber();
                    System.out.println(simulateTabs(5) + "YOUR FOX SOLD FOR $ " + formatNumber(foxTotal));

                    double erMineTotal = erminePrice * pelts.get(ERMINE_ENTRY).getNumber();
                    System.out.print("YOUR ERMINE SOLD FOR $ " + formatNumber(erMineTotal));

                    double minkTotal = minkPrice * pelts.get(MINK_ENTRY).getNumber();
                    System.out.println(simulateTabs(5) + "YOUR MINK SOLD FOR $ " + formatNumber(minkTotal));

                    savings += beaverTotal + foxTotal + erMineTotal + minkTotal;
                    System.out.println();
                    System.out.println("YOU NOW HAVE $" + formatNumber(savings) + " INCLUDING YOUR PREVIOUS SAVINGS");

                    gameState = GAME_STATE.TRADE_AGAIN;
                    break;

                case TRADE_AGAIN:
                    if (yesEntered(displayTextAndGetInput("DO YOU WANT TO TRADE FURS NEXT YEAR? "))) {
                        gameState = GAME_STATE.STARTUP;
                    } else {
                        gameState = GAME_STATE.GAME_OVER;
                    }

            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    /**
     * Create all pelt types with a count of zero
     *
     * @return Arraylist of initialised Pelt objects.
     */
    private ArrayList<Pelt> initPelts() {

        ArrayList<Pelt> tempPelts = new ArrayList<>();
        tempPelts.add(new Pelt(MINK, 0));
        tempPelts.add(new Pelt(BEAVER, 0));
        tempPelts.add(new Pelt(ERMINE, 0));
        tempPelts.add(new Pelt(FOX, 0));
        return tempPelts;
    }

    /**
     * Display a message about trading at each fort and confirm if the player wants to trade
     * at ANOTHER fort
     *
     * @param fort the fort in question
     * @return true if YES was typed by player
     */
    private boolean confirmFort(int fort) {
        switch (fort) {
            case FORT_HOCHELAGA_MONTREAL:
                System.out.println("YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT");
                System.out.println("IS FAR FROM ANY SEAPORT.  THE VALUE");
                System.out.println("YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST");
                System.out.println("OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK.");
                break;
            case FORT_STADACONA_QUEBEC:
                System.out.println("YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION,");
                System.out.println("HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN");
                System.out.println("THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE");
                System.out.println("FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE.");
                break;
            case FORT_NEW_YORK:
                System.out.println("YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT");
                System.out.println("FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE");
                System.out.println("FOR YOUR FURS.  THE COST OF YOUR SUPPLIES");
                System.out.println("WILL BE LOWER THAN AT ALL THE OTHER FORTS.");
                break;
        }

        System.out.println("DO YOU WANT TO TRADE AT ANOTHER FORT?");
        return yesEntered(displayTextAndGetInput("ANSWER YES OR NO "));

    }

    /**
     * Trade at the safest fort - Fort Hochelaga
     * No chance of anything bad happening, so just calculate amount per pelt
     * and return
     */
    private void tradeAtFortHochelagaMontreal() {
        savings -= 160.0;
        System.out.println();
        System.out.println("SUPPLIES AT FORT HOCHELAGA COST $150.00.");
        System.out.println("YOUR TRAVEL EXPENSES TO HOCHELAGA WERE $10.00.");
        minkPrice = ((.2 * Math.random() + .7) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
        erminePrice = ((.2 * Math.random() + .65) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
        beaverPrice = ((.2 * Math.random() + .75) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
        foxPrice = ((.2 * Math.random() + .8) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
    }


    private void tradeAtFortStadaconaQuebec() {
        savings -= 140.0;
        System.out.println();
        minkPrice = ((.2 * Math.random() + .85) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
        erminePrice = ((.2 * Math.random() + .8) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
        beaverPrice = ((.2 * Math.random() + .9) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);

        // What happened during the trip to the fort?
        int tripResult = (int) (Math.random() * 10) + 1;
        if (tripResult <= 2) {
            // Find the Beaver pelt in our Arraylist
            Pelt beaverPelt = pelts.get(BEAVER_ENTRY);
            // Pelts got stolen, so update to a count of zero
            beaverPelt.lostPelts();
            // Update it back in the ArrayList
            pelts.set(BEAVER_ENTRY, beaverPelt);
            System.out.println("YOUR BEAVER WERE TOO HEAVY TO CARRY ACROSS");
            System.out.println("THE PORTAGE.  YOU HAD TO LEAVE THE PELTS, BUT FOUND");
            System.out.println("THEM STOLEN WHEN YOU RETURNED.");
        } else if (tripResult <= 6) {
            System.out.println("YOU ARRIVED SAFELY AT FORT STADACONA.");
        } else if (tripResult <= 8) {
            System.out.println("YOUR CANOE UPSET IN THE LACHINE RAPIDS.  YOU");
            System.out.println("LOST ALL YOUR FURS.");
            // Clear out all pelts.
            resetPelts();
        } else if (tripResult <= 10) {
            // Fox pelts not cured
            System.out.println("YOUR FOX PELTS WERE NOT CURED PROPERLY.");
            System.out.println("NO ONE WILL BUY THEM.");
            // Bug because Fox furs were not calculated above in original basic program
            // Find the Beaver pelt in our Arraylist
            Pelt foxPelt = pelts.get(FOX_ENTRY);
            // Pelts got stolen, so update to a count of zero
            foxPelt.lostPelts();
            // Update it back in the ArrayList
            pelts.set(FOX_ENTRY, foxPelt);
        }

        System.out.println("SUPPLIES AT FORT STADACONA COST $125.00.");
        System.out.println("YOUR TRAVEL EXPENSES TO STADACONA WERE $15.00.");
    }

    private boolean tradeAtFortNewYork() {

        boolean playerAlive = true;
        savings -= 105.0;
        System.out.println();
        minkPrice = ((.2 * Math.random() + 1.05) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
        foxPrice = ((.2 * Math.random() + 1.1) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);

        // What happened during the trip to the fort?
        int tripResult = (int) (Math.random() * 10) + 1;
        if (tripResult <= 2) {
            System.out.println("YOU WERE ATTACKED BY A PARTY OF IROQUOIS.");
            System.out.println("ALL PEOPLE IN YOUR TRADING GROUP WERE");
            System.out.println("KILLED.  THIS ENDS THE GAME.");
            playerAlive = false;
        } else if (tripResult <= 6) {
            System.out.println("YOU WERE LUCKY.  YOU ARRIVED SAFELY");
            System.out.println("AT FORT NEW YORK.");
        } else if (tripResult <= 8) {
            System.out.println("YOU NARROWLY ESCAPED AN IROQUOIS RAIDING PARTY.");
            System.out.println("HOWEVER, YOU HAD TO LEAVE ALL YOUR FURS BEHIND.");
            // Clear out all pelts.
            resetPelts();
        } else if (tripResult <= 10) {
            beaverPrice /= 2;
            minkPrice /= 2;
            System.out.println("YOUR MINK AND BEAVER WERE DAMAGED ON YOUR TRIP.");
            System.out.println("YOU RECEIVE ONLY HALF THE CURRENT PRICE FOR THESE FURS.");
        }

        if (playerAlive) {
            System.out.println("SUPPLIES AT NEW YORK COST $80.00.");
            System.out.println("YOUR TRAVEL EXPENSES TO NEW YORK WERE $25.00.");
        }

        return playerAlive;
    }

    /**
     * Reset pelt count for all Pelt types to zero.
     */
    private void resetPelts() {
        for (int i = 0; i < pelts.size(); i++) {
            Pelt pelt = pelts.get(i);
            pelt.lostPelts();
            pelts.set(i, pelt);
        }
    }

    /**
     * Return a pelt object containing the user entered number of pelts.
     *
     * @param peltName Name of pelt (Type)
     * @return number of pelts assigned by player
     */
    private int getPeltCount(String peltName) {
        return displayTextAndGetNumber("HOW MANY " + peltName + " PELTS DO YOU HAVE? ");
    }

    private void extendedTradingInfo() {
        System.out.println("YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2,");
        System.out.println("OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)");
        System.out.println("AND IS UNDER THE PROTECTION OF THE FRENCH ARMY.");
        System.out.println("FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE");
        System.out.println("PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST");
        System.out.println("MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS.");
        System.out.println("FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL.");
        System.out.println("YOU MUST CROSS THROUGH IROQUOIS LAND.");
        System.out.println();

    }

    private void gameStartupMessage() {
        System.out.println(simulateTabs(31) + "FUR TRADER");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
    }

    private void intro() {
        System.out.println("YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN ");
        System.out.println("1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET");
        System.out.println("SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE");
        System.out.println("FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES");
        System.out.println("AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND");
        System.out.println("ON THE FORT THAT YOU CHOOSE.");
        System.out.println();
    }

    /**
     * Format a double number to two decimal points for output.
     *
     * @param number double number
     * @return formatted number as a string
     */
    private String formatNumber(double number) {
        return String.format("%.2f", number);
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to an Integer
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
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

    /**
     * Checks whether player entered Y or YES to a question.
     *
     * @param text player string from kb
     * @return true of Y or YES was entered, otherwise false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * Check whether a string equals one of a variable number of values
     * Useful to check for Y or YES for example
     * Comparison is case insensitive.
     *
     * @param text   source string
     * @param values a range of values to compare against the source string
     * @return true if a comparison was found in one of the variable number of strings passed
     */
    private boolean stringIsAnyValue(String text, String... values) {

        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));
    }
}

```