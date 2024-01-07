# `basic-computer-games\38_Fur_Trader\c\furtrader.c`

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
#define STATE_TRADING       4

int main( void )

```