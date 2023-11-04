# BasicComputerGames源码解析 51

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `49_Hockey/python/hockey.py`

这段代码是一个冰球游戏的模拟，它模拟了一个原始冰球游戏的场景。作者是罗伯特·puopolo，修改者是史蒂夫·北。在2022年，马丁·托马斯将其翻译成Python。

具体来说，这段代码的功能是模拟一个冰球游戏场景。它使用`dataclasses`库来简化冰球数据的定义和场景元素的创建。它还使用`random`库来生成冰球比赛的随机数。

这段代码的主要目的是提供一个冰球游戏模拟，让玩家可以体验冰球比赛的乐趣。


```
"""
HOCKEY

A simulation of an ice hockey game.

The original author is Robert Puopolo;
modifications by Steve North of Creative Computing.

Ported to Python by Martin Thoma in 2022
"""

from dataclasses import dataclass, field
from random import randint
from typing import List, Tuple

```

这段代码定义了一个名为Team的类，用于表示一个足球队伍。它有以下属性和方法：

- name: 队伍名称，类型为字符串。
- players: 队伍成员名单，类型为列表，每个元素为字符串类型。
- shots_on_net: 队伍进球数，类型为整数。
- goals: 队伍进球总数，类型为列表，每个元素为整数类型。
- assists: 队伍助攻数，类型为列表，每个元素为整数类型。
- score: 队伍得分，类型为整数。

此外，它还有一个show_lineup方法，用于打印队伍的阵容。


```
NB_PLAYERS = 6


@dataclass
class Team:
    # TODO: It would be better to use a Player-class (name, goals, assits)
    #       and have the attributes directly at each player. This would avoid
    #       dealing with indices that much
    #
    #       I'm also rather certain that I messed up somewhere with the indices
    #       - instead of using those, one could use actual player positions:
    #       LEFT WING,    CENTER,        RIGHT WING
    #       LEFT DEFENSE, RIGHT DEFENSE, GOALKEEPER
    name: str
    players: List[str]  # 6 players
    shots_on_net: int = 0
    goals: List[int] = field(default_factory=lambda: [0 for _ in range(NB_PLAYERS)])
    assists: List[int] = field(default_factory=lambda: [0 for _ in range(NB_PLAYERS)])
    score: int = 0

    def show_lineup(self) -> None:
        print(" " * 10 + f"{self.name} STARTING LINEUP")
        for player in self.players:
            print(player)


```



This code defines two functions and an optimist approach to ask for user input, which is then processed in-place without being saved to disk or returning error messages. 

The first function `ask_binary` takes a `prompt` string and an `error_msg` string as input. It displays a message to the user and repeatedly prompts the user to provide a binary answer (`y` or `yes` or `no`). If the user enters `y` or `yes`, the function returns `True`. Otherwise, it returns `False`. If the user enters `n` or `no`, the function prints the `error_msg`.

The second function `get_team_names` takes a `while` loop and displays a message to the user asking them to provide the names of two teams. It then waits for the user to enter the names separated by a comma. If the user enters the names, the function returns them as a tuple. If the user does not provide any input or enters a non-single name, the function prints an error message.

The `optimist_ approach` is a simple implementation of the ask_binary function that converts the user's input to lowercase and returns `True` if the user enters `y` or `yes` and `False` otherwise.


```
def ask_binary(prompt: str, error_msg: str) -> bool:
    while True:
        answer = input(prompt).lower()
        if answer in ["y", "yes"]:
            return True
        if answer in ["n", "no"]:
            return False
        print(error_msg)


def get_team_names() -> Tuple[str, str]:
    while True:
        answer = input("ENTER THE TWO TEAMS: ")
        if answer.count(",") == 1:
            return answer.split(",")  # type: ignore
        print("separated by a single comma")


```

这两函数是在尝试从用户那里获取输入并返回一个整数。函数get_pass()用于获取用户输入的 pass 数量，并判断输入是否是一个数字 0 到 3。如果输入是数字 0 到 3，函数将返回传入的值。如果输入不是数字，函数将提示用户重新输入。

函数get_minutes_per_game()用于获取用户输入的游戏中的每分钟时间。函数将尝试从用户那里获取一个数字，用于判断用户是否输入了有效的分钟数。如果输入的数字不是数字，函数将提示用户重新输入。


```
def get_pass() -> int:
    while True:
        answer = input("PASS? ")
        try:
            passes = int(answer)
            if passes >= 0 and passes <= 3:
                return passes
        except ValueError:
            print("ENTER A NUMBER BETWEEN 0 AND 3")


def get_minutes_per_game() -> int:
    while True:
        answer = input("ENTER THE NUMBER OF MINUTES IN A GAME ")
        try:
            minutes = int(answer)
            if minutes >= 1:
                return minutes
        except ValueError:
            print("ENTER A NUMBER")


```

该代码定义了两个函数，函数1 `make_shot` 和函数2 `get_player_names`。

函数1 `make_shot` 接收一个控制队（设为 1 的整数）和两个队伍的成员列表，以及一个整数（设为 7 的整数）和两个整数（设为 0 和 1 的整数）。该函数用于模拟玩家在冰球比赛中的射门，根据玩家点击数字 1、2、3 或 4 进行模拟。

函数2 `get_player_names` 接收一个提示字符串（设为 `"PLAYER 1: "` 和 `"PLAYER 2: "` 和 `"PLAYER 3: "` 和 `"PLAYER 4: "`）。该函数用于获取球员名单，并返回一个列表。

`get_player_names` 函数的作用是接收玩家输入他们的名字，然后输出给调用者。


```
def get_player_names(prompt: str) -> List[str]:
    players = []
    print(prompt)
    for i in range(1, 7):
        player = input(f"PLAYER {i}: ")
        players.append(player)
    return players


def make_shot(
    controlling_team: int, team_a: Team, team_b: Team, player_index: List[int], j: int
) -> Tuple[int, int, int, int]:
    while True:
        try:
            s = int(input("SHOT? "))
        except ValueError:
            continue
        if s >= 1 and s <= 4:
            break
    if controlling_team == 1:
        print(team_a.players[player_index[j - 1]])
    else:
        print(team_b.players[player_index[j - 1]])
    g = player_index[j - 1]
    g1 = 0
    g2 = 0
    if s == 1:
        print(" LET'S A BOOMER GO FROM THE RED LINE!!\n")  # line 400
        z = 10
    elif s == 2:
        print(" FLIPS A WRISTSHOT DOWN THE ICE\n")  # line 420
        # Probable missing line 430 in original
    elif s == 3:
        print(" BACKHANDS ONE IN ON THE GOALTENDER\n")
        z = 25
    elif s == 4:
        print(" SNAPS A LONG FLIP SHOT\n")
        # line 460
        z = 17
    return z, g, g1, g2


```

print_header() -> None:
print(" " * 33 + "HOCKEY")
print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n")

instructions() -> None:
wants_it = ask_binary("WOULD YOU LIKE THE INSTRUCTIONS? ", "ANSWER YES OR NO!!")
if wants_it:
   print()
   print("THIS IS A SIMULATED HOCKEY GAME.")
   print("QUESTION     RESPONSE")
   print("PASS         TYPE IN THE NUMBER OF PASSES YOU WOULD")
   print("             LIKE TO MAKE, FROM 0 TO 3.")
   print("SHOT         TYPE THE NUMBER CORRESPONDING TO THE SHOT")
   print("            You want to make.  Enter:")
   print("             1 FOR A SLAPSHOT")
   print("             2 FOR A WRISTSHOT")
   print("             3 FOR A BACKHAND")
   print("             4 FOR A SNAP SHOT")
   print("AREA         TYPE IN THE NUMBER CORRESPONDING TO")
   print("             THE AREA YOU ARE AIMING AT.  ENTER:")
   print("             1 FOR UPPER LEFT HAND CORNER")
   print("             2 FOR UPPER RIGHT HAND CORNER")
   print("             3 FOR LOWER LEFT HAND CORNER")
   print("             4 FOR LOWER RIGHT HAND CORNER")
   print("AT THE START OF THE GAME, YOU WILL BE ASKED FOR THE NAMES")
   print("OF YOUR PLAYERS.  THEY ARE ENTERED IN THE ORDER: ")
   print("LEFT WING, CENTER, RIGHT WING, LEFT DEFENSE,")
   print("RIGHT DEFENSE, GOALKEEPER.  ANY OTHER INPUT REQUIRED WILL")
   print("HAVE EXPLAINATORY INSTRUCTIONS.")


```
def print_header() -> None:
    print(" " * 33 + "HOCKEY")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


def instructions() -> None:
    wants_it = ask_binary("WOULD YOU LIKE THE INSTRUCTIONS? ", "ANSWER YES OR NO!!")
    if wants_it:
        print()
        print("THIS IS A SIMULATED HOCKEY GAME.")
        print("QUESTION     RESPONSE")
        print("PASS         TYPE IN THE NUMBER OF PASSES YOU WOULD")
        print("             LIKE TO MAKE, FROM 0 TO 3.")
        print("SHOT         TYPE THE NUMBER CORRESPONDING TO THE SHOT")
        print("             YOU WANT TO MAKE.  ENTER:")
        print("             1 FOR A SLAPSHOT")
        print("             2 FOR A WRISTSHOT")
        print("             3 FOR A BACKHAND")
        print("             4 FOR A SNAP SHOT")
        print("AREA         TYPE IN THE NUMBER CORRESPONDING TO")
        print("             THE AREA YOU ARE AIMING AT.  ENTER:")
        print("             1 FOR UPPER LEFT HAND CORNER")
        print("             2 FOR UPPER RIGHT HAND CORNER")
        print("             3 FOR LOWER LEFT HAND CORNER")
        print("             4 FOR LOWER RIGHT HAND CORNER")
        print("AT THE START OF THE GAME, YOU WILL BE ASKED FOR THE NAMES")
        print("OF YOUR PLAYERS.  THEY ARE ENTERED IN THE ORDER: ")
        print("LEFT WING, CENTER, RIGHT WING, LEFT DEFENSE,")
        print("RIGHT DEFENSE, GOALKEEPER.  ANY OTHER INPUT REQUIRED WILL")
        print("HAVE EXPLANATORY INSTRUCTIONS.")


```

It looks like you are trying to analyze a soccer game and determine the pass value (1 for a pass, 2 for a long pass, or 3 for a fake pass) and the passing situation. The pass value is determined by the number of players the ball is passed to, and the passing situation is determined by the position of the players on the field. The passing situation is classified as a 1 if the pass is from a player to a player, a 2 if the pass is from a player to the goal, or a 3 if the pass is from a player to an opponent. The number of goals scored is also determined by the position of the players on the field. If the ball is passed to a player in the goal area, the score is one goal for the team with the ball and two goals for the opponent. If the ball is passed to a player outside the goal area, the score is zero goals for either team.



```
def team1_action(
    pass_value: int, player_index: List[int], team_a: Team, team_b: Team, j: int
) -> Tuple[int, int, int, int]:
    if pass_value == 1:
        print(
            team_a.players[player_index[j - 2]]
            + " LEADS "
            + team_a.players[player_index[j - 1]]
            + " WITH A PERFECT PASS.\n"
        )
        print(team_a.players[player_index[j - 1]] + " CUTTING IN!!!\n")
        scoring_player = player_index[j - 1]
        goal_assistant1 = player_index[j - 2]
        goal_assistant2 = 0
        z1 = 3
    elif pass_value == 2:
        print(
            team_a.players[player_index[j - 2]]
            + " GIVES TO A STREAKING "
            + team_a.players[player_index[j - 1]]
        )
        print(
            team_a.players[player_index[j - 3]]
            + " COMES DOWN ON "
            + team_b.players[4]
            + " AND "
            + team_b.players[3]
        )
        scoring_player = player_index[j - 3]
        goal_assistant1 = player_index[j - 1]
        goal_assistant2 = player_index[j - 2]
        z1 = 2
    elif pass_value == 3:
        print("OH MY GOD!! A ' 4 ON 2 ' SITUATION\n")
        print(
            team_a.players[player_index[j - 3]]
            + " LEADS "
            + team_a.players[player_index[j - 2]]
            + "\n"
        )
        print(team_a.players[player_index[j - 2]] + " IS WHEELING THROUGH CENTER.\n")
        print(
            team_a.players[player_index[j - 2]]
            + " GIVES AND GOEST WITH "
            + team_a.players[player_index[j - 1]]
        )
        print("PRETTY PASSING!\n")
        print(
            team_a.players[player_index[j - 1]]
            + " DROPS IT TO "
            + team_a.players[player_index[j - 4]]
        )
        scoring_player = player_index[j - 4]
        goal_assistant1 = player_index[j - 1]
        goal_assistant2 = player_index[j - 2]
        z1 = 1
    return scoring_player, goal_assistant1, goal_assistant2, z1


```

It looks like you have defined a function called `pass_value`, which is a dictionary that maps the different pass values to their corresponding messages.

The function takes four arguments: `team_a`, `pass_value`, `player_index`, and `j`. It returns a tuple containing the `scoring_player`, `goal_assistant1`, `goal_assistant2`, and `z1` values, which are the maximum pass value and the maximum scoring opportunities for each team, respectively.

The function first checks the pass value and then looks up the corresponding message in the `pass_value` dictionary. If the pass value is 0, the function prints a message for a 0-0 draw. If the pass value is 1 or 2, the function prints messages for a scoring opportunity or a pass-and-score opportunity, respectively. If the pass value is 3, the function prints a message for a 3-2 or 3-3 game.

The function then returns the maximum pass value, the maximum scoring opportunities for each team, and the maximum scoring opportunities for each team.


```
def team2_action(
    pass_value: int, player_index: List[int], team_a: Team, team_b: Team, j: int
) -> Tuple[int, int, int, int]:
    if pass_value == 1:
        print(
            team_b.players[player_index[j - 1]]
            + " HITS "
            + team_b.players[player_index[j - 2]]
            + " FLYING DOWN THE LEFT SIDE\n"
        )
        scoring_player = player_index[j - 2]
        goal_assistant1 = player_index[j - 1]
        goal_assistant2 = 0
        z1 = 3
    elif pass_value == 2:
        print("IT'S A ' 3 ON 2 '!\n")
        print(
            "ONLY " + team_a.players[3] + " AND " + team_a.players[4] + " ARE BACK.\n"
        )
        print(
            team_b.players[player_index[j - 2]]
            + " GIVES OFF TO "
            + team_b.players[player_index[j - 1]]
        )
        print(
            team_b.players[player_index[j - 1]]
            + " DROPS TO "
            + team_b.players[player_index[j - 3]]
        )
        scoring_player = player_index[j - 3]
        goal_assistant1 = player_index[j - 1]
        goal_assistant2 = player_index[j - 2]
        z1 = 2
    elif pass_value == 3:
        print(" A '3 ON 2 ' WITH A ' TRAILER '!\n")
        print(
            team_b.players[player_index[j - 4]]
            + " GIVES TO "
            + team_b.players[player_index[j - 2]]
            + " WHO SHUFFLES IT OFF TO\n"
        )
        print(
            team_b.players[player_index[j - 1]] + " WHO FIRES A WING TO WING PASS TO \n"
        )
        print(team_b.players[player_index[j - 3]] + " AS HE CUTS IN ALONE!!\n")
        scoring_player = player_index[j - 3]
        goal_assistant1 = player_index[j - 1]
        goal_assistant2 = player_index[j - 2]
        z1 = 1
    return scoring_player, goal_assistant1, goal_assistant2, z1


```

这是一个Python函数，名为`final_message`，它接收两个团队和一名球员的索引作为参数。函数在输出的最后两行中显示了比赛得分和得分者信息。但是，在输出之前，函数先用`print("THAT'S THE SIREN")`输出一声警告。然后用`print("FINAL SCORE")`输出比赛得分，如果另一个团队的得分低于另一个团队，则输出比赛结果。否则，则输出另一个团队的得分。接着输出一个总结，显示了比赛的得分者和得分。


```
def final_message(team_a: Team, team_b: Team, player_index: List[int]) -> None:
    # Bells chime
    print("THAT'S THE SIREN\n")
    print("\n")
    print(" " * 15 + "FINAL SCORE:\n")
    if team_b.score <= team_a.score:
        print(f"{team_a.name}: {team_a.score}\t{team_b.name}: {team_b.score}\n")
    else:
        print(f"{team_b.name}: {team_b.score}\t{team_a.name}\t:{team_a.score}\n")
    print("\n")
    print(" " * 10 + "SCORING SUMMARY\n")
    print("\n")
    print(" " * 25 + team_a.name + "\n")
    print("\tNAME\tGOALS\tASSISTS\n")
    print("\t----\t-----\t-------\n")
    for i in range(1, 6):
        print(f"\t{team_a.players[i]}\t{team_a.goals[i]}\t{team_a.assists[i]}\n")
    print("\n")
    print(" " * 25 + team_b.name + "\n")
    print("\tNAME\tGOALS\tASSISTS\n")
    print("\t----\t-----\t-------\n")
    for t in range(1, 6):
        print(f"\t{team_b.players[t]}\t{team_b.goals[t]}\t{team_b.assists[t]}\n")
    print("\n")
    print("SHOTS ON NET\n")
    print(f"{team_a.name}: {team_a.shots_on_net}\n")
    print(team_b.name + f": {team_b.shots_on_net}\n")


```

这是一个Python的函数式程序，名为`main`。函数的主要作用是模拟一个足球比赛场景，让两个团队进行比赛，并在比赛过程中计算剩余时间。

具体来说，这个程序首先会引入一些标准库，包括`print_header`、`print_lineup`、`print_score`、`print_time`等，然后定义了一系列函数，包括`print_header`用于在比赛开始时输出比赛双方的阵容，`print_lineup`用于在比赛进行中输出各队球员的阵容，以及`print_score`、`print_time`等用于计算比赛得分和剩余时间等。

接下来，程序会调用`get_team_names`、`get_minutes_per_game`和`get_player_names`等函数，从这些函数中获取比赛的相关信息，包括比赛双方的名称、每队球员数量以及球员姓名等。

在获取到所有信息之后，程序会创建两个`Team`类对象`team_a`和`team_b`，分别代表比赛中的两个团队。接着，程序会要求用户输入比赛主持人，并输出比赛双方的阵容。随后，程序会进入一个循环，模拟每一轮比赛，计算剩余时间，并在循环结束时输出比赛结果。

总之，这个程序的主要作用是模拟一个丰富多彩的足球比赛场景，并在比赛过程中让用户体验到比赛的紧张和刺激。


```
def main() -> None:
    # Intro
    print_header()
    player_index: List[int] = [0 for _ in range(21)]
    print("\n" * 3)
    instructions()

    # Gather input
    team_name_a, team_name_b = get_team_names()
    print()
    minutes_per_game = get_minutes_per_game()
    print()
    players_a = get_player_names(f"WOULD THE {team_name_a} COACH ENTER HIS TEAM")
    print()
    players_b = get_player_names(f"WOULD THE {team_name_b} COACH DO THE SAME")
    team_a = Team(team_name_a, players_a)
    team_b = Team(team_name_b, players_b)
    print()
    referee = input("INPUT THE REFEREE FOR THIS GAME: ")
    print()
    team_a.show_lineup()
    print()
    team_b.show_lineup()
    print("WE'RE READY FOR TONIGHTS OPENING FACE-OFF.")
    print(
        f"{referee} WILL DROP THE PUCK BETWEEN "
        f"{team_a.players[0]} AND {team_b.players[0]}"
    )
    remaining_time = minutes_per_game

    # Play the game
    while remaining_time > 0:
        cont, remaining_time = simulate_game_round(
            team_a, team_b, player_index, remaining_time
        )
        remaining_time -= 1
        if cont == "break":
            break

    # Outro
    final_message(team_a, team_b, player_index)


```

This is a Python program that simulates a soccer game. It has two teams, `team_a` and `team_b`, each with a list of players. The program also has a bell system that rewards players for executing a pass within certain areas of the field.

The program starts by printing out the players of the teams, followed by a "GOAL" message. If the players of the teams are in the same team, the program will print out the score of that team. Otherwise, it will print out the score of the other team.

The program then enters a loop where the teams take turns executing a player's pass. The program will calculate the impact of the pass on the ball, the player, and the team. If the player executes a pass to an assist, the program will print out the assists.

The program also has a bell system that rewards players based on where they pass the ball. The program will print out a message for each assist, and it will keep track of the number of assists for each player.

The program has a variable called `goal_assistant_count` that keeps track of how many passes the player has assisted for each goal. When a goal is scored, the program will print out a message indicating the goal scorers, and it will update the score for the team and the goal scorers.

The program runs for a certain number of iterations and then exits.

Overall, this program simulates a realistic soccer game with two teams, and it takes into account various aspects of the game, such as player movements, player assists, and the bell system.


```
def handle_hit(
    controlling_team: int,
    team_a: Team,
    team_b: Team,
    player_index: List[int],
    goal_player: int,
    goal_assistant1: int,
    goal_assistant2: int,
    hit_area: int,
    z: int,
) -> int:
    while True:
        player_index[20] = randint(1, 100)
        if player_index[20] % z != 0:
            break
        a2 = randint(1, 100)
        if a2 % 4 == 0:
            if controlling_team == 1:
                print(f"SAVE {team_b.players[5]} --  REBOUND\n")
            else:
                print(f"SAVE {team_a.players[5]} --  FOLLOW up\n")
            continue
        else:
            hit_area += 1
    if player_index[20] % z != 0:
        if controlling_team == 1:
            print(f"GOAL {team_a.name}\n")
            team_a.score += 1
        else:
            print(f"SCORE {team_b.name}\n")
            team_b.score += 1
        # Bells in origninal
        print("\n")
        print("SCORE: ")
        if team_b.score <= team_a.score:
            print(f"{team_a.name}: {team_a.score}\t{team_b.name}: {team_b.score}\n")
        else:
            print(f"{team_b.name}: {team_b.score}\t{team_a.name}: {team_a.score}\n")
        if controlling_team == 1:
            team = team_a
        else:
            team = team_b
        print("GOAL SCORED BY: " + team.players[goal_player] + "\n")
        if goal_assistant1 != 0:
            if goal_assistant2 != 0:
                print(
                    f" ASSISTED BY: {team.players[goal_assistant1]}"
                    f" AND {team.players[goal_assistant2]}"
                )
            else:
                print(f" ASSISTED BY: {team.players[goal_assistant1]}")
            team.assists[goal_assistant1] += 1
            team.assists[goal_assistant2] += 1
        else:
            print(" UNASSISTED.\n")
        team.goals[goal_player] += 1

    return hit_area


```

If the specified `remaining_time` is negative or zero, the game will continue until the end of the period or the end of the season.

If the specified `remaining_time` is non-zero, the game will continue until the end of the period or the end of the season, regardless of whether the team making the shot has possession or not.

If the specified `remaining_time` is negative, it will be treated as a power-outage and the game will end with the score at the end of the period.

If the specified `remaining_time` is non-zero and the shot missed, it will be treated as a missed shot and the game will continue with the next shot.

If the specified `remaining_time` is negative and the shot missed, it will be treated as a missed shot and the game will end with the score at the end of the period.

If the specified `remaining_time` is non-zero and the shot hit the net, it will be treated as a power-play goal and the goal will count as one for the opponent.

If the specified `remaining_time` is negative or zero and the team that missed the shot had possession, it will be treated as a missed shot and the game will continue with the next shot.

If the specified `remaining_time` is negative or zero and the team that missed the shot did not have possession, it will be treated as a missed shot and the game will end with the score at the end of the period.

If the specified `remaining_time` is negative and the team making the shot had possession, it will be treated as a missed shot and the game will continue with the next shot.

If the specified `remaining_time` is non-zero and the player in the net with the ball missed the shot, it will be treated as a missed shot and the game will end with the score at the end of the period.

If the specified `remaining_time` is negative or zero and the player in the net with the ball hit the net, it will be treated as a power-play goal and the goal will count as one for the opponent.

If the specified `remaining_time` is non-zero and the player in the net with the ball missed the shot, it will be treated as a missed shot and the game will end with the score at the end of the period.

If the specified `remaining_time` is negative or zero and the player in the net with the ball had possession, it will be treated as a missed shot and the game will continue with the next shot.

If the specified `remaining_time` is non-zero and the player in the net with the ball had possession and missed the shot, it will be treated as a missed shot and the game will end with the score at the end of the period.

If the specified `remaining_time` is negative or zero and the player in the net with the ball had possession and hit the net, it will be treated as a power-play goal and the goal will count as one for the opponent.

If the specified `remaining_time` is non-zero and the player in the net with the ball had possession and missed the shot, it will be treated as a missed shot and the game will end with the score at the end of the period.


```
def handle_miss(
    controlling_team: int,
    team_a: Team,
    team_b: Team,
    remaining_time: int,
    goal_player: int,
) -> Tuple[str, int]:
    saving_player = randint(1, 7)
    if controlling_team == 1:
        if saving_player == 1:
            print("KICK SAVE AND A BEAUTY BY " + team_b.players[5] + "\n")
            print("CLEARED OUT BY " + team_b.players[3] + "\n")
            remaining_time -= 1
            return ("continue", remaining_time)
        if saving_player == 2:
            print("WHAT A SPECTACULAR GLOVE SAVE BY " + team_b.players[5] + "\n")
            print("AND " + team_b.players[5] + " GOLFS IT INTO THE CROWD\n")
            return ("break", remaining_time)
        if saving_player == 3:
            print("SKATE SAVE ON A LOW STEAMER BY " + team_b.players[5] + "\n")
            remaining_time -= 1
            return ("continue", remaining_time)
        if saving_player == 4:
            print(f"PAD SAVE BY {team_b.players[5]} OFF THE STICK\n")
            print(
                f"OF {team_a.players[goal_player]} AND "
                f"{team_b.players[5]} COVERS UP\n"
            )
            return ("break", remaining_time)
        if saving_player == 5:
            print(f"WHISTLES ONE OVER THE HEAD OF {team_b.players[5]}\n")
            remaining_time -= 1
            return ("continue", remaining_time)
        if saving_player == 6:
            print(f"{team_b.players[5]} MAKES A FACE SAVE!! AND HE IS HURT\n")
            print(f"THE DEFENSEMAN {team_b.players[5]} COVERS UP FOR HIM\n")
            return ("break", remaining_time)
    else:
        if saving_player == 1:
            print(f"STICK SAVE BY {team_a.players[5]}\n")
            print(f"AND CLEARED OUT BY {team_a.players[3]}\n")
            remaining_time -= 1
            return ("continue", remaining_time)
        if saving_player == 2:
            print(
                "OH MY GOD!! "
                f"{team_b.players[goal_player]} RATTLES ONE OFF THE POST\n"
            )
            print(
                f"TO THE RIGHT OF {team_a.players[5]} AND "
                f"{team_a.players[5]} COVERS "
            )
            print("ON THE LOOSE PUCK!\n")
            return ("break", remaining_time)
        if saving_player == 3:
            print("SKATE SAVE BY " + team_a.players[5] + "\n")
            print(team_a.players[5] + " WHACKS THE LOOSE PUCK INTO THE STANDS\n")
            return ("break", remaining_time)
        if saving_player == 4:
            print(
                "STICK SAVE BY " + team_a.players[5] + " AND HE CLEARS IT OUT HIMSELF\n"
            )
            remaining_time -= 1
            return ("continue", remaining_time)
        if saving_player == 5:
            print("KICKED OUT BY " + team_a.players[5] + "\n")
            print("AND IT REBOUNDS ALL THE WAY TO CENTER ICE\n")
            remaining_time -= 1
            return ("continue", remaining_time)
        if saving_player == 6:
            print("GLOVE SAVE " + team_a.players[5] + " AND HE HANGS ON\n")
            return ("break", remaining_time)
    return ("continue", remaining_time)


```

It looks like the game is a sport-based game where two teams compete to score goals. Each team has a goal player, and an attacker and a defender. The attacker tries to score a goal by taking a shot at the goal, while the defender tries to prevent the attacker from scoring. The team with the goal wins the game.

The game starts with a series of user inputs asking the player to control the team that will attack or defend. After the user has controlled the team, the game displays a graphical interface showing the game. The game ends when one team scores a goal or when the game is ended due to a tied score.

Each shot attempt by the attacker is represented by a number (1-4) and a string "SHOT?". If the user tries to make a shot when the game is not over, the game will end with a shot out of bounds.

The game also includes some rules for scoring. If the ball passes completely through the net, the attacking team scores a goal. If the ball hits the net but does not pass completely through, the attacking team will take a shot. If the ball hits the net and passes completely through the net, the defending team scores a goal. If the ball hits the net, but does not pass completely through, the defending team will take a shot. If the ball hits the goal posts, a goal is scored if the ball passes through the net without being handled by the goalkeeper.


```
def simulate_game_round(
    team_a: Team, team_b: Team, player_index: List[int], remaining_time: int
) -> Tuple[str, int]:
    controlling_team = randint(1, 2)
    if controlling_team == 1:
        print(f"{team_a.name} HAS CONTROL OF THE PUCK.")
    else:
        print(f"{team_b.name} HAS CONTROL.")
    pass_value = get_pass()
    for i in range(1, 4):
        player_index[i] = 0

    # Line 310:
    while True:
        j = 0
        for j in range(1, (pass_value + 2) + 1):
            player_index[j] = randint(1, 5)
        if player_index[j - 1] == player_index[j - 2] or (
            pass_value + 2 >= 3
            and (
                player_index[j - 1] == player_index[j - 3]
                or player_index[j - 2] == player_index[j - 3]
            )
        ):
            break
    if pass_value == 0:  # line 350
        z, goal_player, goal_assistant1, goal_assistant2 = make_shot(
            controlling_team, team_a, team_b, player_index, j
        )
    else:
        if controlling_team == 1:
            goal_player, goal_assistant1, goal_assistant2, z1 = team1_action(
                pass_value, player_index, team_a, team_b, j
            )
        else:
            goal_player, goal_assistant1, goal_assistant2, z1 = team2_action(
                pass_value, player_index, team_a, team_b, j
            )
        while True:
            shot_type = int(input("SHOT? "))
            if not (shot_type < 1 or shot_type > 4):
                break
        if controlling_team == 1:
            print(team_a.players[goal_player], end="")
        else:
            print(team_b.players[goal_player], end="")
        if shot_type == 1:
            print(" LET'S A BIG SLAP SHOT GO!!\n")
            z = 4
            z += z1
        if shot_type == 2:
            print(" RIPS A WRIST SHOT OFF\n")
            z = 2
            z += z1
        if shot_type == 3:
            print(" GETS A BACKHAND OFF\n")
            z = 3
            z += z1
        if shot_type == 4:
            print(" SNAPS OFF A SNAP SHOT\n")
            z = 2
            z += z1
    while True:
        goal_area = int(input("AREA? "))
        if not (goal_area < 1 or goal_area > 4):
            break
    if controlling_team == 1:
        team_a.shots_on_net += 1
    else:
        team_b.shots_on_net += 1
    hit_area = randint(1, 5)
    if goal_area == hit_area:
        hit_area = handle_hit(
            controlling_team,
            team_a,
            team_b,
            player_index,
            goal_player,
            goal_assistant1,
            goal_assistant2,
            hit_area,
            z,
        )
    if goal_area != hit_area:
        return handle_miss(
            controlling_team, team_a, team_b, remaining_time, goal_player
        )
    print("AND WE'RE READY FOR THE FACE-OFF\n")
    return ("continue", remaining_time)


```

这段代码是一个if语句，它的作用是判断当前脚本是否被意为 __main__ 所调用。如果当前脚本被意为 __main__ 所调用，那么程序将跳转到 __main__ 函数中执行。

换句话说，这段代码会检查当前脚本是否被用户在命令行中使用“/path/to/script.py”这样的路径运行。如果是，那么程序将跳转到 __main__ 函数中执行，否则不会执行任何操作。

在 Python 中，__name__ 属性是一个特殊属性，它的值为当前脚本的名称，如果没有赋值，它的值为 "script.py"。如果当前脚本被名为 __main__ 的话，那么 __name__ 的值为 "__main__"，程序将跳转到 __main__ 函数中执行。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


## Porting Notes

Variables:

* C: Do you want instructions?
* A$(7): Team name + player names (6 players)
* B$(7): Team name + player names (6 players)
* T6: Minutes per game
* R: REFEREE

Functions:

* REM: A line comment
* `INT(2*RND(X))+1`: X is constantly 1. That means that this expression is simpler expressed as `randint(1,2)`

---

Looking at the JS implementation:

* as[7] / bs[7]: The team name
* ha[8] : Score of team B
* ha[9] : Score of team A


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Horserace

This program simulates a one-mile horse race for three-year old throughbreds. Up to ten people may place bets on the race up to $10,000 each. However, you may only bet to win. You place your bet by inputting the number of the horse, a comma, and the amount of your bet. The computer then shows the position of the horses at seven points around the track and at the finish. Payoffs and winnings are shown at the end.

The program was written by Laurie Chevalier while a student at South Portland High School.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=92)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=107)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `50_Horserace/javascript/horserace.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是将一个字符串 `str` 打印到页面上，并将其包含在一个 `div` 元素中，该元素有一个名为 `output` 的子元素。具体实现是通过在文档中创建一个文本节点，然后将其添加到 `div` 元素中，并将 `str` 作为文本内容。

`input` 函数的作用是从用户那里获取一个字符串，并在输入框中显示该字符串。它通过创建一个带有 `type="text"` 属性的 `INPUT` 元素来获取用户的输入，并在页面上显示它。它还绑定了 `keydown` 事件，以便在用户按下键盘上的 13 键时，将获取的输入字符串打印出来，并将其包含在 `div` 元素中，通过调用 `print` 函数将其打印到页面上，并打印出来。


```
// HORSERACE
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

这段代码定义了一个名为 "tab" 的函数，用于将一个字符串中的空格替换成指定的字符串。

该函数接收一个参数 "space"，它会代表字符串中每个空格的位置。函数内部先创建一个空字符串，然后使用 while 循环从 space 变量中不断取出一个空格，将该位置的字符 " " 添加到空字符串中，并将 space 自减 1。当 space 的值为 0 时，循环结束，返回生成的字符串。

代码中定义了多个变量，包括 str、ws、da、pa、ma 和 ya，它们都被赋值为空字符串。这些变量在代码后续的处理中可能会被用来存储字符串中的不同部分，以便进行后续操作。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var sa = [];
var ws = [];
var da = [];
var qa = [];
var pa = [];
var ma = [];
var ya = [];
```

This is a program that reads the race results of a horse race and displays them to the audience. The race is assumed to be conducted on a set of virtual tracks, and the program displays the results of each horse in the race, including their placing, the distance they ran, and the winer if applicable. The program also allows the audience to place bets on the outcome of the race.



```
var vs = [];

// Main program
async function main()
{
    print(tab(31) + "HORSERACE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("WELCOME TO SOUTH PORTLAND HIGH RACETRACK\n");
    print("                      ...OWNED BY LAURIE CHEVALIER\n");
    print("DO YOU WANT DIRECTIONS");
    str = await input();
    if (str == "YES") {
        print("UP TO 10 MAY PLAY.  A TABLE OF ODDS WILL BE PRINTED.  YOU\n");
        print("MAY BET ANY + AMOUNT UNDER 100000 ON ONE HORSE.\n");
        print("DURING THE RACE, A HORSE WILL BE SHOWN BY ITS\n");
        print("NUMBER.  THE HORSES RACE DOWN THE PAPER!\n");
        print("\n");
    }
    print("HOW MANY WANT TO BET");
    c = parseInt(await input());
    print("WHEN ? APPEARS,TYPE NAME\n");
    for (a = 1; a <= c; a++) {
        ws[a] = await input();
    }
    do {
        print("\n");
        print("HORSE\t\tNUMBERS\tODDS\n");
        print("\n");
        for (i = 1; i <= 8; i++) {
            sa[i] = 0;
        }
        r = 0;
        for (a = 1; a <= 8; a++) {
            da[a] = Math.floor(10 * Math.random() + 1);
        }
        for (a = 1; a <= 8; a++) {
            r = r + da[a];
        }
        vs[1] = "JOE MAN";
        vs[2] = "L.B.J.";
        vs[3] = "MR.WASHBURN";
        vs[4] = "MISS KAREN";
        vs[5] = "JOLLY";
        vs[6] = "HORSE";
        vs[7] = "JELLY DO NOT";
        vs[8] = "MIDNIGHT";
        for (n = 1; n <= 8; n++) {
            print(vs[n] + "\t\t" + n + "\t" + (r / da[n]) + ":1\n");
        }
        print("--------------------------------------------------\n");
        print("PLACE YOUR BETS...HORSE # THEN AMOUNT\n");
        for (j = 1; j <= c; j++) {
            while (1) {
                print(ws[j]);
                str = await input();
                qa[j] = parseInt(str);
                pa[j] = parseInt(str.substr(str.indexOf(",") + 1));
                if (pa[j] < 1 || pa[j] >= 100000) {
                    print("  YOU CAN'T DO THAT!\N");
                } else {
                    break;
                }
            }
        }
        print("\n");
        print("1 2 3 4 5 6 7 8\n");
        t = 0;
        do {
            print("XXXXSTARTXXXX\n");
            for (i = 1; i <= 8; i++) {
                m = i;
                ma[i] = m;
                ya[ma[i]] = Math.floor(100 * Math.random() + 1);
                if (ya[ma[i]] < 10) {
                    ya[ma[i]] = 1;
                    continue;
                }
                s = Math.floor(r / da[i] + 0.5);
                if (ya[ma[i]] < s + 17) {
                    ya[ma[i]] = 2;
                    continue;
                }
                if (ya[ma[i]] < s + 37) {
                    ya[ma[i]] = 3;
                    continue;
                }
                if (ya[ma[i]] < s + 57) {
                    ya[ma[i]] = 4;
                    continue;
                }
                if (ya[ma[i]] < s + 77) {
                    ya[ma[i]] = 5;
                    continue;
                }
                if (ya[ma[i]] < s + 92) {
                    ya[ma[i]] = 6;
                    continue;
                }
                ya[ma[i]] = 7;
            }
            m = i;
            for (i = 1; i <= 8; i++) {
                sa[ma[i]] = sa[ma[i]] + ya[ma[i]];
            }
            i = 1;
            for (l = 1; l <= 8; l++) {
                for (i = 1; i <= 8 - l; i++) {
                    if (sa[ma[i]] < sa[ma[i + 1]])
                        continue;
                    h = ma[i];
                    ma[i] = ma[i + 1];
                    ma[i + 1] = h;
                }
            }
            t = sa[ma[8]];
            for (i = 1; i <= 8; i++) {
                b = sa[ma[i]] - sa[ma[i - 1]];
                if (b != 0) {
                    for (a = 1; a <= b; a++) {
                        print("\n");
                        if (sa[ma[i]] > 27)
                            break;
                    }
                    if (a <= b)
                        break;
                }
                print(" " + ma[i] + " ");
            }
            for (a = 1; a < 28 - t; a++) {
                print("\n");
            }
            print("XXXXFINISHXXXX\n");
            print("\n");
            print("\n");
            print("---------------------------------------------\n");
            print("\n");
        } while (t < 28) ;
        print("THE RACE RESULTS ARE:\n");
        z9 = 1;
        for (i = 8; i >= 1; i--) {
            f = ma[i];
            print("\n");
            print("" + z9 + " PLACE HORSE NO. " + f + " AT " + (r / da[f]) + ":1\n");
            z9++;
        }
        for (j = 1; j <= c; j++) {
            if (qa[j] != ma[8])
                continue;
            n = qa[j];
            print("\n");
            print(ws[j] + " WINS $" + (r / da[n]) * pa[j] + "\n");
        }
        print("DO YOU WANT TO BET ON THE NEXT RACE ?\n");
        print("YES OR NO");
        str = await input();
    } while (str == "YES") ;
}

```

这道题目要求解释以下代码的作用，不要输出源代码。根据我的理解，这道题目需要我解释 main() 函数的作用，因此我将给出对 main() 函数的分析和解释。

首先，我们需要了解 main() 函数在程序中的作用。在许多程序中，main() 函数是保存所有用户输入并执行特定任务的关键部分。在 main() 函数中，程序可以访问和操作用户提供的数据，如用户名、密码、年龄等。此外，在 main() 函数中，程序还可以执行特定的任务，如打印消息、生成报告等。

总的来说，main() 函数是程序中的入口点，负责执行程序的所有操作和操作。对于这道题目，我建议您查看程序的具体内容和功能，以更好地理解 main() 函数的作用。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `50_Horserace/python/horserace.py`

这段代码定义了一个名为 `basic_print` 的函数，该函数接受两个参数 `zones` 和 `**kwargs`，并使用 `math`、`random` 和 `time` 模块进行计算。函数的作用是模拟从 BASIC 语言中的 `print` 命令到某个区域（或多个区域）的输出。

具体来说，这段代码实现了一个类似于 BASIC 语言的 `print` 命令的功能，但它在不同的区域（或多个区域）输出不同的内容。 `indent` 参数指定输出的缩进量，`end` 参数指定输出结束时是否添加换行符。当多个区域同时指定时，它们将被连接成一个区域，直到达到指定的 `end` 字符串。

例如，你可以这样使用这个函数：
```python
math.random.shuffle(range(10, 21))
basic_print(["out", "a", "b", "c"], indent=4)
```
这将输出一个类似于这样的字符串：
```
out a b c
```



```
import math
import random
import time
from typing import List, Tuple


def basic_print(*zones, **kwargs) -> None:
    """Simulates the PRINT command from BASIC to some degree.
    Supports `printing zones` if given multiple arguments."""

    line = ""
    if len(zones) == 1:
        line = str(zones[0])
    else:
        line = "".join([f"{str(zone):<14}" for zone in zones])
    identation = kwargs.get("indent", 0)
    end = kwargs.get("end", "\n")
    print(" " * identation + line, end=end)


```

这段代码定义了一个名为“basic_input”的函数，它会向用户输入一个命令行提示符（prompt），并要求输入一个可识别的类型，如果没有提供类型转换，则自动将输入转换为该类型的含义。该函数在无限循环中进行尝试，直到从用户输入中得到有效的输入，并允许类型转换。如果输入出现无效值，函数会输出“INVALID INPUT!”的错误消息。

函数的作用是接受一个提示字符串作为输入，并将其传递给用户。如果用户输入的类型不是可识别的类型，函数将尝试将输入转换为可识别的类型。如果用户输入的类型无法转换或输入无效，函数将输出错误消息并继续尝试从用户输入中得到有效的输入。函数将保持交互式并始终尝试从用户输入中得到有效的输入，直到收到有效的输入来退出无限循环。


```
def basic_input(prompt: str, type_conversion=None):
    """BASIC INPUT command with optional type conversion"""

    while True:
        try:
            inp = input(f"{prompt}? ")
            if type_conversion is not None:
                inp = type_conversion(inp)
            break
        except ValueError:
            basic_print("INVALID INPUT!")
    return inp


# horse names do not change over the program, therefore making it a global.
```

这段代码是一个Python程序，其目的是让玩家参与一场虚拟的赛马游戏。在这个游戏中，对马匹的排序将用于标识它们。

程序的主要部分是一个名为`introduction`的函数，它会在屏幕上打印介绍和说明赛马游戏的相关信息。然后，程序会询问玩家是否需要指示，如果玩家选择不需要指示，函数就会返回到最初的状态。

如果玩家需要指示，程序将返回一个包含提示信息的列表，其中包括赛马游戏的规则。程序还定义了一个包含九匹马的名字的列表，这些名字将用于在游戏过程中识别马匹。


```
# throught the game, the ordering of the horses is used to indentify them
HORSE_NAMES = [
    "JOE MAW",
    "L.B.J.",
    "MR.WASHBURN",
    "MISS KAREN",
    "JOLLY",
    "HORSE",
    "JELLY DO NOT",
    "MIDNIGHT",
]


def introduction() -> None:
    """Print the introduction, and optional the instructions"""

    basic_print("HORSERACE", indent=31)
    basic_print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY", indent=15)
    basic_print("\n\n")
    basic_print("WELCOME TO SOUTH PORTLAND HIGH RACETRACK")
    basic_print("                      ...OWNED BY LAURIE CHEVALIER")
    y_n = basic_input("DO YOU WANT DIRECTIONS")

    # if no instructions needed, return
    if y_n.upper() == "NO":
        return

    basic_print("UP TO 10 MAY PLAY.  A TABLE OF ODDS WILL BE PRINTED.  YOU")
    basic_print("MAY BET ANY + AMOUNT UNDER 100000 ON ONE HORSE.")
    basic_print("DURING THE RACE, A HORSE WILL BE SHOWN BY ITS")
    basic_print("NUMBER.  THE HORSES RACE DOWN THE PAPER!")
    basic_print("")


```

这两段代码定义了 `setup_players()` 和 `setup_horses()` 函数。它们的目的是收集玩家数量和他们的名字，以及为每匹马生成随机赔率。

`setup_players()` 函数通过一个基本的输入操作，要求用户输入想要参与游戏的用户数量。然后，该函数收集每个玩家的名字，并返回一个包含这些名字的字符列表。

`setup_horses()` 函数生成每个马的随机赔率，并将这些赔率返回作为一个包含浮点数列表的元组。该列表的索引与全局 `HORSE_NAMES` 列表中每个名称的顺序相同。


```
def setup_players() -> List[str]:
    """Gather the number of players and their names"""

    # ensure we get an integer value from the user
    number_of_players = basic_input("HOW MANY WANT TO BET", int)

    # for each user query their name and return the list of names
    player_names = []
    basic_print("WHEN ? APPEARS,TYPE NAME")
    for _ in range(number_of_players):
        player_names.append(basic_input(""))
    return player_names


def setup_horses() -> List[float]:
    """Generates random odds for each horse. Returns a list of
    odds, indexed by the order of the global HORSE_NAMES."""

    odds = [random.randrange(1, 10) for _ in HORSE_NAMES]
    total = sum(odds)

    # rounding odds to two decimals for nicer output,
    # this is not in the origin implementation
    return [round(total / odd, 2) for odd in odds]


```



这段代码定义了两个函数，分别是 `print_horse_odds` 和 `get_bets`。

`print_horse_odds` 函数接收一个 `odds` 参数，并输出对于每个马的赔率。具体来说，函数会首先输出一个空格，然后用循环遍历 `HORSE_NAMES` 列表中的每个元素，对于每个元素，输出该马的编号和赔率，最后再输出一个空格。

`get_bets` 函数接收一个包含玩家名字的列表 `player_names` 参数，然后根据每个玩家的名字，获取他们要下注的赛马号码和投注金额。具体来说，函数会首先输出一个空格，然后循环遍历 `player_names` 列表中的每个元素，对于每个元素，程序会提示用户输入该赛马的编号，直到用户输入的编号是数字，并且用户输入的金额是数字和数字之间的小数，否则程序会提示用户重新输入。接着，程序会将该玩家的投注记录在 `bets` 列表中，并将 `bets` 列表中的元素打印出来。

这两个函数一起构成了一个赌博游戏的后台，用户可以根据自己的喜好输入自己的投注号码和金额，程序会计算出该投注的赔率和该玩家可能获得的奖金，并在用户点击“下注”按钮之后将所有投注记录在 `bets` 列表中。


```
def print_horse_odds(odds) -> None:
    """Print the odds for each horse"""

    basic_print("")
    for i in range(len(HORSE_NAMES)):
        basic_print(HORSE_NAMES[i], i, f"{odds[i]}:1")
    basic_print("")


def get_bets(player_names: List[str]) -> List[Tuple[int, float]]:
    """For each player, get the number of the horse to bet on,
    as well as the amount of money to bet"""

    basic_print("--------------------------------------------------")
    basic_print("PLACE YOUR BETS...HORSE # THEN AMOUNT")

    bets: List[Tuple[int, float]] = []
    for name in player_names:
        horse = basic_input(name, int)
        amount = None
        while amount is None:
            amount = basic_input("", float)
            if amount < 1 or amount >= 100000:
                basic_print("  YOU CAN'T DO THAT!")
                amount = None
        bets.append((horse, amount))

    basic_print("")

    return bets


```

这段代码定义了一个名为 `get_distance` 的函数，用于计算在赛马模拟中，一匹马在一小步之间的距离。

函数接收一个参数 `odd`，表示这一步的步数，这个参数是一个浮点数，范围是 1 到 100。函数返回一个整数，表示马在这一步中走过的距离。

函数内部使用 `random.randrange(1, 100)` 来生成一个 1 到 100 之间的随机数，用于模拟马步数。然后用这个随机数乘以一个变量 `s`，表示一匹马在一小步中相对于起跑线的距离，再根据马步数和距离的关系，选择返回 1、2、3、4 或 7。

整函数的作用是，根据 odds 随机选择马的一小步，并计算出这一步的距离。


```
def get_distance(odd: float) -> int:
    """Advances a horse during one step of the racing simulation.
    The amount travelled is random, but scaled by the odds of the horse"""

    d = random.randrange(1, 100)
    s = math.ceil(odd)
    if d < 10:
        return 1
    elif d < s + 17:
        return 2
    elif d < s + 37:
        return 3
    elif d < s + 57:
        return 4
    elif d < s + 77:
        return 5
    elif d < s + 92:
        return 6
    else:
        return 7


```

这段代码定义了一个名为 `print_race_state` 的函数，它用于输出有关赛马比赛当前状态和停止的信息。函数接收两个参数：赛马的总距离 `total_distance` 和赛马的位置列表 `race_pos`。

函数的主要目的是输出赛马比赛的当前状态和停止，以便骑手和观众了解比赛的情况。在函数内部，首先创建一个名为 "XXXXSTARTXXXX" 的字符串，用于表示赛马比赛的开始。然后使用 for 循环来打印比赛中的所有 28 个单位。在循环中，首先确保我们有下一匹要打印的赛马，然后打印该赛马的位置。如果下一匹要打印的赛马不在当前位置，就打印一个新的一行。在循环结束后，输出一个名为 "XXXXFINISHXXXX" 的字符串，表示赛马比赛的结束。

具体来说，这段代码的工作原理是：首先创建一个字符串来表示赛马比赛的开始，然后使用 for 循环来打印比赛中的所有 28 个单位。在循环内部，使用 while 循环来打印当前位置的骑手，并在该位置为空时输出自己的位置。如果当前位置有骑手，就打印该骑手的位置，否则就输出一个空行。在循环结束后，输出一个字符串，该字符串表示赛马比赛的结束。


```
def print_race_state(total_distance, race_pos) -> None:
    """Outputs the current state/stop of the race.
    Each horse is placed according to the distance they have travelled. In
    case some horses travelled the same distance, their numbers are printed
    on the same name"""

    # we dont want to modify the `race_pos` list, since we need
    # it later. Therefore we generating an interator from the list
    race_pos_iter = iter(race_pos)

    # race_pos is stored by last to first horse in the race.
    # we get the next horse we need to print out
    next_pos = next(race_pos_iter)

    # start line
    basic_print("XXXXSTARTXXXX")

    # print all 28 lines/unit of the race course
    for line in range(28):

        # ensure we still have a horse to print and if so, check if the
        # next horse to print is not the current line
        # needs iteration, since multiple horses can share the same line
        while next_pos is not None and line == total_distance[next_pos]:
            basic_print(f"{next_pos} ", end="")
            next_pos = next(race_pos_iter, None)
        else:
            # if no horses are left to print for this line, print a new line
            basic_print("")

    # finish line
    basic_print("XXXXFINISHXXXX")


```

This is an implementation of a race tracker in Python. It uses two arrays, one to track the total distance each horse has traveled and


```
def simulate_race(odds) -> List[int]:
    num_horses = len(HORSE_NAMES)

    # in spirit of the original implementation, using two arrays to
    # track the total distance travelled, and create an index from
    # race position -> horse index
    total_distance = [0] * num_horses

    # race_pos maps from the position in the race, to the index of the horse
    # it will later be sorted from last to first horse, based on the
    # distance travelled by each horse.
    # e.g. race_pos[0] => last horse
    #      race_pos[-1] => winning horse
    race_pos = list(range(num_horses))

    basic_print("\n1 2 3 4 5 6 7 8")

    while True:

        # advance each horse by a random amount
        for i in range(num_horses):
            total_distance[i] += get_distance(odds[i])

        # bubble sort race_pos based on total distance travelled
        # in the original implementation, race_pos is reset for each
        # simulation step, so we keep this behaviour here
        race_pos = list(range(num_horses))
        for line in range(num_horses):
            for i in range(num_horses - 1 - line):
                if total_distance[race_pos[i]] < total_distance[race_pos[i + 1]]:
                    continue
                race_pos[i], race_pos[i + 1] = race_pos[i + 1], race_pos[i]

        # print current state of the race
        print_race_state(total_distance, race_pos)

        # goal line is defined as 28 units from start
        # check if the winning horse is already over the finish line
        if total_distance[race_pos[-1]] >= 28:
            return race_pos

        # this was not in the original BASIC implementation, but it makes the
        # race visualization a nice animation (if the terminal size is set to 31 rows)
        time.sleep(1)


```

这段代码定义了一个名为 `print_race_results` 的函数，它接受四个参数：`race_positions` 是一个包含四个元素的列表，表示赛马场的位置，每个元素都是一位数字；`odds` 是一个包含四个元素的列表，表示每个赛马位置的赔率；`bets` 是一个包含四个元素的列表，表示每个投注者的赌注；`player_names` 是一个包含四个元素的列表，表示每个投注者的姓名。

函数中首先打印赛马场的所有位置，然后打印每个投注者的胜赔。具体来说，对于每个位置和投注者，函数会打印一条包含他们名称和赢得的赔率的行。如果某个投注者赢得了比赛，函数将另外一条包含该投注者名称和赢得的赔率的行。


```
def print_race_results(race_positions, odds, bets, player_names) -> None:
    """Print the race results, as well as the winnings of each player"""

    # print the race positions first
    basic_print("THE RACE RESULTS ARE:")
    for position, horse_idx in enumerate(reversed(race_positions), start=1):
        line = f"{position} PLACE HORSE NO. {horse_idx} AT {odds[horse_idx]}:1"
        basic_print("")
        basic_print(line)

    # followed by the amount the players won
    winning_horse_idx = race_positions[-1]
    for idx, name in enumerate(player_names):
        (horse, amount) = bets[idx]
        if horse == winning_horse_idx:
            basic_print("")
            basic_print(f"{name} WINS ${amount * odds[winning_horse_idx]}")


```

这段代码是一个Python游戏循环，主要作用是让玩家在每一轮游戏中下注，然后计算出下一轮的赔率。

具体来说，代码首先定义了一个`main_loop`函数，它接受两个参数：一个字符串`player_names`表示每个玩家的名字，一个浮点数`horse_odds`表示赔率。

接下来，代码进入了一个无限循环，每次会输出当前的赔率，然后要求玩家输入是否要下一注。如果玩家输入"YES"，则程序会进入下一轮游戏，否则退出循环。

在每次游戏循环中，程序会先调用一个名为`simulate_race`的函数，这个函数会根据当前的赔率生成一系列比赛结果，并返回这些结果的位置。

接着，程序会调用一个名为`print_race_results`的函数，这个函数会打印出所有比赛结果，包括玩家和赔率。

最后，程序会再次调用`basic_input`函数，要求玩家输入"YES"或"NO"，如果玩家输入"NO"，则退出当前游戏循环。


```
def main_loop(player_names, horse_odds) -> None:
    """Main game loop"""

    while True:
        print_horse_odds(horse_odds)
        bets = get_bets(player_names)
        final_race_positions = simulate_race(horse_odds)
        print_race_results(final_race_positions, horse_odds, bets, player_names)

        basic_print("DO YOU WANT TO BET ON THE NEXT RACE ?")
        one_more = basic_input("YES OR NO")
        if one_more.upper() != "YES":
            break


```

这段代码定义了一个名为 `main` 的函数，它导出了 `None`。函数内部包含以下几行代码：

1. 引入外部函数 `setup_players` 和 `setup_horses`。
2. 生成介绍信、选手姓名和马匹赔率。
3. 生成游戏的主要循环，玩家可以多次参加比赛，每次比赛的结果相同。
4. 在游戏循环中，马匹的赔率保持不变。

如果运行了这段代码，将会首先调用 `setup_players` 和 `setup_horses` 函数，然后进入游戏的主要循环。在游戏循环中，玩家可以参加多个比赛，而每次比赛的结果在所有比赛结束后才会更新。


```
def main() -> None:
    # introduction, player names and horse odds are only generated once
    introduction()
    player_names = setup_players()
    horse_odds = setup_horses()

    # main loop of the game, the player can play multiple races, with the
    # same odds
    main_loop(player_names, horse_odds)


if __name__ == "__main__":
    main()

```