# `basic-computer-games\46_Hexapawn\python\hexapawn.py`

```

"""
HEXAPAWN

A machine learning game, an interpretation of HEXAPAWN game as
presented in Martin Gardner's "The Unexpected Hanging and Other
Mathematical Diversions", Chapter Eight: A Matchbox Game-Learning
Machine.

Original version for H-P timeshare system by R.A. Kaapke 5/5/76
Instructions by Jeff Dalton
Conversion to MITS BASIC by Steve North


Port to Python by Dave LeCompte
"""

# PORTING NOTES:
#
# I printed out the BASIC code and hand-annotated what each little block
# of code did, which feels amazingly retro.
#
# I encourage other porters that have a complex knot of GOTOs and
# semi-nested subroutines to do hard-copy hacking, it might be a
# different perspective that helps.
#
# A spoiler - the objective of the game is not documented, ostensibly to
# give the human player a challenge. If a player (human or computer)
# advances a pawn across the board to the far row, that player wins. If
# a player has no legal moves (either by being blocked, or all their
# pieces having been captured), that player loses.
#
# The original BASIC had 2 2-dimensional tables stored in DATA at the
# end of the program. This encoded all 19 different board configurations
# (Hexapawn is a small game), with reflections in one table, and then in
# a parallel table, for each of the 19 rows, a list of legal moves was
# encoded by turning them into 2-digit decimal numbers. As gameplay
# continued, the AI would overwrite losing moves with 0 in the second
# array.
#
# My port takes this "parallel array" structure and turns that
# information into a small Python class, BoardLayout. BoardLayout stores
# the board description and legal moves, but stores the moves as (row,
# column) 2-tuples, which is easier to read. The logic for checking if a
# BoardLayout matches the current board, as well as removing losing move
# have been moved into methods of this class.

import random
from typing import Iterator, List, NamedTuple, Optional, Tuple

PAGE_WIDTH = 64

HUMAN_PIECE = 1
EMPTY_SPACE = 0
COMPUTER_PIECE = -1


class ComputerMove(NamedTuple):
    board_index: int
    move_index: int
    m1: int
    m2: int


wins = 0
losses = 0


def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


def print_instructions() -> None:
    # Print the instructions for the game
    print(
        """
THIS PROGRAM PLAYS THE GAME OF HEXAPAWN.
HEXAPAWN IS PLAYED WITH CHESS PAWNS ON A 3 BY 3 BOARD.
THE PAWNS ARE MOVED AS IN CHESS - ONE SPACE FORWARD TO
AN EMPTY SPACE OR ONE SPACE FORWARD AND DIAGONALLY TO
CAPTURE AN OPPOSING MAN.  ON THE BOARD, YOUR PAWNS
ARE 'O', THE COMPUTER'S PAWNS ARE 'X', AND EMPTY
SQUARES ARE '.'.  TO ENTER A MOVE, TYPE THE NUMBER OF
THE SQUARE YOU ARE MOVING FROM, FOLLOWED BY THE NUMBER
OF THE SQUARE YOU WILL MOVE TO.  THE NUMBERS MUST BE
SEPERATED BY A COMMA.

THE COMPUTER STARTS A SERIES OF GAMES KNOWING ONLY WHEN
THE GAME IS WON (A DRAW IS IMPOSSIBLE) AND HOW TO MOVE.
IT HAS NO STRATEGY AT FIRST AND JUST MOVES RANDOMLY.
HOWEVER, IT LEARNS FROM EACH GAME.  THUS, WINNING BECOMES
MORE AND MORE DIFFICULT.  ALSO, TO HELP OFFSET YOUR
INITIAL ADVANTAGE, YOU WILL NOT BE TOLD HOW TO WIN THE
GAME BUT MUST LEARN THIS BY PLAYING.

THE NUMBERING OF THE BOARD IS AS FOLLOWS:
          123
          456
          789

FOR EXAMPLE, TO MOVE YOUR RIGHTMOST PAWN FORWARD,
YOU WOULD TYPE 9,6 IN RESPONSE TO THE QUESTION
'YOUR MOVE ?'.  SINCE I'M A GOOD SPORT, YOU'LL ALWAYS
GO FIRST.

"""
    )


def prompt_yes_no(msg: str) -> bool:
    # Prompt the user with a message and return True if the response starts with 'Y', False if it starts with 'N'
    while True:
        print(msg)
        response = input().upper()
        if response[0] == "Y":
            return True
        elif response[0] == "N":
            return False


def reverse_space_name(space_name: int) -> int:
    # Reverse a space name in the range 1-9 left to right
    assert 1 <= space_name <= 9

    reflections = {1: 3, 2: 2, 3: 1, 4: 6, 5: 5, 6: 4, 7: 9, 8: 8, 9: 7}
    return reflections[space_name]


def is_space_in_center_column(space_name: int) -> bool:
    # Check if a space is in the center column
    return reverse_space_name(space_name) == space_name


class BoardLayout:
    def __init__(self, cells: List[int], move_list: List[Tuple[int, int]]) -> None:
        self.cells = cells
        self.moves = move_list

    def _check_match_no_mirror(self, cell_list: List[int]) -> bool:
        # Check if the board layout matches without mirroring
        return all(
            board_contents == cell_list[space_index]
            for space_index, board_contents in enumerate(self.cells)
        )

    def _check_match_with_mirror(self, cell_list: List[int]) -> bool:
        # Check if the board layout matches with mirroring
        for space_index, board_contents in enumerate(self.cells):
            reversed_space_index = reverse_space_name(space_index + 1) - 1
            if board_contents != cell_list[reversed_space_index]:
                return False
        return True

    def check_match(self, cell_list: List[int]) -> Tuple[bool, Optional[bool]]:
        # Check if the board layout matches with or without mirroring
        if self._check_match_with_mirror(cell_list):
            return True, True
        elif self._check_match_no_mirror(cell_list):
            return True, False
        return False, None

    def get_random_move(
        self, reverse_board: Optional[bool]
    ) -> Optional[Tuple[int, int, int]]:
        # Get a random move from the board layout
        if not self.moves:
            return None
        move_index = random.randrange(len(self.moves))

        m1, m2 = self.moves[move_index]
        if reverse_board:
            m1 = reverse_space_name(m1)
            m2 = reverse_space_name(m2)

        return move_index, m1, m2


boards = [
    BoardLayout([-1, -1, -1, 1, 0, 0, 0, 1, 1], [(2, 4), (2, 5), (3, 6)]),
    # ... (other board layouts)
]


def get_move(board_index: int, move_index: int) -> Tuple[int, int]:
    # Get a move from a specific board layout
    assert board_index >= 0 and board_index < len(boards)
    board = boards[board_index]

    assert move_index >= 0 and move_index < len(board.moves)

    return board.moves[move_index]


def remove_move(board_index: int, move_index: int) -> None:
    # Remove a move from a specific board layout
    assert board_index >= 0 and board_index < len(boards)
    board = boards[board_index]

    assert move_index >= 0 and move_index < len(board.moves)

    del board.moves[move_index]


def init_board() -> List[int]:
    # Initialize the game board
    return [COMPUTER_PIECE] * 3 + [EMPTY_SPACE] * 3 + [HUMAN_PIECE] * 3

# ... (other functions and classes)

def main() -> None:
    # Main function to start the game
    print_header("HEXAPAWN")
    if prompt_yes_no("INSTRUCTIONS (Y-N)?"):
        print_instructions()

    global wins, losses
    wins = 0
    losses = 0

    while True:
        play_game()
        show_scores()


if __name__ == "__main__":
    main()

```