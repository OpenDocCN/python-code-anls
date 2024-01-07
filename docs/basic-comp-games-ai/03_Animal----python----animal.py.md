# `basic-computer-games\03_Animal\python\animal.py`

```

"""
Animal

"""

from typing import Optional


class Node:
    """
    Node of the binary tree of questions.
    """

    def __init__(
        self, text: str, yes_node: Optional["Node"], no_node: Optional["Node"]
    ):
        # the nodes that are leafs have as text the animal's name, otherwise
        # a yes/no question
        self.text = text
        self.yes_node = yes_node
        self.no_node = no_node

    def update_node(
        self, new_question: str, answer_new_ques: str, new_animal: str
    ) -> None:
        # update the leaf with a question
        old_animal = self.text
        # we replace the animal with a new question
        self.text = new_question

        if answer_new_ques == "y":
            self.yes_node = Node(new_animal, None, None)
            self.no_node = Node(old_animal, None, None)
        else:
            self.yes_node = Node(old_animal, None, None)
            self.no_node = Node(new_animal, None, None)

    # the leafs have as children None
    def is_leaf(self) -> bool:
        return self.yes_node is None and self.no_node is None


def list_known_animals(root_node: Optional[Node]) -> None:
    """Traversing the tree by recursion until we reach the leafs."""
    if root_node is None:
        return

    if root_node.is_leaf():
        print(root_node.text, end=" " * 11)
        return

    if root_node.yes_node:
        list_known_animals(root_node.yes_node)

    if root_node.no_node:
        list_known_animals(root_node.no_node)


def parse_input(message: str, check_list: bool, root_node: Optional[Node]) -> str:
    """only accepts yes or no inputs and recognizes list operation"""
    token = ""
    while token not in ["y", "n"]:
        inp = input(message)

        if check_list and inp.lower() == "list":
            print("Animals I already know are:")
            list_known_animals(root_node)
            print("\n")

        if len(inp) > 0:
            token = inp[0].lower()
        else:
            token = ""

    return token


def avoid_void_input(message: str) -> str:
    answer = ""
    while answer == "":
        answer = input(message)
    return answer


def print_intro() -> None:
    print(" " * 32 + "Animal")
    print(" " * 15 + "Creative Computing Morristown, New Jersey\n")
    print("Play ´Guess the Animal´")
    print("Think of an animal and the computer will try to guess it.\n")


########################################################
# Porting Notes
#
#   The data structure used for storing questions and
#   animals is a binary tree where each non-leaf node
#   has a question, while the leafs store the animals.
#
#   As the original program, this program doesn't store
#   old questions and animals. A good modification would
#   be to add a database to store the tree.
#    Also as the original program, this one can be easily
#   modified to not only make guesses about animals, by
#   modyfing the initial data of the tree, the questions
#   that are asked to the user and the initial message
#   function  (Lines 120 to 130, 135, 158, 160, 168, 173)

########################################################

if __name__ == "__main__":
    main()



注释：这段代码是一个关于动物猜测的程序。它使用了二叉树数据结构来存储问题和动物，其中非叶节点存储问题，叶节点存储动物。程序包括了一些函数，如更新节点、列出已知动物、解析输入等。同时，还包括了一些关于程序修改和扩展的说明。最后，通过`if __name__ == "__main__":`来调用主函数`main()`。
```