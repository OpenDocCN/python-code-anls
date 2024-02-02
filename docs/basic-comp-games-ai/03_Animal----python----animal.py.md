# `basic-computer-games\03_Animal\python\animal.py`

```py
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
    # 返回变量 token 的值
    return token
# 避免空输入，要求用户输入非空字符串
def avoid_void_input(message: str) -> str:
    answer = ""
    while answer == "":
        answer = input(message)
    return answer

# 打印游戏介绍
def print_intro() -> None:
    print(" " * 32 + "Animal")
    print(" " * 15 + "Creative Computing Morristown, New Jersey\n")
    print("Play ´Guess the Animal´")
    print("Think of an animal and the computer will try to guess it.\n")

# 主函数
def main() -> None:
    # 初始化树
    yes_child = Node("Fish", None, None)
    no_child = Node("Bird", None, None)
    root = Node("Does it swim?", yes_child, no_child)

    # 游戏的主循环
    print_intro()
    # 解析用户输入，判断是否继续游戏
    keep_playing = parse_input("Are you thinking of an animal? ", True, root) == "y"
    # 当继续游戏时
    while keep_playing:
        # 继续询问
        keep_asking = True
        # 从根节点开始遍历树
        actual_node: Node = root

        # 当需要继续询问时
        while keep_asking:

            # 如果当前节点不是叶子节点
            if not actual_node.is_leaf():

                # 我们需要继续询问，即遍历节点
                answer = parse_input(actual_node.text, False, None)

                # 由于这是一个内部节点，两个子节点都不是空的
                if answer == "y":
                    assert actual_node.yes_node is not None
                    actual_node = actual_node.yes_node
                else:
                    assert actual_node.no_node is not None
                    actual_node = actual_node.no_node
            else:
                # 我们已经到达可能的答案
                answer = parse_input(f"Is it a {actual_node.text}? ", False, None)
                if answer == "n":
                    # 将新动物添加到树中
                    new_animal = avoid_void_input(
                        "The animal you were thinking of was a ? "
                    )
                    new_question = avoid_void_input(
                        "Please type in a question that would distinguish a "
                        f"{new_animal} from a {actual_node.text}: "
                    )
                    answer_new_question = parse_input(
                        f"for a {new_animal} the answer would be: ", False, None
                    )

                    actual_node.update_node(
                        new_question + "?", answer_new_question, new_animal
                    )

                else:
                    print("Why not try another animal?")

                keep_asking = False

        # 当玩家继续游戏时，询问是否在想着一个动物
        keep_playing = parse_input("Are you thinking of an animal? ", True, root) == "y"
# 如果当前脚本被直接执行，则执行 main() 函数
if __name__ == "__main__":
    main()
```