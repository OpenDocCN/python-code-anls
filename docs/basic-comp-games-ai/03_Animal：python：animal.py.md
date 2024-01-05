# `03_Animal\python\animal.py`

```
# 定义一个函数，根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
"""
    Update the current node with new text and yes/no nodes.
    """
        self.text = text
        self.yes_node = yes_node
        self.no_node = no_node
        self, new_question: str, answer_new_ques: str, new_animal: str
    ) -> None:
    # update the leaf with a question
    old_animal = self.text  # 保存当前节点的文本信息到变量old_animal
    self.text = new_question  # 用新的问题替换当前节点的文本信息

    if answer_new_ques == "y":  # 如果回答是"y"
        self.yes_node = Node(new_animal, None, None)  # 创建一个新的节点作为"是"的子节点，文本信息为new_animal
        self.no_node = Node(old_animal, None, None)  # 创建一个新的节点作为"否"的子节点，文本信息为old_animal
    else:  # 如果回答不是"y"
        self.yes_node = Node(old_animal, None, None)  # 创建一个新的节点作为"是"的子节点，文本信息为old_animal
        self.no_node = Node(new_animal, None, None)  # 创建一个新的节点作为"否"的子节点，文本信息为new_animal

# the leafs have as children None
def is_leaf(self) -> bool:
    return self.yes_node is None and self.no_node is None  # 判断当前节点是否为叶子节点，即没有子节点


def list_known_animals(root_node: Optional[Node]) -> None:
    """Traversing the tree by recursion until we reach the leafs."""
    # 通过递归遍历树，直到达到叶子节点
    if root_node is None:
        return

    if root_node.is_leaf():
        # 如果当前节点是叶子节点，则打印节点文本
        print(root_node.text, end=" " * 11)
        return

    if root_node.yes_node:
        # 如果存在“是”节点，则继续遍历“是”节点
        list_known_animals(root_node.yes_node)

    if root_node.no_node:
        # 如果存在“否”节点，则继续遍历“否”节点
        list_known_animals(root_node.no_node)


def parse_input(message: str, check_list: bool, root_node: Optional[Node]) -> str:
    """only accepts yes or no inputs and recognizes list operation"""
    # 只接受“是”或“否”的输入，并识别列表操作
    token = ""
    while token not in ["y", "n"]:
        inp = input(message)
        # 如果check_list不为空且输入的内容为"list"，则打印已知动物列表
        if check_list and inp.lower() == "list":
            print("Animals I already know are:")
            list_known_animals(root_node)
            print("\n")

        # 如果输入内容长度大于0，则将输入的第一个字符转换为小写作为token，否则token为空字符串
        if len(inp) > 0:
            token = inp[0].lower()
        else:
            token = ""

    # 返回token
    return token


# 避免空输入，要求用户输入非空消息
def avoid_void_input(message: str) -> str:
    answer = ""
    # 当用户输入为空时，持续要求用户输入消息
    while answer == "":
        answer = input(message)
    # 返回用户输入的消息
    return answer
def print_intro() -> None:
    # 打印游戏介绍
    print(" " * 32 + "Animal")
    print(" " * 15 + "Creative Computing Morristown, New Jersey\n")
    print("Play ´Guess the Animal´")
    print("Think of an animal and the computer will try to guess it.\n")


def main() -> None:
    # 初始化树的根节点和子节点
    yes_child = Node("Fish", None, None)
    no_child = Node("Bird", None, None)
    root = Node("Does it swim?", yes_child, no_child)

    # 游戏的主循环
    print_intro()
    # 解析用户输入，确定是否继续游戏
    keep_playing = parse_input("Are you thinking of an animal? ", True, root) == "y"
    while keep_playing:
        keep_asking = True
        # 开始通过根节点遍历树
        actual_node: Node = root  # 初始化当前节点为根节点

        while keep_asking:  # 当需要继续询问时执行循环

            if not actual_node.is_leaf():  # 如果当前节点不是叶子节点

                # 我们需要继续询问，即遍历节点
                answer = parse_input(actual_node.text, False, None)  # 解析用户输入的答案

                # 由于这是一个内部节点，两个子节点都不是空的
                if answer == "y":  # 如果答案是"y"
                    assert actual_node.yes_node is not None  # 确保当前节点的"是"子节点不为空
                    actual_node = actual_node.yes_node  # 将当前节点移动到"是"子节点
                else:
                    assert actual_node.no_node is not None  # 确保当前节点的"否"子节点不为空
                    actual_node = actual_node.no_node  # 将当前节点移动到"否"子节点
            else:
                # 我们已经到达可能的答案
                answer = parse_input(f"Is it a {actual_node.text}? ", False, None)  # 解析用户输入的答案，询问用户是否是当前节点的值
                if answer == "n":  # 如果答案是"n"
                    # 将新动物添加到树中
                    new_animal = avoid_void_input(
                        "The animal you were thinking of was a ? "
                    )
                    # 避免空输入，提示用户输入一个可以区分新动物和现有动物的问题
                    new_question = avoid_void_input(
                        "Please type in a question that would distinguish a "
                        f"{new_animal} from a {actual_node.text}: "
                    )
                    # 解析用户输入的问题的答案
                    answer_new_question = parse_input(
                        f"for a {new_animal} the answer would be: ", False, None
                    )

                    # 更新当前节点的信息，添加新问题、新问题的答案和新动物
                    actual_node.update_node(
                        new_question + "?", answer_new_question, new_animal
                    )

                else:
                    # 提示用户尝试另一个动物
                    print("Why not try another animal?")

                # 停止继续询问
                keep_asking = False
        keep_playing = parse_input("Are you thinking of an animal? ", True, root) == "y"
```
这行代码的作用是询问用户是否想到了一个动物，并根据用户的回答更新keep_playing变量的值。

```
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
```
这部分是注释，解释了程序使用的数据结构以及对程序的一些修改建议。建议可以将树的数据存储到数据库中，还提到了程序可以轻松修改以不仅仅是对动物进行猜测，还可以修改树的初始数据、向用户提出的问题以及初始消息函数。
# 如果当前脚本被直接执行，则执行 main() 函数
if __name__ == "__main__":
    main()
```
这段代码用于判断当前脚本是否被直接执行，如果是，则调用 main() 函数。这是一种常见的编程习惯，可以使代码更具可重用性和模块化。
```