# `D:\src\scipysrc\sympy\sympy\combinatorics\rewritingsystem_fsm.py`

```
class State:
    '''
    A representation of a state managed by a ``StateMachine``.

    Attributes:
        name (instance of FreeGroupElement or string) -- State name which is also assigned to the Machine.
        transisitons (OrderedDict) -- Represents all the transitions of the state object.
        state_type (string) -- Denotes the type (accept/start/dead) of the state.
        rh_rule (instance of FreeGroupElement) -- right hand rule for dead state.
        state_machine (instance of StateMachine object) -- The finite state machine that the state belongs to.
    '''

    def __init__(self, name, state_machine, state_type=None, rh_rule=None):
        self.name = name                              # 设置状态的名称
        self.transitions = {}                         # 初始化状态的转移字典为空
        self.state_machine = state_machine             # 将状态机对象与状态关联起来
        self.state_type = state_type[0]                # 设置状态的类型（首字母）
        self.rh_rule = rh_rule                        # 设置状态的右手规则

    def add_transition(self, letter, state):
        '''
        Add a transition from the current state to a new state.

        Keyword Arguments:
            letter -- The alphabet element the current state reads to make the state transition.
            state -- This will be an instance of the State object which represents a new state after in the transition after the alphabet is read.

        '''
        self.transitions[letter] = state              # 在转移字典中添加从当前状态读取字母到新状态的映射关系

class StateMachine:
    '''
    Representation of a finite state machine the manages the states and the transitions of the automaton.

    Attributes:
        states (dictionary) -- Collection of all registered `State` objects.
        name (str) -- Name of the state machine.
    '''

    def __init__(self, name, automaton_alphabet):
        self.name = name                              # 设置状态机的名称
        self.automaton_alphabet = automaton_alphabet  # 设置状态机使用的字母表
        self.states = {}                              # 初始化状态字典为空，用于存储所有的状态对象
        self.add_state('start', state_type='s')       # 向状态机添加起始状态

    def add_state(self, state_name, state_type=None, rh_rule=None):
        '''
        Instantiate a state object and stores it in the 'states' dictionary.

        Arguments:
            state_name (instance of FreeGroupElement or string) -- name of the new states.
            state_type (string) -- Denotes the type (accept/start/dead) of the state added.
            rh_rule (instance of FreeGroupElement) -- right hand rule for dead state.

        '''
        new_state = State(state_name, self, state_type, rh_rule)  # 创建新的状态对象
        self.states[state_name] = new_state             # 将新创建的状态对象添加到状态字典中

    def __repr__(self):
        return "%s" % (self.name)                       # 返回状态机的名称作为字符串表示
```