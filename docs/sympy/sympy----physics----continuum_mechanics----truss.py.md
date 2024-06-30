# `D:\src\scipysrc\sympy\sympy\physics\continuum_mechanics\truss.py`

```
    @property
    def nodes(self):
        """
        Returns the nodes of the truss along with their positions.
        """
        return self._nodes

返回当前 `Truss` 对象的节点列表 `_nodes`，这些节点包括它们的位置信息。


    @property
    def node_labels(self):
        """
        Returns the node labels of the truss.
        """
        return self._node_labels

返回当前 `Truss` 对象的节点标签列表 `_node_labels`，用于标识每个节点。


    @property
    def node_positions(self):
        """
        Returns the positions of the nodes of the truss.
        """
        return self._node_positions

返回当前 `Truss` 对象的节点位置列表 `_node_positions`，包含每个节点的坐标信息。
    def members(self):
        """
        Returns the members of the truss along with the start and end points.
        """
        # 返回该桁架的构件列表，包括起始点和终点信息
        return self._members

    @property
    def member_lengths(self):
        """
        Returns the length of each member of the truss.
        """
        # 返回桁架每个构件的长度信息
        return self._member_lengths

    @property
    def supports(self):
        """
        Returns the nodes with provided supports along with the kind of support provided i.e.
        pinned or roller.
        """
        # 返回具有支持的节点列表，以及提供的支持类型（例如固定或滚动支持）
        return self._supports

    @property
    def loads(self):
        """
        Returns the loads acting on the truss.
        """
        # 返回作用在桁架上的荷载信息
        return self._loads

    @property
    def reaction_loads(self):
        """
        Returns the reaction forces for all supports which are all initialized to 0.
        """
        # 返回所有支持节点的反力，所有反力初始化为0
        return self._reaction_loads

    @property
    def internal_forces(self):
        """
        Returns the internal forces for all members which are all initialized to 0.
        """
        # 返回所有构件的内力，所有内力初始化为0
        return self._internal_forces

    def add_node(self, *args):
        """
        This method adds a node to the truss along with its name/label and its location.
        Multiple nodes can be added at the same time.

        Parameters
        ==========
        The input(s) for this method are tuples of the form (label, x, y).

        label:  String or a Symbol
            The label for a node. It is the only way to identify a particular node.

        x: Sympifyable
            The x-coordinate of the position of the node.

        y: Sympifyable
            The y-coordinate of the position of the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0))
        >>> t.nodes
        [('A', 0, 0)]
        >>> t.add_node(('B', 3, 0), ('C', 4, 1))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('C', 4, 1)]
        """

        for i in args:
            label = i[0]
            x = i[1]
            x = sympify(x)  # 将x坐标符号化
            y = i[2]
            y = sympify(y)  # 将y坐标符号化

            # 检查节点是否具有唯一标签
            if label in self._node_coordinates:
                raise ValueError("Node needs to have a unique label")

            # 检查位置是否已经存在节点
            elif [x, y] in self._node_coordinates.values():
                raise ValueError("A node already exists at the given position")

            else:
                # 将节点添加到各种内部数据结构中
                self._nodes.append((label, x, y))
                self._node_labels.append(label)
                self._node_positions.append((x, y))
                self._node_position_x.append(x)
                self._node_position_y.append(y)
                self._node_coordinates[label] = [x, y]
    # 定义一个方法，用于从桁架中移除节点
    def remove_node(self, *args):
        """
        This method removes a node from the truss.
        Multiple nodes can be removed at the same time.

        Parameters
        ==========
        The input(s) for this method are the labels of the nodes to be removed.

        label:  String or Symbol
            The label of the node to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 5, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('C', 5, 0)]
        >>> t.remove_node('A', 'C')
        >>> t.nodes
        [('B', 3, 0)]
        """
        # 遍历每个传入的节点标签
        for label in args:
            # 遍历当前桁架中的节点列表
            for i in range(len(self.nodes)):
                # 检查节点标签是否匹配当前节点
                if self._node_labels[i] == label:
                    # 获取节点的位置坐标
                    x = self._node_position_x[i]
                    y = self._node_position_y[i]

            # 如果节点标签不在节点坐标字典中，引发数值错误异常
            if label not in self._node_coordinates:
                raise ValueError("No such node exists in the truss")

            else:
                # 复制一份成员列表
                members_duplicate = self._members.copy()
                # 遍历复制的成员列表
                for member in members_duplicate:
                    # 如果节点已经有与之相关联的成员，引发数值错误异常
                    if label == self._members[member][0] or label == self._members[member][1]:
                        raise ValueError("The given node already has member attached to it")
                # 从节点列表中移除指定节点
                self._nodes.remove((label, x, y))
                # 从节点标签列表中移除指定标签
                self._node_labels.remove(label)
                # 从节点位置列表中移除指定位置
                self._node_positions.remove((x, y))
                # 从节点 x 坐标列表中移除指定坐标
                self._node_position_x.remove(x)
                # 从节点 y 坐标列表中移除指定坐标
                self._node_position_y.remove(y)
                # 如果节点标签存在于负载字典中，将其移除
                if label in self._loads:
                    self._loads.pop(label)
                # 如果节点标签存在于支持字典中，将其移除
                if label in self._supports:
                    self._supports.pop(label)
                # 从节点坐标字典中移除指定节点
                self._node_coordinates.pop(label)
    # 这个方法在给定桁架中的任意两个节点之间添加一个构件（桁架的一部分）

    # 参数：
    # args: 可变参数，每个参数是一个元组 (label, start, end)，定义了一个构件的标签和连接的起始点和终止点

    # label: 字符串或符号
    #     构件的标签，用于唯一标识一个特定的构件

    # start: 字符串或符号
    #     构件的起始点/节点的标签

    # end: 字符串或符号
    #     构件的结束点/节点的标签

    # 抛出异常：
    # - 如果起始点或结束点不在桁架的节点坐标字典中，或者起始点等于结束点，则抛出 ValueError
    # - 如果桁架中已存在具有相同标签的构件，则抛出 ValueError
    # - 如果两个节点之间已经存在构件，则抛出 ValueError

    # 如果没有抛出异常，将构件信息添加到桁架的成员和长度字典中，并初始化内部力为零
    for i in args:
        label = i[0]
        start = i[1]
        end = i[2]

        if start not in self._node_coordinates or end not in self._node_coordinates or start==end:
            raise ValueError("The start and end points of the member must be unique nodes")

        elif label in self._members:
            raise ValueError("A member with the same label already exists for the truss")

        elif self._nodes_occupied.get((start, end)):
            raise ValueError("A member already exists between the two nodes")

        else:
            self._members[label] = [start, end]
            # 计算构件的长度并存储在内部变量中
            self._member_lengths[label] = sqrt((self._node_coordinates[end][0]-self._node_coordinates[start][0])**2 + (self._node_coordinates[end][1]-self._node_coordinates[start][1])**2)
            # 标记节点之间存在构件
            self._nodes_occupied[start, end] = True
            self._nodes_occupied[end, start] = True
            # 初始化内部力为零
            self._internal_forces[label] = 0
    # 定义一个方法，用于从给定的桁架中移除成员
    def remove_member(self, *args):
        """
        This method removes members from the given truss.

        Parameters
        ==========
        labels: String or Symbol
            The label for the member to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 2, 2))
        >>> t.add_member(('AB', 'A', 'B'), ('AC', 'A', 'C'), ('BC', 'B', 'C'))
        >>> t.members
        {'AB': ['A', 'B'], 'AC': ['A', 'C'], 'BC': ['B', 'C']}
        >>> t.remove_member('AC', 'BC')
        >>> t.members
        {'AB': ['A', 'B']}
        """
        # 遍历传入的标签参数列表
        for label in args:
            # 如果标签不在成员字典中，则引发值错误异常
            if label not in self._members:
                raise ValueError("No such member exists in the Truss")

            else:
                # 从占用节点集合中移除该成员连接的节点对
                self._nodes_occupied.pop((self._members[label][0], self._members[label][1]))
                self._nodes_occupied.pop((self._members[label][1], self._members[label][0]))
                # 从成员字典中移除该成员
                self._members.pop(label)
                # 从成员长度字典中移除该成员的长度信息
                self._member_lengths.pop(label)
                # 从内部力字典中移除该成员的内部力信息
                self._internal_forces.pop(label)
    # 定义一个方法，用于修改指定成员的标签
    def change_member_label(self, *args):
        """
        This method changes the label(s) of the specified member(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (label, new_label)

        label: String or Symbol
            The label of the member for which the label has
            to be changed.

        new_label: String or Symbol
            The new label of the member.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('D', 5, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('D', 5, 0)]
        >>> t.change_node_label(('A', 'C'))
        >>> t.nodes
        [('C', 0, 0), ('B', 3, 0), ('D', 5, 0)]
        >>> t.add_member(('BC', 'B', 'C'), ('BD', 'B', 'D'))
        >>> t.members
        {'BC': ['B', 'C'], 'BD': ['B', 'D']}
        >>> t.change_member_label(('BC', 'BC_new'), ('BD', 'BD_new'))
        >>> t.members
        {'BC_new': ['B', 'C'], 'BD_new': ['B', 'D']}
        """
        # 遍历输入的参数元组列表
        for i in args:
            # 提取当前成员的原始标签和新标签
            label = i[0]
            new_label = i[1]
            # 如果原始标签不在_truss对象的成员字典中，引发值错误异常
            if label not in self._members:
                raise ValueError("No such member exists for the Truss")
            else:
                # 复制当前_truss对象成员字典的键列表
                members_duplicate = list(self._members).copy()
                # 遍历复制后的键列表
                for member in members_duplicate:
                    # 如果当前键等于原始标签
                    if member == label:
                        # 使用新标签替换原始标签，并更新成员相关的长度和内部力字典
                        self._members[new_label] = [self._members[member][0], self._members[member][1]]
                        self._members.pop(label)
                        self._member_lengths[new_label] = self._member_lengths[label]
                        self._member_lengths.pop(label)
                        self._internal_forces[new_label] = self._internal_forces[label]
                        self._internal_forces.pop(label)
    def apply_load(self, *args):
        """
        This method applies external load(s) at the specified node(s).

        Parameters
        ==========
        The input(s) of the method are tuple(s) of the form (location, magnitude, direction).

        location: String or Symbol
            Label of the Node at which load is applied.

        magnitude: Sympifyable
            Magnitude of the load applied. It must always be positive and any changes in
            the direction of the load are not reflected here.

        direction: Sympifyable
            The angle, in degrees, that the load vector makes with the horizontal
            in the counter-clockwise direction. It takes the values 0 to 360,
            inclusive.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> from sympy import symbols
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> P = symbols('P')
        >>> t.apply_load(('A', P, 90), ('A', P/2, 45), ('A', P/4, 90))
        >>> t.loads
        {'A': [[P, 90], [P/2, 45], [P/4, 90]]}
        """
        # 遍历传入的参数元组列表
        for i in args:
            # 获取加载位置、大小和方向
            location = i[0]
            magnitude = i[1]
            direction = i[2]
            # 将大小和方向转换为Sympy对象（如果可能）
            magnitude = sympify(magnitude)
            direction = sympify(direction)

            # 检查加载位置是否在已知节点中
            if location not in self._node_coordinates:
                # 如果位置不在已知节点中，抛出值错误异常
                raise ValueError("Load must be applied at a known node")
            else:
                # 如果位置在已知节点中
                if location in self._loads:
                    # 如果该位置已经有加载，将新的大小和方向添加到现有列表中
                    self._loads[location].append([magnitude, direction])
                else:
                    # 如果该位置没有加载，创建新的列表保存大小和方向
                    self._loads[location] = [[magnitude, direction]]
    def remove_load(self, *args):
        """
        This method removes already
        present external load(s) at specified node(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (location, magnitude, direction).

        location: String or Symbol
            Label of the Node at which load is applied and is to be removed.

        magnitude: Sympifyable
            Magnitude of the load applied.

        direction: Sympifyable
            The angle, in degrees, that the load vector makes with the horizontal
            in the counter-clockwise direction. It takes the values 0 to 360,
            inclusive.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> from sympy import symbols
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> P = symbols('P')
        >>> t.apply_load(('A', P, 90), ('A', P/2, 45), ('A', P/4, 90))
        >>> t.loads
        {'A': [[P, 90], [P/2, 45], [P/4, 90]]}
        >>> t.remove_load(('A', P/4, 90), ('A', P/2, 45))
        >>> t.loads
        {'A': [[P, 90]]}
        """
        # Iterate over each tuple of load specifications
        for i in args:
            # Extract location, magnitude, and direction from the tuple
            location = i[0]
            magnitude = i[1]
            direction = i[2]

            # Convert magnitude and direction to sympy expressions if they are not already
            magnitude = sympify(magnitude)
            direction = sympify(direction)

            # Check if the specified location exists in the node coordinates
            if location not in self._node_coordinates:
                # Raise an error if the location is not recognized
                raise ValueError("Load must be removed from a known node")
            else:
                # Check if the [magnitude, direction] pair exists in the loads dictionary
                if [magnitude, direction] not in self._loads[location]:
                    # Raise an error if the load with the specified magnitude and direction does not exist
                    raise ValueError("No load of this magnitude and direction has been applied at this node")
                else:
                    # Remove the [magnitude, direction] pair from the loads list of the specified location
                    self._loads[location].remove([magnitude, direction])

            # If there are no loads left for the location, remove the location entry from loads
            if self._loads[location] == []:
                self._loads.pop(location)
    def apply_support(self, *args):
        """
        This method adds a pinned or roller support at specified node(s).

        Parameters
        ==========
        The input(s) of this method are of the form (location, type).

        location: String or Symbol
            Label of the Node at which support is added.

        type: String
            Type of the support being provided at the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> t.apply_support(('A', 'pinned'), ('B', 'roller'))
        >>> t.supports
        {'A': 'pinned', 'B': 'roller'}
        """
        # Iterate over each argument provided
        for i in args:
            # Extract the location and type from the argument tuple
            location = i[0]
            type = i[1]
            
            # Check if the specified location is a known node in the truss
            if location not in self._node_coordinates:
                # Raise an error if the location is not a known node
                raise ValueError("Support must be added on a known node")

            else:
                # If the location is valid:
                if location not in self._supports:
                    # If support has not been added to this location yet:
                    if type == 'pinned':
                        # Apply horizontal and vertical reactions for a pinned support
                        self.apply_load((location, Symbol('R_'+str(location)+'_x'), 0))
                        self.apply_load((location, Symbol('R_'+str(location)+'_y'), 90))
                    elif type == 'roller':
                        # Apply vertical reaction for a roller support
                        self.apply_load((location, Symbol('R_'+str(location)+'_y'), 90))
                elif self._supports[location] == 'pinned':
                    # If support type at location is currently pinned:
                    if type == 'roller':
                        # Remove horizontal reaction if changing from pinned to roller
                        self.remove_load((location, Symbol('R_'+str(location)+'_x'), 0))
                elif self._supports[location] == 'roller':
                    # If support type at location is currently roller:
                    if type == 'pinned':
                        # Apply horizontal reaction if changing from roller to pinned
                        self.apply_load((location, Symbol('R_'+str(location)+'_x'), 0))
                
                # Update the support type for the location
                self._supports[location] = type
    # 定义方法，用于移除指定节点的支持条件

    for location in args:
        # 遍历传入的节点参数列表

        if location not in self._node_coordinates:
            # 如果节点不存在于节点坐标字典中，抛出数值错误异常
            raise ValueError("No such node exists in the Truss")

        elif location not in self._supports:
            # 如果节点不存在于支持条件字典中，抛出数值错误异常
            raise ValueError("No support has been added to the given node")

        else:
            # 否则，根据支持类型执行相应操作
            if self._supports[location] == 'pinned':
                # 如果支持类型为 pinned（铰支），移除节点的水平和垂直支反力
                self.remove_load((location, Symbol('R_'+str(location)+'_x'), 0))
                self.remove_load((location, Symbol('R_'+str(location)+'_y'), 90))
            elif self._supports[location] == 'roller':
                # 如果支持类型为 roller（滚轮支），移除节点的垂直支反力
                self.remove_load((location, Symbol('R_'+str(location)+'_y'), 90))
            # 移除节点的支持条件
            self._supports.pop(location)
    # 定义一个方法来绘制节点，需要一个用于替换的字典 subs_dict
    def _draw_nodes(self, subs_dict):
        # 存储节点标记信息的列表
        node_markers = []

        # 遍历所有节点坐标
        for node in self._node_coordinates:
            # 检查节点的第一个坐标元素是否是符号(Symbol)或数量(Quantity)类型
            if (type(self._node_coordinates[node][0]) in (Symbol, Quantity)):
                # 如果在 subs_dict 中找到了对应的替换值，则替换当前节点坐标中的第一个元素
                if self._node_coordinates[node][0] in subs_dict:
                    self._node_coordinates[node][0] = subs_dict[self._node_coordinates[node][0]]
                else:
                    # 如果 subs_dict 中没有找到对应的替换值，则抛出数值错误异常
                    raise ValueError("provided substituted dictionary is not adequate")
            elif (type(self._node_coordinates[node][0]) == Mul):
                # 如果节点坐标的第一个元素是乘法对象(Mul)，则分解成单独的对象进行处理
                objects = self._node_coordinates[node][0].as_coeff_Mul()
                for object in objects:
                    # 对每个对象检查是否是符号(Symbol)或数量(Quantity)类型
                    if type(object) in (Symbol, Quantity):
                        # 如果 subs_dict 为 None 或者对象不在 subs_dict 中，则抛出数值错误异常
                        if subs_dict == None or object not in subs_dict:
                            raise ValueError("provided substituted dictionary is not adequate")
                        else:
                            # 否则，用 subs_dict 中的值替换当前对象，并重新计算节点坐标的第一个元素
                            self._node_coordinates[node][0] /= object
                            self._node_coordinates[node][0] *= subs_dict[object]

            # 检查节点的第二个坐标元素是否是符号(Symbol)或数量(Quantity)类型
            if (type(self._node_coordinates[node][1]) in (Symbol, Quantity)):
                # 如果在 subs_dict 中找到了对应的替换值，则替换当前节点坐标中的第二个元素
                if self._node_coordinates[node][1] in subs_dict:
                    self._node_coordinates[node][1] = subs_dict[self._node_coordinates[node][1]]
                else:
                    # 如果 subs_dict 中没有找到对应的替换值，则抛出数值错误异常
                    raise ValueError("provided substituted dictionary is not adequate")
            elif (type(self._node_coordinates[node][1]) == Mul):
                # 如果节点坐标的第二个元素是乘法对象(Mul)，则分解成单独的对象进行处理
                objects = self._node_coordinates[node][1].as_coeff_Mul()
                for object in objects:
                    # 对每个对象检查是否是符号(Symbol)或数量(Quantity)类型
                    if type(object) in (Symbol, Quantity):
                        # 如果 subs_dict 为 None 或者对象不在 subs_dict 中，则抛出数值错误异常
                        if subs_dict == None or object not in subs_dict:
                            raise ValueError("provided substituted dictionary is not adequate")
                        else:
                            # 否则，用 subs_dict 中的值替换当前对象，并重新计算节点坐标的第二个元素
                            self._node_coordinates[node][1] /= object
                            self._node_coordinates[node][1] *= subs_dict[object]

        # 遍历所有节点坐标，并将节点的标记信息添加到 node_markers 列表中
        for node in self._node_coordinates:
            node_markers.append(
                {
                    'args':[[self._node_coordinates[node][0]], [self._node_coordinates[node][1]]],
                    'marker':'o',
                    'markersize':5,
                    'color':'black'
                }
            )

        # 返回所有节点的标记信息列表
        return node_markers
    # 定义一个方法来绘制支撑节点的标记
    def _draw_supports(self):
        # 存储支撑节点的标记信息的列表
        support_markers = []

        # 初始化用于计算边界的变量，设定为无穷大和负无穷大
        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        # 遍历所有节点坐标，更新最大和最小的 x 和 y 值
        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])

        # 计算 x 和 y 方向上的最大差值
        if abs(1.1*xmax - 0.8*xmin) > abs(1.1*ymax - 0.8*ymin):
            max_diff = 1.1*xmax - 0.8*xmin
        else:
            max_diff = 1.1*ymax - 0.8*ymin

        # 根据支撑类型绘制不同类型的支撑节点标记
        for node in self._supports:
            if self._supports[node] == 'pinned':
                # 添加固定支撑的标记信息
                support_markers.append(
                    {
                        'args': [
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1]]
                        ],
                        'marker': 6,
                        'markersize': 15,
                        'color': 'black',
                        'markerfacecolor': 'none'
                    }
                )
                # 添加固定支撑的方向标记信息
                support_markers.append(
                    {
                        'args': [
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1] - 0.035*max_diff]
                        ],
                        'marker': '_',
                        'markersize': 14,
                        'color': 'black'
                    }
                )

            elif self._supports[node] == 'roller':
                # 添加滚动支撑的标记信息
                support_markers.append(
                    {
                        'args': [
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1] - 0.02*max_diff]
                        ],
                        'marker': 'o',
                        'markersize': 11,
                        'color': 'black',
                        'markerfacecolor': 'none'
                    }
                )
                # 添加滚动支撑的方向标记信息
                support_markers.append(
                    {
                        'args': [
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1] - 0.0375*max_diff]
                        ],
                        'marker': '_',
                        'markersize': 14,
                        'color': 'black'
                    }
                )

        # 返回所有支撑节点的标记信息列表
        return support_markers
    # 定义一个私有方法 `_draw_loads`，用于生成载荷的注释信息列表
    def _draw_loads(self):
        # 初始化载荷注释列表
        load_annotations = []

        # 初始化坐标极限值，初始设定为负无穷大和正无穷大
        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        # 遍历节点坐标字典，更新最大和最小的 x、y 值
        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])

        # 计算最大差异值 max_diff，用于确定载荷注释的位置
        if abs(1.1*xmax - 0.8*xmin) > abs(1.1*ymax - 0.8*ymin):
            max_diff = 1.1*xmax - 0.8*xmin + 5
        else:
            max_diff = 1.1*ymax - 0.8*ymin + 5

        # 遍历节点载荷字典，生成每个载荷的注释信息
        for node in self._loads:
            for load in self._loads[node]:
                # 如果载荷为节点反力，则跳过不处理
                if load[0] in [Symbol('R_' + str(node) + '_x'), Symbol('R_' + str(node) + '_y')]:
                    continue
                # 获取节点的坐标信息
                x = self._node_coordinates[node][0]
                y = self._node_coordinates[node][1]
                # 计算载荷注释的位置和箭头的属性
                load_annotations.append(
                    {
                        'text': '',
                        'xy': (
                            x - math.cos(pi * load[1] / 180) * (max_diff / 100),
                            y - math.sin(pi * load[1] / 180) * (max_diff / 100)
                        ),
                        'xytext': (
                            x - (max_diff / 100 + abs(xmax - xmin) + abs(ymax - ymin)) * math.cos(pi * load[1] / 180) / 20,
                            y - (max_diff / 100 + abs(xmax - xmin) + abs(ymax - ymin)) * math.sin(pi * load[1] / 180) / 20
                        ),
                        'arrowprops': {'width': 1.5, 'headlength': 5, 'headwidth': 5, 'facecolor': 'black'}
                    }
                )
        
        # 返回所有载荷注释的列表
        return load_annotations
```