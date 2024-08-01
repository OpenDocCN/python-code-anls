# `.\DB-GPT-src\web\pages\flow\canvas\index.tsx`

```py
import { addFlow, apiInterceptors, getFlowById, updateFlowById } from '@/client/api';
import MuiLoading from '@/components/common/loading';
import AddNodes from '@/components/flow/add-nodes';
import ButtonEdge from '@/components/flow/button-edge';
import CanvasNode from '@/components/flow/canvas-node';
import { IFlowData, IFlowUpdateParam } from '@/types/flow';
import { checkFlowDataRequied, getUniqueNodeId, mapHumpToUnderline, mapUnderlineToHump } from '@/utils/flow';
import { FrownOutlined, SaveOutlined } from '@ant-design/icons';
import { Button, Checkbox, Divider, Form, Input, Modal, Space, message, notification } from 'antd';
import { useSearchParams } from 'next/navigation';
import React, { DragEvent, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import ReactFlow, { Background, Connection, Controls, ReactFlowProvider, addEdge, useEdgesState, useNodesState, useReactFlow, Node } from 'reactflow';
import 'reactflow/dist/style.css';

const { TextArea } = Input;

interface Props {
  // Define your component props here
}
const nodeTypes = { customNode: CanvasNode };
const edgeTypes = { buttonedge: ButtonEdge };

const Canvas: React.FC<Props> = () => {
  const { t } = useTranslation();
  const [messageApi, contextHolder] = message.useMessage();
  const [form] = Form.useForm<IFlowUpdateParam>();
  const searchParams = useSearchParams();
  const id = searchParams?.get('id') || '';
  const reactFlow = useReactFlow();

  const [loading, setLoading] = useState(false);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [flowInfo, setFlowInfo] = useState<IFlowUpdateParam>();
  const [deploy, setDeploy] = useState(true);

  // 异步函数，获取流程数据
  async function getFlowData() {
    setLoading(true);
    // 发送请求获取流程数据
    const [_, data] = await apiInterceptors(getFlowById(id));
    if (data) {
      // 将下划线命名转换为驼峰命名
      const flowData = mapUnderlineToHump(data.flow_data);
      setFlowInfo(data);
      setNodes(flowData.nodes);
      setEdges(flowData.edges);
    }
    setLoading(false);
  }

  // 当 id 变化时，调用获取流程数据函数
  useEffect(() => {
    id && getFlowData();
  }, [id]);

  // 监听页面关闭事件，提示用户保存未提交的数据
  useEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      event.returnValue = message;
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  // 处理节点点击事件
  function onNodesClick(event: any, clickedNode: Node) {
    reactFlow.setNodes((nds) =>
      nds.map((node) => {
        if (node.id === clickedNode.id) {
          node.data = {
            ...node.data,
            selected: true,
          };
        } else {
          node.data = {
            ...node.data,
            selected: false,
          };
        }
        return node;
      }),
  );

  // 连接建立时的回调函数，添加新的边到图表中
  function onConnect(connection: Connection) {
    // 创建一个新的边对象，继承现有连接信息并添加额外的类型和ID信息
    const newEdge = {
      ...connection,
      type: 'buttonedge',
      id: `${connection.source}|${connection.target}`,
    };
    // 更新边的状态，将新边添加到边数组中
    setEdges((eds) => addEdge(newEdge, eds));
  }

  // 拖放事件的回调函数，处理拖放到流程图区域内的操作
  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();
      // 获取React Flow组件的界限信息
      const reactFlowBounds = reactFlowWrapper.current!.getBoundingClientRect();
      // 从拖放事件中获取节点数据的序列化字符串
      let nodeStr = event.dataTransfer.getData('application/reactflow');
      // 如果节点字符串为空或未定义，直接返回
      if (!nodeStr || typeof nodeStr === 'undefined') {
        return;
      }
      // 将节点数据反序列化为JavaScript对象
      const nodeData = JSON.parse(nodeStr);
      // 计算节点在流程图中的位置
      const position = reactFlow.screenToFlowPosition({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });
      // 获取唯一的节点ID
      const nodeId = getUniqueNodeId(nodeData, reactFlow.getNodes());
      // 设置节点ID，并创建新节点对象
      nodeData.id = nodeId;
      const newNode = {
        id: nodeId,
        position,
        type: 'customNode',
        data: nodeData,
      };
      // 更新节点状态，将新节点添加到节点数组中，并更新节点的选中状态
      setNodes((nds) =>
        nds.concat(newNode).map((node) => {
          if (node.id === newNode.id) {
            node.data = {
              ...node.data,
              selected: true,
            };
          } else {
            node.data = {
              ...node.data,
              selected: false,
            };
          }
          return node;
        }),
      );
    },
    [reactFlow],
  );

  // 拖放区域的悬停事件处理函数，设置允许的拖放效果
  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // 标签输入框内容变化时的处理函数
  function labelChange(e: React.ChangeEvent<HTMLInputElement>) {
    const label = e.target.value;
    // 将空格替换为下划线，将大写字母转换为小写，移除除了数字、字母、下划线和破折号之外的字符
    let result = label
      .replace(/\s+/g, '_')
      .replace(/[^a-z0-9_-]/g, '')
      .toLowerCase();
    // 更新表单中名称字段的值
    form.setFieldsValue({ name: result });
  }

  // 点击保存按钮时的处理函数
  function clickSave() {
    // 获取流程图的数据对象
    const flowData = reactFlow.toObject() as IFlowData;
    // 检查流程图数据的必填项和有效性，返回检查结果、节点对象和错误信息
    const [check, node, message] = checkFlowDataRequied(flowData);
    // 如果检查未通过且有错误信息
    if (!check && message) {
      // 更新节点状态，将出错的节点标记为无效，并显示错误通知
      setNodes((nds) =>
        nds.map((item) => {
          if (item.id === node?.id) {
            item.data = {
              ...item.data,
              invalid: true,
            };
          } else {
            item.data = {
              ...item.data,
              invalid: false,
            };
          }
          return item;
        }),
      );
      // 显示错误通知
      return notification.error({ message: 'Error', description: message, icon: <FrownOutlined className="text-red-600" /> });
    }
    // 显示保存流程的模态框
    setIsModalVisible(true);
  }

  // 处理保存流程的异步函数
  async function handleSaveFlow() {
    // 获取表单中的名称、标签和描述等字段的值
    const { name, label, description = '', editable = false, state = 'deployed' } = form.getFieldsValue();
    console.log(form.getFieldsValue());
    // 将流程图数据对象中的驼峰命名转换为下划线命名
    const reactFlowObject = mapHumpToUnderline(reactFlow.toObject() as IFlowData);
    if (id) {
      // 如果存在id，表示更新流程
      const [, , res] = await apiInterceptors(updateFlowById(id, { name, label, description, editable, uid: id, flow_data: reactFlowObject, state }));
      // 关闭编辑模态框
      setIsModalVisible(false);
      // 如果更新成功，显示保存成功消息
      if (res?.success) {
        messageApi.success(t('save_flow_success'));
      } else if (res?.err_msg) {
        // 如果更新失败，显示错误消息
        messageApi.error(res?.err_msg);
      }
    } else {
      // 如果不存在id，表示新建流程
      const [_, res] = await apiInterceptors(addFlow({ name, label, description, editable, flow_data: reactFlowObject, state }));
      // 如果成功创建流程
      if (res?.uid) {
        // 显示保存成功消息
        messageApi.success(t('save_flow_success'));
        // 更新浏览器历史记录到新创建流程页面
        const history = window.history;
        history.pushState(null, '', `/flow/canvas?id=${res.uid}`);
      }
      // 关闭编辑模态框
      setIsModalVisible(false);
    }
};

export default function CanvasWrapper() {
  // 返回一个组件，提供 React Flow 的上下文环境
  return (
    <ReactFlowProvider>
      // 渲染 Canvas 组件，该组件受 React Flow 上下文提供的影响
      <Canvas />
    </ReactFlowProvider>
  );
}
```