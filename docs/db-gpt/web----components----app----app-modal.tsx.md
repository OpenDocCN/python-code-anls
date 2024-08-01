# `.\DB-GPT-src\web\components\app\app-modal.tsx`

```py
import { AgentParams, IAgent as IAgentParams, IApp, IDetail } from '@/types/app';
import { Dropdown, Form, Input, Modal, Select, Space, Spin, Tabs } from 'antd';
import React, { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import AddIcon from '../icons/add-icon';
import AgentPanel from './agent-panel';
import { addApp, apiInterceptors, getAgents, getResourceType, getTeamMode, updateApp } from '@/client/api';
import type { TabsProps } from 'antd';
import DagLayout from './dag-layout';
import { IFlow } from '@/types/flow';

type TargetKey = string;

type FieldType = {
  app_name: string;
  app_describe: string;
  language: string;
  team_mode: string;
};

type IAgent = {
  label: string;
  children?: React.ReactNode;
  onClick?: () => void;
  key: number | string;
};

interface IProps {
  handleCancel: () => void;
  open: boolean;
  updateApps: () => void;
  type: string;
  app?: any;
}

type TeamModals = 'awel_layout' | 'singe_agent' | 'auto_plan';

export default function AppModal(props: IProps) {
  // 获取组件属性
  const { handleCancel, open, updateApps, type, app } = props;

  // 使用国际化翻译
  const { t } = useTranslation();

  // 状态管理：是否加载中
  const [spinning, setSpinning] = useState<boolean>(false);

  // 状态管理：当前激活的标签页
  const [activeKey, setActiveKey] = useState<string>();

  // 状态管理：团队模态框选项
  const [teamModal, setTeamModal] = useState<{ label: string; value: string }[]>();

  // 状态管理：代理列表
  const [agents, setAgents] = useState<TabsProps['items']>([]);

  // 状态管理：下拉菜单项
  const [dropItems, setDropItems] = useState<IAgent[]>([]);

  // 状态管理：详情数据
  const [details, setDetails] = useState<IDetail[]>([...(app?.details || [])]);

  // 状态管理：流程数据
  const [flow, setFlow] = useState<IFlow>();

  // 状态管理：资源类型列表
  const [resourceTypes, setResourceTypes] = useState<string[]>();

  // 状态管理：当前团队模态框选择
  const [curTeamModal, setCurTeamModal] = useState<TeamModals>(app.team_modal || 'auto_plan');

  // 使用表单实例
  const [form] = Form.useForm();

  // 语言选项列表
  const languageOptions = [
    { value: 'zh', label: t('Chinese') },
    { value: 'en', label: t('English') },
  ];

  // 标签页变更处理函数
  const onChange = (newActiveKey: string) => {
    setActiveKey(newActiveKey);
  };

  // 创建或更新应用
  const createApp = async (app: IApp) => {
    await apiInterceptors(type === 'add' ? addApp(app) : updateApp(app));
    await updateApps();
  };

  // 初始化应用数据
  const initApp = async () => {
    // 获取应用详情
    const appDetails = app.details;

    // 获取资源类型列表
    const [_, resourceType] = await apiInterceptors(getResourceType());
    // 如果 appDetails 数组存在并且长度大于 0，则执行以下逻辑
    if (appDetails?.length > 0) {
      // 设置 Agents 状态，通过映射转换每个 AgentParams 对象为特定格式的对象
      setAgents(
        appDetails?.map((item: AgentParams) => {
          return {
            // 设置 label 字段为 agent_name 属性值
            label: item?.agent_name,
            // 设置 children 字段为包含 AgentPanel 组件的对象
            children: (
              <AgentPanel
                // 如果类型为 'edit'，则将 item.resources 传递给 editResources 属性
                editResources={type === 'edit' && item.resources}
                // 设置 detail 对象，包含多个属性值的映射
                detail={{
                  key: item?.agent_name,
                  llm_strategy: item?.llm_strategy,
                  agent_name: item?.agent_name,
                  prompt_template: item?.prompt_template,
                  llm_strategy_value: item?.llm_strategy_value,
                }}
                // 将 updateDetailsByAgentKey 方法传递给 updateDetailsByAgentKey 属性
                updateDetailsByAgentKey={updateDetailsByAgentKey}
                // 将 resourceType 变量传递给 resourceTypes 属性
                resourceTypes={resourceType}
              />
            ),
            // 设置 key 字段为 agent_name 属性值
            key: item?.agent_name,
          };
        }),
      );
    }
  };

  // 异步函数 fetchTeamModal，用于获取团队模态框数据
  const fetchTeamModal = async () => {
    // 调用 apiInterceptors 方法获取团队模态框数据
    const [_, data] = await apiInterceptors(getTeamMode());
    // 如果没有返回数据，则返回 null
    if (!data) return null;

    // 将 data 数组映射为包含 value 和 label 属性的对象数组，并设置为 teamModal 状态
    const teamModalOptions = data.map((item) => {
      return { value: item, label: item };
    });
    setTeamModal(teamModalOptions);
  };

  // 异步函数 fetchAgent，用于获取代理数据
  const fetchAgent = async () => {
    // 调用 apiInterceptors 方法获取代理数据
    const [_, data] = await apiInterceptors(getAgents());
    // 如果没有返回数据，则返回 null
    if (!data) {
      return null;
    }

    // 将 data 数组映射为包含 label、key、onClick 和 agent 属性的对象数组，并设置为 dropItems 状态
    setDropItems(
      data
        .map((agent) => {
          return {
            label: agent.name,
            key: agent.name,
            onClick: () => {
              // 点击事件将代理对象作为参数传递给 add 方法
              add(agent);
            },
            agent,
          };
        })
        // 过滤掉在 app.details 中已存在的代理对象
        .filter((item) => {
          if (!app.details || app.details?.length === 0) {
            return item;
          }
          // 仅保留 app.details 中不存在的代理对象
          return app?.details?.every((detail: AgentParams) => detail.agent_name !== item.label);
        }),
    );
  };

  // handleFlowsChange 方法，用于处理流程改变时更新 flow 状态
  const handleFlowsChange = (data: IFlow) => {
    setFlow(data);
  };

  // 异步函数 fetchResourceType，用于获取资源类型数据
  const fetchResourceType = async () => {
    // 调用 apiInterceptors 方法获取资源类型数据
    const [_, data] = await apiInterceptors(getResourceType());
    // 如果有返回数据，则设置为 resourceTypes 状态
    if (data) {
      setResourceTypes(data);
    }
  };

  // useEffect Hook，在组件挂载后执行一次 fetchTeamModal、fetchAgent 和 fetchResourceType 方法
  useEffect(() => {
    fetchTeamModal();
    fetchAgent();
    fetchResourceType();
  }, []);

  // useEffect Hook，当 resourceTypes 状态改变时执行一次 initApp 方法（如果类型为 'edit'）
  useEffect(() => {
    type === 'edit' && initApp();
  }, [resourceTypes]);

  // useEffect Hook，当 app 状态改变时更新 curTeamModal 状态为 app.team_mode 或 'auto_plan'
  useEffect(() => {
    setCurTeamModal(app.team_mode || 'auto_plan');
  }, [app]);

  // updateDetailsByAgentKey 方法，根据 key 更新 details 状态数组中对应的数据
  const updateDetailsByAgentKey = (key: string, data: IDetail) => {
    setDetails((details: IDetail[]) => {
      return details.map((detail: IDetail) => {
        // 如果 key 等于 detail.agent_name 或 detail.key，则更新为传入的 data
        return key === (detail.agent_name || detail.key) ? data : detail;
      });
    });
  };

  // 异步函数 add，向 details 状态数组中添加一个新的代理对象
  const add = async (tabBar: IAgentParams) => {
    // 设置新的激活 key 为 tabBar.name
    const newActiveKey = tabBar.name;
    // 再次调用 apiInterceptors 方法获取资源类型数据
    const [_, data] = await apiInterceptors(getResourceType());

    // 设置当前激活 key 为 newActiveKey
    setActiveKey(newActiveKey);

    // 在 details 状态数组末尾添加一个新的对象，包含 key、name、llm_strategy 属性
    setDetails((details: IDetail[]) => {
      return [...details, { key: newActiveKey, name: '', llm_strategy: 'priority' }];
    });
  };
    // 更新代理列表状态，添加新的代理项
    setAgents((items: any) => {
      return [
        ...items,
        {
          label: newActiveKey,
          children: (
            <AgentPanel
              detail={{ key: newActiveKey, llm_strategy: 'default', agent_name: newActiveKey, prompt_template: '', llm_strategy_value: null }}
              updateDetailsByAgentKey={updateDetailsByAgentKey}
              resourceTypes={data}
            />
          ),
          key: newActiveKey,
        },
      ];
    });

    // 更新下拉菜单项状态，移除指定键名的项
    setDropItems((items) => {
      return items.filter((item) => item.key !== tabBar.name);
    });
  };

  // 移除代理项
  const remove = (targetKey: TargetKey) => {
    let newActiveKey = activeKey;
    let lastIndex = -1;

    // 如果代理列表为空，则返回
    if (!agents) {
      return null;
    }

    // 查找目标键名在代理列表中的位置
    agents.forEach((item, i) => {
      if (item.key === targetKey) {
        lastIndex = i - 1;
      }
    });

    // 过滤掉目标键名对应的代理项
    const newPanes = agents.filter((item) => item.key !== targetKey);

    // 如果仍有剩余代理项且当前激活项是被移除的项，则更新激活项
    if (newPanes.length && newActiveKey === targetKey) {
      if (lastIndex >= 0) {
        newActiveKey = newPanes[lastIndex].key;
      } else {
        newActiveKey = newPanes[0].key;
      }
    }

    // 更新详情列表状态，移除与目标键名不匹配的项
    setDetails((details: IDetail[]) => {
      return details?.filter((detail: any) => {
        return (detail.agent_name || detail.key) !== targetKey;
      });
    });

    // 更新代理列表状态，设置为过滤后的新代理项列表
    setAgents(newPanes);

    // 更新激活的代理项键名
    setActiveKey(newActiveKey);

    // 更新下拉菜单项状态，添加目标键名为新项
    setDropItems((items: any) => {
      return [
        ...items,
        {
          label: targetKey,
          key: targetKey,
          onClick: () => {
            add({ name: targetKey, describe: '', system_message: '' });
          },
        },
      ];
    });
  };

  // 处理标签页编辑操作
  const onEdit = (targetKey: any, action: 'add' | 'remove') => {
    if (action === 'add') {
      // 添加操作暂时略过
    } else {
      // 执行移除操作
      remove(targetKey);
    }
  };

  // 提交表单处理函数
  const handleSubmit = async () => {
    // 表单验证
    const isValidate = await form.validateFields();

    // 如果验证不通过，则直接返回
    if (!isValidate) {
      return;
    }

    // 设置加载状态为 true
    setSpinning(true);

    // 获取表单数据
    const data = {
      ...form.getFieldsValue(),
    };

    // 如果是编辑模式，则添加 app_code 到数据中
    if (type === 'edit') {
      data.app_code = app.app_code;
    }

    // 如果团队模式不是 'awel_layout'，则将详情列表添加到数据中
    if (data.team_mode !== 'awel_layout') {
      data.details = details;
    } else {
      // 否则，复制流对象，并删除 flow_data 字段，添加到 team_context 中
      const tempFlow = { ...flow };
      delete tempFlow.flow_data;
      data.team_context = tempFlow;
    }

    try {
      // 调用创建应用的异步函数
      await createApp(data);
    } catch (error) {
      // 捕获错误，不做处理，直接返回
      return;
    }

    // 设置加载状态为 false
    setSpinning(false);

    // 处理取消操作
    handleCancel();
  };

  // 处理团队模态框变化
  const handleTeamModalChange = (value: TeamModals) => {
    setCurTeamModal(value);
  };

  // 渲染添加图标
  const renderAddIcon = () => {
    return (
      // 下拉菜单渲染
      <Dropdown menu={{ items: dropItems }} trigger={['click']}>
        {/* 点击图标时阻止默认行为 */}
        <a className="h-8 flex items-center" onClick={(e) => e.preventDefault()}>
          {/* 渲染添加图标 */}
          <Space>
            <AddIcon />
          </Space>
        </a>
      </Dropdown>
    );
  };

  // 返回 JSX 组件
  return (
    <div>
      <Modal
        // 设置模态框的确定按钮文本为国际化后的 "Submit"
        okText={t('Submit')}
        // 根据编辑类型设置模态框的标题，编辑时为 "edit_application"，添加时为 "add_application"
        title={type === 'edit' ? t('edit_application') : t('add_application')}
        // 控制模态框的显示与隐藏状态
        open={open}
        // 设置模态框宽度为 65%
        width={'65%'}
        // 点击模态框右上角的取消按钮时的回调函数
        onCancel={handleCancel}
        // 点击模态框右上角的确定按钮时的回调函数
        onOk={handleSubmit}
        // 关闭模态框时销毁内容
        destroyOnClose={true}
      >
        <Spin spinning={spinning}>
          <Form
            // 表单对象，用于管理表单数据
            form={form}
            // 提交表单后不保留已填写的值
            preserve={false}
            // 设置表单的大小为 "large"
            size="large"
            // 设置表单样式类，包括边距、最大高度和滚动条处理
            className="mt-4 max-h-[70vh] overflow-auto h-[90vh]"
            // 设置表单布局为水平排列
            layout="horizontal"
            // 设置表单标签对齐方式为左对齐
            labelAlign="left"
            // 设置表单标签的列属性，每个标签占据 4 格
            labelCol={{ span: 4 }}
            // 初始化表单的默认值，包括应用名称、应用描述、语言和团队模式
            initialValues={{
              app_name: app.app_name,
              app_describe: app.app_describe,
              language: app.language || languageOptions[0].value,
              team_mode: app.team_mode || 'auto_plan',
            }}
            // 关闭浏览器自动完成表单
            autoComplete="off"
            // 表单提交时的回调函数
            onFinish={handleSubmit}
          >
            <Form.Item<FieldType> label={t('app_name')} name="app_name" rules={[{ required: true, message: t('Please_input_the_name') }]}>
              {/* 应用名称输入框 */}
              <Input placeholder={t('Please_input_the_name')} />
            </Form.Item>
            <Form.Item<FieldType>
              // 应用描述标签
              label={t('Description')}
              // 表单项的名称
              name="app_describe"
              // 输入规则，必填，如果未填写则显示国际化提示信息
              rules={[{ required: true, message: t('Please_input_the_description') }]}
            >
              {/* 多行文本框，用于输入应用描述 */}
              <Input.TextArea rows={3} placeholder={t('Please_input_the_description')} />
            </Form.Item>
            <div className="flex w-full">
              <Form.Item<FieldType> labelCol={{ span: 7 }} label={t('language')} name="language" className="w-1/2" rules={[{ required: true }]}>
                {/* 语言选择下拉框 */}
                <Select className="w-2/3 ml-4" placeholder={t('language_select_tips')} options={languageOptions} />
              </Form.Item>
              <Form.Item<FieldType> label={t('team_modal')} name="team_mode" className="w-1/2" labelCol={{ span: 6 }} rules={[{ required: true }]}>
                {/* 团队模式选择下拉框 */}
                <Select
                  defaultValue={app.team_mode || 'auto_plan'}
                  className="ml-4 w-72"
                  onChange={handleTeamModalChange}
                  placeholder={t('Please_input_the_work_modal')}
                  options={teamModal}
                />
              </Form.Item>
            </div>
            {/* 根据当前团队模式选择显示不同的内容 */}
            {curTeamModal !== 'awel_layout' ? (
              <>
                {/* 显示 "Agents" 标题 */}
                <div className="mb-5">{t('Agents')}</div>
                {/* 可编辑选项卡，用于显示代理人信息 */}
                <Tabs addIcon={renderAddIcon()} type="editable-card" onChange={onChange} activeKey={activeKey} onEdit={onEdit} items={agents} />
              </>
            ) : (
              {/* DAG 布局组件，用于显示流程图布局 */}
              <DagLayout onFlowsChange={handleFlowsChange} teamContext={app.team_context} />
            )}
          </Form>
        </Spin>
      </Modal>
    </div>
  );
}



# 这行代码表示一个函数的结束。
```