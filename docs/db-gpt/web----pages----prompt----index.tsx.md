# `.\DB-GPT-src\web\pages\prompt\index.tsx`

```py
import { useState, useEffect, useRef, Ref } from 'react';
import type { ColumnsType } from 'antd/es/table';  // 导入表格列类型定义
import type { FormInstance, MenuProps } from 'antd';  // 导入表单实例和菜单属性类型定义
import { Menu, Table, Button, Tooltip, Modal } from 'antd';  // 导入antd组件
import { PlusOutlined } from '@ant-design/icons';  // 导入加号图标
import GroupsIcon from '@mui/icons-material/Groups';  // 导入分组图标
import PersonIcon from '@mui/icons-material/Person';  // 导入个人图标
import { useTranslation } from 'react-i18next';  // 导入国际化翻译钩子
import { addPrompt, apiInterceptors, getPromptList, postScenes, updatePrompt } from '@/client/api';  // 导入API函数
import { IPrompt } from '@/types/prompt';  // 导入提示类型定义
import PromptForm from '@/components/prompt/prompt-form';  // 导入提示表单组件
import { TFunction } from 'i18next';  // 导入国际化翻译函数类型定义

const getItems = (t: TFunction) => [  // 定义获取菜单项的函数
  {
    label: t('Public') + ' Prompts',  // 公共提示的标签
    key: 'common',  // 键名为common
    icon: <GroupsIcon />,  // 使用分组图标
  },
  {
    label: t('Private') + ' Prompts',  // 私有提示的标签
    key: 'private',  // 键名为private
    icon: <PersonIcon />,  // 使用个人图标
  },
];

const getColumns = (t: TFunction, handleEdit: (prompt: IPrompt) => void): ColumnsType<IPrompt> => [  // 定义获取表格列的函数
  {
    title: t('Prompt_Info_Name'),  // 提示名称列的标题
    dataIndex: 'prompt_name',  // 数据索引为prompt_name
    key: 'prompt_name',  // 键名为prompt_name
  },
  {
    title: t('Prompt_Info_Scene'),  // 提示场景列的标题
    dataIndex: 'chat_scene',  // 数据索引为chat_scene
    key: 'chat_scene',  // 键名为chat_scene
  },
  {
    title: t('Prompt_Info_Sub_Scene'),  // 子场景列的标题
    dataIndex: 'sub_chat_scene',  // 数据索引为sub_chat_scene
    key: 'sub_chat_scene',  // 键名为sub_chat_scene
  },
  {
    title: t('Prompt_Info_Content'),  // 提示内容列的标题
    dataIndex: 'content',  // 数据索引为content
    key: 'content',  // 键名为content
    render: (content) => (  // 自定义渲染内容，使用Tooltip显示完整内容
      <Tooltip placement="topLeft" title={content}>
        {content}
      </Tooltip>
    ),
  },
  {
    title: t('Operation'),  // 操作列的标题
    dataIndex: 'operate',  // 数据索引为operate
    key: 'operate',  // 键名为operate
    render: (_, record) => (  // 自定义渲染内容，点击编辑按钮触发handleEdit函数
      <Button
        onClick={() => {
          handleEdit(record);
        }}
        type="primary"
      >
        {t('Edit')}  // 编辑按钮显示文字由国际化翻译
      </Button>
    ),
  },
];

type FormType = Ref<FormInstance<any>> | undefined;  // 定义表单引用类型

const Prompt = () => {
  const { t } = useTranslation();  // 使用国际化翻译钩子获取翻译函数t

  const [promptType, setPromptType] = useState<string>('common');  // 状态：提示类型，默认为'common'
  const [promptList, setPromptList] = useState<Array<IPrompt>>();  // 状态：提示列表
  const [loading, setLoading] = useState<boolean>(false);  // 状态：加载状态，默认为false
  const [prompt, setPrompt] = useState<IPrompt>();  // 状态：当前提示
  const [showModal, setShowModal] = useState<boolean>(false);  // 状态：模态框显示状态，默认为false
  const [scenes, setScenes] = useState<Array<Record<string, string>>>();  // 状态：场景列表

  const formRef = useRef<FormType>();  // 使用useRef创建表单引用

  const getPrompts = async () => {  // 异步获取提示列表函数
    setLoading(true);  // 设置加载状态为true
    const body = {  // 定义请求体参数
      prompt_type: promptType,
      current: 1,
      pageSize: 1000,
      hideOnSinglePage: true,
      showQuickJumper: true,
    };
    const [_, data] = await apiInterceptors(getPromptList(body));  // 调用API函数获取提示列表数据
    setPromptList(data!);  // 更新提示列表状态
    setLoading(false);  // 设置加载状态为false
  };

  const getScenes = async () => {  // 异步获取场景列表函数
    const [, res] = await apiInterceptors(postScenes());  // 调用API函数获取场景数据
    setScenes(res?.map((scene) => ({ value: scene.chat_scene, label: scene.scene_name })));  // 更新场景列表状态
  };

  const onFinish = async (newPrompt: IPrompt) => {  // 提交表单完成时的处理函数
    if (prompt) {  // 如果存在当前提示
      await apiInterceptors(updatePrompt({ ...newPrompt, prompt_type: promptType }));  // 调用API函数更新提示信息
      // 其他处理逻辑...
    }
    // 其他处理逻辑...
  };
  };

  // 编辑按钮的点击事件处理函数，设置当前要编辑的提示，并显示模态框
  const handleEditBtn = (prompt: IPrompt) => {
    setPrompt(prompt);
    setShowModal(true);
  };

  // 添加按钮的点击事件处理函数，显示模态框并清空当前提示
  const handleAddBtn = () => {
    setShowModal(true);
    setPrompt(undefined);
  };

  // 关闭模态框的函数，隐藏模态框
  const handleClose = () => {
    setShowModal(false);
  };

  // 菜单项点击事件处理函数，根据点击的菜单项设置当前提示类型
  const handleMenuChange: MenuProps['onClick'] = (e) => {
    const type = e.key;
    setPromptType(type);
  };

  useEffect(() => {
    // 在组件加载时获取对应类型的提示
    getPrompts();
  }, [promptType]);

  useEffect(() => {
    // 在组件加载时获取所有场景信息
    getScenes();
  }, []);

  return (
    <div>
      {/* 顶部菜单，根据当前的 promptType 渲染不同的菜单项 */}
      <Menu onClick={handleMenuChange} selectedKeys={[promptType]} mode="horizontal" items={getItems(t)} />
      <div className="px-6 py-4">
        {/* 按钮组，包括添加按钮和特定条件下的添加模板按钮 */}
        <div className="flex flex-row-reverse mb-4">
          <Button className="flex items-center" onClick={handleAddBtn}>
            <PlusOutlined />
            {t('Add')} Prompts
          </Button>
          {/* 当 promptType 为 'common' 时显示禁用状态的添加模板按钮 */}
          {promptType === 'common' && (
            <Button className="mr-2 flex items-center" disabled>
              <PlusOutlined />
              {t('Add')} Prompts {t('template')}
            </Button>
          )}
        </div>
        {/* 展示提示信息的表格 */}
        <Table
          columns={getColumns(t, handleEditBtn)}
          dataSource={promptList}
          loading={loading}
          rowKey={(record) => record.prompt_name}
          scroll={{ y: 600 }}
        />
      </div>
      {/* 编辑或添加提示信息的模态框 */}
      <Modal
        title={`${prompt ? t('Edit') : t('Add')} Prompts`}
        destroyOnClose
        open={showModal}
        onCancel={handleClose}
        cancelText={t('cancel')}
        okText={t('submit')}
        onOk={() => {
          // 提交表单的处理函数
          // @ts-ignore
          formRef.current?.submit();
        }}
      >
        {/* 提示信息表单组件 */}
        <PromptForm scenes={scenes} ref={formRef as FormType} prompt={prompt} onFinish={onFinish} />
      </Modal>
    </div>
  );
};

export default Prompt;
```