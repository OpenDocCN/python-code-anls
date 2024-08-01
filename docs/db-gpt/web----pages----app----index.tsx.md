# `.\DB-GPT-src\web\pages\app\index.tsx`

```py
import AppModal from '@/components/app/app-modal';
import AppCard from '@/components/app/app-card';
import { Button, Spin, Tabs, TabsProps } from 'antd';
import React, { useEffect, useState } from 'react';
import { apiInterceptors, getAppList } from '@/client/api';
import { IApp } from '@/types/app';
import { PlusOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import MyEmpty from '@/components/common/MyEmpty';

type TabKey = 'app' | 'collected';  // 定义 Tab 的键类型为 'app' 或 'collected'

type ModalType = 'edit' | 'add';  // 定义 Modal 的类型为 'edit' 或 'add'

export default function App() {
  const { t } = useTranslation();  // 获取国际化翻译函数 t

  const [open, setOpen] = useState<boolean>(false);  // 设置控制 Modal 是否打开的状态
  const [spinning, setSpinning] = useState<boolean>(false);  // 设置控制 Spin 组件加载状态的状态
  const [activeKey, setActiveKey] = useState<TabKey>('app');  // 设置当前激活的 Tab 键
  const [apps, setApps] = useState<IApp[]>([]);  // 设置应用列表的状态
  const [curApp, setCurApp] = useState<IApp>();  // 设置当前编辑的应用的状态
  const [modalType, setModalType] = useState<ModalType>('add');  // 设置当前 Modal 的类型

  const handleCreate = () => {  // 处理创建应用的事件
    setModalType('add');  // 设置 Modal 类型为 'add'
    setOpen(true);  // 打开 Modal
  };

  const handleCancel = () => {  // 处理取消 Modal 的事件
    setOpen(false);  // 关闭 Modal
  };

  const handleEdit = (app: any) => {  // 处理编辑应用的事件
    setModalType('edit');  // 设置 Modal 类型为 'edit'
    setCurApp(app);  // 设置当前编辑的应用
    setOpen(true);  // 打开 Modal
  };

  const handleTabChange = (activeKey: string) => {  // 处理 Tab 切换的事件
    setActiveKey(activeKey as TabKey);  // 设置当前激活的 Tab 键
    if (activeKey === 'collected') {  // 如果切换到 'collected' Tab
      initData({ is_collected: true });  // 初始化数据，传入参数 is_collected 为 true
    } else {  // 如果切换到 'app' Tab
      initData();  // 初始化数据，不传入参数
    }
  };

  const initData = async (params = {}) => {  // 异步初始化数据的函数
    setSpinning(true);  // 设置加载状态为 true
    const [error, data] = await apiInterceptors(getAppList(params));  // 使用 apiInterceptors 发起 API 请求获取数据
    if (error) {  // 如果请求出错
      setSpinning(false);  // 设置加载状态为 false
      return;  // 结束函数
    }
    if (!data) return;  // 如果没有数据，结束函数

    setApps(data.app_list || []);  // 更新应用列表数据
    setSpinning(false);  // 设置加载状态为 false
  };

  useEffect(() => {
    initData();  // 组件挂载时初始化数据
  }, []);

  const renderAppList = (data: { isCollected: boolean }) => {  // 渲染应用列表的函数
    const isNull = data.isCollected ? apps.every((item) => !item.is_collected) : apps.length === 0;  // 判断是否为空

    return (
      <div>
        {!data.isCollected && (  // 如果不是收藏 Tab
          <Button onClick={handleCreate} type="primary" className="mb-4" icon={<PlusOutlined />}>
            {t('create')}  // 按钮文本使用国际化的 'create' 字符串
          </Button>
        )}
        {!isNull ? (  // 如果列表不为空
          <div className=" w-full flex flex-wrap pb-0 gap-4">
            {apps.map((app, index) => {
              return <AppCard handleEdit={handleEdit} key={index} app={app} updateApps={initData} isCollected={activeKey === 'collected'} />;
            })}
          </div>
        ) : (
          <MyEmpty />  // 如果列表为空，显示空状态组件
        )}
      </div>
    );
  };

  const items: TabsProps['items'] = [  // Tabs 组件的配置项
    {
      key: 'app',
      label: t('App'),  // Tab 标签文本使用国际化的 'App' 字符串
      children: renderAppList({ isCollected: false }),  // 渲染应用列表
    },
    {
      key: 'collected',
      label: t('collected'),  // Tab 标签文本使用国际化的 'collected' 字符串
      children: renderAppList({ isCollected: true }),  // 渲染收藏应用列表
    },
  ];

  return (
    <>
      {/* 在加载状态下显示旋转图标 */}
      <Spin spinning={spinning}>
        {/* 定义一个高度占满屏幕的容器，包含页面内边距和垂直滚动条 */}
        <div className="h-screen w-full p-4 md:p-6 overflow-y-auto">
          {/* 显示一个标签页组件，初始激活的标签为"app"，提供标签项和标签切换处理函数 */}
          <Tabs defaultActiveKey="app" items={items} onChange={handleTabChange} />
          {/* 当 modal 打开时，渲染应用模态框组件 */}
          {open && (
            <AppModal
              // 根据 modalType 确定模态框类型是编辑还是新建
              app={modalType === 'edit' ? curApp : {}}
              type={modalType}
              // 更新应用数据后的回调函数
              updateApps={initData}
              open={open}
              // 取消模态框的处理函数
              handleCancel={handleCancel}
            />
          )}
        </div>
      </Spin>
    </>
}



# 这行代码表示一个单独的右花括号 '}'，用于结束某个代码块或函数的定义或语句块。
```