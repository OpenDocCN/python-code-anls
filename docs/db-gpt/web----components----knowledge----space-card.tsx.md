# `.\DB-GPT-src\web\components\knowledge\space-card.tsx`

```py
import { Popover, ConfigProvider, Modal, Badge } from 'antd';  // 导入 Ant Design 的 Popover、ConfigProvider、Modal 和 Badge 组件
import { useRouter } from 'next/router';  // 导入 Next.js 的 useRouter 钩子函数，用于获取路由对象
import Image from 'next/image';  // 导入 Next.js 的 Image 组件，用于优化图像加载
import { ClockCircleOutlined, DeleteFilled, MessageFilled, UserOutlined, WarningOutlined } from '@ant-design/icons';  // 导入 Ant Design 的一些图标组件
import { ISpace } from '@/types/knowledge';  // 导入名为 ISpace 的类型定义，位于 '@/types/knowledge' 模块
import DocPanel from './doc-panel';  // 导入当前目录下的 DocPanel 组件
import moment from 'moment';  // 导入 moment 库，用于日期时间处理
import { apiInterceptors, delSpace, newDialogue } from '@/client/api';  // 导入名为 apiInterceptors、delSpace 和 newDialogue 的函数，位于 '@/client/api' 模块
import { useTranslation } from 'react-i18next';  // 导入 react-i18next 库的 useTranslation 钩子函数，用于多语言国际化支持
import GptCard from '../common/gpt-card';  // 导入上级目录下的 common 目录中的 gpt-card 组件

interface IProps {  // 定义接口 IProps，描述组件的 props 结构
  space: ISpace;  // 接收一个名为 space 的 ISpace 类型参数
  onAddDoc: (spaceName: string) => void;  // 接收一个返回值为 void 的函数 onAddDoc，参数为 spaceName: string
  getSpaces: () => void;  // 接收一个返回值为 void 的函数 getSpaces，用于获取空间列表
}

const { confirm } = Modal;  // 从 Modal 中解构出 confirm 方法

export default function SpaceCard(props: IProps) {  // 定义默认导出的 SpaceCard 组件，接收类型为 IProps 的 props 参数
  const router = useRouter();  // 使用 useRouter 钩子函数获取路由对象
  const { t } = useTranslation();  // 使用 useTranslation 钩子函数获取翻译函数 t
  const { space, getSpaces } = props;  // 从 props 中解构出 space 和 getSpaces 方法

  const showDeleteConfirm = () => {  // 定义 showDeleteConfirm 方法，显示删除确认对话框
    confirm({  // 调用 Modal.confirm 方法显示确认对话框
      title: t('Tips'),  // 设置对话框标题为国际化后的 'Tips'
      icon: <WarningOutlined />,  // 设置对话框图标为警告图标
      content: `${t('Del_Knowledge_Tips')}?`,  // 设置对话框内容为国际化后的 'Del_Knowledge_Tips' 加上问号
      okText: 'Yes',  // 设置确认按钮文本为 'Yes'
      okType: 'danger',  // 设置确认按钮类型为危险样式
      cancelText: 'No',  // 设置取消按钮文本为 'No'
      async onOk() {  // 设置确认按钮点击后的异步处理函数
        await apiInterceptors(delSpace({ name: space?.name }));  // 调用 apiInterceptors 函数发送删除空间请求
        getSpaces();  // 调用 getSpaces 函数更新空间列表
      },
    });
  };

  function onDeleteDoc() {  // 定义 onDeleteDoc 方法，用于处理文档删除操作
    getSpaces();  // 调用 getSpaces 函数更新空间列表
  }

  const handleChat = async () => {  // 定义 handleChat 方法，处理与用户的聊天交互
    const [_, data] = await apiInterceptors(  // 调用 apiInterceptors 函数获取对话数据
      newDialogue({  // 调用 newDialogue 函数创建新对话
        chat_mode: 'chat_knowledge',  // 设置对话模式为 'chat_knowledge'
      }),
    );
    if (data?.conv_uid) {  // 如果返回的数据中包含对话 ID
      router.push(`/chat?scene=chat_knowledge&id=${data?.conv_uid}&db_param=${space.name}`);  // 使用路由对象跳转到聊天页面，并传递对话 ID 和空间名称参数
    }
  };

  return (  // 返回 JSX 结构，渲染 SpaceCard 组件
    <ConfigProvider  // 使用 ConfigProvider 组件包裹下面的内容，并设置主题配置
      theme={{  // 设置主题对象
        components: {  // 定义组件配置
          Popover: {  // 针对 Popover 组件的配置
            zIndexPopup: 90,  // 设置弹出层的 z-index 为 90
          },
        },
      }}
    <Popover
      // 创建一个弹出框组件，用于显示文档面板
      className="cursor-pointer"
      // 设置组件样式类，添加光标指针样式
      placement="bottom"
      // 弹出框显示在目标元素下方
      trigger="click"
      // 触发弹出框的事件为点击
      content={<DocPanel space={space} onAddDoc={props.onAddDoc} onDeleteDoc={onDeleteDoc} />}
      // 弹出框的内容为文档面板组件，传入相应的参数
    >
      <Badge className="mb-4 min-w-[200px] sm:w-60 lg:w-72" count={space.docs || 0}>
        {/* 
          创建一个带徽章的组件，用于展示空间（space）的文档数量。
          mb-4：设置下边距为4
          min-w-[200px] sm:w-60 lg:w-72：设置最小宽度和在不同屏幕大小下的宽度
        */}
        <GptCard
          // 创建一个卡片组件，展示空间（space）的相关信息
          title={space.name}
          // 设置卡片标题为空间的名称
          desc={space.desc}
          // 设置卡片描述为空间的描述信息
          icon={
            // 根据空间的领域类型和向量类型选择不同的图标
            space.domain_type === 'FinancialReport'
              ? '/models/fin_report.jpg'
              : space.vector_type === 'KnowledgeGraph'
              ? '/models/knowledge-graph.png'
              : space.vector_type === 'FullText'
              ? '/models/knowledge-full-text.jpg'
              : '/models/knowledge-default.jpg'
          }
          // 根据空间的领域类型和向量类型设置卡片图标
          iconBorder={false}
          // 不显示卡片图标的边框
          tags={[
            // 设置卡片的标签，包括所有者和修改时间
            {
              text: (
                <>
                  <UserOutlined className="mr-1" />
                  {space?.owner}
                </>
              ),
            },
            {
              text: (
                <>
                  <ClockCircleOutlined className="mr-1" />
                  {moment(space.gmt_modified).format('YYYY-MM-DD')}
                </>
              ),
            },
          ]}
          operations={[
            // 设置卡片的操作按钮，包括聊天和删除功能
            {
              label: t('Chat'),
              // 设置按钮文本为多语言翻译后的"Chat"
              children: <MessageFilled />,
              // 设置按钮图标为消息填充图标
              onClick: handleChat,
              // 点击按钮触发 handleChat 函数
            },
            {
              label: t('Delete'),
              // 设置按钮文本为多语言翻译后的"Delete"
              children: <DeleteFilled />,
              // 设置按钮图标为删除填充图标
              onClick: () => {
                showDeleteConfirm();
                // 点击按钮时显示删除确认对话框
              },
            },
          ]}
        />
      </Badge>
    </Popover>
}



# 这行代码关闭了一个代码块，对应之前的开放的代码块，不做其他操作。
```