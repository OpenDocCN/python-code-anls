# ArknightMower源码解析 13

# `ui/src/stores/mower.js`

This is a Node.js server that allows users to interact with a task scheduler. The scheduler takes tasks from a list and sends them to the specified endpoint with a specified format. The endpoint can also return the current running tasks.

The server has a WebSocket connection that listens for messages from the scheduler, and a log that displays the current running tasks. The log is stored in a buffer that resets after 500 lines.

When a new task is received, the server splits it up into its parts and extracts the relevant information, such as the task type and the scheduled time. It then formats the task text and wraps it in a string that includes the time and the task type.

The server is able to be configured to use a custom Logger and the file system for storage.

It also has a `get_running` function that retrieves the current running tasks from the API.


```py
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import ReconnectingWebSocket from 'reconnecting-websocket'

import axios from 'axios'

export const useMowerStore = defineStore('mower', () => {
  const log_lines = ref([])

  const log = computed(() => {
    return log_lines.value.join('\n')
  })

  const ws = ref(null)
  const running = ref(false)

  const first_load = ref(true)

  const task_list = ref([])

  function listen_ws() {
    let backend_url
    if (import.meta.env.DEV) {
      backend_url = import.meta.env.VITE_HTTP_URL
    } else {
      backend_url = location.origin
    }
    const ws_url = backend_url.replace(/^http/, 'ws') + '/log'
    ws.value = new ReconnectingWebSocket(ws_url)
    ws.value.onmessage = (event) => {
      log_lines.value = log_lines.value.concat(event.data.split('\n')).slice(-500)
      let task_line
      for (let i = log_lines.value.length - 1; i >= 0; --i) {
        task_line = log_lines.value[i].substring(15)
        if (task_line.startsWith('SchedulerTask')) {
          break
        }
      }
      const scheduler_task = task_line.split('||')
      const date_time_re = /time='[0-9]+-[0-9]+-[0-9]+ ([0-9]+:[0-9]+:[0-9]+)/
      const plan_re = /task_plan={(.*)}/
      const type_re = /task_type='(.*)'/
      let task_text
      task_list.value = scheduler_task.map((x) => {
        const plan_text = plan_re.exec(x)[1].replace(/'/g, '"')
        if (plan_text) {
          task_text = Object.entries(JSON.parse('{' + plan_text + '}')).map(
            (x) => `${x[0]}: ${x[1].join(', ')}`
          )
        } else {
          task_text = [type_re.exec(x)[1]]
        }
        return {
          time: date_time_re.exec(x)[1],
          task: task_text
        }
      })
    }
  }

  async function get_running() {
    const response = await axios.get(`${import.meta.env.VITE_HTTP_URL}/running`)
    running.value = response.data
  }

  return {
    log,
    log_lines,
    ws,
    running,
    listen_ws,
    get_running,
    first_load,
    task_list
  }
})

```

# `ui/src/stores/plan.js`

This is a JavaScript script that appears to be for building a后端API. It does this by:

1. 检查传入的参数（plan, ling_xi, max_resting_count, exhaust_require, rest_in_full, resting_priority, workaholic）是否为空，如果不为空，则执行下一步操作；
2. 遍历传入的计划（plan.value）中的每个子计划（plan.value[i].plans），如果子计划不为空，则执行下一步操作；
3. 如果子计划为空，则在该位置设置exhaust_require的值为exhaust_require的值为true，exhaust_require为true时，将不会执行plan的进一步计算；
4. 根据得出的结果返回结果；
5. 定义groups，其值是plan.value中每个子计划的group的集合，使用es6的computed方法，根据step3的计算结果动态的增加了一个exhaust_require的判断条件，并在groups中增加exhaust_require为true的组。


```py
import { defineStore } from 'pinia'
import { ref, watch, computed } from 'vue'
import axios from 'axios'

export const usePlanStore = defineStore('plan', () => {
  const ling_xi = ref('1')
  const max_resting_count = ref([])
  const exhaust_require = ref([])
  const rest_in_full = ref([])
  const resting_priority = ref([])
  const workaholic = ref([])

  const plan = ref({})

  const operators = ref([])

  const left_side_facility = []

  const facility_operator_limit = { central: 5, meeting: 2, factory: 1, contact: 1 }
  for (let i = 1; i <= 3; ++i) {
    for (let j = 1; j <= 3; ++j) {
      const facility_name = `room_${i}_${j}`
      facility_operator_limit[facility_name] = 3
      left_side_facility.push({ label: facility_name, value: facility_name })
    }
  }
  for (let i = 0; i <= 4; ++i) {
    facility_operator_limit[`dormitory_${i}`] = 5
  }

  async function load_plan() {
    const response = await axios.get(`${import.meta.env.VITE_HTTP_URL}/plan`)
    ling_xi.value = response.data.conf.ling_xi.toString()
    max_resting_count.value = response.data.conf.max_resting_count.toString()
    exhaust_require.value =
      response.data.conf.exhaust_require == '' ? [] : response.data.conf.exhaust_require.split(',')
    rest_in_full.value =
      response.data.conf.rest_in_full == '' ? [] : response.data.conf.rest_in_full.split(',')
    resting_priority.value =
      response.data.conf.resting_priority == ''
        ? []
        : response.data.conf.resting_priority.split(',')
    workaholic.value =
      response.data.conf.workaholic == '' ? [] : response.data.conf.workaholic.split(',')

    const full_plan = response.data.plan1
    for (const i in facility_operator_limit) {
      let count = 0
      if (!full_plan[i]) {
        count = facility_operator_limit[i]
        full_plan[i] = { name: '', plans: [] }
      } else {
        let limit = facility_operator_limit[i]
        if (full_plan[i].name == '发电站') {
          limit = 1
        }
        if (full_plan[i].plans.length < limit) {
          count = limit - full_plan[i].plans.length
        }
      }
      for (let j = 0; j < count; ++j) {
        full_plan[i].plans.push({ agent: '', group: '', replacement: [] })
      }
    }
    plan.value = full_plan
  }

  async function load_operators() {
    const response = await axios.get(`${import.meta.env.VITE_HTTP_URL}/operator`)
    const option_list = []
    for (const i of response.data) {
      option_list.push({
        value: i,
        label: i
      })
    }
    operators.value = option_list
  }

  function remove_empty_agent(input) {
    const result = {
      name: input.name,
      plans: []
    }
    for (const i of input.plans) {
      if (i.agent) {
        result.plans.push(i)
      }
    }
    return result
  }

  function build_plan() {
    const result = {
      default: 'plan1',
      plan1: {},
      conf: {
        ling_xi: parseInt(ling_xi.value),
        max_resting_count: parseInt(max_resting_count.value),
        exhaust_require: exhaust_require.value.join(','),
        rest_in_full: rest_in_full.value.join(','),
        resting_priority: resting_priority.value.join(','),
        workaholic: workaholic.value.join(',')
      }
    }

    const plan1 = result.plan1

    for (const i in facility_operator_limit) {
      if (i.startsWith('room') && plan.value[i].name) {
        plan1[i] = remove_empty_agent(plan.value[i])
      } else {
        let empty = true
        for (const j of plan.value[i].plans) {
          if (j.agent) {
            empty = false
            break
          }
        }
        if (!empty) {
          plan1[i] = remove_empty_agent(plan.value[i])
        }
      }
    }

    return result
  }

  watch(
    [plan, ling_xi, max_resting_count, exhaust_require, rest_in_full, resting_priority, workaholic],
    () => {
      axios.post(`${import.meta.env.VITE_HTTP_URL}/plan`, build_plan())
    },
    { deep: true }
  )

  const groups = computed(() => {
    const result = []
    for (const facility in plan.value) {
      for (const p of plan.value[facility].plans) {
        if (p.group) {
          result.push(p.group)
        }
      }
    }
    return [...new Set(result)]
  })

  return {
    load_plan,
    load_operators,
    ling_xi,
    max_resting_count,
    exhaust_require,
    rest_in_full,
    resting_priority,
    workaholic,
    plan,
    operators,
    facility_operator_limit,
    left_side_facility,
    build_plan,
    groups
  }
})

```

# `ui/src/stores/record.js`

这段代码使用了 Pinia，一个 Vue 3 的 store 库，主要作用是提供了一个 centralized 的 store 机制，用于在整个 application 中统一管理状态。

具体来说，该代码以下列方式导入了两个依赖项：

1. defineStore，它是 Pinia 的 store 函数，定义了一个 store 函数返回一个 store 对象，可以用来展示 store 中存储的数据。

2. axios，是一个第三方库，用于发送 HTTP 请求。

在 defineStore 的内部，定义了一个名为 getMoodRatios 的函数，该函数通过调用 axios.get 发送一个 HTTP GET 请求获取一个名为 "record/getMoodRatios" 的路由的 HTTP 响应，然后返回响应中的数据。

最终，该代码返回了一个 store 对象，其中包含一个名为 "getMoodRatios" 的方法，用于获取情感分数。


```py
import { defineStore } from 'pinia'
import axios from 'axios'

export const useRecordStore = defineStore('record', () => {
  async function getMoodRatios() {
    const response = await axios.get(`${import.meta.env.VITE_HTTP_URL}/record/getMoodRatios`)
    return response.data
  }

  return {
    getMoodRatios
  }
})

```

# `ui/src/utils/dialog.js`

这两个函数使用了 `axios` 库来发送 HTTP GET 请求获取文件或文件夹的路径。它们的作用是分别从服务器端获取一个文件夹和一个文件的路径，用于弹出对话框让用户选择文件或文件夹。

具体来说，`file_dialog()` 函数从服务器端获取一个文件夹的路径，然后在弹出对话框中显示该路径，用户可以选择关闭对话框并返回。`folder_dialog()` 函数从服务器端获取一个文件的路径，然后在弹出对话框中显示该路径，用户可以选择关闭对话框并返回。

这两个函数的实现依赖于 `import.meta.env.VITE_HTTP_URL` 环境变量，它是一个预设的环境变量，表示服务器端的 HTTP 请求 URL。在这个例子中，`VITE_HTTP_URL` 可能是一个预设的 URL，用于在开发模式下发送 HTTP 请求。


```py
import axios from 'axios'

export async function file_dialog() {
  const response = await axios.get(`${import.meta.env.VITE_HTTP_URL}/dialog/file`)
  const file_path = response.data
  return file_path
}

export async function folder_dialog() {
  const response = await axios.get(`${import.meta.env.VITE_HTTP_URL}/dialog/folder`)
  const folder_path = response.data
  return folder_path
}

```