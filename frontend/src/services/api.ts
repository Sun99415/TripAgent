import axios from 'axios'
import type { TripFormData, TripPlanResponse } from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5分钟超时:多智能体串行调用+大JSON生成耗时较长
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('响应错误:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

/**
 * 生成旅行计划
 */
export async function generateTripPlan(formData: TripFormData): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>('/api/trip/plan', formData)
    return response.data
  } catch (error: any) {
    console.error('生成旅行计划失败:', error)
    // 超时:多智能体串行调用,正常也可能需要1-3分钟
    if (error?.code === 'ECONNABORTED' || /timeout/i.test(error?.message || '')) {
      throw new Error('生成超时(超过5分钟),请稍后重试或减少旅行天数')
    }
    // 网络错误:后端未启动 / 端口不对 / CORS被拦截
    if (error?.message === 'Network Error') {
      throw new Error('无法连接后端服务,请确认后端已启动(默认 http://localhost:8000)')
    }
    throw new Error(error.response?.data?.detail || error.message || '生成旅行计划失败')
  }
}

/**
 * 流式事件类型
 */
export interface TripStreamEvent {
  type: 'status' | 'plan_delta' | 'done' | 'error'
  stage?: string
  message?: string
  text?: string
  plan?: any
  from_cache?: boolean
}

/**
 * 流式生成旅行计划(SSE)
 *
 * 通过 fetch 消费后端 text/event-stream,按阶段实时回调,替代整段同步等待。
 */
export async function streamTripPlan(
  formData: TripFormData,
  handlers: {
    onStatus?: (stage: string, message: string) => void
    onDelta?: (text: string) => void
    onDone?: (plan: any, fromCache: boolean) => void
    onError?: (message: string) => void
  }
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/trip/plan/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(formData)
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`流式请求失败(${response.status}): ${detail}`)
  }
  if (!response.body) {
    throw new Error('当前浏览器不支持流式响应')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const dispatch = (event: TripStreamEvent) => {
    switch (event.type) {
      case 'status':
        handlers.onStatus?.(event.stage || '', event.message || '')
        break
      case 'plan_delta':
        handlers.onDelta?.(event.text || '')
        break
      case 'done':
        handlers.onDone?.(event.plan, !!event.from_cache)
        break
      case 'error':
        handlers.onError?.(event.message || '')
        break
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const chunks = buffer.split('\n\n')
    buffer = chunks.pop() ?? ''
    for (const chunk of chunks) {
      const line = chunk.trim()
      if (!line.startsWith('data:')) continue
      const dataStr = line.slice(5).trim()
      if (!dataStr) continue
      try {
        dispatch(JSON.parse(dataStr) as TripStreamEvent)
      } catch (e) {
        console.warn('解析SSE事件失败:', dataStr, e)
      }
    }
  }
}

/**
 * 健康检查
 */
export async function healthCheck(): Promise<any> {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error: any) {
    console.error('健康检查失败:', error)
    throw new Error(error.message || '健康检查失败')
  }
}

export default apiClient

