import { SERVER_CONFIG } from '../../../../config/serverConfig'

// WebSocket管理类
class WebSocketManager {
    private ws: WebSocket | null = null;
    private clientId: string = '';
    private reconnectAttempts: number = 0;
    private maxReconnectAttempts: number = 5;
    private reconnectDelay: number = 1000;
    private messageHandlers: Map<string, Function[]> = new Map();
    private isConnecting: boolean = false;

    constructor() {
        this.setupMessageHandlers();
    }

    private setupMessageHandlers() {
        // 注册默认消息处理器
        this.on('round_complete', this.handleRoundComplete.bind(this));
        this.on('gradient_received', this.handleGradientReceived.bind(this));
        this.on('heartbeat_ack', this.handleHeartbeat.bind(this));
    }

    async connect(clientId: string): Promise<boolean> {
        if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
            return true;
        }

        this.isConnecting = true;
        this.clientId = clientId;

        try {
            const wsUrl = SERVER_CONFIG.baseUrl.replace('http', 'ws') + `/ws/${clientId}`;
            this.ws = new WebSocket(wsUrl);

            return new Promise((resolve, reject) => {
                if (!this.ws) {
                    reject(new Error('WebSocket creation failed'));
                    return;
                }

                this.ws.onopen = () => {
                    console.log('WebSocket connected successfully');
                    this.isConnecting = false;
                    this.reconnectAttempts = 0;
                    this.startHeartbeat();
                    resolve(true);
                };

                this.ws.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data);
                        this.handleMessage(message);
                    } catch (error) {
                        console.error('Failed to parse WebSocket message:', error);
                    }
                };

                this.ws.onclose = () => {
                    console.log('WebSocket connection closed');
                    this.isConnecting = false;
                    this.scheduleReconnect();
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.isConnecting = false;
                    reject(error);
                };

                // 连接超时处理
                setTimeout(() => {
                    if (this.isConnecting) {
                        this.isConnecting = false;
                        reject(new Error('WebSocket connection timeout'));
                    }
                }, 5000);
            });
        } catch (error) {
            this.isConnecting = false;
            console.error('WebSocket connection failed:', error);
            return false;
        }
    }

    private scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            
            console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
            
            setTimeout(() => {
                this.connect(this.clientId);
            }, delay);
        } else {
            console.error('Max reconnection attempts reached');
        }
    }

    private startHeartbeat() {
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.send({
                    type: 'heartbeat',
                    timestamp: Date.now()
                });
            }
        }, 30000); // 每30秒发送心跳
    }

    send(message: any): boolean {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                this.ws.send(JSON.stringify(message));
                return true;
            } catch (error) {
                console.error('Failed to send WebSocket message:', error);
                return false;
            }
        }
        return false;
    }

    private handleMessage(message: any) {
        const { type } = message;
        const handlers = this.messageHandlers.get(type) || [];
        
        handlers.forEach(handler => {
            try {
                handler(message);
            } catch (error) {
                console.error(`Error in message handler for ${type}:`, error);
            }
        });
    }

    on(messageType: string, handler: Function): void {
        if (!this.messageHandlers.has(messageType)) {
            this.messageHandlers.set(messageType, []);
        }
        this.messageHandlers.get(messageType)!.push(handler);
    }

    off(messageType: string, handler: Function): void {
        const handlers = this.messageHandlers.get(messageType);
        if (handlers) {
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }

    private handleRoundComplete(message: any) {
        console.log('Round completed:', message);
        // 触发轮次完成事件
        this.emit('training_round_complete', message);
    }

    private handleGradientReceived(message: any) {
        console.log('Gradient received confirmation:', message);
        // 更新等待状态
        this.emit('gradient_status_update', message);
    }

    private handleHeartbeat(message: any) {
        // 心跳响应，无需特殊处理
    }

    private emit(eventType: string, data: any) {
        // 创建自定义事件
        const event = new CustomEvent(eventType, { detail: data });
        window.dispatchEvent(event);
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    isConnected(): boolean {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }
}

// 全局WebSocket管理器实例
export const wsManager = new WebSocketManager();

// 轮次状态管理
class RoundManager {
    private pendingRounds: Map<number, {
        resolve: Function;
        reject: Function;
        timeout: NodeJS.Timeout;
    }> = new Map();

    constructor() {
        // 监听WebSocket事件
        window.addEventListener('training_round_complete', this.handleRoundComplete.bind(this));
    }

    private handleRoundComplete(event: CustomEvent) {
        const { round_id } = event.detail;
        const pending = this.pendingRounds.get(round_id);
        
        if (pending) {
            clearTimeout(pending.timeout);
            pending.resolve(event.detail);
            this.pendingRounds.delete(round_id);
        }
    }

    waitForRoundComplete(roundId: number, timeoutMs: number = 120000): Promise<any> {
        return new Promise((resolve, reject) => {
            // 设置超时
            const timeout = setTimeout(() => {
                this.pendingRounds.delete(roundId);
                reject(new Error(`Round ${roundId} timeout after ${timeoutMs}ms`));
            }, timeoutMs);

            // 存储Promise解析器
            this.pendingRounds.set(roundId, {
                resolve,
                reject,
                timeout
            });
        });
    }

    cancelRound(roundId: number) {
        const pending = this.pendingRounds.get(roundId);
        if (pending) {
            clearTimeout(pending.timeout);
            pending.reject(new Error(`Round ${roundId} cancelled`));
            this.pendingRounds.delete(roundId);
        }
    }
}

export const roundManager = new RoundManager();

// 简化的梯度提交函数，由服务端管理round_id
export async function submitGradientsWithWebSocket(
    client_id: string, 
    gradient: any, 
    compute_time: number = 0
): Promise<any> {
    try {
        // 确保WebSocket连接
        if (!wsManager.isConnected()) {
            await wsManager.connect(client_id);
        }

        // 提交梯度，不传递round_id，由服务端管理
        const response = await fetch(`${SERVER_CONFIG.baseUrl}${SERVER_CONFIG.endpoints.submitGradients}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                client_id: client_id,
                gradient: gradient,
                compute_time: compute_time
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const responseJson = await response.json();

        // 如果状态是等待，使用WebSocket等待完成
        if (responseJson.status === 'waiting') {
            const round_id = responseJson.round_id;
            console.log(`Waiting for round ${round_id} to complete via WebSocket...`);
            
            // 等待WebSocket通知轮次完成
            const result = await roundManager.waitForRoundComplete(round_id);
            return result;
        }

        return responseJson;
    } catch (error) {
        console.error('Error submitting gradients:', error);
        
        // WebSocket失败时回退到轮询
        console.log('Falling back to polling...');
        return await fallbackToPolling();
    }
}

// 回退轮询机制（作为备用方案）
async function fallbackToPolling(): Promise<any> {
    let responseJson = { status: 'waiting' };
    let attempts = 0;
    const maxAttempts = 150; // 最多尝试150次（约60秒）

    while (responseJson.status === 'waiting' && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 400));
        
        try {
            // 检查当前轮次状态，不需要传递round_id
            const response = await fetch(`${SERVER_CONFIG.baseUrl}${SERVER_CONFIG.endpoints.checkRoundStatus}`, {
                method: 'GET',
                headers: {
                    'cache-control': 'no-cache',
                },
            });
            
            if (response.ok) {
                responseJson = await response.json();
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
        
        attempts++;
    }

    if (attempts >= maxAttempts) {
        throw new Error('Timeout waiting for round completion');
    }

    return responseJson;
}