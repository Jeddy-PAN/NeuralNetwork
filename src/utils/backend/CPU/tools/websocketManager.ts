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
    private pendingRequests: Map<string, {
        resolve: Function;
        reject: Function;
        timeout: NodeJS.Timeout;
    }> = new Map();

    constructor() {
        this.setupMessageHandlers();
    }

    private setupMessageHandlers() {
        // 注册默认消息处理器
        this.on('round_complete', this.handleRoundComplete.bind(this));
        this.on('gradient_received', this.handleGradientReceived.bind(this));
        this.on('heartbeat_ack', this.handleHeartbeat.bind(this));
        this.on('response', this.handleResponse.bind(this));
        this.on('error', this.handleError.bind(this));
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
                    // this.scheduleReconnect();
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

    // 发送请求并等待响应
    async sendRequest(type: string, data: any = {}, timeoutMs: number = 30000): Promise<any> {
        return new Promise((resolve, reject) => {
            const requestId = this.generateRequestId();
            
            // 设置超时
            const timeout = setTimeout(() => {
                this.pendingRequests.delete(requestId);
                reject(new Error(`Request ${type} timeout after ${timeoutMs}ms`));
            }, timeoutMs);

            // 存储Promise解析器
            this.pendingRequests.set(requestId, {
                resolve,
                reject,
                timeout
            });

            // 发送请求
            const success = this.send({
                type,
                request_id: requestId,
                client_id: this.clientId,
                ...data
            });

            if (!success) {
                this.pendingRequests.delete(requestId);
                clearTimeout(timeout);
                reject(new Error('Failed to send WebSocket message'));
            }
        });
    }

    private generateRequestId(): string {
        return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
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

    private handleResponse(message: any) {
        const { request_id, status, data, error } = message;
        const pending = this.pendingRequests.get(request_id);
        
        if (pending) {
            clearTimeout(pending.timeout);
            if (status === 'success') {
                pending.resolve(data);
            } else {
                pending.reject(new Error(error || 'Request failed'));
            }
            this.pendingRequests.delete(request_id);
        }
    }

    private handleError(message: any) {
        const { request_id, error } = message;
        const pending = this.pendingRequests.get(request_id);
        
        if (pending) {
            clearTimeout(pending.timeout);
            pending.reject(new Error(error || 'Unknown error'));
            this.pendingRequests.delete(request_id);
        }
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
        // 清理所有待处理的请求
        for (const [requestId, pending] of this.pendingRequests.entries()) {
            clearTimeout(pending.timeout);
            pending.reject(new Error('WebSocket disconnected'));
        }
        this.pendingRequests.clear();

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    isConnected(): boolean {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }

    // WebSocket API 方法
    async getClientId(): Promise<string> {
        const response = await this.sendRequest('get_client_id');
        return response.client_id;
    }

    async registerDevice(deviceInfo: any): Promise<any> {
        return await this.sendRequest('register_device', deviceInfo);
    }

    async submitBenchmark(benchmark: any): Promise<any> {
        return await this.sendRequest('submit_benchmark', benchmark);
    }

    async getDataset(datasetName: string): Promise<string> {
        const response = await this.sendRequest('get_dataset', { dataset_name: datasetName });
        return response.csv_data;
    }

    async startTrain(datasetName: string): Promise<any> {
        return await this.sendRequest('start_train', { dataset_name: datasetName });
    }

    async submitGradients(gradient: any, computeTime: number): Promise<any> {
        return await this.sendRequest('submit_gradients', {
            gradient,
            compute_time: computeTime
        });
    }

    async getNewGradient(): Promise<any> {
        return await this.sendRequest('get_new_gradient');
    }

    async checkRoundStatus(): Promise<any> {
        return await this.sendRequest('check_round_status');
    }

    async resetServer(): Promise<any> {
        return await this.sendRequest('reset');
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

// 简化的梯度提交函数，使用WebSocket
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

        // 通过WebSocket提交梯度
        const responseJson = await wsManager.submitGradients(gradient, compute_time);

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
        console.error('Error submitting gradients via WebSocket:', error);
        throw error;
    }
}