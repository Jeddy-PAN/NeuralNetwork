from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from typing import List, Dict, Optional
from fastapi.responses import FileResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import uuid
from fastapi.middleware.cors import CORSMiddleware
import json
import time
import asyncio
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Federated Learning WebSocket Server",
    description="A federated learning server with full WebSocket support",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
csv_filename = '../public/Datasets/'
gradient_storage: Dict[int, Dict[str, Dict]] = {}
new_gradient: List[List[float]] = []
round_start_times: Dict[int, float] = {}
current_round_id = 0  # 服务端统一管理的轮次ID

# 客户端设备和性能信息
client_device_info: Dict[str, Dict] = {}
client_performance: Dict[str, Dict] = {}

# 配置常量
MAX_WAIT_TIME_PC = 30  # PC设备最大等待时间(秒)
MAX_WAIT_TIME_MOBILE = 60  # 移动设备最大等待时间(秒)
PERFORMANCE_WEIGHT_THRESHOLD = 0.3  # 性能权重阈值

class DeviceInfo(BaseModel):
    client_id: str = Field(..., description="客户端唯一标识符")
    device_type: str = Field(..., description="设备类型")
    gpu_memory: int = Field(ge=0, description="GPU内存大小（MB）")
    cpu_cores: int = Field(ge=1, le=64, description="CPU核心数")
    ram_size: int = Field(ge=0, description="RAM大小（MB）")
    webgpu_supported: bool = Field(description="是否支持WebGPU")

class GradientData(BaseModel):
    client_id: str = Field(..., description="客户端唯一标识符")
    gradient: List[List[float]] = Field(..., description="梯度数据")
    compute_time: float = Field(ge=0, description="客户端计算时间")

class PerformanceBenchmark(BaseModel):
    client_id: str = Field(..., description="客户端唯一标识符")
    benchmark_time: float = Field(ge=0, description="基准测试时间")
    batch_size: int = Field(ge=1, description="测试时使用的批次大小")
    tensor_operations_per_sec: float = Field(ge=0, description="每秒张量操作数")

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_times: Dict[str, float] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        try:
            await websocket.accept()
            self.active_connections[client_id] = websocket
            self.connection_times[client_id] = time.time()
            logger.info(f"WebSocket connected: {client_id}")
            
            # 发送连接确认
            await self.send_personal_message({
                "type": "connection_established",
                "client_id": client_id,
                "server_time": time.time()
            }, client_id)
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket for {client_id}: {e}")
            raise
    
    async def disconnect(self, client_id: str):
        """异步断开连接并处理训练逻辑"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            if client_id in self.connection_times:
                del self.connection_times[client_id]
            logger.info(f"WebSocket disconnected: {client_id}")
            
            # 检查是否影响当前训练轮次
            await self._handle_client_disconnect(client_id)
    
    def _disconnect_sync(self, client_id: str):
        """同步版本的断开连接，不处理训练逻辑"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            if client_id in self.connection_times:
                del self.connection_times[client_id]
            logger.info(f"WebSocket sync disconnected: {client_id}")
    
    async def send_personal_message(self, message: dict, client_id: str) -> bool:
        if client_id not in self.active_connections:
            return False
            
        try:
            await self.active_connections[client_id].send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.warning(f"Failed to send message to {client_id}: {e}")
            # 同步调用disconnect的同步版本
            self._disconnect_sync(client_id)
            return False
    
    async def broadcast_to_connected(self, message: dict) -> int:
        """向所有连接的客户端广播消息，返回成功发送的数量"""
        success_count = 0
        failed_clients = []
        
        for client_id in list(self.active_connections.keys()):
            if await self.send_personal_message(message, client_id):
                success_count += 1
            else:
                failed_clients.append(client_id)
        
        if failed_clients:
            logger.warning(f"Failed to send broadcast to clients: {failed_clients}")
            
        return success_count
    
    def get_connected_clients(self) -> List[str]:
        return list(self.active_connections.keys())
    
    async def _handle_client_disconnect(self, disconnected_client_id: str):
        """处理客户端断开连接对训练的影响"""
        global current_round_id, gradient_storage, new_gradient
        
        # 检查当前轮次是否受影响
        if current_round_id in gradient_storage:
            connected_clients = self.get_connected_clients()
            submitted_clients = list(gradient_storage[current_round_id].keys())
            
            logger.info(f"Client {disconnected_client_id} disconnected. Current round {current_round_id}: "
                       f"{len(submitted_clients)} submitted, {len(connected_clients)} remaining connected")
            
            # 如果断开的客户端已经提交了梯度，从提交列表中移除
            if disconnected_client_id in submitted_clients:
                logger.info(f"Removing gradient from disconnected client {disconnected_client_id}")
                del gradient_storage[current_round_id][disconnected_client_id]
            
            # 使用统一的聚合判断逻辑
            if len(connected_clients) > 0:  # 确保还有连接的客户端
                should_aggregate = await _check_should_aggregate(current_round_id, connected_clients)
                
                if should_aggregate:
                    logger.info(f"Triggering aggregation after client disconnect for round {current_round_id}")
                    
                    # 开始聚合
                    success = await _aggregate_and_broadcast(current_round_id, connected_clients)
                    
                    if success:
                        # 广播完成消息给剩余客户端
                        await self.broadcast_to_connected({
                            "type": "round_complete",
                            "round_id": current_round_id,
                            "status": "complete",
                            "message": f"gradients aggregated after client {disconnected_client_id} left"
                        })
                        
                        # 增加轮次ID
                        current_round_id += 1
                        logger.info(f"Round {current_round_id - 1} completed, starting round {current_round_id}")
                else:
                    # 通知剩余客户端等待状态更新
                    remaining_submitted = list(gradient_storage[current_round_id].keys())
                    waiting_count = len(connected_clients) - len(remaining_submitted)
                    await self.broadcast_to_connected({
                        "type": "participant_left",
                        "left_client": disconnected_client_id,
                        "remaining_participants": len(connected_clients),
                        "current_round": current_round_id,
                        "waiting_for": waiting_count,
                        "status": "waiting"
                    })
            else:
                # 没有剩余客户端，清理当前轮次
                logger.warning(f"No clients remaining, clearing round {current_round_id}")
                if current_round_id in gradient_storage:
                    del gradient_storage[current_round_id]
                if current_round_id in round_start_times:
                    del round_start_times[current_round_id]

manager = ConnectionManager()

# WebSocket消息处理器
async def handle_websocket_request(websocket: WebSocket, client_id: str, message: dict):
    """处理WebSocket请求并返回响应"""
    message_type = message.get("type")
    request_id = message.get("request_id")
    
    try:
        if message_type == "get_client_id":
            response_data = {"client_id": str(uuid.uuid4())}
            
        elif message_type == "register_device":
            device_info = DeviceInfo(**message)
            response_data = await register_device_handler(device_info)
            
        elif message_type == "submit_benchmark":
            benchmark = PerformanceBenchmark(**message)
            response_data = await submit_benchmark_handler(benchmark)
            
        elif message_type == "get_dataset":
            dataset_name = message.get("dataset_name")
            response_data = await get_dataset_handler(client_id, dataset_name)
            
        elif message_type == "start_train":
            dataset_name = message.get("dataset_name")
            response_data = await start_train_handler(dataset_name)
            
        elif message_type == "submit_gradients":
            gradient_data = GradientData(
                client_id=client_id,
                gradient=message.get("gradient"),
                compute_time=message.get("compute_time", 0)
            )
            response_data = await submit_gradients_handler(gradient_data)
            
        elif message_type == "get_new_gradient":
            response_data = await get_new_gradient_handler()
            
        elif message_type == "check_round_status":
            response_data = await check_round_status_handler()
            
        elif message_type == "reset":
            response_data = await reset_server_handler()
            
        else:
            raise ValueError(f"Unknown message type: {message_type}")
        
        # 发送成功响应
        await manager.send_personal_message({
            "type": "response",
            "request_id": request_id,
            "status": "success",
            "data": response_data
        }, client_id)
        
    except Exception as e:
        logger.error(f"Error handling {message_type} for {client_id}: {e}")
        # 发送错误响应
        await manager.send_personal_message({
            "type": "error",
            "request_id": request_id,
            "error": str(e)
        }, client_id)

# 请求处理函数
async def register_device_handler(device_info: DeviceInfo):
    client_id = device_info.client_id
    client_device_info[client_id] = {
        "device_type": device_info.device_type,
        "gpu_memory": device_info.gpu_memory,
        "cpu_cores": device_info.cpu_cores,
        "ram_size": device_info.ram_size,
        "webgpu_supported": device_info.webgpu_supported,
        "registration_time": time.time()
    }
    
    # 初始化性能记录
    client_performance[client_id] = {
        "avg_compute_time": 0.0,
        "benchmark_score": 0.0,
        "reliability_score": 1.0,
        "performance_weight": 1.0
    }
    
    return {
        "status": "success",
        "message": "device registered successfully",
        "client_id": client_id,
        "recommended_batch_size": _calculate_batch_size(device_info)
    }

async def submit_benchmark_handler(benchmark: PerformanceBenchmark):
    client_id = benchmark.client_id
    
    if client_id not in client_performance:
        raise ValueError("client not registered")
    
    # 更新基准测试分数
    client_performance[client_id]["benchmark_score"] = benchmark.tensor_operations_per_sec
    
    # 计算性能权重
    _update_performance_weights()
    
    # 生成个性化配置
    config = _generate_client_config(client_id)
    
    return {
        "status": "success",
        "performance_tier": _get_performance_tier(client_id),
        "recommended_config": config
    }

async def get_dataset_handler(client_id: str, dataset_name: str):
    connected_clients = manager.get_connected_clients()
    
    if client_id not in connected_clients:
        raise ValueError("client is not connected")
    
    if len(connected_clients) == 0:
        raise ValueError("no clients connected")
    
    csv_file = csv_filename + dataset_name
    await split_csv_async(csv_file, len(connected_clients))
    client_part_index = connected_clients.index(client_id)
    part_filename = csv_file[:-4] + f'_part{client_part_index}.csv'
    
    # 读取CSV文件内容
    try:
        with open(part_filename, 'r') as f:
            csv_data = f.read()
        return {"csv_data": csv_data}
    except Exception as e:
        raise ValueError(f"Failed to read dataset file: {e}")

async def start_train_handler(dataset_name: str):
    global current_round_id
    
    # 验证数据集
    dataset_path = Path(csv_filename) / dataset_name
    if not dataset_path.exists():
        raise ValueError(f"Dataset {dataset_name} not found")
    
    # 重置状态
    current_round_id = 0
    
    connected_clients = manager.get_connected_clients()
    
    if not connected_clients:
        raise ValueError("No clients connected")
    
    # 广播训练开始消息
    success_count = await manager.broadcast_to_connected({
        "type": "training_start",
        "dataset": dataset_name,
        "participants": len(connected_clients),
        "round_id": current_round_id
    })
    
    logger.info(f"Training started with dataset {dataset_name}, {len(connected_clients)} participants")
    
    return {
        "status": "success",
        "message": f"training started with dataset {dataset_name}",
        "participants": len(connected_clients),
        "notified_clients": success_count
    }

async def submit_gradients_handler(data: GradientData):
    global new_gradient, current_round_id
    
    client_id = data.client_id
    gradient = data.gradient
    compute_time = data.compute_time
    connected_clients = manager.get_connected_clients()
    
    if client_id not in connected_clients:
        raise ValueError("client is not connected")

    # 初始化梯度存储
    if current_round_id not in gradient_storage:
        gradient_storage[current_round_id] = {}
        round_start_times[current_round_id] = time.time()

    gradient_storage[current_round_id][client_id] = {
        "gradient": gradient,
        "compute_time": compute_time,
        "submit_time": time.time()
    }

    logger.info(f"Gradient received from client: {client_id}, round: {current_round_id} ({len(gradient_storage[current_round_id])}/{len(connected_clients)})")

    # 统一的聚合判断逻辑
    should_aggregate = await _check_should_aggregate(current_round_id)
    
    if should_aggregate:
        success = await _aggregate_and_broadcast(current_round_id, connected_clients)
        
        if success:
            # 向所有连接的客户端广播完成消息
            await manager.broadcast_to_connected({
                "type": "round_complete",
                "round_id": current_round_id,
                "status": "complete",
                "message": "gradients aggregated and ready"
            })
            
            response = {
                "status": "complete",
                "message": "gradients aggregated",
                "round_id": current_round_id
            }
            
            # 增加轮次ID，准备下一轮
            current_round_id += 1
            
            return response
        else:
            raise ValueError("Gradient aggregation failed")
    else:
        # 向当前客户端发送等待确认
        await manager.send_personal_message({
            "type": "gradient_received",
            "round_id": current_round_id,
            "status": "waiting",
            "waiting_for": len(connected_clients) - len(gradient_storage[current_round_id])
        }, client_id)
        
        return {
            "status": "waiting", 
            "round_id": current_round_id,
            "message": f"{len(connected_clients) - len(gradient_storage[current_round_id])} more clients needed"
        }

async def get_new_gradient_handler():
    if len(new_gradient) == 0:
        return {"new_gradient": None}
    return {"new_gradient": new_gradient}

async def check_round_status_handler():
    """检查当前轮次状态"""
    if current_round_id not in gradient_storage:
        return {
            "status": "complete",
            "message": "current round complete or not started",
            "round_id": current_round_id,
        }
    
    connected_clients = manager.get_connected_clients()
    if len(gradient_storage[current_round_id]) < len(connected_clients):
        return {
            "status": "waiting",
            "round_id": current_round_id,
            "message": "waiting other clients to submit gradients"
        }
    else:
        return {
            "status": "complete",
            "message": "all gradients received",
            "round_id": current_round_id,
        }

async def reset_server_handler():
    global current_round_id
    
    # 通知所有客户端训练停止
    await manager.broadcast_to_connected({
        "type": "training_stopped",
        "reason": "server_reset"
    })
    
    # 重置状态
    new_gradient.clear()
    gradient_storage.clear()
    round_start_times.clear()
    client_device_info.clear()
    client_performance.clear()
    current_round_id = 0
    
    logger.info("Server reset completed")
    return {"status": "success", "message": "server reset"}

# 辅助函数
def _calculate_batch_size(device_info: DeviceInfo) -> int:
    """根据设备信息计算推荐的批次大小"""
    if device_info.device_type == "pc":
        if device_info.gpu_memory > 4000:
            return min(64, device_info.gpu_memory // 100)
        elif device_info.gpu_memory > 2000:
            return 32
        else:
            return 16
    else:  # mobile
        if device_info.ram_size > 6000:
            return 16
        elif device_info.ram_size > 4000:
            return 8
        else:
            return 4

def _update_performance_weights():
    """更新所有客户端的性能权重"""
    if not client_performance:
        return
    
    scores = [
        perf["benchmark_score"] 
        for perf in client_performance.values() 
        if perf["benchmark_score"] > 0
    ]
    
    if not scores:
        return
    
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    for client_id, perf in client_performance.items():
        if perf["benchmark_score"] > 0:
            if score_range > 0:
                normalized_score = (perf["benchmark_score"] - min_score) / score_range
            else:
                normalized_score = 1.0
            perf["performance_weight"] = 0.1 + 0.9 * normalized_score

def _generate_client_config(client_id: str) -> dict:
    """为客户端生成个性化训练配置"""
    if (client_id not in client_performance or 
        client_id not in client_device_info):
        return {
            "framerate": 15,
            "learning_rate": 0.01,
            "local_epochs": 1
        }
    
    performance_weight = client_performance[client_id]["performance_weight"]
    device_type = client_device_info[client_id]["device_type"]
    
    configs = {
        "high": {
            "framerate": 10 if device_type == "pc" else 20,
            "learning_rate": 0.01,
            "local_epochs": 1
        },
        "medium": {
            "framerate": 15 if device_type == "pc" else 25,
            "learning_rate": 0.008,
            "local_epochs": 1
        },
        "low": {
            "framerate": 20 if device_type == "pc" else 30,
            "learning_rate": 0.005,
            "local_epochs": 1
        }
    }
    
    if performance_weight > 0.7:
        return configs["high"]
    elif performance_weight > 0.4:
        return configs["medium"]
    else:
        return configs["low"]

def _get_performance_tier(client_id: str) -> str:
    """获取客户端性能等级"""
    if client_id not in client_performance:
        return "unknown"
    
    weight = client_performance[client_id]["performance_weight"]
    if weight > 0.7:
        return "high"
    elif weight > 0.4:
        return "medium"
    else:
        return "low"

async def split_csv_async(file_path: str, total_part_num: int) -> bool:
    """异步分割CSV文件"""
    try:
        df = pd.read_csv(file_path)
        df_len = len(df)
        part_len = df_len // total_part_num
        
        for i in range(total_part_num):
            if i == total_part_num - 1:
                df_part = df.iloc[i * part_len:]
            else:
                df_part = df.iloc[i * part_len: (i + 1) * part_len]
            
            part_filename = file_path[:-4] + f'_part{i}.csv'
            df_part.to_csv(part_filename, index=False)
            
            logger.info(f"Creating part {i} with {len(df_part)} rows")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to split CSV {file_path}: {e}")
        return False

async def _aggregate_and_broadcast(round_id: int, connected_clients: List[str]) -> bool:
    """聚合梯度并准备广播"""
    global new_gradient
    
    try:
        round_data = gradient_storage[round_id]
        selected_clients = list(round_data.keys())
        
        if not selected_clients:
            logger.warning(f"No gradients to aggregate for round {round_id}")
            return False
        
        # 获取客户端权重
        if len(selected_clients) < len(connected_clients):
            # 使用加权聚合
            weights = [
                client_performance.get(cid, {}).get("performance_weight", 1.0) 
                for cid in selected_clients
            ]
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(selected_clients) for _ in selected_clients]
            
            client_gradients_data = [round_data[cid]["gradient"] for cid in selected_clients]
            
            new_gradient.clear()
            for tensors in zip(*client_gradients_data):
                tensor_avg = [sum(w * val for w, val in zip(weights, values)) 
                             for values in zip(*tensors)]
                new_gradient.append(tensor_avg)
                
            logger.info(f"Weighted aggregation completed for round {round_id} with {len(selected_clients)} / {len(connected_clients)} clients")
        else:
            # 传统平均聚合
            client_gradients_data = [round_data[cid]["gradient"] for cid in selected_clients]
            
            new_gradient.clear()
            for tensors in zip(*client_gradients_data):
                tensor_avg = [sum(values) / len(selected_clients) for values in zip(*tensors)]
                new_gradient.append(tensor_avg)
                
            logger.info(f"Standard aggregation completed for round {round_id} with {len(selected_clients)} / {len(connected_clients)} clients")
        
        # 清理已完成的轮次数据
        del gradient_storage[round_id]
        if round_id in round_start_times:
            del round_start_times[round_id]
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to aggregate gradients for round {round_id}: {e}")
        return False

async def _check_should_aggregate(round_id: int, connected_clients: List[str] = None) -> bool:
    """简化的聚合条件检查函数 - 基于剩余设备权重加和的等待策略"""
    if round_id not in gradient_storage:
        return False
    
    submitted_clients = list(gradient_storage[round_id].keys())
    
    if connected_clients is None:
        connected_clients = manager.get_connected_clients()
    
    logger.info(f"Checking aggregation for round {round_id}: {len(submitted_clients)}/{len(connected_clients)} clients submitted")
    
    # 基础条件：至少需要2个客户端或所有客户端都已提交
    if len(submitted_clients) >= len(connected_clients):
        logger.info(f"All {len(connected_clients)} connected clients have submitted gradients")
        return True
    
    # if len(submitted_clients) < 2:
    #     logger.info(f"Need at least 2 clients, currently have {len(submitted_clients)}")
    #     return False
    
    # 计算剩余设备权重总和
    remaining_weight_sum = 0
    for client_id in connected_clients:
        if client_id not in submitted_clients:
            weight = client_performance.get(client_id, {}).get("performance_weight", 0.5)
            remaining_weight_sum += weight
    
    # 生成随机数并比较
    random_value = np.random.random()
    should_wait = random_value < remaining_weight_sum
    
    logger.info(f"简化等待决策 - 轮次 {round_id}: "
               f"剩余权重总和={remaining_weight_sum:.3f}, "
               f"随机值={random_value:.3f}, "
               f"决策={'等待' if should_wait else '聚合'}")
    
    return not should_wait

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # 接收并处理WebSocket消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "heartbeat":
                await manager.send_personal_message({"type": "heartbeat_ack"}, client_id)
            elif message.get("type") == "join_round":
                round_id = message.get("round_id")
                await manager.send_personal_message({
                    "type": "round_joined",
                    "round_id": round_id,
                    "status": "waiting"
                }, client_id)
            elif message.get("request_id"):
                # 处理请求-响应模式的消息
                await handle_websocket_request(websocket, client_id, message)
            else:
                logger.warning(f"Unknown message type from {client_id}: {message.get('type')}")
                
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        await manager.disconnect(client_id)
