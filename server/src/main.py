from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from typing import List, Dict
from fastapi.responses import FileResponse
import numpy as np
import pandas
from pydantic import BaseModel
import uuid
from fastapi.middleware.cors import CORSMiddleware
import json
import time
import asyncio

app = FastAPI()

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
ready_clients: List[str] = []
new_gradient: List[List[float]] = []
round_start_times: Dict[int, float] = {}

# 客户端设备和性能信息
client_device_info: Dict[str, Dict] = {}
client_performance: Dict[str, Dict] = {}

# 配置常量
MAX_WAIT_TIME_PC = 30  # PC设备最大等待时间(秒)
MAX_WAIT_TIME_MOBILE = 60  # 移动设备最大等待时间(秒)
PERFORMANCE_WEIGHT_THRESHOLD = 0.3  # 性能权重阈值
EXPECTED_CLIENTS = 1

class DeviceInfo(BaseModel):
	client_id: str
	device_type: str  # "pc" 或 "mobile"
	gpu_memory: int  # GPU内存大小（MB）
	cpu_cores: int   # CPU核心数
	ram_size: int    # RAM大小（MB）
	webgpu_supported: bool  # 是否支持WebGPU

class GradientData(BaseModel):
	client_id: str
	gradient: List[List[float]]
	round_id: int
	compute_time: float  # 客户端计算时间

class PerformanceBenchmark(BaseModel):
	client_id: str
	benchmark_time: float  # 基准测试时间
	batch_size: int       # 测试时使用的批次大小
	tensor_operations_per_sec: float  # 每秒张量操作数

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"WebSocket connected: {client_id}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"WebSocket disconnected: {client_id}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                print(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_round(self, message: dict, round_id: int):
        """向参与特定轮次的所有客户端广播消息"""
        for client_id in ready_clients:
            await self.send_personal_message(message, client_id)

manager = ConnectionManager()

@app.get("/get_client_id")
async def get_client_id():
	return {"client_id": str(uuid.uuid4())}

@app.post("/register_device")
async def register_device(device_info: DeviceInfo):
	"""注册客户端设备信息"""
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
		"reliability_score": 1.0,  # 可靠性评分
		"performance_weight": 1.0   # 性能权重
	}
	
	return {
		"status": "success",
		"message": "device registered successfully",
		"client_id": client_id,
		"recommended_batch_size": _calculate_batch_size(device_info)
	}

@app.post("/submit_benchmark")
async def submit_benchmark(benchmark: PerformanceBenchmark):
	"""提交性能基准测试结果"""
	client_id = benchmark.client_id
	
	if client_id not in client_performance:
		return {"status": "error", "message": "client not registered"}
	
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

def _calculate_batch_size(device_info: DeviceInfo) -> int:
	"""根据设备信息计算推荐的批次大小"""
	if device_info.device_type == "pc":
		if device_info.gpu_memory > 4000:  # > 4GB GPU
			return 64
		elif device_info.gpu_memory > 2000:  # > 2GB GPU
			return 32
		else:
			return 16
	else:  # mobile
		if device_info.ram_size > 6000:  # > 6GB RAM
			return 16
		elif device_info.ram_size > 4000:  # > 4GB RAM
			return 8
		else:
			return 4

def _update_performance_weights():
	"""更新所有客户端的性能权重"""
	if not client_performance:
		return
	
	scores = [perf["benchmark_score"] for perf in client_performance.values() if perf["benchmark_score"] > 0]
	if not scores:
		return
	
	min_score = min(scores)
	max_score = max(scores)
	
	for client_id, perf in client_performance.items():
		if perf["benchmark_score"] > 0:
			if max_score > min_score:
				normalized_score = (perf["benchmark_score"] - min_score) / (max_score - min_score)
			else:
				normalized_score = 1.0
			perf["performance_weight"] = 0.1 + 0.9 * normalized_score  # 0.1 到 1.0 范围

def _generate_client_config(client_id: str) -> dict:
	"""为客户端生成个性化训练配置"""
	if client_id not in client_performance or client_id not in client_device_info:
		return {}
	
	performance_weight = client_performance[client_id]["performance_weight"]
	device_type = client_device_info[client_id]["device_type"]
	
	if performance_weight > 0.7:  # 高性能
		return {
			"framerate": 10 if device_type == "pc" else 20,
			"learning_rate": 0.01,
			"local_epochs": 1
		}
	elif performance_weight > 0.4:  # 中性能
		return {
			"framerate": 15 if device_type == "pc" else 25,
			"learning_rate": 0.008,
			"local_epochs": 1
		}
	else:  # 低性能
		return {
			"framerate": 20 if device_type == "pc" else 30,
			"learning_rate": 0.005,
			"local_epochs": 1
		}

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

@app.get("/ready_to_train/{client_id}")
async def ready_to_train(client_id: str):
	global EXPECTED_CLIENTS

	if(client_id not in ready_clients):
		ready_clients.append(client_id)
		EXPECTED_CLIENTS = len(ready_clients)

		return {
			"status": "success",
			"message": "client is ready to train",
			"client_id": client_id
		}

@app.get("/not_ready_to_train/{client_id}")
async def not_ready_to_train(client_id: str):
	global EXPECTED_CLIENTS

	if (client_id in ready_clients):
		ready_clients.remove(client_id)
		EXPECTED_CLIENTS = len(ready_clients)
		return {
			"status": "success",
			"message": "Client is removed",
			"client_id": client_id
		}
	else:
		return {
			"status": "fail",
			"message": "There is no such client or other errors exist",
			"client_id": client_id
		}


def split_csv(file_path, total_part_num):
	df = pandas.read_csv(file_path)
	df_len = len(df)
	part_len = df_len // total_part_num
	for i in range(total_part_num):
		if i == total_part_num - 1:
			df_part = df.iloc[i * part_len:]
		else:
			df_part = df.iloc[i * part_len: (i + 1) * part_len]
		print("part", i, "len:", len(df_part))
		df_part.to_csv(file_path[:-4] + f'_part{i}.csv', index=False)

@app.get("/get_dataset/{client_id}/{dataset_name}")
async def get_dataset(client_id: str, dataset_name: str):
	if client_id not in ready_clients:
		return {
				"status": "error",
				"message": "client is not ready to train",
				"client_id": client_id
		}
	
	if len(ready_clients) == 0:
		return {
				"status": "error",
				"message": "client is not ready to train",
				"client_id": client_id
		}
	
	csv_file = csv_filename + dataset_name
	split_csv(csv_file, len(ready_clients))
	client_part_index = ready_clients.index(client_id)
	part_filename = csv_file[:-4] + f'_part{client_part_index}.csv'
	return FileResponse(part_filename, media_type='text/csv', filename=f'part{client_id}.csv')

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
	await manager.connect(websocket, client_id)
	try:
		while True:
			# 保持连接活跃，处理心跳或其他消息
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
	except WebSocketDisconnect:
		manager.disconnect(client_id)
	except Exception as e:
		print(f"WebSocket error for {client_id}: {e}")
		manager.disconnect(client_id)

@app.post("/submit_gradients")
async def submit_gradients(data: GradientData):
	global new_gradient
	
	round_id = data.round_id
	client_id = data.client_id
	gradient = data.gradient
	compute_time = data.compute_time
	
	if client_id not in ready_clients:
		return {
			"status": "error",
			"message": "client is not ready to train",
			"client_id": client_id
		}

	# 更新性能记录
	if client_id in client_performance:
		old_time = client_performance[client_id]["avg_compute_time"]
		if old_time == 0:
			client_performance[client_id]["avg_compute_time"] = compute_time
		else:
			client_performance[client_id]["avg_compute_time"] = (old_time + compute_time) / 2

	# 初始化梯度存储
	if round_id not in gradient_storage:
		gradient_storage[round_id] = {}
		round_start_times[round_id] = time.time()

	gradient_storage[round_id][client_id] = {
		"gradient": gradient,
		"compute_time": compute_time,
		"submit_time": time.time()
	}

	print(f"Gradient received from client: {client_id}, round: {round_id}")

	# 检查是否应该触发聚合
	if _should_aggregate_gradients(round_id):
		await _aggregate_and_broadcast(round_id)
		
		# 向所有客户端广播完成消息
		await manager.broadcast_to_round({
			"type": "round_complete",
			"round_id": round_id,
			"status": "complete",
			"message": "gradients aggregated and ready"
		}, round_id)
		
		return {
			"status": "complete",
			"message": "gradients aggregated",
			"round_id": round_id
		}
	else:
		# 向当前客户端发送等待确认
		await manager.send_personal_message({
			"type": "gradient_received",
			"round_id": round_id,
			"status": "waiting",
			"waiting_for": len(ready_clients) - len(gradient_storage[round_id])
		}, client_id)
		
		return {
			"status": "waiting", 
			"round_id": round_id,
			"message": f"{len(ready_clients) - len(gradient_storage[round_id])} more clients needed"
		}

async def _aggregate_and_broadcast(round_id: int):
	"""聚合梯度并准备广播"""
	global new_gradient
	
	round_data = gradient_storage[round_id]
	selected_clients = list(round_data.keys())
	
	# 获取客户端权重
	if len(selected_clients) < len(ready_clients):
		# 使用加权聚合
		weights = [client_performance.get(cid, {}).get("performance_weight", 1.0) for cid in selected_clients]
		total_weight = sum(weights)
		if total_weight > 0:
			weights = [w / total_weight for w in weights]  # 归一化
		else:
			weights = [1.0 / len(selected_clients) for _ in selected_clients]
		
		client_gradients_data = [round_data[cid]["gradient"] for cid in selected_clients]
		
		new_gradient.clear()
		for tensors in zip(*client_gradients_data):
			tensor_avg = [sum(w * val for w, val in zip(weights, values)) 
						 for values in zip(*tensors)]
			new_gradient.append(tensor_avg)
			
		print(f"Weighted aggregation completed for round {round_id} with {len(selected_clients)} clients")
	else:
		# 传统平均聚合
		client_gradients_data = [round_data[cid]["gradient"] for cid in selected_clients]
		
		new_gradient.clear()
		for tensors in zip(*client_gradients_data):
			tensor_avg = [sum(values) / len(selected_clients) for values in zip(*tensors)]
			new_gradient.append(tensor_avg)
			
		print(f"Standard aggregation completed for round {round_id}")
	
	# 清理已完成的轮次数据
	del gradient_storage[round_id]
	if round_id in round_start_times:
		del round_start_times[round_id]

def _should_aggregate_gradients(round_id: int) -> bool:
	"""智能判断是否应该聚合梯度"""
	if round_id not in gradient_storage:
		return False
	
	submitted_clients = list(gradient_storage[round_id].keys())
	
	# 策略1: 所有客户端都提交了
	if len(submitted_clients) == len(ready_clients):
		return True
	
	# 策略2: 超时策略
	if round_id in round_start_times:
		elapsed_time = time.time() - round_start_times[round_id]
		
		# 检查是否有PC客户端参与
		pc_clients = [cid for cid in submitted_clients 
					 if client_device_info.get(cid, {}).get("device_type") == "pc"]
		
		if pc_clients and elapsed_time > MAX_WAIT_TIME_PC:
			return True
		elif elapsed_time > MAX_WAIT_TIME_MOBILE:
			return True
	
	# 策略3: 性能权重策略
	total_submitted_weight = sum(client_performance.get(cid, {}).get("performance_weight", 1.0) 
							   for cid in submitted_clients)
	total_possible_weight = sum(client_performance.get(cid, {}).get("performance_weight", 1.0) 
							  for cid in ready_clients)
	
	if total_possible_weight > 0 and (total_submitted_weight / total_possible_weight) > PERFORMANCE_WEIGHT_THRESHOLD:
		return True
	
	return False

@app.get("/check_round_status/")
async def check_round_status(round_id: int):
	if (round_id not in gradient_storage):
		return {
			"status": "complete",
			"message": "all gradients received",
			"round_id": round_id,
		}
	if (len(gradient_storage[round_id]) < len(ready_clients)):
		return {
			"status": "waiting",
			"round_id": round_id,
			"message": "waiting other clients to submit gradients"
		}
	else:
		return {
			"status": "complete",
			"message": "all gradients received",
			"round_id": round_id,
		}
	
@app.get("/get_new_gradient")
async def get_new_gradient():
	if len(new_gradient) == 0:
		return {"new_gradient": None}
	
	return {"new_gradient": new_gradient}

@app.get("/reset")
async def reset_server():
	ready_clients.clear()
	new_gradient.clear()
	gradient_storage.clear()
	round_start_times.clear()
	client_device_info.clear()
	client_performance.clear()
	return {"status": "success", "message": "server reset"}
