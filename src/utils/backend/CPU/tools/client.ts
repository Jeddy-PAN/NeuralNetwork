// client.ts

import { SERVER_CONFIG } from '../../../../config/serverConfig';
import { wsManager } from './websocketManager';

// 定义一个函数来请求客户端 ID
const serverUrl = SERVER_CONFIG.baseUrl;
async function fetchClientId(): Promise<string | null> {
	try {
		const response = await fetch(`${serverUrl}${SERVER_CONFIG.endpoints.getClientId}`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
				'Cache-Control': 'no-cache', // 禁用缓存
			},
		});
		if (!response.ok) {
			throw new Error('Network response was not ok');
		}

		const data = await response.json();
		console.log('LOG: ', data);
		const clientId = data.client_id;

		// 将客户端 ID 存储在浏览器的 localStorage 中
		localStorage.setItem('client_id', clientId);

		return clientId;
	} catch (error) {
		console.error('There has been a problem with your fetch operation:', error);
		return null;
	}
}

// 从 localStorage 中获取客户端 ID，如果不存在则通过WebSocket获取
export async function getClientId(): Promise<string> {
	let c_id = localStorage.getItem('client_id');
	if (c_id == null) {
		// 使用临时ID建立WebSocket连接来获取真实的client_id
		const tempId = `temp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
		await wsManager.connect(tempId);
		c_id = await wsManager.getClientId();
		localStorage.setItem('client_id', c_id);
		console.log('get client_id from server via WebSocket', c_id);
		
		// 断开临时连接，后续会用真实ID重新连接
		wsManager.disconnect();
	}

	if (c_id == null) {
		throw new Error('Failed to get client id');
	}
	return c_id;
}

//api/start-train/{dataset_name}
export async function startTrain(datasetName: string) {
	try {
		const result = await wsManager.startTrain(datasetName);
		console.log('start train via WebSocket', datasetName, result);
		return result;
	} catch (error) {
		console.error('There has been a problem with WebSocket start train:', error);
		throw error;
	}
}

// 获取数据集 - 通过WebSocket
export async function fetchDataset(client_id: string, dataset_name: string): Promise<string> {
	try {
		if (!wsManager.isConnected()) {
			await wsManager.connect(client_id);
		}

		const csvData = await wsManager.getDataset(dataset_name);
		return csvData;
	} catch (error) {
		console.error('Error fetching dataset via WebSocket:', error);
		return '';
	}
}

// 重置服务器状态 - 通过WebSocket
export async function resetServer(): Promise<void> {
	try {
		await wsManager.resetServer();
		console.log('Server reset via WebSocket');
	} catch (error) {
		console.error('There has been a problem with WebSocket server reset', error);
		throw error;
	}
}
