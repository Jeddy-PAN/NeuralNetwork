// client.ts

import { SERVER_CONFIG } from '../../../../config/serverConfig';

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

// 从 localStorage 中获取客户端 ID
export async function getClientId(): Promise<string> {
	let c_id = localStorage.getItem('client_id');
	if (c_id == null) {
		c_id = await fetchClientId();
		console.log('get client_id from server', c_id);
	}

	if (c_id == null) {
		throw new Error('Failed to get client id');
	}
	return c_id;
}

//api/start-train/{dataset_name}
export async function startTrain(datasetName: string) {
	try {
		const response = await fetch(`${serverUrl}${SERVER_CONFIG.endpoints.startTrain}/${datasetName}`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
				'Cache-Control': 'no-cache', // 禁用缓存
			},
		});
		if (!response.ok) {
			throw new Error('Network response was not ok');
		}

		const json = await response.json();
		console.log('start train', datasetName, json);
	} catch (error) {
		console.error('There has been a problem with your fetch operation:', error);
	}
}

// /api/dataset/{dataset_name}
export async function fetchDataset(client_id: string, dataset_name: string): Promise<string> {
	try {
		const response = await fetch(`${serverUrl}${SERVER_CONFIG.endpoints.getDataset}/${client_id}/${dataset_name}`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
				Accept: 'text/csv', // 设置接收类型为 CSV
				'Cache-Control': 'no-cache', // 禁用缓存
			},
		});
		if (!response.ok) {
			const errorMsg = await response.json();
			throw new Error(errorMsg.message || 'Network response was not ok');
		}

		return response.text();
	} catch (error) {
		console.error('Error:', error);
		return '';
	}
}

export async function setReadyForTrain(client_id: string): Promise<void> {
	try {
		console.log('set ready for train', client_id);
		const response = await fetch(`${serverUrl}${SERVER_CONFIG.endpoints.readyToTrain}/${client_id}`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
				'Cache-Control': 'no-cache', // 禁用缓存
			},
		});
		if (!response.ok) {
			throw new Error('Network response was not ok');
		}
		const json = await response.json();
		console.log('LOG: ', json);
	} catch (error) {
		console.error('There has been a problem with your fetch operation:', error);
	}
}

export async function NotReadyForTrain(client_id: string): Promise<void> {
		try {
		console.log('set not ready for train', client_id);
		const response = await fetch(`${serverUrl}${SERVER_CONFIG.endpoints.notReadyToTrain}/${client_id}`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
				'Cache-Control': 'no-cache', // 禁用缓存
			},
		});
		if (!response.ok) {
			throw new Error('Network response was not ok');
		}
		const json = await response.json();
		console.log('LOG: ', json);
	} catch (error) {
		console.error('There has been a problem with your fetch operation:', error);
	}
}

// ///api/stop-train
// export async function stopTrain(client_id: string): Promise<void> {
// 	try {
// 		const response = await fetch(`http://localhost:8000/api/stop-train/`, {
// 			method: 'GET',
// 			headers: {
// 				'Content-Type': 'application/json',
// 				'Cache-Control': 'no-cache', // 禁用缓存
// 			},
// 		});
// 		if (!response.ok) {
// 			throw new Error('Network response was not ok');
// 		}
// 	} catch (error) {
// 		console.error('There has been a problem with your fetch operation:', error);
// 	}
// }

export async function resetServer(): Promise<void> {
	try {
		const response = await fetch(`${serverUrl}${SERVER_CONFIG.endpoints.reset}/`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
				'Cache-Control': 'no-cache', // 禁用缓存
			},
		});
		if (!response.ok) {
			throw new Error('Network response was not ok');
		}
	} catch (error) {
		console.error('There has been a problem with your fetch operation', error);
	}
}
