import { wsManager } from '../../CPU/tools/websocketManager.ts';

// 通过WebSocket提交梯度（保留作为备用，实际应该直接使用wsManager.submitGradients）
export async function postGradients(client_id, gradient, compute_time = 0) {
	try {
		return await wsManager.submitGradients(gradient, compute_time);
	} catch (error) {
		console.error('Error submitting gradients via WebSocket:', error);
		throw error;
	}
}

// 通过WebSocket检查轮次状态
export async function checkRoundStatus() {
	try {
		return await wsManager.checkRoundStatus();
	} catch (error) {
		console.error('Error checking round status via WebSocket:', error);
		throw error;
	}
}

// 通过WebSocket获取新梯度
export async function getNewGradient() {
	try {
		return await wsManager.getNewGradient();
	} catch (error) {
		console.error('Error getting new gradient via WebSocket:', error);
		throw error;
	}
}
