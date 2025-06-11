// 动态获取服务器基础URL
function getServerBaseUrl(): string {
	// 如果在浏览器环境中，使用当前页面的hostname
	if (typeof window !== 'undefined' && window.location) {
		const { protocol, hostname } = window.location;
		// 如果前端使用HTTPS，后端也应该使用HTTPS
		const serverProtocol = 'http';
		return 'http://' + hostname + ':8000';
	}
	// fallback到原来的地址（用于SSR或其他环境）
	return 'http://172.23.228.12:8000';
}

export const SERVER_CONFIG = {
	get baseUrl() {
		return getServerBaseUrl();
	},
	endpoints: {
		getClientId: '/get_client_id',
		registerDevice: '/register_device',
		submitBenchmark: '/submit_benchmark',
		startTrain: '/api/start-train',
		getDataset: '/get_dataset',
		readyToTrain: '/ready_to_train',
		notReadyToTrain: '/not_ready_to_train',
		reset: '/reset',
		submitGradients: '/submit_gradients',
		checkRoundStatus: '/check_round_status',
		getNewGradient: '/get_new_gradient',
	},
} as const;

export default SERVER_CONFIG;
