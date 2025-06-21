import { wsManager } from './websocketManager';

export interface DeviceInfo {
    client_id: string;
    device_type: 'pc' | 'mobile';
    gpu_memory: number;
    cpu_cores: number;
    ram_size: number;
    webgpu_supported: boolean;
}

export interface BenchmarkInfo {
    client_id: string;
    benchmark_time: number;
    batch_size: number;
    tensor_operations_per_sec: number;
}

// 设备类型检测
export function detectDeviceType(): 'pc' | 'mobile' {
	const userAgent = navigator.userAgent;
	const mobileAgents = ["Android", "iPhone", "SymbianOS", "Windows Phone", "iPad", "iPod"];
	
	for (const agent of mobileAgents) {
		if (userAgent.indexOf(agent) > 0) {
			return 'mobile';
		}
	}
	return 'pc';
}

// 获取设备硬件信息
export async function getDeviceInfo(): Promise<Partial<DeviceInfo>> {
	const deviceType = detectDeviceType();
	
	// 获取基本设备信息
	const deviceInfo: Partial<DeviceInfo> = {
		device_type: deviceType,
		cpu_cores: navigator.hardwareConcurrency || 4,
		webgpu_supported: 'gpu' in navigator
	};

	// 尝试获取GPU内存信息
	if ('gpu' in navigator) {
		try {
			const adapter = await (navigator as any).gpu.requestAdapter();
			if (adapter) {
				const limits = adapter.limits;
				deviceInfo.gpu_memory = limits.maxStorageBufferBindingSize ? 
					Math.floor(limits.maxStorageBufferBindingSize / (1024 * 1024)) : 2048;
			}
		} catch (error) {
			console.warn('Failed to get GPU info:', error);
			deviceInfo.gpu_memory = deviceType === 'pc' ? 2048 : 1024;
		}
	} else {
		deviceInfo.gpu_memory = 0;
	}

	// 估算RAM大小
	if ('memory' in (navigator as any)) {
		deviceInfo.ram_size = Math.floor((navigator as any).memory.jsHeapSizeLimit / (1024 * 1024));
	} else {
		deviceInfo.ram_size = deviceType === 'pc' ? 8192 : 4096;
	}

	return deviceInfo;
}

// 注册设备信息到服务器
export async function registerDevice(clientId: string): Promise<any> {
    try {
        // 确保WebSocket连接
        if (!wsManager.isConnected()) {
            await wsManager.connect(clientId);
        }

        const deviceInfo = await getDeviceInfo();
        const fullDeviceInfo = {
            client_id: clientId,
            ...deviceInfo
        };

        const result = await wsManager.registerDevice(fullDeviceInfo);
        console.log('Device registered via WebSocket:', result);
        return result;
    } catch (error) {
        console.error('Failed to register device via WebSocket:', error);
        throw error;
    }
}

// 提交性能基准测试结果
export async function submitBenchmark(clientId: string): Promise<any> {
    try {
        // 确保WebSocket连接
        if (!wsManager.isConnected()) {
            await wsManager.connect(clientId);
        }

        // 运行简单的性能基准测试
        const benchmarkResult = await runPerformanceBenchmark();
        
        const benchmarkInfo: BenchmarkInfo = {
            client_id: clientId,
            benchmark_time: benchmarkResult.time,
            batch_size: benchmarkResult.batchSize,
            tensor_operations_per_sec: benchmarkResult.opsPerSec
        };

        const result = await wsManager.submitBenchmark(benchmarkInfo);
        console.log('Benchmark submitted via WebSocket:', result);
        return result;
    } catch (error) {
        console.error('Failed to submit benchmark via WebSocket:', error);
        throw error;
    }
}

async function runPerformanceBenchmark(): Promise<{
    time: number;
    batchSize: number;
    opsPerSec: number;
}> {
    const startTime = performance.now();
    const batchSize = 32;
    const iterations = 100;

    // 简单的数学运算基准测试
    for (let i = 0; i < iterations; i++) {
        const data = new Float32Array(batchSize * 100);
        for (let j = 0; j < data.length; j++) {
            data[j] = Math.sin(Math.cos(Math.tan(j * 0.01))) * Math.sqrt(j);
        }
    }

    const endTime = performance.now();
    const totalTime = endTime - startTime;
    const opsPerSec = (iterations * batchSize * 100) / (totalTime / 1000);

    return {
        time: totalTime,
        batchSize,
        opsPerSec
    };
}