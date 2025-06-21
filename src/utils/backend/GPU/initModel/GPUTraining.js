// import structs
import structs from '../wgsl_operations/structs.wgsl';

// forward computation imports
import copyInput from '../wgsl_operations/forwardOperations/copyInput.wgsl';
import forward from '../wgsl_operations/forwardOperations/forward.wgsl';
import addTensors from '../wgsl_operations/forwardOperations/addTensors.wgsl';
import correlateTensors from '../wgsl_operations/forwardOperations/correlateTensors.wgsl';
import multiplyTensors from '../wgsl_operations/forwardOperations/multiplyTensors.wgsl';
import ReLUTensor from '../wgsl_operations/forwardOperations/ReLUTensor.wgsl';
import softmaxTensor from '../wgsl_operations/forwardOperations/softmaxTensor.wgsl';
import CETensors from '../wgsl_operations/forwardOperations/CETensors.wgsl';
import MSETensor from '../wgsl_operations/forwardOperations/MSETensor.wgsl';
import OneHotTensor from '../wgsl_operations/forwardOperations/OneHotTensor.wgsl';
const forwards =
	copyInput + addTensors + correlateTensors + multiplyTensors + ReLUTensor + softmaxTensor + CETensors + OneHotTensor + MSETensor + forward;

// partial derivative computation imports
import computePartialDerivatives from '../wgsl_operations/partialDerivativesOperations/computePartialDerivatives.wgsl';
import pd_add from '../wgsl_operations/partialDerivativesOperations/pd_add.wgsl';
import pd_multiply from '../wgsl_operations/partialDerivativesOperations/pd_multiply.wgsl';
import pd_correlate from '../wgsl_operations/partialDerivativesOperations/pd_correlate.wgsl';
import pd_softmaxCE from '../wgsl_operations/partialDerivativesOperations/pd_softmaxCE.wgsl';
import pd_MSE from '../wgsl_operations/partialDerivativesOperations/pd_MSE.wgsl';
const partialDerivatives = pd_add + pd_multiply + pd_correlate + pd_softmaxCE + pd_MSE + computePartialDerivatives;

// partial derivative computation imports
import computeGradients from '../wgsl_operations/addGradientsOperations/computeGradients.wgsl';
import gr_add from '../wgsl_operations/addGradientsOperations/gr_add.wgsl';
import gr_multiply from '../wgsl_operations/addGradientsOperations/gr_multiply.wgsl';
import gr_correlate from '../wgsl_operations/addGradientsOperations/gr_correlate.wgsl';
import gr_ReLU from '../wgsl_operations/addGradientsOperations/gr_ReLU.wgsl';
import gr_softmaxCE from '../wgsl_operations/addGradientsOperations/gr_softmaxCE.wgsl';
import gr_MSE from '../wgsl_operations/addGradientsOperations/gr_MSE.wgsl';
const addGradients = gr_add + gr_multiply + gr_correlate + gr_ReLU + gr_softmaxCE + gr_MSE + computeGradients;

import updateData from '../wgsl_operations/updateDataOperations/updateData.wgsl';

import main from '../wgsl_operations/main.wgsl';

import { getxValues, getPredValues, getTrueValues, getErrorValue, getGradientValues } from './testSet.js';
import { ref } from 'vue';
import { useComputeGraphStore } from '../../../../store/computeGraphStore.js';
import { getNewGradient, checkRoundStatus, postGradients } from './network.js';
import { SERVER_CONFIG } from '../../../../config/serverConfig.ts';
import { registerDevice, submitBenchmark, detectDeviceType } from '../../CPU/tools/deviceDetector.ts';
import { getClientId } from '../../CPU/tools/client.ts';
import { submitGradientsWithWebSocket, wsManager } from '../../CPU/tools/websocketManager.ts';
import VConsole from 'vconsole';

// 默认客户端训练配置
let clientTrainingConfig = {
    framerate: 10,
    batchSize: 32,
    learningRate: 0.01
};

// 客户端初始化函数
async function initializeClient() {
    try {
        // 获取客户端ID
        const clientId = await getClientId();
        
        // 注册设备信息
        const deviceResponse = await registerDevice(clientId);
        console.log('Device registered:', deviceResponse);
        
        // 运行性能基准测试
        const benchmarkResponse = await submitBenchmark(clientId);
        console.log('Benchmark completed:', benchmarkResponse);
        
        // 获取训练配置
        if (benchmarkResponse.recommended_config) {
            clientTrainingConfig = {
                ...clientTrainingConfig,
                ...benchmarkResponse.recommended_config
            };
            console.log('Client training config:', clientTrainingConfig);
        }
        
        return {
            clientId,
            config: clientTrainingConfig,
            performance_tier: benchmarkResponse.performance_tier
        };
    } catch (error) {
        console.error('Failed to initialize client:', error);
        return null;
    }
}

const isPc = () => {
	const userAgentInfo = navigator.userAgent;
    const Agents = ["Android", "iPhone",
        "SymbianOS", "Windows Phone",
        "iPad", "iPod"];
    let flag = true;
    for (let v = 0; v < Agents.length; v++) {
        if (userAgentInfo.indexOf(Agents[v]) > 0) {
            flag = false;
            break;
        }
    }
    return flag;
}

if (process.env.NODE_ENV != "prod" && !isPc()) {
	console.log(process.env.NODE_ENV);
	const vConsole = new VConsole();
}

const stopFlag = ref(false);

function setFlagTrain() {
	stopFlag.value = false;
}

function setFlagStop() {
	stopFlag.value = true;
}

async function MatMul(Offsets, FlatData, BackwardTape, GradientTape, _iterations, data, model, forwardTape, gradientTape, backwardTape) {
	const store = useComputeGraphStore();

	// 初始化客户端配置
	const clientInfo = await initializeClient();
	if (!clientInfo) {
		console.error('Failed to initialize client, using default configuration');
	}

const numIterations = _iterations;
	const server_domain = SERVER_CONFIG.baseUrl;

	// 使用动态配置的帧率
	const framerate = clientTrainingConfig.framerate;
	
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		console.log('no adapter');
		return;
	}
	const device = await adapter.requestDevice();

	const gpuBufferOffsets = device.createBuffer({
		mappedAtCreation: true,
		size: Offsets.byteLength,
		usage: GPUBufferUsage.STORAGE,
	});
	const arrayBufferOffsets = new Float32Array(gpuBufferOffsets.getMappedRange());
	arrayBufferOffsets.set(Offsets);
	gpuBufferOffsets.unmap();

	const gpuBufferFlatData = device.createBuffer({
		mappedAtCreation: true,
		size: FlatData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
	});
	const arrayBufferFlatData = new Float32Array(gpuBufferFlatData.getMappedRange());
	arrayBufferFlatData.set(FlatData);
	gpuBufferFlatData.unmap();

	const gpuBufferBackwardTape = device.createBuffer({
		mappedAtCreation: true,
		size: BackwardTape.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	const arrayBufferBackwardTape = new Float32Array(gpuBufferBackwardTape.getMappedRange());
	arrayBufferBackwardTape.set(BackwardTape);
	gpuBufferBackwardTape.unmap();

	const gpuBufferGradientTape = device.createBuffer({
		mappedAtCreation: true,
		size: GradientTape.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});
	const arrayBufferGradientTape = new Float32Array(gpuBufferGradientTape.getMappedRange());
	arrayBufferGradientTape.set(GradientTape);
	gpuBufferGradientTape.unmap();

	const resultMatrixBuffer = device.createBuffer({
		mappedAtCreation: true,
		size: FlatData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});
	const resultMatrixArray = new Float32Array(resultMatrixBuffer.getMappedRange());
	resultMatrixArray.set(FlatData);
	resultMatrixBuffer.unmap();

	const controlBuffer = device.createBuffer({
		size: 20,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});

	data.shuffle();
	let inputData = new Float32Array(data.getInputDataBuffer());

	const inputDataBuffer = device.createBuffer({
		size: inputData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, // GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC // | GPUBufferUsage.MAP_WRITE
	});

	let trueValues = new Float32Array(data.getTrueValuesAny());
	const trueValuesBuffer = device.createBuffer({
		size: trueValues.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, // GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC // | GPUBufferUsage.MAP_WRITE
	});

	let EmptyAccuracies = new Float32Array(numIterations);

	const gpuBufferAvgAccuracy = device.createBuffer({
		mappedAtCreation: true,
		size: EmptyAccuracies.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
	});
	const arrayBufferAvgAccuracy = gpuBufferAvgAccuracy.getMappedRange();
	new Float32Array(arrayBufferAvgAccuracy).set(EmptyAccuracies);
	gpuBufferAvgAccuracy.unmap();

	//init layout
	const bindGroupLayout = device.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'read-only-storage',
				},
			},
			{
				binding: 1,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},

			{
				binding: 2,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},
			{
				binding: 3,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},
			{
				binding: 4,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},
			{
				binding: 5,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},
			{
				binding: 6,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},
			{
				binding: 7,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},
		],
	});

	const bindGroup = device.createBindGroup({
		layout: bindGroupLayout,
		entries: [
			{
				binding: 0,
				resource: {
					buffer: gpuBufferOffsets,
				},
			},
			{
				binding: 1,
				resource: {
					buffer: gpuBufferFlatData,
				},
			},
			{
				binding: 2,
				resource: {
					buffer: gpuBufferBackwardTape,
				},
			},
			{
				binding: 3,
				resource: {
					buffer: controlBuffer,
				},
			},
			{
				binding: 4,
				resource: {
					buffer: inputDataBuffer,
				},
			},
			{
				binding: 5,
				resource: {
					buffer: gpuBufferGradientTape,
				},
			},
			{
				binding: 6,
				resource: {
					buffer: trueValuesBuffer,
				},
			},
			{
				binding: 7,
				resource: {
					buffer: gpuBufferAvgAccuracy,
				},
			},
		],
	});

	const shaderModule = device.createShaderModule({
		code: structs + forwards + partialDerivatives + addGradients + updateData + main,
	});

	const computePipeline = device.createComputePipeline({
		layout: device.createPipelineLayout({
			bindGroupLayouts: [bindGroupLayout],
		}),
		compute: {
			module: shaderModule,
			entryPoint: 'main',
		},
	});

	const gpuReadBuffer = device.createBuffer({
		size: FlatData.byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	});

	const gpuReadAvgAccuracyBuffer = device.createBuffer({
		size: EmptyAccuracies.byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	});

	const gpuSetBuffer = device.createBuffer({
		size: FlatData.byteLength,
		usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
	});

	// var startTime = performance.now();
	let xValues_all = [];
	let predValues_all = [];
	let trueValues_all = [];
	let errorsArray = [];

	console.log('enter training with config:', clientTrainingConfig);
	const startTime = performance.now();
	
	// 检查WebSocket连接状态
	const clientId = localStorage.getItem('client_id');
	if (!clientId) {
		throw new Error('Client ID not found. Please start training from the UI.');
	}
	
	if (!wsManager.isConnected()) {
		throw new Error('WebSocket not connected. Please start training from the UI.');
	}
	
	console.log('Using existing WebSocket connection for training');

	let localIteration = 0;
	
	// 使用无限循环，由服务端控制训练轮次
	while (true) {
		if (stopFlag.value == true) {
			console.log('Training stopped by user');
			return;
		}
		
		const iterationStartTime = performance.now();
		
		data.shuffle();

		inputData = new Float32Array(data.getInputDataBuffer());
		trueValues = new Float32Array(data.getTrueValuesAny());
		device.queue.writeBuffer(inputDataBuffer, 0, inputData.buffer, 0, inputData.byteLength);

		let inputTensorId = data.tensorInputId;
		let commandEncoder = device.createCommandEncoder();
		let passEncoder = commandEncoder.beginComputePass();
		let control = new Float32Array([inputTensorId, -1, -1, 0, localIteration]);
		device.queue.writeBuffer(controlBuffer, 0, control.buffer, 0, control.byteLength);

		passEncoder.setPipeline(computePipeline);
		passEncoder.setBindGroup(0, bindGroup);
		let workgroupCountX = Math.ceil(model.tensors[inputTensorId].rows / 16);
		let workgroupCountY = Math.ceil(model.tensors[inputTensorId].cols / 16);
		passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
		passEncoder.end();
		let gpuCommands = commandEncoder.finish();
		device.queue.submit([gpuCommands]);

		//compute type 0 - update trueValues
		device.queue.writeBuffer(trueValuesBuffer, 0, trueValues.buffer, 0, trueValues.byteLength);
		let trueValuesTensorId = data.tensorTrueId;
		commandEncoder = device.createCommandEncoder();
		passEncoder = commandEncoder.beginComputePass();
		control = new Float32Array([trueValuesTensorId, -1, -1, 0, localIteration]);
		device.queue.writeBuffer(controlBuffer, 0, control.buffer, 0, control.byteLength);
		passEncoder.setPipeline(computePipeline);
		passEncoder.setBindGroup(0, bindGroup);
		workgroupCountX = Math.ceil(model.tensors[trueValuesTensorId].rows / 16);
		workgroupCountY = Math.ceil(model.tensors[trueValuesTensorId].cols / 16);
		passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
		passEncoder.end();
		gpuCommands = commandEncoder.finish();
		device.queue.submit([gpuCommands]);

		//compute type 1 - forward
		const numInferences = forwardTape.length;
		for (let i = 0; i < numInferences; i++) {
			const curTensorId = forwardTape[i];

			const commandEncoder = device.createCommandEncoder();
			const passEncoder = commandEncoder.beginComputePass();

			let control = new Float32Array([curTensorId, -1, -1, 1, localIteration]);
			device.queue.writeBuffer(controlBuffer, 0, control.buffer, 0, control.byteLength);

			passEncoder.setPipeline(computePipeline);
			passEncoder.setBindGroup(0, bindGroup);

			let workgroupCountX = Math.ceil(model.tensors[curTensorId].rows / 16);
			let workgroupCountY = Math.ceil(model.tensors[curTensorId].cols / 16);

			passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
			passEncoder.end();

			const gpuCommands = commandEncoder.finish();
			device.queue.submit([gpuCommands]);
		}

		await device.queue.onSubmittedWorkDone();

		const readCommandEncoder = device.createCommandEncoder();
		readCommandEncoder.copyBufferToBuffer(
			gpuBufferFlatData,
			0,
			gpuReadBuffer,
			0,
			FlatData.byteLength
		);
		readCommandEncoder.copyBufferToBuffer(gpuBufferAvgAccuracy, 0, gpuReadAvgAccuracyBuffer, 0, EmptyAccuracies.byteLength);

		const readCommands = readCommandEncoder.finish();
		device.queue.submit([readCommands]);

		await gpuReadBuffer.mapAsync(GPUMapMode.READ);
		await gpuReadAvgAccuracyBuffer.mapAsync(GPUMapMode.READ);
		const arrayBuffer = new Float32Array(gpuReadBuffer.getMappedRange());

		predValues_all.push(getPredValues(arrayBuffer, model, Offsets));
		trueValues_all.push(getTrueValues(arrayBuffer, model, Offsets));
		errorsArray.push(getErrorValue(arrayBuffer, model, Offsets));
		xValues_all.push(getxValues(arrayBuffer, data, Offsets));

		gpuReadBuffer.unmap();
		gpuReadAvgAccuracyBuffer.unmap();

		// 每帧率次数更新UI
		if (localIteration % framerate === 0) {
			let xVals = [].concat(...xValues_all);
			let predVals = [].concat(...predValues_all);
			let trueVals = [].concat(...trueValues_all);
			let avgError = errorsArray.reduce((sum, error) => sum + error, 0) / errorsArray.length;

			console.log('avgError', avgError, 'localIteration', localIteration);

			store.setModelIterations(localIteration);
			store.setAvgError(avgError);
			store.setPredVals(predVals);
			store.setTrueVals(trueVals);
			store.setXVals(xVals);

			if (avgError < 0.2) {
				const endTime = performance.now();
				const elapsedTime = endTime - startTime;
				console.log('Elapsed time for whole Training', elapsedTime, 'ms');
				console.log('Training complete with avgError:', avgError);
				store.setTrainingComplete(true);
				stopFlag.value = true;
				break;
			}

			// 清空数据数组，准备下一轮数据收集
			predValues_all = [];
			trueValues_all = [];
			xValues_all = [];
			errorsArray = [];
		}

		// compute type 2 - compute partial derivatives
		const numPds = gradientTape.length / 2;
		for (let i = 0; i < numPds; i++) {
			const parTensorId = gradientTape[2 * i];
			const curTensorId = gradientTape[2 * i + 1];

			const commandEncoder = device.createCommandEncoder();
			const passEncoder = commandEncoder.beginComputePass();
			let control = new Float32Array([curTensorId, parTensorId, -1, 2, localIteration]);
			device.queue.writeBuffer(controlBuffer, 0, control.buffer, 0, control.byteLength);
			passEncoder.setPipeline(computePipeline);
			passEncoder.setBindGroup(0, bindGroup);

			let workgroupCountX = 1;
			let workgroupCountY = 1;

			if (model.tensors[curTensorId].type == 1) {
				workgroupCountX = Math.ceil(model.tensors[curTensorId].rows / 16);
				workgroupCountY = Math.ceil(model.tensors[curTensorId].cols / 16);
			} else if (model.tensors[curTensorId].type == 2) {
				let isRightMultiplicator = model.tensors[parTensorId].isRightMultiplicator;
				if (isRightMultiplicator) {
					workgroupCountX = Math.ceil(model.tensors[curTensorId].rows / 16);
					workgroupCountY = Math.ceil(model.tensors[parTensorId].rows / 16);
				} else {
					workgroupCountX = Math.ceil(model.tensors[parTensorId].cols / 16);
					workgroupCountY = Math.ceil(model.tensors[curTensorId].cols / 16);
				}
			} else if (model.tensors[curTensorId].type == 4) {
				workgroupCountX = Math.ceil(model.tensors[parTensorId].rows / 16);
				workgroupCountY = Math.ceil(model.tensors[parTensorId].cols / 16);
			} else if (model.tensors[curTensorId].type == 5) {
				workgroupCountX = Math.ceil(model.tensors[parTensorId].rows / 16);
				workgroupCountY = Math.ceil(model.tensors[parTensorId].cols / 16);
			} else if (model.tensors[curTensorId].type == 7) {
				workgroupCountX = Math.ceil(model.tensors[parTensorId].rows / 16);
				workgroupCountY = Math.ceil(model.tensors[parTensorId].cols / 16);
			}

			passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
			passEncoder.end();

			let gpuCommands = commandEncoder.finish();
			device.queue.submit([gpuCommands]);
		}

		const numGrds = gradientTape.length / 2;

		for (let i = 0; i < numGrds; i++) {
			const currTensorId = gradientTape[2 * i];
			const currChildId = gradientTape[2 * i + 1];

			let commandEncoder = device.createCommandEncoder();
			let passEncoder = commandEncoder.beginComputePass();
			let control = new Float32Array([
				currTensorId,
				-1,
				currChildId,
				3,
				localIteration,
			]);
			device.queue.writeBuffer(controlBuffer, 0, control.buffer, 0, control.byteLength);
			passEncoder.setPipeline(computePipeline);
			passEncoder.setBindGroup(0, bindGroup);

			let workgroupCountX = Math.ceil(model.tensors[currTensorId].rows / 16);
			let workgroupCountY = Math.ceil(model.tensors[currTensorId].cols / 16);
			passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
			passEncoder.end();

			let gpuCommands = commandEncoder.finish();
			device.queue.submit([gpuCommands]);
		}

		// 获取梯度
		commandEncoder = device.createCommandEncoder();
		commandEncoder.copyBufferToBuffer(gpuBufferFlatData, 0, gpuReadBuffer, 0, FlatData.byteLength);
		gpuCommands = commandEncoder.finish();
		device.queue.submit([gpuCommands]);

		await gpuReadBuffer.mapAsync(GPUMapMode.READ);
		const dataReadBuffer = new Float32Array(gpuReadBuffer.getMappedRange());
		const gradientValues = getGradientValues(dataReadBuffer, model, Offsets);

		// 计算这轮的计算时间
		const iterationTime = performance.now() - iterationStartTime;

		// 使用WebSocket提交梯度，由服务端管理round_id
		let responseJson;
		try {
			responseJson = await submitGradientsWithWebSocket(
				localStorage.getItem('client_id'), 
				gradientValues, 
				iterationTime
			);

			if (responseJson.status !== 'complete') {
				throw new Error('Error: Round not completed');
			}

		} catch (error) {
			console.error('WebSocket gradient submission failed:', error);
			
			// 如果WebSocket失败，仍然尝试WebSocket方式（不再回退到HTTP）
			console.log('Retrying via WebSocket...');
			try {
				responseJson = await postGradients(
					localStorage.getItem('client_id'), 
					gradientValues, 
					iterationTime
				);

				// WebSocket轮询等待
				while (responseJson.status == 'waiting') {
					await new Promise((resolve) => setTimeout(resolve, 400));

					if (stopFlag.value == true) return;

					responseJson = await checkRoundStatus();
					console.log('LOG: Waiting for other clients (WebSocket retry): ', responseJson);
				}

				if (responseJson.status !== 'complete') {
					throw new Error('Error: Round not completed');
				}
			} catch (retryError) {
				console.error('WebSocket retry also failed:', retryError);
				throw retryError;
			}
		}

		gpuReadBuffer.unmap();

		// 获取新梯度从服务器 - 通过WebSocket
		const responseNewGradJson = await getNewGradient();

		const newGradientValues = responseNewGradJson.new_gradient;
		const flattenedGradientValues = newGradientValues.flat();

		const newGradientValuesBuffer = new Float32Array(flattenedGradientValues);

		// 更新模型参数
		const N = 15;
		let gradientOffset = 0;
		const sourceBufferSize = newGradientValuesBuffer.byteLength;
		const destinationBufferSize = gpuBufferFlatData.size;

		for (let tensor of model.tensors) {
			const rows = tensor.rows;
			const cols = tensor.cols;
			const tensorSizeInElements = rows * cols;

			const baseIndex = 3 + tensor.id * N;
			const offsetDataIndex = baseIndex + 6;
			const flatDataOffset = Offsets[offsetDataIndex] * 4;

			if (gradientOffset + tensorSizeInElements * 4 > sourceBufferSize) {
				console.log('sourceBufferSize', sourceBufferSize);
				throw new Error('源数据的偏移量和大小超过了源缓冲区的大小。');
			}

			if (flatDataOffset + tensorSizeInElements * 4 > destinationBufferSize) {
				throw new Error('目标缓冲区的偏移量和大小超过了目标缓冲区的大小。');
			}
			device.queue.writeBuffer(
				gpuBufferFlatData,
				flatDataOffset,
				newGradientValuesBuffer,
				gradientOffset,
				tensorSizeInElements
			);
			gradientOffset += tensorSizeInElements;
		}

		await device.queue.onSubmittedWorkDone();

		// compute type 4 - update data
		const numUpdates = backwardTape.length;
		for (let i = 3; i < numUpdates; ++i) {
			const currTensorId = backwardTape[i];

			let commandEncoder = device.createCommandEncoder();
			let passEncoder = commandEncoder.beginComputePass();
			let control = new Float32Array([
				currTensorId,
				-1,
				-1,
				4,
				localIteration,
			]);
			device.queue.writeBuffer(controlBuffer, 0, control.buffer, 0, control.byteLength);
			passEncoder.setPipeline(computePipeline);
			passEncoder.setBindGroup(0, bindGroup);

			let workgroupCountX = Math.ceil(model.tensors[currTensorId].rows / 16);
			let workgroupCountY = Math.ceil(model.tensors[currTensorId].cols / 16);
			passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
			passEncoder.end();

			let gpuCommands = commandEncoder.finish();
			device.queue.submit([gpuCommands]);
		}

		localIteration++;
	}

	console.log('Training complete');
	
	return;
}

export { MatMul, setFlagStop, setFlagTrain };
