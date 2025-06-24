import { Tensor } from '../tools/TensorClass';
import { Model } from '../tools/ModelClass';
import { MatMul } from '../../GPU/initModel/GPUTraining';
import Data from '../tools/DataClass';
import { Ref } from 'vue';

let params = {
	batch_size: 32,
    learning_rate: 0.3,
    iterations: 2000
}
export async function writeParams(object: any) {
	params.batch_size = object.batch_size;
	params.iterations = object.iterations;
	params.learning_rate = object.learning_rate;
}

function setUpModel(data: Data) {
	// #TODO Hardcode tensors
	const tensors = [
		{
			id: 0,
			type: 'none',
			initialization: 'ones',
			rows: 1,
			cols: 48,
			requiresGradient: false,
			parents: [],
			children: [4],
			isRightMultiplicator: true,
			isLast: false,
			isTrue: false,
			isInput: false,
			isOutput: false,
			metaDims: [1],
		},
		{
			id: 1,
			type: 'none',
			initialization: 'random',
			rows: 64,
			cols: 1,
			requiresGradient: true,
			parents: [],
			children: [4],
			isRightMultiplicator: false,
			isLast: false,
			isTrue: false,
			isInput: false,
			isOutput: false,
			metaDims: [64],
		},
		{
			id: 2,
			type: 'input',
			initialization: 'zeros',
			rows: 2,
			cols: 48,
			requiresGradient: false,
			parents: [],
			children: [5],
			isRightMultiplicator: true,
			isLast: false,
			isTrue: false,
			isInput: true,
			isOutput: false,
			metaDims: [2],
		},
		{
			id: 3,
			type: 'none',
			initialization: 'random',
			rows: 64,
			cols: 2,
			requiresGradient: true,
			parents: [],
			children: [5],
			isRightMultiplicator: false,
			isLast: false,
			isTrue: false,
			isInput: false,
			isOutput: false,
			metaDims: [64],
		},
		{
			id: 4,
			type: 'mult',
			initialization: 'zeros',
			rows: 64,
			cols: 48,
			requiresGradient: true,
			parents: [1, 0],
			children: [6],
			isRightMultiplicator: true,
			isLast: false,
			isTrue: false,
			isInput: false,
			isOutput: false,
			metaDims: [64],
		},
		{
			id: 5,
			type: 'mult',
			initialization: 'zeros',
			rows: 64,
			cols: 48,
			requiresGradient: true,
			parents: [3, 2],
			children: [6],
			isRightMultiplicator: false,
			isLast: false,
			isTrue: false,
			isInput: false,
			isOutput: false,
			metaDims: [64],
		},
		{
			id: 6,
			type: 'add',
			initialization: 'zeros',
			rows: 64,
			cols: 48,
			requiresGradient: true,
			parents: [5, 4],
			children: [8],
			isRightMultiplicator: false,
			isLast: false,
			isTrue: false,
			isInput: false,
			isOutput: false,
			metaDims: [64],
		},
		{
			id: 7,
			type: 'none',
			initialization: 'random',
			rows: 2,
			cols: 64,
			requiresGradient: true,
			parents: [],
			children: [9],
			isRightMultiplicator: false,
			isLast: false,
			isTrue: false,
			isInput: false,
			isOutput: false,
			metaDims: [2],
		},
		{
			id: 8,
			type: 'ReLU',
			initialization: 'zeros',
			rows: 64,
			cols: 48,
			requiresGradient: true,
			parents: [6],
			children: [9],
			isRightMultiplicator: true,
			isLast: false,
			isTrue: false,
			isInput: false,
			isOutput: false,
			metaDims: [64],
		},
		{
			id: 9,
			type: 'mult',
			initialization: 'zeros',
			rows: 2,
			cols: 48,
			requiresGradient: true,
			parents: [7, 8],
			children: [11],
			isRightMultiplicator: false,
			isLast: false,
			isTrue: false,
			isInput: false,
			isOutput: false,
			metaDims: [2],
		},
		{
			id: 10,
			type: 'isTrue',
			initialization: 'zeros',
			rows: 2,
			cols: 48,
			requiresGradient: false,
			parents: [],
			children: [12],
			isRightMultiplicator: false,
			isLast: false,
			isTrue: true,
			isInput: false,
			isOutput: false,
			metaDims: [2],
		},
		{
			id: 11,
			type: 'softmax',
			initialization: 'zeros',
			rows: 2,
			cols: 48,
			requiresGradient: true,
			parents: [9],
			children: [12],
			isRightMultiplicator: true,
			isLast: false,
			isTrue: false,
			isInput: false,
			isOutput: true,
			metaDims: [2],
		},
		{
			id: 12,
			type: 'CE',
			initialization: 'zeros',
			rows: 1,
			cols: 48,
			requiresGradient: true,
			parents: [10, 11],
			children: [],
			isRightMultiplicator: false,
			isLast: true,
			isTrue: false,
			isInput: false,
			isOutput: false,
			metaDims: [1],
		},
	];

	console.log('params', params);
	// const learningRate = params.learning_rate;
	// const momentum = 0.9;
	// const batchSize = params.batch_size;
	// const iterations = params.iterations;
	const learningRate = 0.3;
	const momentum = 0.9;
	const batchSize = 64;
	const iterations = 3000;

	const model = new Model();
	model.numTensors = tensors.length - 1;

	// init tensors in model
	for (let i = 0; i < tensors.length; i++) {
		const type = tensors[i].type;
		const tensorId = Number(tensors[i].id);
		const rows = tensors[i].rows;
		const cols = tensors[i].cols;
		const requiresGradient = tensors[i].requiresGradient;
		const tensor = new Tensor(type, rows, cols, requiresGradient, tensorId);
		const initialization = tensors[i].initialization;

		tensor.parents = tensors[i].parents;
		tensor.children = tensors[i].children;
		tensor.isRightMultiplicator = tensors[i].isRightMultiplicator;

		if (initialization == 'random' && type == 'none') {
			tensor.setRandomData();
		} else if (initialization == 'ones' && type == 'none') {
			tensor.setAllOnes();
		}
		if (tensors[i].isLast || type == 'MSE' || type == 'CE') {
			model.lastTensor = tensorId;
		}
		if (tensors[i].isInput) {
			data.setInputTensor(tensor);

			// data.input_rows = tensor.rows;
			// data.input_cols = tensor.cols;
			// data.tensorInputId = tensor.id;
			tensor.requiresGradient = false;
		}
		if (tensors[i].isTrue) {
			data.setTrueTensor(tensor);
			// data.true_rows = tensor.rows;
			// data.true_cols = tensor.cols;
			// data.tensorTrueId = tensor.id;
			model.trueTensor = tensorId;

			tensor.requiresGradient = false;
		}
		if (tensors[i].isOutput) {
			model.outputTensor = tensorId;
		}
		model.tensors.push(tensor);
	}

	const maxNumIterations = iterations;
	model.learningRate = learningRate;
	model.momentum = momentum;
	model.batchSize = batchSize;

	let maxBlockSize: number;
	maxBlockSize = 0;
	for (let i = 0; i < model.numTensors + 1; i++) {
		if (model.tensors[i].rows > maxBlockSize) {
			maxBlockSize = model.tensors[i].rows;
		}
		if (model.tensors[i].cols > maxBlockSize) {
			maxBlockSize = model.tensors[i].cols;
		}
	}

	// init partial Derivatives to zero with proper lengths
	for (let i = 0; i < model.numTensors + 1; i++) {
		if (model.tensors[i].type == 0) {
			continue;
		} else if (model.tensors[i].type == 1) {
			const par1 = model.tensors[i].parents[0];
			const n = model.tensors[par1].cols;
			model.tensors[i].partialDerivativeLeft = new Array<number>(n * n).fill(0);
			model.tensors[i].partialDerivativeRight = new Array<number>(n * n).fill(0);
		}

		// multiply
		else if (model.tensors[i].type == 2) {
			const par1 = model.tensors[i].parents[0];
			const par2 = model.tensors[i].parents[1];
			const m = model.tensors[par1].rows;
			const n = model.tensors[par1].cols; // ( = model.tensors[par2].rows  )
			const k = model.tensors[par2].cols;
			model.tensors[i].partialDerivativeLeft = new Array<number>(k * n).fill(0);
			model.tensors[i].partialDerivativeRight = new Array<number>(m * n).fill(0);
		}

		// ReLU
		else if (model.tensors[i].type == 3) {
			const par1 = model.tensors[i].parents[0];
			const m = model.tensors[par1].rows;
			const n = model.tensors[par1].cols;
			model.tensors[i].partialDerivativeLeft = new Array<number>(m * n).fill(0);
			model.tensors[i].partialDerivativeRight = new Array<number>(1).fill(0);
		}

		// softmax
		else if (model.tensors[i].type == 4) {
			const par1 = model.tensors[i].parents[0];
			const m = model.tensors[par1].rows;
			const n = model.tensors[par1].cols;
			model.tensors[i].partialDerivativeLeft = new Array<number>(m * n).fill(0);
			model.tensors[i].partialDerivativeRight = new Array<number>(1).fill(0);
		}

		// CE
		else if (model.tensors[i].type == 5) {
			const par1 = model.tensors[i].parents[0];
			// const m = model.tensors[par1].rows;
			const n = model.tensors[par1].cols;
			model.tensors[i].partialDerivativeLeft = new Array<number>(n * n).fill(0);
			model.tensors[i].partialDerivativeRight = new Array<number>(n * n).fill(0);
		}

		// OneHot
		else if (model.tensors[i].type == 6) {
		}

		// MSE
		else if (model.tensors[i].type == 7) {
			const par1 = model.tensors[i].parents[0];
			// const m = model.tensors[par1].rows;
			const n = model.tensors[par1].cols;
			model.tensors[i].partialDerivativeLeft = new Array<number>(n * n).fill(0);
			model.tensors[i].partialDerivativeRight = new Array<number>(n * n).fill(0);
		}

		// conv2d
		else if (model.tensors[i].type == 8) {
			const par1 = model.tensors[i].parents[0]; // is X
			const par2 = model.tensors[i].parents[1]; // is kernel
			const m = model.tensors[par1].rows;
			const n = model.tensors[par1].cols;
			const m_k = model.tensors[par2].rows;
			const n_k = model.tensors[par2].cols;
			model.tensors[i].partialDerivativeLeft = new Array<number>(m_k * n_k).fill(0); // derivateive will be X
			model.tensors[i].partialDerivativeRight = new Array<number>(m * n).fill(0); // derivate will be rotate180(kernel)
		}
	}

	let flatData: any[] = [];
	const offsets: number[] = []; // where does each tensor start in the flattened array
	let tensorOffsets: number[] = [];
	let offset = 0;

	tensorOffsets.push(model.numTensors);
	tensorOffsets.push(0); // set to total number of floats after completely filled
	tensorOffsets.push(maxBlockSize);

	for (let i = 0; i < model.numTensors + 1; i++) {
		offsets.push(offset); // tensor i starts at index offset in flatData
		flatData.push(model.tensors[i].type);
		flatData.push(model.tensors[i].isRightMultiplicator);
		flatData.push(model.tensors[i].requiresGradient);
		flatData.push(model.tensors[i].rows);
		flatData.push(model.tensors[i].cols);
		flatData = flatData.concat(Array.from(model.tensors[i].data));
		flatData = flatData.concat(Array.from(model.tensors[i].gradientData));
		flatData = flatData.concat(Array.from(model.tensors[i].velocity_momentum));
		flatData = flatData.concat(Array.from(model.tensors[i].velocity_RMSProp));
		flatData = flatData.concat(Array.from(model.tensors[i].children));
		flatData = flatData.concat(Array.from(model.tensors[i].parents));
		flatData = flatData.concat(Array.from(model.tensors[i].partialDerivativeLeft));
		flatData = flatData.concat(Array.from(model.tensors[i].partialDerivativeRight));
		flatData.push(model.tensors[i].metaDims.length);
		flatData = flatData.concat(Array.from(model.tensors[i].metaDims));

		if (i == data.tensorInputId) {
			data.inputOffset = offset + 6;
		}
		if (i == data.tensorTrueId) {
			data.trueOffset = offset + 6;
		}
		// calc offset
		tensorOffsets = tensorOffsets.concat([
			offset,
			++offset,
			++offset,
			++offset,
			++offset, // m (rows)
			++offset, // n (cols)
			offset + model.tensors[i].data.length,
			offset + model.tensors[i].data.length + model.tensors[i].gradientData.length,
			offset + model.tensors[i].data.length + model.tensors[i].gradientData.length + model.tensors[i].velocity_momentum.length,
			offset + model.tensors[i].data.length + model.tensors[i].gradientData.length + model.tensors[i].velocity_momentum.length * 2, // times 2 because we have the velocity_RMSProp as well
			offset +
				model.tensors[i].data.length +
				model.tensors[i].gradientData.length +
				model.tensors[i].velocity_momentum.length * 2 +
				model.tensors[i].children.length,
			offset +
				model.tensors[i].data.length +
				model.tensors[i].gradientData.length +
				model.tensors[i].velocity_momentum.length * 2 +
				model.tensors[i].children.length +
				model.tensors[i].parents.length,
			offset +
				model.tensors[i].data.length +
				model.tensors[i].gradientData.length +
				model.tensors[i].velocity_momentum.length * 2 +
				model.tensors[i].children.length +
				model.tensors[i].parents.length +
				model.tensors[i].partialDerivativeLeft.length,
			offset +
				model.tensors[i].data.length +
				model.tensors[i].gradientData.length +
				model.tensors[i].velocity_momentum.length * 2 +
				model.tensors[i].children.length +
				model.tensors[i].parents.length +
				model.tensors[i].partialDerivativeLeft.length +
				model.tensors[i].partialDerivativeRight.length,
			offset +
				model.tensors[i].data.length +
				model.tensors[i].gradientData.length +
				model.tensors[i].velocity_momentum.length * 2 +
				model.tensors[i].children.length +
				model.tensors[i].parents.length +
				model.tensors[i].partialDerivativeLeft.length +
				model.tensors[i].partialDerivativeRight.length +
				1,
		]);
		// update offset
		offset =
			offset +
			model.tensors[i].data.length +
			model.tensors[i].gradientData.length +
			model.tensors[i].velocity_momentum.length * 2 +
			model.tensors[i].children.length +
			model.tensors[i].parents.length +
			model.tensors[i].partialDerivativeLeft.length +
			model.tensors[i].partialDerivativeRight.length +
			1 +
			model.tensors[i].metaDims.length;
	}

	tensorOffsets[1] = flatData.length;

	// build sequence for the backwards pass for computing the partialDerivatives
	let queueCount: number[] = [];
	for (let i = 0; i < model.numTensors + 1; i++) {
		queueCount.push(0);
	}
	const backwardTape: number[] = [];
	backwardTape.push(0); //numTensors + 1
	backwardTape.push(model.learningRate);
	backwardTape.push(model.momentum);

	let queue: number[] = [model.lastTensor];
	while (queue.length > 0) {
		const curTensor = queue.shift();
		if (curTensor === undefined) {
			break;
		}
		const parents = model.tensors[curTensor].parents;
		backwardTape.push(curTensor);
		for (const parTensor of parents) {
			if (model.tensors[parTensor].requiresGradient) {
				queueCount[parTensor]++;
				// all children of par tensor are traversed already
				if (queueCount[parTensor] == model.tensors[parTensor].children.length) {
					queue.push(parTensor);
				}
			}
		}
	}

	backwardTape[0] = model.numTensors + 1; //#TODO why plus 1

	// build sequence for the backwards pass for computing the gradients
	queueCount = [];
	for (let i = 0; i < model.numTensors + 1; i++) {
		queueCount.push(0);
	}
	const gradientTape: number[] = [];
	queue = [model.lastTensor];
	while (queue.length > 0) {
		const curTensor = queue.shift();
		if (curTensor === undefined) {
			break;
		}
		const parents = model.tensors[curTensor].parents;
		for (const parTensor of parents) {
			if (model.tensors[parTensor].requiresGradient) {
				queueCount[parTensor]++;
				gradientTape.push(parTensor);
				gradientTape.push(curTensor);
				if (queueCount[parTensor] == model.tensors[parTensor].children.length) {
					queue.push(parTensor);
				}
			}
		}
	}

	// build sequence for the forwards pass
	queueCount = [];
	for (let i = 0; i < model.numTensors + 1; i++) {
		queueCount.push(0);
	}
	const forwardTape: number[] = [];
	queue = [model.lastTensor];
	while (queue.length > 0) {
		const curTensor = queue.shift();
		if (curTensor === undefined) {
			break;
		}
		const parents = model.tensors[curTensor].parents;
		forwardTape.push(curTensor);
		for (const par of parents) {
			if (model.tensors[par].type != 0) {
				queueCount[par]++;
				if (queueCount[par] == model.tensors[par].children.length) {
					queue.push(par);
				}
			}
		}
	}
	forwardTape.reverse(); // forward flows from parent to child

	const f32TensorOffsets = new Float32Array(tensorOffsets);
	const f32FlatData = new Float32Array(flatData);
	const f32BackwardTape = new Float32Array(backwardTape);
	const f32GradientTape = new Float32Array(gradientTape);

	console.log('data', data);
	MatMul(f32TensorOffsets, f32FlatData, f32BackwardTape, f32GradientTape, maxNumIterations, data, model, forwardTape, gradientTape, backwardTape);
}

export default setUpModel;
