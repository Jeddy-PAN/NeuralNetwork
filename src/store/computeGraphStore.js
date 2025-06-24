import { defineStore } from 'pinia';

export const useComputeGraphStore = defineStore('computeGraph', {
	state: () => ({
		tensors: [],
		clicked: 0,
		lastTensor: -1,
		xVals: [],
		predVals: [],
		trueVals: [],
		avgError: 1,
		iterations: 1,
		learningRate: 0.01,
		batchSize: 48,
		momentum: 0.9,
		edgesActive: false,
		dataset: '',
		predefinedModel: 'Classification1raw',
		breakTraining: false,
	}),
	actions: {
		setInitialState(payload) {
			Object.assign(this.$state, payload);
		},
		setTensorNodes(payload) {
			this.tensors = payload;
		},
		setClicked() {
			this.clicked += 1;
		},
		setLastTensor(payload) {
			this.lastTensor = payload;
		},
		setXVals(payload) {
			this.xVals = payload;
		},
		setPredVals(payload) {
			this.predVals = payload;
		},
		setTrueVals(payload) {
			this.trueVals = payload;
		},
		setAvgError(payload) {
			if (isNaN(payload)) {
				return;
			}
			this.avgError = payload;
		},
		setModelIterations(payload) {
			this.iterations = payload;
		},
		setModelLearningRate(payload) {
			this.learningRate = payload;
		},
		setModelMomentum(payload) {
			this.momentum = payload;
		},
		setEdgesActive(payload) {
			this.edgesActive = payload;
		},
		setDataSetName(payload) {
			this.dataset = payload;
		},
		setBatchSize(payload) {
			this.batchSize = payload;
		},
		setPredefinedModel(payload) {
			this.predefinedModel = payload;
		},
		setBreakTraining(payload) {
			this.breakTraining = payload;
		},
	},
});
