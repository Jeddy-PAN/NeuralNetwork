<script setup>
import { useComputeGraphStore } from '../store/computeGraphStore';
import { onMounted, onUnmounted, watch } from 'vue';
import * as d3 from 'd3';

const props = defineProps(['resetFlag']);
const emit = defineEmits(['classifyResetComplete']);

const store = useComputeGraphStore();

const width = 640;
const height = 400;
const marginTop = 20;
const marginRight = 20;
const marginBottom = 30;
const marginLeft = 40;

const classColorMap = {
	0: 'rgba(255, 100, 0, 0.2)',
	1: ' rgba(0, 60, 255, 0.3)',
};

const TOTAL_CLASS = 2;

const data = [];

const getMaxIndex = (arr) => {
	return arr.reduce((maxIndex, curVal, curIndex, arr) => {
		return curVal > arr[maxIndex] ? curIndex : maxIndex;
	}, 0);
};
const chunkArray = (array, n) => {
	const chunkedArray = [];
	for (let i = 0; i < array.length; i += n) {
		chunkedArray.push(array.slice(i, i + n));
	}
	return chunkedArray;
};
const unsubscribe = store.$onAction(
	({
		name, // action 名称
		store, // store 实例，类似 `someStore`
		args, // 传递给 action 的参数数组
		after, // 在 action 返回或解决后的钩子
		onError, // action 抛出或拒绝的钩子
	}) => {
		after(() => {
			if (name === 'setXVals') {
				if (store.xVals.length !== store.predVals.length || store.predVals.length !== store.trueVals.length) {
					console.error('Input and output length mismatch');
					return;
				}
				data.length = 0;

				// seperate data every 2 or TOTAL_CLASS as a group
				const inputVals = chunkArray(store.xVals, 2); // [[x1, y1], [x2, y2]]
				const outputVals = chunkArray(store.predVals, TOTAL_CLASS);
				const targetVals = chunkArray(store.trueVals, TOTAL_CLASS);
				// console.log('ClassifyPlot: ', inputVals, outputVals, targetVals);

				for (let i = 0; i < inputVals.length; i++) {
					const outputClass = getMaxIndex(outputVals[i]);
					const targetClass = getMaxIndex(targetVals[i]);
					const x = inputVals[i][0];
					const y = inputVals[i][1];
					if (isNaN(x) || isNaN(y)) {
						console.error(`Invalid coordinates at index ${i}: x=${x}, y=${y}`);
						continue; // 跳过无效数据点
					}
					data.push({
						x: inputVals[i][0],
						y: inputVals[i][1],
						outputClass: outputClass,
						isTrue: outputClass === targetClass ? true : false,
					});
				}

				drawClassifyPlot();
			}
		});
	}
);
watch(
	() => props.resetFlag,
	(newVal) => {
		if (newVal == true) {
			data.length = 0;
			drawClassifyPlot();
			emit('classifyResetComplete');
		}
	}
);

const drawClassifyPlot = () => {
	const svg = d3.select('.classify').attr('width', width).attr('height', height);
	svg.selectAll('*').remove();

	const xScale = d3.scaleLinear().domain([0, 1]).range([0, width]);
	const yScale = d3.scaleLinear().domain([0, 1]).range([0, height]);
	// console.log(
	// 	data.map((d) => {
	// 		return [Math.floor(xScale(d.x)), Math.floor(yScale(d.y))];
	// 	})
	// );
	const points = data.map((d) => [xScale(d.x), yScale(d.y)]);
	const delaunay = d3.Delaunay.from(points);
	const voronoi = delaunay.voronoi([0, 0, width, height]);

	for (let i = 0; i < points.length; i++) {
		svg
			.append('path')
			.attr('d', voronoi.renderCell(i))
			.attr('fill', () => {
				const pointData = data[i];
				if (pointData.outputClass in classColorMap) {
					return classColorMap[pointData.outputClass];
				} else {
					// 定义错误分类的颜色
					return 'gray';
				}
			})
			.attr('stroke', 'grey');

		if (data[i].isTrue) {
			svg
				.append('path')
				.attr('transform', `translate(${points[i][0]}, ${points[i][1]})`)
				.attr('d', d3.symbol().type(d3.symbolCircle).size(16))
				.attr('fill', 'rgba(20, 81, 125, 1)');
		} else {
			svg
				.append('path')
				.attr('transform', `translate(${points[i][0]}, ${points[i][1]})`)
				.attr('d', d3.symbol().type(d3.symbolDiamond2).size(28))
				.attr('fill', 'rgba(216, 56, 58, 1)');
		}
	}
};

onMounted(() => {
	drawClassifyPlot();
});

onUnmounted(() => {
	unsubscribe();
});
</script>

<template>
	<div id="container">
		<h3>Classify Result</h3>
		<svg class="classify"></svg>
	</div>
</template>
