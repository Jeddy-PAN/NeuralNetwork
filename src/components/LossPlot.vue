<script setup>
import { useComputeGraphStore } from '../store/computeGraphStore';
import { onMounted, onUnmounted, watch } from 'vue';
import * as d3 from 'd3';

const props = defineProps(['resetFlag']);
const emit = defineEmits(['resetComplete']);

const store = useComputeGraphStore();

const width = 640;
const height = 400;
const marginTop = 20;
const marginRight = 20;
const marginBottom = 30;
const marginLeft = 40;

const data = [{ x: 0, y: 1 }];

const unsubscribe = store.$onAction(
	({
		name, // action 名称
		store, // store 实例，类似 `someStore`
		args, // 传递给 action 的参数数组
		after, // 在 action 返回或解决后的钩子
		onError, // action 抛出或拒绝的钩子
	}) => {
		after(() => {
			if (name === 'setAvgError') {
				data.push({ x: store.iterations, y: store.avgError });
				drawLossPlot();
			}
		});
	}
);

watch(
	() => props.resetFlag,
	(newVal) => {
		if (newVal == true) {
			data.length = 1;
			drawLossPlot();
			emit('resetComplete');
		}
	}
);

const drawLossPlot = () => {
	const svg = d3.select('.loss-plot').attr('width', width).attr('height', height);
	svg.selectAll('*').remove();

	if (data.length === 0) {
		console.warn('Data array is empty');
		return;
	}

	const x = d3
		.scaleLinear()
		.domain(d3.extent(data, (d) => d.x))
		.range([marginLeft, width - marginRight]);

	const y = d3
		.scaleLinear()
		.domain([
			0,
			Math.min(
				d3.max(data, (d) => d.y),
				1
			),
		])
		.range([height - marginBottom, marginTop])
		.clamp(true);

	const line = d3
		.line()
		.x((d) => x(d.x))
		.y((d) => y(d.y))
		.curve(d3.curveBasis);
	const xAxis = d3.axisBottom(x).ticks(5);
	const yAxis = d3.axisLeft(y).ticks(5);

	svg
		.append('g')
		.attr('transform', `translate(0,${height - marginBottom})`)
		.call(xAxis);
	svg.append('g').attr('transform', `translate(${marginLeft},0)`).call(yAxis);
	svg.append('g').append('path').datum(data).attr('d', line).attr('fill', 'none').attr('stroke', 'steelblue');
};

onMounted(() => {
	drawLossPlot();
});

onUnmounted(() => {
	unsubscribe();
});
</script>

<template>
	<div id="container">
		<h3>Loss Plot</h3>
		<svg class="loss-plot"></svg>
	</div>
</template>
