<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import {
	DatasetComponent,
	GridComponent,
	LegendComponent,
	TitleComponent,
	TooltipComponent,
	TransformComponent,
} from "echarts/components";
import { BarChart, GaugeChart, LineChart, PieChart } from "echarts/charts";
import { LabelLayout, UniversalTransition } from "echarts/features";
import { init, use, type EChartsOption, type EChartsType } from "echarts/core";
import { CanvasRenderer, SVGRenderer } from "echarts/renderers";

use([
	BarChart,
	GaugeChart,
	LineChart,
	PieChart,
	GridComponent,
	LegendComponent,
	TitleComponent,
	TooltipComponent,
	DatasetComponent,
	TransformComponent,
	LabelLayout,
	UniversalTransition,
	CanvasRenderer,
	SVGRenderer,
]);

const props = withDefaults(
	defineProps<{
		option: EChartsOption;
		title?: string;
		height?: number | string;
		renderer?: "canvas" | "svg";
	}>(),
	{
		title: "",
		height: 360,
		renderer: "canvas",
	},
);

const root = ref<HTMLDivElement | null>(null);

let chart: EChartsType | null = null;
let resizeObserver: ResizeObserver | null = null;

const heightStyle = computed(() =>
	typeof props.height === "number" ? `${props.height}px` : props.height,
);

function renderChart() {
	if (!root.value) return;

	if (!chart) {
		chart = init(root.value, undefined, {
			renderer: props.renderer,
		});
	}

	chart.setOption(props.option, true);
	chart.resize();
}

function disposeChart() {
	chart?.dispose();
	chart = null;
}

onMounted(() => {
	renderChart();

	if (root.value) {
		resizeObserver = new ResizeObserver(() => {
			chart?.resize();
		});
		resizeObserver.observe(root.value);
	}
});

watch(
	() => props.option,
	() => {
		renderChart();
	},
	{ deep: true },
);

watch(
	() => props.renderer,
	() => {
		disposeChart();
		renderChart();
	},
);

onBeforeUnmount(() => {
	resizeObserver?.disconnect();
	disposeChart();
});
</script>

<template>
	<div class="viz-card">
		<div v-if="title" class="viz-title">{{ title }}</div>
		<div ref="root" class="chart-root" :style="{ height: heightStyle }"></div>
	</div>
</template>

<style scoped>
.viz-card {
	border: 1px solid var(--line-divider);
	border-radius: 1rem;
	padding: 1rem;
	background: color-mix(in oklab, var(--card-bg) 88%, transparent);
}

.viz-title {
	margin-bottom: 0.9rem;
	font-size: 0.95rem;
	font-weight: 700;
	letter-spacing: 0.02em;
	color: var(--tw-prose-headings);
}

.chart-root {
	width: 100%;
	min-height: 240px;
}
</style>
