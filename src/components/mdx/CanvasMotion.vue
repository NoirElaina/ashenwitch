<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";

const props = withDefaults(
	defineProps<{
		title?: string;
		height?: number | string;
		preset?: "waves" | "particles" | "orbits";
	}>(),
	{
		title: "",
		height: 280,
		preset: "waves",
	},
);

const canvas = ref<HTMLCanvasElement | null>(null);

let animationFrame = 0;
let resizeObserver: ResizeObserver | null = null;

const heightStyle = computed(() =>
	typeof props.height === "number" ? `${props.height}px` : props.height,
);

function palette() {
	const isDark = document.documentElement.classList.contains("dark");
	return isDark
		? {
				background: "rgba(15, 23, 42, 0.92)",
				primary: "rgba(125, 211, 252, 0.95)",
				secondary: "rgba(56, 189, 248, 0.45)",
				accent: "rgba(165, 243, 252, 0.22)",
			}
		: {
				background: "rgba(241, 245, 249, 0.9)",
				primary: "rgba(14, 116, 144, 0.95)",
				secondary: "rgba(6, 182, 212, 0.35)",
				accent: "rgba(14, 165, 233, 0.14)",
			};
}

function resizeCanvas() {
	if (!canvas.value) return null;

	const element = canvas.value;
	const ratio = window.devicePixelRatio || 1;
	const rect = element.getBoundingClientRect();

	element.width = Math.max(1, Math.floor(rect.width * ratio));
	element.height = Math.max(1, Math.floor(rect.height * ratio));

	const context = element.getContext("2d");
	if (!context) return null;

	context.setTransform(ratio, 0, 0, ratio, 0, 0);
	return {
		context,
		width: rect.width,
		height: rect.height,
	};
}

function drawWaves(
	context: CanvasRenderingContext2D,
	width: number,
	height: number,
	time: number,
) {
	const colors = palette();

	context.fillStyle = colors.background;
	context.fillRect(0, 0, width, height);

	for (const [index, amplitude, widthFactor] of [
		[0, 12, 38],
		[1, 18, 52],
		[2, 24, 66],
	] as const) {
		context.beginPath();
		context.lineWidth = 2 + index;
		context.strokeStyle = index === 0 ? colors.primary : colors.secondary;

		for (let x = 0; x <= width; x += 4) {
			const y =
				height * (0.35 + index * 0.16) +
				Math.sin(x / widthFactor + time * (0.0014 + index * 0.0004)) *
					amplitude;

			if (x === 0) {
				context.moveTo(x, y);
			} else {
				context.lineTo(x, y);
			}
		}

		context.stroke();
	}
}

function drawParticles(
	context: CanvasRenderingContext2D,
	width: number,
	height: number,
	time: number,
) {
	const colors = palette();

	context.fillStyle = colors.background;
	context.fillRect(0, 0, width, height);

	const particleCount = Math.max(24, Math.floor(width / 22));

	for (let index = 0; index < particleCount; index++) {
		const base = index / particleCount;
		const x =
			(base * width +
				Math.sin(time * 0.0012 + index * 1.3) * width * 0.06 +
				width) %
			width;
		const y =
			(height * (0.18 + (index % 6) * 0.12) +
				Math.cos(time * 0.0017 + index * 0.8) * 18 +
				height) %
			height;
		const radius = 1.5 + ((index * 7) % 9) * 0.35;

		context.beginPath();
		context.fillStyle = index % 3 === 0 ? colors.primary : colors.secondary;
		context.arc(x, y, radius, 0, Math.PI * 2);
		context.fill();

		context.beginPath();
		context.strokeStyle = colors.accent;
		context.moveTo(x, y);
		context.lineTo(width / 2, height / 2);
		context.stroke();
	}
}

function drawOrbits(
	context: CanvasRenderingContext2D,
	width: number,
	height: number,
	time: number,
) {
	const colors = palette();
	const centerX = width / 2;
	const centerY = height / 2;

	context.fillStyle = colors.background;
	context.fillRect(0, 0, width, height);

	for (const radius of [36, 64, 92]) {
		context.beginPath();
		context.strokeStyle = colors.accent;
		context.lineWidth = 1;
		context.arc(centerX, centerY, radius, 0, Math.PI * 2);
		context.stroke();
	}

	for (const [index, radius, speed] of [
		[0, 36, 0.0018],
		[1, 64, 0.0012],
		[2, 92, 0.0009],
	] as const) {
		const angle = time * speed + index * 2.3;
		const x = centerX + Math.cos(angle) * radius;
		const y = centerY + Math.sin(angle) * radius;

		context.beginPath();
		context.fillStyle = index === 0 ? colors.primary : colors.secondary;
		context.arc(x, y, 6 + index * 1.5, 0, Math.PI * 2);
		context.fill();
	}

	context.beginPath();
	context.fillStyle = colors.primary;
	context.arc(centerX, centerY, 10, 0, Math.PI * 2);
	context.fill();
}

function animate(time: number) {
	const resized = resizeCanvas();
	if (!resized) return;

	const { context, width, height } = resized;

	switch (props.preset) {
		case "particles":
			drawParticles(context, width, height, time);
			break;
		case "orbits":
			drawOrbits(context, width, height, time);
			break;
		default:
			drawWaves(context, width, height, time);
			break;
	}

	animationFrame = window.requestAnimationFrame(animate);
}

function restartAnimation() {
	cancelAnimationFrame(animationFrame);
	animationFrame = window.requestAnimationFrame(animate);
}

onMounted(() => {
	restartAnimation();

	if (canvas.value) {
		resizeObserver = new ResizeObserver(() => {
			restartAnimation();
		});
		resizeObserver.observe(canvas.value);
	}
});

watch(
	() => props.preset,
	() => {
		restartAnimation();
	},
);

onBeforeUnmount(() => {
	resizeObserver?.disconnect();
	cancelAnimationFrame(animationFrame);
});
</script>

<template>
	<div class="viz-card">
		<div v-if="title" class="viz-title">{{ title }}</div>
		<div class="canvas-shell" :style="{ height: heightStyle }">
			<canvas ref="canvas" class="motion-canvas"></canvas>
		</div>
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

.canvas-shell {
	overflow: hidden;
	border-radius: 0.9rem;
}

.motion-canvas {
	display: block;
	width: 100%;
	height: 100%;
}
</style>
