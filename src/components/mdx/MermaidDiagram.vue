<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";

type MermaidModule = typeof import("mermaid");

const props = withDefaults(
	defineProps<{
		code: string;
		title?: string;
	}>(),
	{
		title: "",
	},
);

const container = ref<HTMLDivElement | null>(null);
const errorMessage = ref("");

let mermaidModule: MermaidModule["default"] | null = null;
const instanceId =
	typeof crypto !== "undefined" && "randomUUID" in crypto
		? crypto.randomUUID()
		: `mermaid-${Math.random().toString(36).slice(2, 10)}`;
let renderIndex = 0;
let themeObserver: MutationObserver | null = null;

function getMermaidTheme() {
	return document.documentElement.classList.contains("dark")
		? "dark"
		: "default";
}

async function ensureMermaid() {
	if (!mermaidModule) {
		const imported = await import("mermaid");
		mermaidModule = imported.default;
	}

	mermaidModule.initialize({
		startOnLoad: false,
		securityLevel: "loose",
		theme: getMermaidTheme(),
		fontFamily: "Roboto, system-ui, sans-serif",
	});

	return mermaidModule;
}

async function renderDiagram() {
	if (!container.value) return;

	try {
		const mermaid = await ensureMermaid();
		const diagramId = `mermaid-diagram-${instanceId}-${renderIndex++}`;
		const { svg, bindFunctions } = await mermaid.render(diagramId, props.code);

		container.value.innerHTML = svg;
		bindFunctions?.(container.value);
		errorMessage.value = "";
	} catch (error) {
		container.value.innerHTML = "";
		errorMessage.value =
			error instanceof Error
				? error.message
				: "Failed to render Mermaid diagram.";
	}
}

onMounted(() => {
	renderDiagram();

	themeObserver = new MutationObserver(() => {
		renderDiagram();
	});

	themeObserver.observe(document.documentElement, {
		attributes: true,
		attributeFilter: ["class"],
	});
});

watch(
	() => props.code,
	() => {
		renderDiagram();
	},
);

onBeforeUnmount(() => {
	themeObserver?.disconnect();
});
</script>

<template>
	<div class="viz-card">
		<div v-if="title" class="viz-title">{{ title }}</div>
		<div ref="container" class="mermaid-host"></div>
		<pre v-if="errorMessage" class="viz-error">{{ errorMessage }}</pre>
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

.mermaid-host {
	overflow-x: auto;
}

.mermaid-host :deep(svg) {
	display: block;
	max-width: 100%;
	height: auto;
	margin: 0 auto;
}

.viz-error {
	margin-top: 0.9rem;
	white-space: pre-wrap;
	border-radius: 0.75rem;
	padding: 0.85rem 1rem;
	background: rgba(220, 38, 38, 0.12);
	color: rgb(220, 38, 38);
	font-size: 0.9rem;
}
</style>
