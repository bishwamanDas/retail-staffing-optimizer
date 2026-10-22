/**
 * Store Staffing Analyzer — Chart.js builders.
 *
 * Every builder is a pure function of `(canvas, data)` returning a Chart
 * instance. If the Chart.js CDN is unreachable the module degrades to a
 * friendly placeholder instead of throwing.
 */

/**
 * Qualitative palette used across the dashboard.
 *
 * Deliberately desaturated and close in luminance so that series read as
 * one system rather than a rainbow. The first entry matches `--accent`
 * so a single-series chart always agrees with the UI chrome.
 */
export const PALETTE = [
  '#d3a04c', // brass  — accent
  '#6f9dc4', // steel blue
  '#7fa88c', // sage
  '#bd7d92', // dusty rose
  '#9189ab', // muted violet
  '#b08968', // tan
  '#5f9e9e', // teal
  '#8a8f7a', // olive
];

/** Fallback token values, used before the stylesheet has loaded. */
const FALLBACK = {
  text: '#ececee',
  muted: '#9b9ba3',
  faint: '#6c6c76',
  border: '#2b2b30',
  borderSoft: '#222227',
  surface3: '#212125',
  accent: '#d3a04c',
  teal: '#57a89b',
  green: '#4aa06c',
  red: '#d1604f',
  amber: '#d3a04c',
  violet: '#9189ab',
};

const registry = new Map(); // canvasId -> { builder, data }
const instances = new Map(); // canvasId -> Chart instance

let chartReadyPromise = null;

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

/** Read a design-token value from the document root. */
export function cssVar(name, fallback) {
  try {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name);
    return value && value.trim() ? value.trim() : fallback;
  } catch (error) {
    return fallback;
  }
}

/** Convert a `#rrggbb` (or `#rgb`) colour into an rgba() string. */
export function withAlpha(color, alpha) {
  const hex = String(color || '').trim();
  const short = /^#([0-9a-f])([0-9a-f])([0-9a-f])$/i.exec(hex);
  const long = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  const parts = short
    ? [short[1] + short[1], short[2] + short[2], short[3] + short[3]]
    : long
      ? [long[1], long[2], long[3]]
      : null;
  if (!parts) return hex;
  const [r, g, b] = parts.map((p) => parseInt(p, 16));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** True when the Chart.js global is present. */
export function chartAvailable() {
  return typeof window !== 'undefined' && typeof window.Chart !== 'undefined';
}

/**
 * Resolve once Chart.js is available (or after `timeout` milliseconds).
 * The CDN script is loaded with `defer`, so it normally resolves immediately.
 */
export function ensureChart(timeout = 4000) {
  if (chartAvailable()) return Promise.resolve(true);
  if (chartReadyPromise) return chartReadyPromise;

  chartReadyPromise = new Promise((resolve) => {
    const started = Date.now();
    const tick = () => {
      if (chartAvailable()) {
        resolve(true);
        return;
      }
      if (Date.now() - started > timeout) {
        resolve(false);
        return;
      }
      setTimeout(tick, 60);
    };
    tick();
  });

  return chartReadyPromise;
}

/** Apply global chart defaults from the active theme. */
function applyDefaults() {
  if (!chartAvailable()) return;
  const C = window.Chart;
  const fontFamily = cssVar('--font-sans', 'Inter, -apple-system, Segoe UI, sans-serif');
  const text = cssVar('--text', FALLBACK.text);
  const muted = cssVar('--text-muted', FALLBACK.muted);
  const border = cssVar('--border', FALLBACK.border);
  const surface = cssVar('--surface-3', FALLBACK.surface3);

  C.defaults.font.family = fontFamily;
  C.defaults.font.size = 11;
  C.defaults.color = muted;
  C.defaults.borderColor = border;
  C.defaults.responsive = true;
  C.defaults.maintainAspectRatio = false;
  C.defaults.animation = { duration: 450 };
  C.defaults.plugins.legend.labels.usePointStyle = true;
  C.defaults.plugins.legend.labels.boxWidth = 8;
  C.defaults.plugins.legend.labels.boxHeight = 8;
  C.defaults.plugins.legend.labels.padding = 14;
  C.defaults.plugins.legend.labels.color = muted;
  C.defaults.plugins.tooltip.backgroundColor = surface;
  C.defaults.plugins.tooltip.borderColor = border;
  C.defaults.plugins.tooltip.borderWidth = 1;
  C.defaults.plugins.tooltip.titleColor = text;
  C.defaults.plugins.tooltip.bodyColor = muted;
  C.defaults.plugins.tooltip.padding = 10;
  C.defaults.plugins.tooltip.cornerRadius = 8;
  C.defaults.plugins.tooltip.displayColors = true;
}

/** Shared scale configuration. */
function baseScales(yTitle) {
  const border = cssVar('--border-soft', FALLBACK.borderSoft);
  const muted = cssVar('--text-faint', FALLBACK.faint);
  return {
    x: {
      grid: { color: border, drawTicks: false },
      border: { color: border },
      ticks: { color: muted, maxRotation: 0, autoSkipPadding: 16 },
    },
    y: {
      title: yTitle ? { display: true, text: yTitle, color: muted } : undefined,
      grid: { color: border, drawTicks: false },
      border: { display: false },
      ticks: { color: muted, padding: 6 },
      beginAtZero: false,
    },
  };
}

/** Shared tooltip/legend options. */
function basePlugins(extra = {}) {
  return {
    legend: { display: true, position: 'top', align: 'start' },
    tooltip: { intersect: false, mode: 'index' },
    ...extra,
  };
}

/** Show a non-fatal placeholder inside the chart frame. */
function showPlaceholder(canvas, title, detail) {
  const frame = canvas && canvas.parentElement;
  if (!frame) return;
  let placeholder = frame.querySelector('.chart-placeholder');
  if (!placeholder) {
    placeholder = document.createElement('div');
    placeholder.className = 'chart-placeholder';
    frame.insertBefore(placeholder, canvas);
  }
  placeholder.innerHTML =
    '<svg class="icon" aria-hidden="true"><use href="#i-chart"></use></svg>' +
    `<strong>${title}</strong><span>${detail}</span>`;
  canvas.hidden = true;
}

/** Remove any placeholder and reveal the canvas again. */
function clearPlaceholder(canvas) {
  const frame = canvas && canvas.parentElement;
  if (!frame) return;
  const placeholder = frame.querySelector('.chart-placeholder');
  if (placeholder) placeholder.remove();
  canvas.hidden = false;
}

/* -------------------------------------------------------------------------- */
/* Public chart API                                                           */
/* -------------------------------------------------------------------------- */

/**
 * Render (or re-render) the chart bound to `canvasId`.
 *
 * @param {string}   canvasId id of the `<canvas>` element
 * @param {Function} builder  `(canvas, data) => Chart`
 * @param {*}       data      payload handed to the builder
 * @returns {Promise<Chart|null>}
 */
export function renderChart(canvasId, builder, data) {
  destroyChart(canvasId);
  registry.set(canvasId, { builder, data });

  return ensureChart()
    .then((available) => {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return null;
      if (!available) {
        showPlaceholder(
          canvas,
          'Chart library unavailable',
          'Chart.js could not be loaded from the CDN, so visualisations are disabled. All numeric results remain available below.',
        );
        return null;
      }
      applyDefaults();
      clearPlaceholder(canvas);
      const chart = builder(canvas, data);
      instances.set(canvasId, chart);
      return chart;
    })
    .catch((error) => {
      const canvas = document.getElementById(canvasId);
      if (canvas) {
        showPlaceholder(
          canvas,
          'Chart could not be rendered',
          error && error.message ? error.message : 'Unknown rendering error.',
        );
      }
      return null;
    });
}

/** Destroy a single chart instance. */
export function destroyChart(canvasId) {
  const chart = instances.get(canvasId);
  if (chart) {
    try {
      chart.destroy();
    } catch (error) {
      /* ignore teardown failures */
    }
    instances.delete(canvasId);
  }
}

/** Rebuild every chart — used after a theme switch so colours stay in sync. */
export function rerenderCharts() {
  if (!chartAvailable()) {
    applyDefaults();
    return;
  }
  registry.forEach((entry, canvasId) => {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    try {
      clearPlaceholder(canvas);
      const chart = entry.builder(canvas, entry.data);
      instances.set(canvasId, chart);
    } catch (error) {
      showPlaceholder(
        canvas,
        'Chart could not be rendered',
        error && error.message ? error.message : 'Unknown rendering error.',
      );
    }
  });
}

/**
 * Resize every chart that lives inside `root`.
 * Charts created while their panel was hidden have no measurable size until
 * the panel becomes visible again.
 */
export function resizeChartsIn(root) {
  if (!chartAvailable() || !root) return;
  const canvases = root.querySelectorAll('canvas');
  canvases.forEach((canvas) => {
    const chart = instances.get(canvas.id);
    if (!chart) return;
    try {
      chart.resize();
    } catch (error) {
      /* ignore resize failures */
    }
  });
}

/* -------------------------------------------------------------------------- */
/* Sparklines (dependency-free SVG)                                           */
/* -------------------------------------------------------------------------- */

/**
 * Build a tiny inline SVG sparkline.
 *
 * @param {number[]} values
 * @param {object}  options `{ color, height }`
 * @returns {SVGElement}
 */
export function buildSparkline(values, options = {}) {
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 100 32');
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('class', 'spark');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');

  const points = Array.isArray(values)
    ? values.map((v) => (typeof v === 'number' && Number.isFinite(v) ? v : null))
    : [];
  const usable = points.filter((v) => v !== null);
  if (usable.length < 2) return svg;

  const min = Math.min(...usable);
  const max = Math.max(...usable);
  const span = max - min || 1;
  const step = 100 / (points.length - 1);

  const coords = points.map((value, index) => {
    const x = index * step;
    const y = value === null ? null : 30 - ((value - min) / span) * 26 - 2;
    return { x, y };
  });

  const valid = coords.filter((c) => c.y !== null);
  const first = valid[0];
  const last = valid[valid.length - 1];

  const areaPath = `M ${first.x} 32 L ${valid
    .map((c) => `${c.x.toFixed(2)} ${c.y.toFixed(2)}`)
    .join(' L ')} L ${last.x.toFixed(2)} 32 Z`;
  const linePath = `M ${valid
    .map((c) => `${c.x.toFixed(2)} ${c.y.toFixed(2)}`)
    .join(' L ')}`;

  const color = options.color || cssVar('--accent', FALLBACK.accent);

  const area = document.createElementNS(NS, 'path');
  area.setAttribute('d', areaPath);
  area.setAttribute('fill', withAlpha(color, 0.16));
  area.setAttribute('stroke', 'none');

  const line = document.createElementNS(NS, 'path');
  line.setAttribute('d', linePath);
  line.setAttribute('fill', 'none');
  line.setAttribute('stroke', color);
  line.setAttribute('stroke-width', '1.8');
  line.setAttribute('stroke-linecap', 'round');
  line.setAttribute('stroke-linejoin', 'round');
  line.setAttribute('vector-effect', 'non-scaling-stroke');

  const dot = document.createElementNS(NS, 'circle');
  dot.setAttribute('cx', last.x.toFixed(2));
  dot.setAttribute('cy', last.y.toFixed(2));
  dot.setAttribute('r', '1.9');
  dot.setAttribute('fill', color);
  dot.setAttribute('vector-effect', 'non-scaling-stroke');

  svg.appendChild(area);
  svg.appendChild(line);
  svg.appendChild(dot);
  return svg;
}

/* -------------------------------------------------------------------------- */
/* Builders                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Historical sales + forecast with a 95% prediction band.
 * data: { months, sales, forecastMonths, values, lower95, upper95 }
 */
export function buildForecastChart(canvas, data) {
  const accent = cssVar('--accent', FALLBACK.accent);
  const teal = cssVar('--teal', FALLBACK.teal);
  const border = cssVar('--border', FALLBACK.border);

  const months = data.months || [];
  const sales = data.sales || [];
  const forecastMonths = data.forecastMonths || [];
  const values = data.values || [];
  const lower = data.lower95 || [];
  const upper = data.upper95 || [];

  // The band width follows the run's configured confidence level.
  const level = Number.isFinite(data.confidenceLevel) ? data.confidenceLevel : 0.95;
  const bandPct = Math.round(level * 100);
  const bandLabel = `${bandPct}% prediction band`;
  const lowerLabel = `${bandPct}% band lower`;

  const labels = months.map((m) => `M${m}`).concat(forecastMonths.map((m) => `M${m}`));
  const historyLength = months.length;
  const lastActual = sales.length ? sales[sales.length - 1] : null;

  const padded = (source, anchor) => {
    const out = new Array(historyLength).fill(null);
    if (anchor !== null && anchor !== undefined && historyLength > 0) {
      out[historyLength - 1] = anchor;
    }
    return out.concat(source);
  };

  const datasets = [
    {
      label: 'Actual sales',
      data: sales,
      borderColor: accent,
      backgroundColor: accent,
      borderWidth: 2,
      tension: 0.32,
      pointRadius: 0,
      pointHoverRadius: 4,
      fill: false,
      order: 4,
    },
    {
      label: bandLabel,
      data: padded(upper, lastActual),
      borderColor: 'transparent',
      backgroundColor: withAlpha(teal, 0.16),
      borderWidth: 0,
      pointRadius: 0,
      fill: false,
      order: 1,
    },
    {
      label: lowerLabel,
      data: padded(lower, lastActual),
      borderColor: 'transparent',
      backgroundColor: withAlpha(teal, 0.16),
      borderWidth: 0,
      pointRadius: 0,
      fill: '-1',
      order: 1,
      legendFilter: false,
    },
    {
      label: 'Forecast',
      data: padded(values, lastActual),
      borderColor: teal,
      backgroundColor: teal,
      borderWidth: 2,
      borderDash: [6, 4],
      tension: 0.32,
      pointRadius: 2.5,
      pointHoverRadius: 5,
      fill: false,
      order: 2,
    },
  ];

  return new window.Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: baseScales('Sales (L)'),
      plugins: basePlugins({
        tooltip: {
          filter: (item) => item.dataset.label !== lowerLabel,
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.formattedValue}`,
          },
        },
        legend: {
          display: true,
          position: 'top',
          align: 'start',
          labels: {
            usePointStyle: true,
            boxWidth: 8,
            // Chart.js passes the legend item (plus a context object) to this
            // filter; the owning dataset is resolved from the local list so the
            // behaviour is identical across Chart.js minor versions.
            filter: (item) => {
              const dataset =
                item && typeof item.datasetIndex === 'number'
                  ? datasets[item.datasetIndex]
                  : item && item.dataset;
              return !dataset || dataset.legendFilter !== false;
            },
          },
        },
      }),
      elements: { line: { borderJoinStyle: 'round' } },
    },
    plugins: [
      {
        id: 'storeStaffingBaseline',
        beforeDatasetsDraw(chart) {
          const { ctx, chartArea, scales } = chart;
          if (!chartArea || !scales.y) return;
          const zero = scales.y.getPixelForValue(0);
          if (zero < chartArea.top || zero > chartArea.bottom) return;
          ctx.save();
          ctx.beginPath();
          ctx.setLineDash([3, 4]);
          ctx.strokeStyle = border;
          ctx.lineWidth = 1;
          ctx.moveTo(chartArea.left, zero);
          ctx.lineTo(chartArea.right, zero);
          ctx.stroke();
          ctx.restore();
        },
      },
    ],
  });
}

/**
 * Monthly gross margin area chart.
 * data: { months, grossMargin }
 */
export function buildMarginChart(canvas, data) {
  const accent = cssVar('--accent', FALLBACK.accent);
  const months = data.months || [];
  const values = data.grossMargin || [];

  return new window.Chart(canvas, {
    type: 'line',
    data: {
      labels: months.map((m) => `M${m}`),
      datasets: [
        {
          label: 'Gross margin',
          data: values,
          borderColor: accent,
          backgroundColor: withAlpha(accent, 0.18),
          borderWidth: 2,
          tension: 0.35,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: baseScales('Margin (L)'),
      plugins: basePlugins({
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => `Gross margin: ${ctx.formattedValue} L` } },
      }),
    },
  });
}

/**
 * Headcount vs. sales bubble chart (bubble radius scales with margin).
 * data: { points: [{ x, y, r, margin }] }
 */
export function buildScatterChart(canvas, data) {
  const accent = cssVar('--accent', FALLBACK.accent);
  const points = data.points || [];

  return new window.Chart(canvas, {
    type: 'bubble',
    data: {
      datasets: [
        {
          label: 'Month',
          data: points,
          backgroundColor: withAlpha(accent, 0.45),
          borderColor: accent,
          borderWidth: 1.2,
          hoverBackgroundColor: withAlpha(accent, 0.7),
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Employees', color: cssVar('--text-faint', FALLBACK.faint) },
          grid: { color: cssVar('--border-soft', FALLBACK.borderSoft), drawTicks: false },
          border: { color: cssVar('--border-soft', FALLBACK.borderSoft) },
          ticks: {
            color: cssVar('--text-faint', FALLBACK.faint),
            stepSize: 1,
            precision: 0,
          },
        },
        y: {
          title: { display: true, text: 'Sales (L)', color: cssVar('--text-faint', FALLBACK.faint) },
          grid: { color: cssVar('--border-soft', FALLBACK.borderSoft), drawTicks: false },
          border: { display: false },
          ticks: { color: cssVar('--text-faint', FALLBACK.faint) },
        },
      },
      plugins: basePlugins({
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const raw = ctx.raw || {};
              return [
                `Employees: ${raw.x}`,
                `Sales: ${raw.y} L`,
                `Gross margin: ${raw.margin === undefined ? '—' : raw.margin} L`,
              ];
            },
          },
        },
      }),
    },
  });
}

/**
 * Walk-forward backtest: actual vs predicted.
 * data: { backtest: [{ origin_month, actual, predicted, error }] }
 */
export function buildBacktestChart(canvas, data) {
  const accent = cssVar('--accent', FALLBACK.accent);
  const teal = cssVar('--teal', FALLBACK.teal);
  const backtest = data.backtest || [];

  // Each origin contributes one plotted point: the value at the end of
  // that origin's forecast window.
  const point = (row, key) => {
    const value = row ? row[key] : null;
    if (Array.isArray(value)) return value.length ? value[value.length - 1] : null;
    return value;
  };

  return new window.Chart(canvas, {
    type: 'line',
    data: {
      labels: backtest.map((row) => `M${row.origin_month}`),
      datasets: [
        {
          label: 'Actual',
          data: backtest.map((row) => point(row, 'actual')),
          borderColor: accent,
          backgroundColor: accent,
          borderWidth: 2,
          tension: 0.3,
          pointRadius: 3,
          pointHoverRadius: 6,
          fill: false,
        },
        {
          label: 'Predicted',
          data: backtest.map((row) => point(row, 'predicted')),
          borderColor: teal,
          backgroundColor: teal,
          borderWidth: 2,
          borderDash: [6, 4],
          tension: 0.3,
          pointRadius: 3,
          pointHoverRadius: 6,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: baseScales('Sales (L)'),
      plugins: basePlugins({
        tooltip: {
          callbacks: {
            afterBody: (items) => {
              const row = backtest[items[0].dataIndex];
              if (!row) return '';
              return `Error: ${row.error} L`;
            },
          },
        },
      }),
    },
  });
}

/**
 * Bar chart comparing scenario gross margins.
 * data: { scenarios: [{ key, name, margin, isBest }] }
 */
export function buildScenarioBarChart(canvas, data) {
  const scenarios = data.scenarios || [];
  const accent = cssVar('--accent', FALLBACK.accent);
  const teal = cssVar('--teal', FALLBACK.teal);
  const green = cssVar('--green', FALLBACK.green);

  return new window.Chart(canvas, {
    type: 'bar',
    data: {
      labels: scenarios.map((s) => `Scenario ${s.key}`),
      datasets: [
        {
          label: 'Total gross margin',
          data: scenarios.map((s) => s.margin),
          backgroundColor: scenarios.map((s) =>
            s.isBest ? withAlpha(green, 0.75) : withAlpha(accent, 0.55),
          ),
          borderColor: scenarios.map((s) => (s.isBest ? green : accent)),
          borderWidth: 1.2,
          borderRadius: 6,
          maxBarThickness: 62,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { display: false },
          border: { color: cssVar('--border-soft', FALLBACK.borderSoft) },
          ticks: { color: cssVar('--text-muted', FALLBACK.muted) },
        },
        y: {
          title: { display: true, text: 'Gross margin (L)', color: cssVar('--text-faint', FALLBACK.faint) },
          grid: { color: cssVar('--border-soft', FALLBACK.borderSoft), drawTicks: false },
          border: { display: false },
          ticks: { color: cssVar('--text-faint', FALLBACK.faint) },
        },
      },
      plugins: basePlugins({
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => {
              const scenario = scenarios[items[0].dataIndex];
              return scenario ? scenario.name : '';
            },
            label: (ctx) => `Gross margin: ${ctx.formattedValue} L`,
          },
        },
      }),
    },
  });
}

/**
 * Cumulative gross margin per scenario.
 * data: { scenarios: [{ key, name, isBest, cumulative: [{ month, value }] }] }
 */
export function buildScenarioCumulativeChart(canvas, data) {
  const scenarios = data.scenarios || [];
  const palette = [accentSafe(), tealSafe(), violetSafe(), amberSafe()];
  const labels = [];

  scenarios.forEach((scenario) => {
    (scenario.cumulative || []).forEach((point) => {
      if (!labels.includes(point.month)) labels.push(point.month);
    });
  });
  labels.sort((a, b) => a - b);

  return new window.Chart(canvas, {
    type: 'line',
    data: {
      labels: labels.map((m) => `M${m}`),
      datasets: scenarios.map((scenario, index) => {
        const lookup = new Map(
          (scenario.cumulative || []).map((point) => [point.month, point.value]),
        );
        return {
          label: `${scenario.key} · ${scenario.name}`,
          data: labels.map((m) => (lookup.has(m) ? lookup.get(m) : null)),
          borderColor: palette[index % palette.length],
          backgroundColor: palette[index % palette.length],
          borderWidth: scenario.isBest ? 2.6 : 1.8,
          borderDash: scenario.isBest ? [] : [5, 4],
          tension: 0.3,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: false,
        };
      }),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: baseScales('Cumulative margin (L)'),
      plugins: basePlugins({
        tooltip: {
          callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.formattedValue} L` },
        },
      }),
    },
  });
}

/**
 * Headcount timeline with joiner / leaver markers.
 * data: { timeline: [{ month, employees, joined: [], left: [] }] }
 */
export function buildHeadcountChart(canvas, data) {
  const timeline = data.timeline || [];
  const accent = cssVar('--accent', FALLBACK.accent);
  const green = cssVar('--green', FALLBACK.green);
  const red = cssVar('--red', FALLBACK.red);
  const numeric = (value) =>
    typeof value === 'number' && Number.isFinite(value) ? value : null;

  const joiners = timeline
    .filter((row) => Array.isArray(row.joined) && row.joined.length > 0)
    .map((row) => ({ x: numeric(row.month), y: numeric(row.employees), joined: row.joined.length }));
  const leavers = timeline
    .filter((row) => Array.isArray(row.left) && row.left.length > 0)
    .map((row) => ({ x: numeric(row.month), y: numeric(row.employees), left: row.left.length }));

  return new window.Chart(canvas, {
    type: 'line',
    data: {
      labels: timeline.map((row) => `M${row.month}`),
      datasets: [
        {
          label: 'Employees',
          data: timeline.map((row) => numeric(row.employees)),
          borderColor: accent,
          backgroundColor: withAlpha(accent, 0.16),
          borderWidth: 2,
          tension: 0.3,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: true,
          spanGaps: true,
          order: 3,
        },
        {
          label: 'Joiners',
          data: joiners,
          type: 'scatter',
          backgroundColor: withAlpha(green, 0.9),
          borderColor: green,
          borderWidth: 1,
          pointRadius: 5,
          pointHoverRadius: 7,
          order: 1,
        },
        {
          label: 'Leavers',
          data: leavers,
          type: 'scatter',
          backgroundColor: withAlpha(red, 0.9),
          borderColor: red,
          borderWidth: 1,
          pointRadius: 5,
          pointHoverRadius: 7,
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: baseScales('Employees'),
      plugins: basePlugins({
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const raw = ctx.raw || {};
              if (ctx.dataset.label === 'Joiners') return `Joiners: ${raw.joined}`;
              if (ctx.dataset.label === 'Leavers') return `Leavers: ${raw.left}`;
              return `Employees: ${ctx.formattedValue}`;
            },
          },
        },
      }),
    },
  });
}

/**
 * Diminishing returns: sales per employee (line) + gross margin (bars).
 * data: { byHeadcount: [{ employees, avg_sales_per_employee, avg_gross_margin }] }
 */
export function buildDiminishingReturnsChart(canvas, data) {
  const rows = data.byHeadcount || [];
  const accent = cssVar('--accent', FALLBACK.accent);
  const teal = cssVar('--teal', FALLBACK.teal);

  return new window.Chart(canvas, {
    type: 'bar',
    data: {
      labels: rows.map((row) => `${row.employees}`),
      datasets: [
        {
          label: 'Avg gross margin (L)',
          data: rows.map((row) => row.avg_gross_margin),
          backgroundColor: withAlpha(teal, 0.4),
          borderColor: withAlpha(teal, 0.85),
          borderWidth: 1,
          borderRadius: 5,
          maxBarThickness: 34,
          yAxisID: 'y1',
          order: 2,
        },
        {
          label: 'Avg sales / employee (L)',
          data: rows.map((row) => row.avg_sales_per_employee),
          type: 'line',
          borderColor: accent,
          backgroundColor: accent,
          borderWidth: 2.2,
          tension: 0.3,
          pointRadius: 3,
          pointHoverRadius: 6,
          fill: false,
          yAxisID: 'y',
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: {
          title: { display: true, text: 'Headcount', color: cssVar('--text-faint', FALLBACK.faint) },
          grid: { display: false },
          border: { color: cssVar('--border-soft', FALLBACK.borderSoft) },
          ticks: { color: cssVar('--text-muted', FALLBACK.muted), precision: 0 },
        },
        y: {
          title: { display: true, text: 'Sales / employee (L)', color: cssVar('--text-faint', FALLBACK.faint) },
          grid: { color: cssVar('--border-soft', FALLBACK.borderSoft), drawTicks: false },
          border: { display: false },
          ticks: { color: cssVar('--text-faint', FALLBACK.faint) },
        },
        y1: {
          position: 'right',
          title: { display: true, text: 'Gross margin (L)', color: cssVar('--text-faint', FALLBACK.faint) },
          grid: { display: false },
          border: { display: false },
          ticks: { color: cssVar('--text-faint', FALLBACK.faint) },
        },
      },
      plugins: basePlugins(),
    },
  });
}

/**
 * Tornado chart: diverging horizontal bars per variable.
 * data: { tornado: [{ variable, low, high, low_margin, high_margin, swing }] }
 */
export function buildTornadoChart(canvas, data) {
  const tornado = (data.tornado || []).slice(0, 10);
  const green = cssVar('--green', FALLBACK.green);
  const red = cssVar('--red', FALLBACK.red);

  const rows = tornado
    .map((row) => {
      const low = typeof row.low_margin === 'number' ? row.low_margin : 0;
      const high = typeof row.high_margin === 'number' ? row.high_margin : 0;
      const center = (low + high) / 2;
      return {
        label: row.variable,
        lowDelta: low - center,
        highDelta: high - center,
        swing: typeof row.swing === 'number' ? row.swing : Math.abs(high - low),
        low,
        high,
        lowSetting: typeof row.low === 'number' ? row.low : null,
        highSetting: typeof row.high === 'number' ? row.high : null,
      };
    })
    .sort((a, b) => b.swing - a.swing);

  return new window.Chart(canvas, {
    type: 'bar',
    data: {
      labels: rows.map((row) => row.label),
      datasets: [
        {
          label: 'Low setting',
          data: rows.map((row) => row.lowDelta),
          backgroundColor: withAlpha(red, 0.72),
          borderColor: red,
          borderWidth: 1,
          borderRadius: 4,
          maxBarThickness: 18,
          raw: rows,
        },
        {
          label: 'High setting',
          data: rows.map((row) => row.highDelta),
          backgroundColor: withAlpha(green, 0.72),
          borderColor: green,
          borderWidth: 1,
          borderRadius: 4,
          maxBarThickness: 18,
          raw: rows,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          stacked: true,
          title: {
            display: true,
            text: 'Gross margin vs. midpoint (L)',
            color: cssVar('--text-faint', FALLBACK.faint),
          },
          grid: { color: cssVar('--border-soft', FALLBACK.borderSoft), drawTicks: false },
          border: { display: false },
          ticks: { color: cssVar('--text-faint', FALLBACK.faint) },
        },
        y: {
          stacked: true,
          grid: { display: false },
          border: { color: cssVar('--border-soft', FALLBACK.borderSoft) },
          ticks: { color: cssVar('--text-muted', FALLBACK.muted) },
        },
      },
      plugins: basePlugins({
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const row = (ctx.dataset.raw || [])[ctx.dataIndex];
              if (!row) return '';
              const sign = ctx.dataset.label === 'Low setting' ? '−' : '+';
              return `${ctx.dataset.label}: ${sign}${Math.abs(ctx.raw).toFixed(2)} L`;
            },
            afterBody: (items) => {
              const row = (items[0].dataset.raw || [])[items[0].dataIndex];
              if (!row) return '';
              const lines = [
                `Margin at low setting: ${row.low} L`,
                `Margin at high setting: ${row.high} L`,
                `Swing: ${row.swing} L`,
              ];
              if (row.lowSetting !== null && row.highSetting !== null) {
                lines.push(`Variable range: ${row.lowSetting} → ${row.highSetting}`);
              }
              return lines;
            },
          },
        },
      }),
    },
  });
}

/**
 * Monte-Carlo histogram — normal approximation synthesised from the
 * reported percentiles, mean and standard deviation.
 * data: { p10, p50, p90, mean, std }
 */
export function buildMonteCarloChart(canvas, data) {
  const accent = cssVar('--accent', FALLBACK.accent);
  const mean = typeof data.mean === 'number' ? data.mean : 0;
  const std = typeof data.std === 'number' && data.std > 0 ? data.std : 1;
  const binCount = 25;

  const lower = mean - 3 * std;
  const upper = mean + 3 * std;
  const width = (upper - lower) / binCount || 1;

  const bins = [];
  for (let i = 0; i < binCount; i += 1) {
    const from = lower + i * width;
    const to = from + width;
    const midpoint = (from + to) / 2;
    const z = (midpoint - mean) / std;
    const density = Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
    bins.push({ x: midpoint, y: density, from, to });
  }

  const verticalLines = [
    { value: data.p10, label: 'p10', color: cssVar('--red', FALLBACK.red) },
    { value: data.p50, label: 'p50', color: cssVar('--accent', FALLBACK.accent) },
    { value: data.p90, label: 'p90', color: cssVar('--green', FALLBACK.green) },
  ].filter((line) => typeof line.value === 'number' && Number.isFinite(line.value));

  return new window.Chart(canvas, {
    type: 'bar',
    data: {
      datasets: [
        {
          label: 'Probability density (approx.)',
          data: bins,
          backgroundColor: withAlpha(accent, 0.55),
          borderColor: withAlpha(accent, 0.9),
          borderWidth: 1,
          borderRadius: 3,
          maxBarThickness: 26,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 16 } },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Gross margin (L)', color: cssVar('--text-faint', FALLBACK.faint) },
          grid: { display: false },
          border: { color: cssVar('--border-soft', FALLBACK.borderSoft) },
          ticks: {
            color: cssVar('--text-faint', FALLBACK.faint),
            maxRotation: 0,
            autoSkipPadding: 12,
            callback: (value) => Number(value).toFixed(0),
          },
        },
        y: {
          title: { display: true, text: 'Density', color: cssVar('--text-faint', FALLBACK.faint) },
          grid: { color: cssVar('--border-soft', FALLBACK.borderSoft), drawTicks: false },
          border: { display: false },
          ticks: { color: cssVar('--text-faint', FALLBACK.faint) },
        },
      },
      plugins: {
        ...basePlugins({
          tooltip: {
            callbacks: {
              title: (items) => {
                const raw = items && items[0] ? items[0].raw : null;
                return raw && typeof raw.from === 'number'
                  ? `${raw.from.toFixed(1)} – ${raw.to.toFixed(1)} L`
                  : '';
              },
              label: (ctx) => `Density: ${Number(ctx.parsed.y).toFixed(4)}`,
            },
          },
        }),
        storeStaffingVLines: { lines: verticalLines },
      },
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
    },
    plugins: [verticalLinesPlugin],
  });
}

/* -------------------------------------------------------------------------- */
/* Local plugin                                                               */
/* -------------------------------------------------------------------------- */

const verticalLinesPlugin = {
  id: 'storeStaffingVLines',
  afterDatasetsDraw(chart) {
    const options = chart.options.plugins && chart.options.plugins.storeStaffingVLines;
    const lines = options && options.lines;
    if (!lines || !lines.length) return;

    const { ctx, chartArea, scales } = chart;
    const xScale = scales && scales.x;
    if (!chartArea || !xScale) return;

    const fontFamily = cssVar('--font-sans', 'Inter, sans-serif');
    ctx.save();
    lines.forEach((line) => {
      const px = xScale.getPixelForValue(line.value);
      if (!Number.isFinite(px)) return;
      if (px < chartArea.left || px > chartArea.right) return;
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = line.color;
      ctx.lineWidth = 1.4;
      ctx.moveTo(px, chartArea.top);
      ctx.lineTo(px, chartArea.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = line.color;
      ctx.font = `600 10px ${fontFamily}`;
      ctx.textAlign = 'center';
      ctx.fillText(line.label, px, chartArea.top - 5);
    });
    ctx.restore();
  },
};

/* Palette helpers (kept in functions so theme changes are picked up). */
function accentSafe() {
  return cssVar('--accent', FALLBACK.accent);
}
function tealSafe() {
  return cssVar('--teal', FALLBACK.teal);
}
function violetSafe() {
  return cssVar('--violet', FALLBACK.violet);
}
function amberSafe() {
  return cssVar('--amber', FALLBACK.amber);
}
