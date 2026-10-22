/**
 * Store Staffing Analyzer — application bootstrap, state and rendering orchestration.
 *
 * The module owns:
 *   • theme handling (persisted to localStorage)
 *   • section navigation (ARIA tab pattern)
 *   • data loading from the REST API with graceful degradation
 *   • rendering of every dashboard panel
 *   • toasts, error banner and loading skeletons
 */

import { api, pick, asNumber, asArray } from './api.js';
import * as charts from './charts.js';
import { createSimulator } from './simulator.js';

/* -------------------------------------------------------------------------- */
/* Application state                                                           */
/* -------------------------------------------------------------------------- */

const state = {
  analysis: null,
  sensitivity: null,
  simulation: null,
  datasets: [],
  runs: [],
  theme: 'dark',
  healthOk: false,
  activePanel: 'overview',
  loading: true,
};

const THEME_KEY = 'store_staffing.theme';
const HEALTH_INTERVAL = 30000;

let simulator = null;
let suppressSimulationErrors = false;
let healthTimer = null;

/* -------------------------------------------------------------------------- */
/* Small DOM helpers                                                           */
/* -------------------------------------------------------------------------- */

function el(id) {
  return document.getElementById(id);
}

function show(node, visible) {
  if (!node) return;
  if (visible) node.removeAttribute('hidden');
  else node.setAttribute('hidden', '');
}

function setText(node, value) {
  if (!node) return;
  node.textContent = value === null || value === undefined || value === '' ? '—' : String(value);
}

/* -------------------------------------------------------------------------- */
/* Formatting                                                                  */
/* -------------------------------------------------------------------------- */

const formatterCache = new Map();

function formatter(decimals) {
  if (!formatterCache.has(decimals)) {
    formatterCache.set(
      decimals,
      new Intl.NumberFormat('en-IN', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      }),
    );
  }
  return formatterCache.get(decimals);
}

/** Format a value in Lakhs, e.g. `Rs. 103.40 L` for 103.4. */
export function fmtL(value, decimals = 2) {
  const n = asNumber(value);
  if (n === null) return '—';
  return `Rs. ${formatter(decimals).format(n)} L`;
}

/** Format a plain number with a fixed precision. */
export function fmtNum(value, decimals = 2) {
  const n = asNumber(value);
  if (n === null) return '—';
  return formatter(decimals).format(n);
}

/** Format an integer with Indian digit grouping. */
export function fmtInt(value) {
  const n = asNumber(value);
  if (n === null) return '—';
  return formatter(0).format(Math.round(n));
}

/** Format a value that is already expressed in percent. */
export function fmtPct(value, decimals = 1) {
  const n = asNumber(value);
  if (n === null) return '—';
  return `${formatter(decimals).format(n)}%`;
}

/**
 * Format a percentage that may be stored either as a fraction (0..1) or as an
 * already-scaled percentage (0..100).
 */
export function fmtFlexiblePct(value, decimals = 1) {
  const n = asNumber(value);
  if (n === null) return '—';
  const scaled = Math.abs(n) <= 1 ? n * 100 : n;
  return `${formatter(decimals).format(scaled)}%`;
}

/** Format a signed number with an explicit `+` / `−` prefix. */
export function fmtSigned(value, decimals = 2) {
  const n = asNumber(value);
  if (n === null) return '—';
  const sign = n > 0 ? '+' : n < 0 ? '−' : '';
  return `${sign}${formatter(decimals).format(Math.abs(n))}`;
}

/** Escape a value for safe interpolation into HTML. */
export function esc(value) {
  if (value === null || value === undefined) return '';
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** CSS class for a signed delta. */
function deltaClass(value) {
  const n = asNumber(value);
  if (n === null) return 'delta--flat';
  if (n > 0) return 'delta--up';
  if (n < 0) return 'delta--down';
  return 'delta--flat';
}

/** Badge tone for a confidence label. */
function confidenceTone(confidence) {
  const value = String(confidence || '').toLowerCase();
  if (value.includes('high')) return 'high';
  if (value.includes('medium') || value.includes('moderate')) return 'medium';
  if (value.includes('low')) return 'low';
  return 'neutral';
}

/** First non-empty string from a list of candidates. */
function firstString(...candidates) {
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim()) return candidate.trim();
  }
  return null;
}

/* -------------------------------------------------------------------------- */
/* Toasts                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * Show a transient notification.
 *
 * @param {string} title
 * @param {string} text
 * @param {string} tone  `success` | `error` | `info`
 */
export function toast(title, text = '', tone = 'info', timeout = 4600) {
  const region = el('toast-region');
  if (!region) return;

  const node = document.createElement('div');
  node.className = `toast toast--${tone}`;
  const iconId = tone === 'success' ? 'i-check' : tone === 'error' ? 'i-alert' : 'i-info';
  node.innerHTML =
    `<svg class="icon" aria-hidden="true"><use href="#${iconId}"></use></svg>` +
    '<div class="toast__body">' +
    `<p class="toast__title">${esc(title)}</p>` +
    (text ? `<p class="toast__text">${esc(text)}</p>` : '') +
    '</div>' +
    '<button class="toast__close" type="button" aria-label="Dismiss notification">' +
    '<svg class="icon icon--sm" aria-hidden="true"><use href="#i-x"></use></svg>' +
    '</button>';

  const remove = () => {
    if (node.parentNode) node.parentNode.removeChild(node);
  };
  node.querySelector('.toast__close').addEventListener('click', remove);
  region.appendChild(node);
  window.setTimeout(remove, timeout);
}

/* -------------------------------------------------------------------------- */
/* Theme                                                                       */
/* -------------------------------------------------------------------------- */

function applyTheme(theme) {
  const next = theme === 'light' ? 'light' : 'dark';
  state.theme = next;
  document.documentElement.setAttribute('data-theme', next);
  const toggle = el('theme-toggle');
  if (toggle) {
    toggle.setAttribute(
      'aria-label',
      next === 'dark' ? 'Switch to light theme' : 'Switch to dark theme',
    );
  }
  try {
    window.localStorage.setItem(THEME_KEY, next);
  } catch (error) {
    /* storage may be unavailable — theme still applies for this session */
  }
}

function initTheme() {
  let stored = null;
  try {
    stored = window.localStorage.getItem(THEME_KEY);
  } catch (error) {
    stored = null;
  }
  const prefersLight =
    window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
  applyTheme(stored || (prefersLight ? 'light' : 'dark'));

  const toggle = el('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      applyTheme(state.theme === 'dark' ? 'light' : 'dark');
      charts.rerenderCharts();
    });
  }
}

/* -------------------------------------------------------------------------- */
/* Navigation                                                                  */
/* -------------------------------------------------------------------------- */

function panelFor(name) {
  return el(`panel-${name}`);
}

function selectPanel(name, focusPanel = false) {
  const tabs = Array.from(document.querySelectorAll('.nav-item'));
  const active = tabs.find((tab) => tab.dataset.panel === name) || tabs[0];
  if (!active) return;

  tabs.forEach((tab) => {
    const selected = tab === active;
    tab.setAttribute('aria-selected', selected ? 'true' : 'false');
    tab.setAttribute('tabindex', selected ? '0' : '-1');
    const panel = panelFor(tab.dataset.panel);
    show(panel, selected);
  });

  state.activePanel = active.dataset.panel;
  if (focusPanel) {
    const panel = panelFor(state.activePanel);
    if (panel) panel.focus({ preventScroll: true });
  }

  // Charts inside a hidden panel have no measurable size — resize on reveal.
  window.setTimeout(() => {
    const panel = panelFor(state.activePanel);
    if (panel) charts.resizeChartsIn(panel);
  }, 30);
}

function initNav() {
  const tabs = Array.from(document.querySelectorAll('.nav-item'));
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => selectPanel(tab.dataset.panel));
    tab.addEventListener('keydown', (event) => {
      const keys = ['ArrowDown', 'ArrowRight', 'ArrowUp', 'ArrowLeft', 'Home', 'End'];
      if (!keys.includes(event.key)) return;
      event.preventDefault();
      let target = index;
      if (event.key === 'ArrowDown' || event.key === 'ArrowRight') target = (index + 1) % tabs.length;
      if (event.key === 'ArrowUp' || event.key === 'ArrowLeft')
        target = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') target = 0;
      if (event.key === 'End') target = tabs.length - 1;
      tabs[target].focus();
      selectPanel(tabs[target].dataset.panel);
    });
  });
}

/* -------------------------------------------------------------------------- */
/* Health polling                                                              */
/* -------------------------------------------------------------------------- */

function setHealthPill(ok) {
  state.healthOk = Boolean(ok);
  const pill = el('api-status');
  if (!pill) return;
  pill.classList.remove('is-busy', 'is-online', 'is-offline');
  pill.classList.add(ok ? 'is-online' : 'is-offline');
  const label = pill.querySelector('.status-text');
  if (label) label.textContent = ok ? 'connected' : 'offline';
}

async function pollHealth() {
  try {
    const data = await api.health();
    const status = String(pick(data, 'status', 'ok')).toLowerCase();
    setHealthPill(status === 'ok' || status === 'up' || status === 'healthy');
  } catch (error) {
    setHealthPill(false);
  }
}

function startHealthPolling() {
  pollHealth();
  if (healthTimer) window.clearInterval(healthTimer);
  healthTimer = window.setInterval(pollHealth, HEALTH_INTERVAL);
}

/* -------------------------------------------------------------------------- */
/* Error banner                                                                */
/* -------------------------------------------------------------------------- */

function showError(title, text) {
  const banner = el('error-banner');
  if (!banner) return;
  setText(el('error-title'), title);
  setText(el('error-text'), text);
  show(banner, true);
}

function hideError() {
  show(el('error-banner'), false);
}

/* -------------------------------------------------------------------------- */
/* Loading skeletons                                                           */
/* -------------------------------------------------------------------------- */

function skeletonCards(count) {
  let html = '';
  for (let i = 0; i < count; i += 1) {
    html += '<div class="skeleton skeleton--card"></div>';
  }
  return html;
}

function setLoading(loading) {
  state.loading = Boolean(loading);
  const grid = el('kpi-grid');
  const submit = el('params-submit');
  const note = el('params-note');

  if (loading) {
    if (grid) grid.innerHTML = skeletonCards(6);
    const verdict = el('verdict-body');
    if (verdict) {
      verdict.innerHTML =
        '<div class="skeleton-stack">' +
        '<div class="skeleton skeleton--title"></div>' +
        '<div class="skeleton skeleton--line"></div>' +
        '<div class="skeleton skeleton--line"></div>' +
        '<div class="skeleton skeleton--line skeleton--short"></div>' +
        '</div>';
    }
    const impact = el('impact-table');
    if (impact) {
      impact.innerHTML =
        '<div class="skeleton-stack">' +
        '<div class="skeleton skeleton--line"></div>'.repeat(5) +
        '</div>';
    }
    if (submit) submit.disabled = true;
    if (note) note.textContent = 'Loading the analysis…';
  } else {
    if (submit) submit.disabled = false;
    if (note) {
      note.textContent =
        'Re-running executes the full forecasting & financial pipeline on the server.';
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Data loading                                                                */
/* -------------------------------------------------------------------------- */

async function loadDatasets() {
  try {
    const data = await api.datasets();
    state.datasets = asArray(pick(data, 'datasets', []));
  } catch (error) {
    state.datasets = [];
  }
  populateDatasetSelect();
}

function populateDatasetSelect() {
  const select = el('param-dataset');
  if (!select) return;
  const previous = select.value;
  select.innerHTML = '';

  const entries = state.datasets
    .map((dataset, index) => ({
      id: pick(dataset, 'id', pick(dataset, 'label', null)),
      label: pick(dataset, 'label', pick(dataset, 'id', 'Dataset')),
      months: asNumber(pick(dataset, 'months')),
      salespersons: asNumber(pick(dataset, 'salespersons')),
      index,
    }))
    .filter((entry) => typeof entry.id === 'string' && entry.id.length > 0);

  if (!entries.length) {
    // No dataset list yet (the API may still be loading): leave the select
    // empty rather than inventing a file name, and let the server pick.
    entries.push({
      id: '',
      label: 'Server default',
      months: null,
      salespersons: null,
      index: -1,
    });
  }

  const ids = entries.map((entry) => entry.id);

  entries.forEach((entry) => {
    const option = document.createElement('option');
    option.value = entry.id;
    const detail =
      entry.months !== null || entry.salespersons !== null
        ? ` — ${entry.months ?? '?'} months, ${entry.salespersons ?? '?'} salespersons`
        : '';
    option.textContent = `${entry.label}${detail}`;
    select.appendChild(option);
  });

  const analysisDataset = pick(state.analysis, 'dataset.name', null);
  if (analysisDataset && ids.includes(analysisDataset)) select.value = analysisDataset;
  else if (previous && ids.includes(previous)) select.value = previous;
}

function readParams() {
  const dataset = el('param-dataset');
  const salary = el('param-salary');
  const commission = el('param-commission');
  const months = el('param-forecast-months');
  const start = el('param-start-month');
  return {
    // An empty selection means "use the server's default", so no file
    // name is ever hard-coded here.
    dataset: dataset && dataset.value ? dataset.value : '',
    salary: asNumber(salary ? salary.value : null),
    commission: asNumber(commission ? commission.value : null),
    forecast_months: asNumber(months ? months.value : null),
    start_month: asNumber(start ? start.value : null),
  };
}

/**
 * Explain the money scale using the *current* run's own figures, so the
 * hint is never a hard-coded example from one particular dataset.
 */
function renderUnitsHint() {
  const node = el('units-hint');
  if (!node) return;

  const kpis = pick(state.analysis, 'kpis', {});
  const sales = asNumber(pick(kpis, 'avg_monthly_sales'));
  const months = asNumber(pick(state.analysis, 'dataset.months'));
  const salespersons = asNumber(pick(state.analysis, 'dataset.salespersons'));

  const parts = [];
  if (sales !== null && sales > 0) {
    // 1 crore = 100 lakh, so the conversion is always derived.
    const crore = sales / 100;
    parts.push(
      crore >= 1
        ? `“${fmtL(sales)}” is Rs. ${crore.toFixed(3)} crore.`
        : `“${fmtL(sales)}”.`,
    );
  }
  if (months !== null && salespersons !== null) {
    parts.push(
      `Figures below cover ${months} month${months === 1 ? '' : 's'} across ` +
        `${salespersons} salesperson${salespersons === 1 ? '' : 's'}.`,
    );
  }

  node.textContent = parts.length ? ` ${parts.join(' ')}` : '';
}

/**
 * Seed the parameter form from the analysis that is currently loaded, so
 * the controls always describe the run on screen instead of a fixed set
 * of defaults.
 */
function syncParamsFromAnalysis() {
  const assumptions = pick(state.analysis, 'assumptions', {});
  if (!assumptions || typeof assumptions !== 'object') return;

  const setValue = (id, value) => {
    const node = el(id);
    const num = asNumber(value);
    if (node && num !== null) node.value = String(num);
  };

  setValue('param-salary', pick(assumptions, 'salary'));
  setValue('param-commission', pick(assumptions, 'commission'));
  setValue('param-forecast-months', pick(assumptions, 'forecast_months'));
  setValue('param-start-month', pick(assumptions, 'start_month'));
}

function mostRecentRunId() {
  const runs = state.runs.slice();
  if (!runs.length) return null;
  runs.sort((a, b) => {
    const aTime = Date.parse(pick(a, 'created_at', '') || '') || 0;
    const bTime = Date.parse(pick(b, 'created_at', '') || '') || 0;
    return bTime - aTime;
  });
  return pick(runs, '0.run_id', null);
}

async function loadAnalysis(options = {}) {
  setLoading(true);
  hideError();

  try {
    if (!state.datasets.length) await loadDatasets();

    let analysis = null;

    // Prefer a cached run to avoid re-running the whole pipeline on every visit.
    try {
      const runsPayload = await api.runs();
      state.runs = asArray(pick(runsPayload, 'runs', []));
      const runId = mostRecentRunId();
      if (runId) analysis = await api.analysis(runId);
    } catch (error) {
      analysis = null;
    }

    if (!analysis) {
      analysis = await api.createAnalysis(readParams());
    }

    state.analysis = analysis;
    await loadSensitivity();
    populateDatasetSelect();
    syncParamsFromAnalysis();
    renderAll();
    setLoading(false);
    syncReportLink();
    if (simulator) {
      simulator.setEnabled(true);
      // The sandbox must perturb the assumptions of the run on screen.
      simulator.seed(pick(state.analysis, 'assumptions', {}));
    }

    if (!options.silent) {
      toast('Analysis ready', 'The staffing analysis has been loaded.', 'success');
    }

    // Seed the simulator with the current run so the panel is never empty.
    if (simulator) {
      suppressSimulationErrors = true;
      simulator
        .run({ auto: true })
        .catch(() => null)
        .finally(() => {
          suppressSimulationErrors = false;
        });
    }
  } catch (error) {
    setLoading(false);
    const message =
      error && error.message
        ? error.message
        : 'The analysis API could not be reached.';
    showError(
      'Analysis API unreachable',
      `${message} Start the FastAPI service and retry to load the dashboard.`,
    );
    renderEmptyState();
  }
}

async function loadSensitivity() {
  const embedded = pick(state.analysis, 'sensitivity', null);
  const hasEmbedded =
    embedded &&
    (asArray(pick(embedded, 'tornado', [])).length > 0 ||
      asArray(pick(embedded, 'grid', [])).length > 0);
  if (hasEmbedded) {
    state.sensitivity = embedded;
    return;
  }
  const runId = pick(state.analysis, 'run_id', null);
  if (!runId) {
    state.sensitivity = embedded;
    return;
  }
  try {
    const payload = await api.sensitivity(runId);
    state.sensitivity = payload && payload.tornado !== undefined ? payload : embedded;
  } catch (error) {
    state.sensitivity = embedded;
  }
}

function renderEmptyState() {
  const grid = el('kpi-grid');
  if (grid) {
    grid.innerHTML =
      '<div class="empty-state"><strong>No analysis data</strong>' +
      '<span>The dashboard could not load a run. Use Retry in the banner above.</span></div>';
  }
  ['verdict-body', 'impact-table', 'leaderboard', 'ensemble', 'backtest-table'].forEach((id) => {
    const node = el(id);
    if (node) {
      node.innerHTML =
        '<div class="empty-state"><strong>Unavailable</strong><span>Waiting for a successful analysis run.</span></div>';
    }
  });
}

/* -------------------------------------------------------------------------- */
/* Shared render helpers                                                       */
/* -------------------------------------------------------------------------- */

/**
 * Render a data table into a container.
 *
 * @param {HTMLElement} container
 * @param {object} spec `{ head: [{label, num}], rows: [[html]], rowClass }`
 */
function renderTable(container, spec) {
  if (!container) return;
  const head = spec.head || [];
  const rows = spec.rows || [];
  if (!rows.length) {
    container.innerHTML =
      '<div class="empty-state"><strong>No rows</strong><span>The API returned no records for this view.</span></div>';
    return;
  }
  const headHtml = head
    .map(
      (column) =>
        `<th scope="col" class="${column.num ? 'num' : ''}">${esc(column.label)}</th>`,
    )
    .join('');
  const bodyHtml = rows
    .map((row, rowIndex) => {
      const rowClass = typeof spec.rowClass === 'function' ? spec.rowClass(row, rowIndex) : '';
      const cells = row
        .map((cell, cellIndex) => {
          const column = head[cellIndex] || {};
          return `<td class="${column.num ? 'num' : ''}">${cell}</td>`;
        })
        .join('');
      return `<tr class="${rowClass}">${cells}</tr>`;
    })
    .join('');
  container.innerHTML = `<table class="data"><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
}

/** Scenario keys in canonical order. */
const SCENARIO_KEYS = ['A', 'B', 'C', 'D'];

/** Normalised scenario records for cards / charts / tables. */
function scenarioRecords(scenarios, bestKey) {
  if (!scenarios || typeof scenarios !== 'object') return [];
  return SCENARIO_KEYS.filter((key) => scenarios[key]).map((key) => {
    const scenario = scenarios[key];
    const monthly = asArray(pick(scenario, 'monthly', []));
    let running = 0;
    const cumulative = monthly.map((row) => {
      running += asNumber(pick(row, 'gross_margin'), 0) || 0;
      return { month: asNumber(pick(row, 'month'), 0), value: running };
    });
    return {
      key,
      name: String(pick(scenario, 'name', `Scenario ${key}`)),
      totalSales: asNumber(pick(scenario, 'total_sales')),
      totalMargin: asNumber(pick(scenario, 'total_gross_margin')),
      totalCost: asNumber(pick(scenario, 'total_staff_cost')),
      avgMargin: asNumber(pick(scenario, 'avg_monthly_margin')),
      delta: asNumber(pick(scenario, 'delta_vs_best')),
      risk: String(pick(scenario, 'risk', '') || ''),
      advantages: asArray(pick(scenario, 'advantages', [])).filter(Boolean),
      disadvantages: asArray(pick(scenario, 'disadvantages', [])).filter(Boolean),
      probPositive: asNumber(pick(scenario, 'prob_positive')),
      p10: asNumber(pick(scenario, 'p10_margin')),
      p90: asNumber(pick(scenario, 'p90_margin')),
      monthly,
      cumulative,
      isBest: bestKey ? key === bestKey : false,
      // The simulator can restrict the comparison to one policy; the
      // out-of-scope ones must say so rather than look like peers.
      inScope: pick(scenario, 'in_scope', true) !== false,
    };
  });
}

/** Render the scenario card grid. */
function renderScenarioCards(container, records, months) {
  if (!container) return;
  const horizon = Number.isFinite(months) && months > 0 ? Math.round(months) : 6;
  const horizonLabel = `${horizon}m`;
  if (!records.length) {
    container.innerHTML =
      '<div class="empty-state"><strong>No scenarios</strong><span>The API returned no scenario data.</span></div>';
    return;
  }

  container.innerHTML = records
    .map((scenario) => {
      const delta = scenario.delta;
      const deltaMarkup =
        delta === null
          ? '<span class="metric__value delta delta--flat">—</span>'
          : `<span class="metric__value delta ${deltaClass(delta)}">${esc(fmtSigned(delta))} L</span>`;

      const pros = scenario.advantages.length
        ? `<ul>${scenario.advantages
            .map((item) => `<li>${esc(item)}</li>`)
            .join('')}</ul>`
        : '<ul><li>No advantages recorded.</li></ul>';

      const cons = scenario.disadvantages.length
        ? `<ul>${scenario.disadvantages
            .map((item) => `<li>${esc(item)}</li>`)
            .join('')}</ul>`
        : '<ul><li>No disadvantages recorded.</li></ul>';

      const probability =
        scenario.probPositive === null
          ? ''
          : `<span class="badge badge--neutral">P(margin &gt; 0) ${esc(
              fmtFlexiblePct(scenario.probPositive),
            )}</span>`;

      return `
        <article class="scenario-card${scenario.isBest ? ' is-best' : ''}${
          scenario.inScope ? '' : ' is-excluded'
        }">
          <div class="scenario-card__head">
            <span class="key-badge">${esc(scenario.key)}</span>
            <div>
              <p class="scenario-card__name">${esc(scenario.name)}</p>
              <div class="row">${
                scenario.isBest
                  ? '<span class="badge badge--positive">Best</span>'
                  : scenario.inScope
                    ? '<span class="badge badge--neutral">Alternative</span>'
                    : '<span class="badge badge--neutral">Not compared</span>'
              }${probability}</div>
            </div>
          </div>
          <div class="scenario-card__metrics">
            <div class="metric">
              <span class="metric__label">Gross margin (${esc(horizonLabel)})</span>
              <span class="metric__value">${esc(fmtL(scenario.totalMargin))}</span>
            </div>
            <div class="metric">
              <span class="metric__label">Delta vs. best</span>
              ${deltaMarkup}
            </div>
            <div class="metric">
              <span class="metric__label">Avg monthly margin</span>
              <span class="metric__value">${esc(fmtL(scenario.avgMargin))}</span>
            </div>
            <div class="metric">
              <span class="metric__label">Staff cost (${esc(horizonLabel)})</span>
              <span class="metric__value">${esc(fmtL(scenario.totalCost))}</span>
            </div>
            <div class="metric">
              <span class="metric__label">P10 – P90 margin</span>
              <span class="metric__value">${
                scenario.p10 === null && scenario.p90 === null
                  ? '—'
                  : `${esc(fmtNum(scenario.p10, 1))} – ${esc(fmtNum(scenario.p90, 1))}`
              }</span>
            </div>
          </div>
          ${scenario.risk ? `<p class="risk-text">${esc(scenario.risk)}</p>` : ''}
          <div class="pro-con">
            <div class="pro-con__group"><h4>Strengths</h4>${pros}</div>
            <div class="pro-con__group pro-con__group--cons"><h4>Risks</h4>${cons}</div>
          </div>
        </article>`;
    })
    .join('');
}

/** Render the scenario comparison table. */
function renderScenarioTable(container, records) {
  const rows = records.map((scenario) => [
    `<span class="key-badge">${esc(scenario.key)}</span>`,
    esc(scenario.name),
    esc(fmtL(scenario.totalSales)),
    esc(fmtL(scenario.totalMargin)),
    esc(fmtL(scenario.totalCost)),
    esc(fmtL(scenario.avgMargin)),
    `<span class="delta ${deltaClass(scenario.delta)}">${esc(fmtSigned(scenario.delta))} L</span>`,
    esc(fmtFlexiblePct(scenario.probPositive)),
  ]);

  renderTable(container, {
    head: [
      { label: 'Key' },
      { label: 'Scenario' },
      { label: 'Total sales', num: true },
      { label: 'Gross margin', num: true },
      { label: 'Staff cost', num: true },
      { label: 'Avg monthly margin', num: true },
      { label: 'Delta vs best', num: true },
      { label: 'P(margin > 0)', num: true },
    ],
    rows,
    rowClass: (row, index) => (records[index] && records[index].isBest ? 'is-top' : ''),
  });
}

/** Build a CSV download of the scenario comparison. */
function downloadScenarioCsv() {
  const records = scenarioRecords(
    pick(state.analysis, 'scenarios', {}),
    pick(state.analysis, 'executive.recommended_scenario', null),
  );
  if (!records.length) {
    toast('Nothing to export', 'No scenario data is available yet.', 'error');
    return;
  }
  const header = [
    'scenario_key',
    'name',
    'total_sales_l',
    'total_gross_margin_l',
    'total_staff_cost_l',
    'avg_monthly_margin_l',
    'delta_vs_best_l',
    'prob_positive',
  ];
  const lines = [header.join(',')];
  records.forEach((scenario) => {
    lines.push(
      [
        scenario.key,
        `"${String(scenario.name).replace(/"/g, '""')}"`,
        scenario.totalSales ?? '',
        scenario.totalMargin ?? '',
        scenario.totalCost ?? '',
        scenario.avgMargin ?? '',
        scenario.delta ?? '',
        scenario.probPositive ?? '',
      ].join(','),
    );
  });

  try {
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'store_staffing-scenario-comparison.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.setTimeout(() => URL.revokeObjectURL(url), 2000);
    toast('Download started', 'Scenario comparison exported as CSV.', 'success');
  } catch (error) {
    toast('Export failed', 'The browser blocked the CSV download.', 'error');
  }
}

/* -------------------------------------------------------------------------- */
/* Section renderers                                                           */
/* -------------------------------------------------------------------------- */

function renderKpis() {
  const container = el('kpi-grid');
  if (!container) return;
  const analysis = state.analysis;
  const kpis = pick(analysis, 'kpis', {});
  const series = pick(analysis, 'series', {});

  const sales = asArray(pick(series, 'sales', []));
  const margins = asArray(pick(series, 'gross_margin', []));
  const forecastValues = asArray(pick(analysis, 'forecast.values', []));

  const accent = charts.cssVar('--accent', '#d3a04c');
  const teal = charts.cssVar('--teal', '#57a89b');
  const violet = charts.cssVar('--violet', '#9189ab');

  // The optimiser answers "what would the model like the team size to be",
  // which is a different question from the near-term hiring decision.
  // Showing both side by side stops the two numbers reading as a conflict.
  const scenarios = pick(analysis, 'scenarios', {});
  const bestKey = pick(analysis, 'executive.recommended_scenario', null);
  const bestScenario = bestKey ? pick(scenarios, `${bestKey}.monthly`, []) : [];
  const recommendedHeadcount = asNumber(
    bestScenario.length ? pick(bestScenario[0], 'employees') : null,
  );
  const optimum = asNumber(pick(kpis, 'optimal_headcount'));
  const headcountSub =
    optimum === null
      ? ''
      : recommendedHeadcount !== null && recommendedHeadcount !== optimum
        ? `Capacity optimum ${fmtInt(optimum)} · decide ${fmtInt(recommendedHeadcount)} now`
        : `Capacity optimum ${fmtInt(optimum)}`;

  const cards = [
    {
      label: 'Current headcount',
      value: fmtInt(pick(kpis, 'current_employees')),
      sub: headcountSub,
      accent,
    },
    {
      label: 'Recommended action',
      value: String(pick(analysis, 'executive.recommendation', '—')),
      text: true,
      sub: `Scenario ${pick(analysis, 'executive.recommended_scenario', '—')} · confidence ${pick(
        analysis,
        'executive.confidence',
        '—',
      )}`,
      accent,
    },
    {
      label: 'Avg monthly sales',
      value: fmtL(pick(kpis, 'avg_monthly_sales')),
      sub: `Forecast avg ${fmtL(pick(kpis, 'forecast_avg_sales'))}`,
      spark: sales,
      sparkColor: accent,
    },
    {
      label: 'Avg monthly margin',
      value: fmtL(pick(kpis, 'avg_monthly_margin')),
      sub: `Total ${fmtL(pick(kpis, 'total_margin'))} · best ${fmtL(pick(kpis, 'best_margin'))}`,
      spark: margins,
      sparkColor: accent,
    },
    {
      label: 'Break-even / employee',
      value: fmtL(pick(kpis, 'break_even_sales_per_employee')),
      sub: `Current avg ${fmtL(pick(kpis, 'avg_sales_per_employee'))}`,
      accent,
    },
    {
      label: 'Forecast model',
      value: String(pick(kpis, 'best_forecast_model', '—')),
      text: true,
      sub: `RMSE ${fmtL(pick(kpis, 'validation_rmse'))} · MAPE ${fmtFlexiblePct(
        pick(kpis, 'validation_mape'),
      )}`,
      spark: forecastValues,
      sparkColor: violet,
    },
  ];

  container.innerHTML = cards
    .map(
      (card) => `
      <article class="stat" data-accent="${esc(card.accent)}">
        <span class="stat__label">${esc(card.label)}</span>
        <span class="stat__value${card.text ? ' stat__value--text' : ''}">${esc(card.value)}</span>
        <span class="stat__sub">${esc(card.sub)}</span>
        ${card.spark ? '<div class="stat__spark" data-spark></div>' : ''}
      </article>`,
    )
    .join('');

  // Paint the per-card accent stripe and attach sparklines.
  const nodes = Array.from(container.querySelectorAll('.stat'));
  cards.forEach((card, index) => {
    const node = nodes[index];
    if (!node) return;
    node.style.setProperty('--stat-accent', card.accent || accent);
    const holder = node.querySelector('[data-spark]');
    if (holder && card.spark) {
      holder.appendChild(charts.buildSparkline(card.spark, { color: card.sparkColor || accent }));
    }
  });
}

function renderVerdict() {
  const analysis = state.analysis;
  const confidence = String(pick(analysis, 'executive.confidence', '—'));
  const badge = el('verdict-confidence');
  if (badge) {
    badge.textContent = confidence;
    badge.className = `badge badge--${confidenceTone(confidence)}`;
  }

  const body = el('verdict-body');
  if (!body) return;

  const headline =
    pick(analysis, 'executive.headline', null) ||
    pick(analysis, 'executive.recommendation', 'No recommendation available');
  const reasoning = asArray(pick(analysis, 'executive.reasoning', [])).filter(Boolean);
  const runnerUp = pick(analysis, 'executive.runner_up', null);
  const recommended = pick(analysis, 'executive.recommended_scenario', null);

  body.innerHTML = `
    <div class="verdict">
      <p class="verdict__headline">${esc(headline)}</p>
      <div class="verdict__meta">
        ${
          recommended
            ? `<span class="badge badge--accent">Recommended scenario ${esc(recommended)}</span>`
            : ''
        }
        ${
          runnerUp
            ? `<span class="badge badge--neutral badge--wrap">Runner-up · ${esc(runnerUp)}</span>`
            : ''
        }
      </div>
      ${
        reasoning.length
          ? `<ul class="reasoning">${reasoning
              .map((reason) => `<li>${esc(reason)}</li>`)
              .join('')}</ul>`
          : '<p class="text-muted text-sm">The API returned no reasoning bullets for this run.</p>'
      }
    </div>`;
}

function renderImpact() {
  const container = el('impact-table');
  if (!container) return;
  const impact = pick(state.analysis, 'executive.expected_impact', {});
  const advantage = asNumber(pick(impact, 'margin_advantage_over_runner_up'));
  const vsBaseline = asNumber(pick(impact, 'margin_advantage_vs_do_nothing'));
  const months = asNumber(pick(state.analysis, 'assumptions.forecast_months')) || 6;
  const window = `${months} month${months === 1 ? '' : 's'}`;

  const caption = el('impact-caption');
  if (caption) {
    caption.textContent = `${months}-month projection for the recommended plan.`;
  }

  const rows = [
    [`Total sales (${window})`, esc(fmtL(pick(impact, 'total_sales_6m')))],
    [`Total gross margin (${window})`, esc(fmtL(pick(impact, 'total_gross_margin_6m')))],
    [`Total staff cost (${window})`, esc(fmtL(pick(impact, 'total_staff_cost_6m')))],
    ['Average monthly margin', esc(fmtL(pick(impact, 'avg_monthly_margin')))],
    [
      'Margin gain vs. leaving the vacancy open',
      `<span class="delta ${deltaClass(vsBaseline)}">${esc(fmtSigned(vsBaseline))} L</span>`,
    ],
    [
      'Margin advantage over runner-up',
      `<span class="delta ${deltaClass(advantage)}">${esc(fmtSigned(advantage))} L</span>`,
    ],
  ];

  renderTable(container, {
    head: [{ label: 'Metric' }, { label: 'Value', num: true }],
    rows,
  });
}

function renderOverviewCharts() {
  const analysis = state.analysis;
  const series = pick(analysis, 'series', {});
  const forecast = pick(analysis, 'forecast', {});

  const modelBadge = el('forecast-model-badge');
  if (modelBadge) {
    const model = String(pick(forecast, 'model', '—'));
    modelBadge.textContent = model;
  }

  charts.renderChart('chart-forecast', charts.buildForecastChart, {
    months: asArray(pick(series, 'months', [])),
    sales: asArray(pick(series, 'sales', [])),
    forecastMonths: asArray(pick(forecast, 'months', [])),
    values: asArray(pick(forecast, 'values', [])),
    lower95: asArray(pick(forecast, 'lower_95', [])),
    upper95: asArray(pick(forecast, 'upper_95', [])),
    confidenceLevel: asNumber(pick(analysis, 'assumptions.confidence_level')),
  });

  // The chart caption and the accessible label follow the same level.
  const band = asNumber(pick(analysis, 'assumptions.confidence_level'));
  const bandPct = band === null ? 95 : Math.round(band * 100);
  const bandSub = el('forecast-band-sub');
  if (bandSub) {
    bandSub.textContent =
      `Historical monthly sales (solid) with the selected model's forecast ` +
      `(dashed) and a ${bandPct}% prediction band.`;
  }
  const forecastCanvas = el('chart-forecast');
  if (forecastCanvas) {
    forecastCanvas.setAttribute(
      'aria-label',
      `Monthly sales with a ${bandPct} percent confidence forecast band`,
    );
  }

  charts.renderChart('chart-margin', charts.buildMarginChart, {
    months: asArray(pick(series, 'months', [])),
    grossMargin: asArray(pick(series, 'gross_margin', [])),
  });

  const marginNote = el('margin-panel-sub');
  if (marginNote) {
    const kpis = pick(state.analysis, 'kpis', {});
    marginNote.textContent =
      `Monthly gross margin in Lakhs after commission and staff cost — best ${fmtL(
        pick(kpis, 'best_margin'),
      )}, weakest ${fmtL(pick(kpis, 'worst_margin'))}. ` +
      `Sales volatility ${fmtFlexiblePct(pick(kpis, 'sales_volatility'))}, ` +
      `productivity CV ${fmtFlexiblePct(pick(kpis, 'productivity_cv'))}.`;
  }

  const months = asArray(pick(series, 'months', []));
  const employees = asArray(pick(series, 'employees', []));
  const sales = asArray(pick(series, 'sales', []));
  const grossMargin = asArray(pick(series, 'gross_margin', []));

  const marginValues = grossMargin.map((value) => asNumber(value) ?? 0);
  const maxMargin = marginValues.length ? Math.max(...marginValues, 1) : 1;

  const points = months.map((month, index) => {
    const margin = asNumber(grossMargin[index]);
    const ratio = maxMargin ? Math.max(0, (margin ?? 0) / maxMargin) : 0;
    return {
      x: asNumber(employees[index], 0) ?? 0,
      y: asNumber(sales[index], 0) ?? 0,
      r: Math.max(3, Math.min(16, 3 + ratio * 12)),
      margin,
      month,
    };
  });

  charts.renderChart('chart-scatter', charts.buildScatterChart, { points });
}

function renderForecast() {
  const forecast = pick(state.analysis, 'forecast', {});

  const metrics = asArray(pick(forecast, 'metrics', []));
  renderTable(el('leaderboard'), {
    head: [
      { label: 'Rank', num: true },
      { label: 'Model' },
      { label: 'RMSE (L)', num: true },
      { label: 'MAE (L)', num: true },
      { label: 'MAPE', num: true },
      { label: 'MASE', num: true },
      { label: 'R²', num: true },
    ],
    rows: metrics.map((metric) => [
      esc(fmtInt(pick(metric, 'rank'))),
      esc(pick(metric, 'model', '—')),
      esc(fmtL(pick(metric, 'rmse'), 3)),
      esc(fmtL(pick(metric, 'mae'), 3)),
      esc(fmtFlexiblePct(pick(metric, 'mape'))),
      esc(fmtNum(pick(metric, 'mase'), 3)),
      esc(fmtNum(pick(metric, 'r2'), 3)),
    ]),
    rowClass: (row) => (String(row[0]) === '1' ? 'is-top' : ''),
  });

  const members = asArray(pick(forecast, 'ensemble_members', []));
  const ensembleContainer = el('ensemble');
  if (ensembleContainer) {
    if (!members.length) {
      ensembleContainer.innerHTML =
        '<div class="empty-state"><strong>No ensemble</strong><span>The selected model is not an ensemble.</span></div>';
    } else {
      const maxWeight = Math.max(
        ...members.map((member) => asNumber(pick(member, 'weight'), 0) ?? 0),
        0.0001,
      );
      ensembleContainer.innerHTML = `<div class="stack">${members
        .map((member) => {
          const weight = asNumber(pick(member, 'weight'), 0) ?? 0;
          const width = Math.max(2, Math.min(100, (weight / maxWeight) * 100));
          return `
            <div>
              <div class="row spread">
                <span class="text-sm">${esc(pick(member, 'model', '—'))}</span>
                <span class="text-sm muted">${esc(fmtFlexiblePct(weight))} · RMSE ${esc(
                  fmtL(pick(member, 'rmse'), 3),
                )}</span>
              </div>
              <span class="mini-bar"><span class="mini-bar__fill" data-width="${width}"></span></span>
            </div>`;
        })
        .join('')}</div>`;
      ensembleContainer.querySelectorAll('[data-width]').forEach((node) => {
        node.style.setProperty('--w', `${node.getAttribute('data-width')}%`);
      });
    }
  }

  const backtest = asArray(pick(forecast, 'backtest', []));

  // The API exposes a scalar per origin plus the full path; accept either
  // so the table and chart stay correct across payload versions.
  const btValue = (row, key) => {
    const value = pick(row, key);
    if (Array.isArray(value)) return value.length ? value[value.length - 1] : null;
    return value;
  };
  const btSeries = (row, key) => {
    const value = pick(row, key);
    if (Array.isArray(value)) return value;
    const single = asNumber(value);
    return single === null ? [] : [single];
  };

  renderTable(el('backtest-table'), {
    head: [
      { label: 'Origin month', num: true },
      { label: 'Actual (L)', num: true },
      { label: 'Predicted (L)', num: true },
      { label: 'Error (L)', num: true },
    ],
    rows: backtest.map((row) => [
      esc(fmtInt(pick(row, 'origin_month'))),
      esc(fmtL(btValue(row, 'actual'), 3)),
      esc(fmtL(btValue(row, 'predicted'), 3)),
      esc(fmtSigned(btValue(row, 'error'), 3)),
    ]),
  });

  charts.renderChart('chart-backtest', charts.buildBacktestChart, {
    backtest: backtest.map((row) => ({
      origin_month: pick(row, 'origin_month'),
      actual: btSeries(row, 'actual'),
      predicted: btSeries(row, 'predicted'),
    })),
  });

  const methodology = el('forecast-methodology');
  if (methodology) {
    const model = String(pick(forecast, 'model', '—'));
    const text = String(pick(forecast, 'methodology', '') || 'No methodology note supplied.');
    const residual = asNumber(pick(forecast, 'residual_std'));
    methodology.innerHTML = `
      <div class="callout callout--accent">
        <svg class="icon" aria-hidden="true"><use href="#i-info"></use></svg>
        <div>
          <p><strong>${esc(model)}</strong> — ${esc(text)}</p>
          <p class="text-sm">Residual standard deviation: ${esc(fmtL(residual, 3))}</p>
        </div>
      </div>`;
  }
}

function renderScenarios() {
  const scenarios = pick(state.analysis, 'scenarios', {});
  const bestKey = pick(state.analysis, 'executive.recommended_scenario', null);
  const records = scenarioRecords(scenarios, bestKey);

  const caption = el('scenarios-sub');
  if (caption) {
    const count = Object.keys(scenarios || {}).length;
    caption.textContent =
      `${count} staffing strateg${count === 1 ? 'y' : 'ies'} compared over the ` +
      'forecast horizon, including ramp-up of a new hire.';
  }

  renderScenarioCards(
    el('scenario-cards'),
    records,
    asNumber(pick(state.analysis, 'assumptions.forecast_months')),
  );
  renderScenarioTable(el('scenario-table'), records);

  charts.renderChart('chart-scenario-bar', charts.buildScenarioBarChart, {
    scenarios: records.map((record) => ({
      key: record.key,
      name: record.name,
      margin: record.totalMargin,
      isBest: record.isBest,
    })),
  });

  charts.renderChart('chart-scenario-cumulative', charts.buildScenarioCumulativeChart, {
    scenarios: records,
  });
}

function renderWorkforce() {
  const productivity = pick(state.analysis, 'productivity', {});
  const attrition = pick(productivity, 'attrition', {});

  const attritionCards = [
    {
      label: 'Total leavers',
      value: fmtInt(pick(attrition, 'total_leavers')),
      sub: 'Across the analysed period',
    },
    {
      label: 'Average tenure',
      value: `${fmtNum(pick(attrition, 'avg_tenure_months'), 1)} months`,
      sub: 'Mean months of employment',
    },
    {
      label: 'Leavers (last 12 months)',
      value: fmtInt(pick(attrition, 'recent_12m_leavers')),
      sub: 'Recent attrition pressure',
    },
    {
      label: 'Diminishing returns',
      value: pick(productivity, 'diminishing_detected') === true ? 'Detected' : 'Not detected',
      sub: 'Extra hires add less per head over time',
    },
  ];

  const container = el('attrition-cards');
  if (container) {
    container.innerHTML = attritionCards
      .map(
        (card) => `
        <article class="stat">
          <span class="stat__label">${esc(card.label)}</span>
          <span class="stat__value stat__value--text">${esc(card.value)}</span>
          <span class="stat__sub">${esc(card.sub)}</span>
        </article>`,
      )
      .join('');
  }

  const employees = asArray(pick(productivity, 'employees', []));
  renderTable(el('roster-table'), {
    head: [
      { label: 'ID' },
      { label: 'Total sales', num: true },
      { label: 'Avg monthly sales', num: true },
      { label: 'Months active', num: true },
      { label: 'First month', num: true },
      { label: 'Last month', num: true },
      { label: 'Status' },
      { label: 'Share of sales' },
    ],
    rows: employees.map((employee) => {
      const share = asNumber(pick(employee, 'share_of_sales'));
      const sharePct = share === null ? null : Math.abs(share) <= 1 ? share * 100 : share;
      const width = Math.max(0, Math.min(100, sharePct ?? 0));
      const active = pick(employee, 'still_active');
      const statusBadge =
        active === true
          ? '<span class="badge badge--positive">Active</span>'
          : active === false
            ? '<span class="badge badge--neutral">Left</span>'
            : '<span class="badge badge--neutral">—</span>';
      return [
        `<span class="mono">${esc(pick(employee, 'id', '—'))}</span>`,
        esc(fmtL(pick(employee, 'total_sales'))),
        esc(fmtL(pick(employee, 'avg_monthly_sales'))),
        esc(fmtInt(pick(employee, 'months_active'))),
        esc(fmtInt(pick(employee, 'first_month'))),
        esc(fmtInt(pick(employee, 'last_month'))),
        statusBadge,
        `<div class="mini-bar-cell"><span class="mini-bar"><span class="mini-bar__fill" data-width="${width}"></span></span><span>${esc(
          sharePct === null ? '—' : fmtPct(sharePct),
        )}</span></div>`,
      ];
    }),
  });

  const roster = el('roster-table');
  if (roster) {
    roster.querySelectorAll('[data-width]').forEach((node) => {
      node.style.setProperty('--w', `${node.getAttribute('data-width')}%`);
    });
  }

  charts.renderChart('chart-headcount', charts.buildHeadcountChart, {
    timeline: asArray(pick(state.analysis, 'series.headcount_timeline', [])),
  });

  charts.renderChart('chart-diminishing', charts.buildDiminishingReturnsChart, {
    byHeadcount: asArray(pick(productivity, 'by_headcount', [])),
  });
}

function renderSensitivity() {
  const sensitivity = state.sensitivity || pick(state.analysis, 'sensitivity', {}) || {};
  const tornado = asArray(pick(sensitivity, 'tornado', []));
  const monteCarlo = pick(sensitivity, 'monte_carlo', {}) || {};
  const grid = asArray(pick(sensitivity, 'grid', []));

  const mcCards = [
    { label: 'P10 margin (downside)', value: fmtL(pick(monteCarlo, 'p10')) },
    { label: 'P50 margin (median)', value: fmtL(pick(monteCarlo, 'p50')) },
    { label: 'P90 margin (upside)', value: fmtL(pick(monteCarlo, 'p90')) },
    { label: 'Mean margin', value: fmtL(pick(monteCarlo, 'mean')) },
    { label: 'Standard deviation', value: fmtL(pick(monteCarlo, 'std')) },
    { label: 'Probability of positive margin', value: fmtFlexiblePct(pick(monteCarlo, 'prob_positive')) },
    { label: 'Probability better than hold', value: fmtFlexiblePct(pick(monteCarlo, 'prob_better_than_hold')) },
  ];

  const container = el('mc-cards');
  if (container) {
    container.innerHTML = mcCards
      .map(
        (card) => `
        <article class="stat">
          <span class="stat__label">${esc(card.label)}</span>
          <span class="stat__value stat__value--text">${esc(card.value)}</span>
        </article>`,
      )
      .join('');
  }

  charts.renderChart('chart-tornado', charts.buildTornadoChart, { tornado });
  charts.renderChart('chart-montecarlo', charts.buildMonteCarloChart, {
    p10: asNumber(pick(monteCarlo, 'p10')),
    p50: asNumber(pick(monteCarlo, 'p50')),
    p90: asNumber(pick(monteCarlo, 'p90')),
    mean: asNumber(pick(monteCarlo, 'mean')),
    std: asNumber(pick(monteCarlo, 'std')),
  });

  renderHeatmap(grid);
}

/**
 * Read a sensitivity-grid row, tolerating the different field spellings
 * the API has used (snake_case and the older CamelCase form).
 */
function gridField(row, ...names) {
  for (const name of names) {
    const value = pick(row, name);
    if (value !== null && value !== undefined) return value;
  }
  return null;
}

function gridNumber(row, ...names) {
  return asNumber(gridField(row, ...names));
}

function renderHeatmap(grid) {
  const container = el('heatmap');
  if (!container) return;

  if (!grid.length) {
    container.innerHTML =
      '<div class="empty-state"><strong>No grid</strong><span>The API returned no sensitivity grid.</span></div>';
    return;
  }

  const salaries = Array.from(
    new Set(grid.map((row) => gridNumber(row, 'salary', 'Salary')).filter((v) => v !== null)),
  ).sort((a, b) => a - b);
  const commissions = Array.from(
    new Set(grid.map((row) => gridNumber(row, 'commission_rate', 'commission', 'CommissionRate')).filter((v) => v !== null)),
  ).sort((a, b) => a - b);

  const margins = grid
    .map((row) => gridNumber(row, 'gross_margin', 'margin', 'GrossMargin'))
    .filter((value) => value !== null);
  const min = margins.length ? Math.min(...margins) : 0;
  const max = margins.length ? Math.max(...margins) : 0;
  const span = max - min || 1;

  let bestCell = null;
  grid.forEach((row) => {
    const margin = gridNumber(row, 'gross_margin', 'margin', 'GrossMargin');
    if (margin !== null && (!bestCell || margin > bestCell.margin)) {
      bestCell = { salary: gridNumber(row, 'salary', 'Salary'), commission: gridNumber(row, 'commission_rate', 'commission', 'CommissionRate'), margin };
    }
  });

  const cellFor = (salary, commission) =>
    grid.find(
      (row) =>
        gridNumber(row, 'salary', 'Salary') === salary &&
        gridNumber(row, 'commission_rate', 'commission', 'CommissionRate') === commission,
    );

container.innerHTML = '';

  const corner = document.createElement('div');
  corner.className = 'heatmap__corner';
  corner.setAttribute('role', 'columnheader');
  corner.textContent = 'Salary \\ Commission';
  container.appendChild(corner);

  commissions.forEach((commission) => {
    const node = document.createElement('div');
    node.className = 'heatmap__col';
    node.setAttribute('role', 'columnheader');
    node.textContent = fmtFlexiblePct(commission);
    container.appendChild(node);
  });

  salaries.forEach((salary) => {
    const rowLabel = document.createElement('div');
    rowLabel.className = 'heatmap__row';
    rowLabel.setAttribute('role', 'rowheader');
    rowLabel.textContent = fmtL(salary, 1);
    container.appendChild(rowLabel);

    commissions.forEach((commission) => {
      const row = cellFor(salary, commission);
      const cell = document.createElement('div');
      cell.className = 'heatmap__cell';
      cell.setAttribute('role', 'cell');

      if (!row) {
        cell.innerHTML = '<b>—</b><small>no data</small>';
        container.appendChild(cell);
        return;
      }

      const margin = gridNumber(row, 'gross_margin', 'margin', 'GrossMargin');
      const ratio = margin === null ? 0 : (margin - min) / span;
      const hue = 8 + (165 - 8) * ratio;
      const alpha = 0.12 + 0.4 * ratio;
      const isBest =
        bestCell && bestCell.salary === salary && bestCell.commission === commission;
      const action = String(gridField(row, 'recommended_action', 'RecommendedAction') || '');

      if (isBest) cell.classList.add('heatmap__cell--best');
      cell.style.setProperty('--heat', `hsla(${hue.toFixed(0)}, 62%, 50%, ${alpha.toFixed(2)})`);
      cell.setAttribute(
        'title',
        `Salary ${salary} L · Commission ${fmtFlexiblePct(commission)} · Margin ${fmtL(margin)}` +
          (action ? ` · ${action}` : ''),
      );

      const value = document.createElement('b');
      value.textContent = fmtNum(margin, 1);
      const sub = document.createElement('small');
      sub.textContent = action || '—';
      cell.appendChild(value);
      cell.appendChild(sub);
      container.appendChild(cell);
    });
  });

  container.style.setProperty(
    '--hm-cols',
    `minmax(104px, auto) repeat(${commissions.length}, minmax(88px, 1fr))`,
  );

  const minLabel = el('heat-min');
  const maxLabel = el('heat-max');
  if (minLabel) minLabel.textContent = `Low ${fmtL(min)}`;
  if (maxLabel) maxLabel.textContent = `High ${fmtL(max)}`;
}

function renderMethodology() {
  const assumptions = pick(state.analysis, 'assumptions', {});
  const payback = pick(state.analysis, 'payback', {});

  renderTable(el('assumptions-table'), {
    head: [{ label: 'Assumption' }, { label: 'Value', num: true }],
    rows: [
      ['Monthly salary per employee', esc(fmtL(pick(assumptions, 'salary')))],
      ['Commission rate', esc(fmtFlexiblePct(pick(assumptions, 'commission')))],
      ['COGS rate', esc(fmtFlexiblePct(pick(assumptions, 'cogs_rate')))],
      ['Forecast horizon', esc(`${fmtInt(pick(assumptions, 'forecast_months'))} months`)],
      ['Confidence level', esc(fmtFlexiblePct(pick(assumptions, 'confidence_level')))],
      ['Start month', esc(fmtInt(pick(assumptions, 'start_month')))],
      ['Dataset completeness', esc(fmtFlexiblePct(pick(state.analysis, 'kpis.data_completeness')))],
      ['Anomalies detected', esc(fmtInt(pick(state.analysis, 'kpis.anomalies_detected')))],
      [
        `Staff cost (${asNumber(pick(state.analysis, 'assumptions.forecast_months')) || 6} months)`,
        esc(fmtL(pick(state.analysis, 'kpis.staff_cost_6m'))),
      ],      [
        'Payback period',
        esc(`${fmtNum(pick(payback, 'months_to_payback', pick(state.analysis, 'kpis.payback_months')), 1)} months`),
      ],
      ['Hiring investment', esc(fmtL(pick(payback, 'investment')))],
      [
        'Return on investment',
        esc(fmtPct(pick(payback, 'roi_pct', pick(state.analysis, 'kpis.roi_pct')))),
      ],
      ['Schema version', esc(pick(state.analysis, 'schema_version', '—'))],
      ['Run created', esc(String(pick(state.analysis, 'created_at', '—')).replace('T', ' ').replace('Z', ''))],
    ],
  });

  const dataset = pick(state.analysis, 'dataset', {});
  const business = el('business-assumptions');
  if (business) {
    // Derive every duration and rate from the payload so the prose stays
    // correct for any dataset, horizon or ramp configuration.
    const ramp = asArray(pick(assumptions, 'new_hire_ramp', []));
    const rampFirst = ramp.length ? asNumber(ramp[0]) : null;
    const rampLast = ramp.length ? asNumber(ramp[ramp.length - 1]) : null;
    const rampText =
      rampFirst !== null && rampLast !== null
        ? `A new hire is ramped from ${Math.round(rampFirst * 100)}% to ` +
          `${Math.round(rampLast * 100)}% productivity over ${ramp.length} months.`
        : 'A new hire follows the modelled ramp-up curve.';
    const window = asNumber(pick(assumptions, 'forecast_months')) || 6;
    const rolling = asNumber(
      pick(state.analysis, 'optimal_rule.rolling_avg_trigger.window_months'),
    );
    const rollingText =
      rolling !== null
        ? `The adaptive rule follows a ${rolling}-month rolling average of sales ` +
          'so hiring is triggered by sustained demand rather than a single spike.'
        : 'The adaptive rule follows a rolling average of sales so hiring is ' +
          'triggered by sustained demand rather than a single spike.';
    const confidence = asNumber(pick(assumptions, 'confidence_level'));
    const bandText =
      confidence !== null
        ? `a ${Math.round(confidence * 100)}% prediction band`
        : 'a prediction band';

    const items = [
      {
        title: 'Gross margin',
        text:
          'Gross margin is computed as (1 − COGS rate − commission rate) × Sales − ' +
          `salary × employees. ${rampText}`,
      },
      {
        title: 'Decision rule',
        text: rollingText,
      },
      {
        title: 'Dataset',
        text: `${String(
          pick(dataset, 'name', '—'),
        )} · ${String(pick(dataset, 'filename', '—'))} · ${fmtInt(
          pick(dataset, 'rows'),
        )} rows × ${fmtInt(pick(dataset, 'columns'))} columns · ${fmtInt(
          pick(dataset, 'months'),
        )} months · ${fmtInt(pick(dataset, 'salespersons'))} salespersons · months ${fmtInt(
          pick(dataset, 'first_month'),
        )}–${fmtInt(pick(dataset, 'last_month'))}.`,
      },
      {
        title: 'Uncertainty',
        text:
          `Demand is forecast with rolling-origin cross-validation and ${bandText}; ` +
          'risk is quantified with Monte-Carlo sampling and a salary/commission grid.',
      },
      {
        title: 'Horizon',
        text:
          `Every projection in this report covers ${window} month${window === 1 ? '' : 's'} ` +
          'from the end of the historical record.',
      },
    ];
    business.innerHTML = items
      .map(
        (item) => `
        <div class="callout">
          <svg class="icon" aria-hidden="true"><use href="#i-info"></use></svg>
          <div><p><strong>${esc(item.title)}</strong></p><p>${esc(item.text)}</p></div>
        </div>`,
      )
      .join('');
  }

  const formulaSub = el('margin-formula-sub');
  if (formulaSub) {
    const salary = asNumber(pick(assumptions, 'salary'));
    const commission = asNumber(pick(assumptions, 'commission'));
    const cogs = asNumber(pick(assumptions, 'cogs_rate'));
    if (salary !== null && commission !== null && cogs !== null) {
      const rate = 1 - cogs - commission;
      formulaSub.textContent = `With COGS ${fmtFlexiblePct(
        cogs,
      )} and commission ${fmtFlexiblePct(commission)}, each lakh of sales contributes ${fmtL(
        rate,
        4,
      )} of gross margin before paying ${fmtL(salary)} per employee per month.`;
      show(formulaSub, true);
    } else {
      show(formulaSub, false);
    }
  }

  const recommendation = el('recommendation-text');
  if (recommendation) {
    const text = String(pick(state.analysis, 'recommendation_text', '') || '').trim();
    recommendation.textContent = text || 'No narrative was returned for this run.';
  }

  const seasonal = asArray(pick(state.analysis, 'series.seasonal_index', []));  renderTable(el('seasonality-table'), {
    head: [
      { label: 'Month', num: true },
      { label: 'Name' },
      { label: 'Index', num: true },
      { label: 'Strength', num: true },
    ],
    rows: seasonal.map((row) => [
      esc(fmtInt(pick(row, 'month'))),
      esc(pick(row, 'name', '—')),
      esc(fmtNum(pick(row, 'index'), 3)),
      esc(fmtNum(pick(row, 'strength'), 3)),
    ]),
  });

  const anomalies = asArray(pick(state.analysis, 'series.anomalies', []));
  renderTable(el('anomalies-table'), {
    head: [
      { label: 'Month', num: true },
      { label: 'Name' },
      { label: 'Sales', num: true },
      { label: 'Expected', num: true },
      { label: 'Score', num: true },
      { label: 'Direction' },
      { label: 'Severity' },
    ],
    rows: anomalies.map((row) => {
      const direction = String(pick(row, 'direction', '—'));
      const severity = String(pick(row, 'severity', '—'));
      const severityTone =
        severity.toLowerCase().includes('high')
          ? 'critical'
          : severity.toLowerCase().includes('med')
            ? 'warning'
            : 'neutral';
      const directionTone = direction.toLowerCase().includes('spike')
        ? 'positive'
        : direction.toLowerCase().includes('drop')
          ? 'critical'
          : 'neutral';
      return [
        esc(fmtInt(pick(row, 'month'))),
        esc(pick(row, 'name', '—')),
        esc(fmtL(pick(row, 'sales'))),
        esc(fmtL(pick(row, 'expected'))),
        esc(fmtNum(pick(row, 'score'), 3)),
        `<span class="badge badge--${directionTone}">${esc(direction)}</span>`,
        `<span class="badge badge--${severityTone}">${esc(severity)}</span>`,
      ];
    }),
  });

  renderArtifacts();
}

function renderArtifacts() {
  const container = el('artifact-links');
  if (!container) return;
  const runId = pick(state.analysis, 'run_id', null);
  const artifacts = pick(state.analysis, 'artifacts', {});
  const chartsMap = pick(state.analysis, 'charts', {}) || {};

  const links = [];
  if (runId) {
    const pdf = firstString(pick(artifacts, 'pdf_url')) || api.reportUrl(runId);
    const html = firstString(pick(artifacts, 'html_url')) || api.htmlReportUrl(runId);
    const csv = firstString(pick(artifacts, 'csv_url')) || api.csvUrl(runId);
    links.push({ href: pdf, label: 'PDF report', icon: 'i-file' });
    links.push({ href: html, label: 'HTML report', icon: 'i-file' });
    links.push({ href: csv, label: 'Results CSV', icon: 'i-table' });

    Object.keys(chartsMap)
      .slice(0, 8)
      .forEach((name) => {
        links.push({ href: api.chartUrl(runId, name), label: name, icon: 'i-image' });
      });
  }

  if (!links.length) {
    container.innerHTML =
      '<div class="empty-state"><strong>No artifacts</strong><span>Run an analysis to generate documents.</span></div>';
    return;
  }

  container.innerHTML = links
    .map(
      (link) =>
        `<a class="btn btn--sm btn--ghost" href="${esc(link.href)}" target="_blank" rel="noopener">` +
        `<svg class="icon" aria-hidden="true"><use href="#${esc(link.icon)}"></use></svg>` +
        `${esc(link.label)}</a>`,
    )
    .join('');
}

function renderAll() {
  if (!state.analysis) return;
  renderKpis();
  renderVerdict();
  renderImpact();
  renderUnitsHint();
  renderOverviewCharts();
  renderForecast();
  renderScenarios();
  renderWorkforce();
  renderSensitivity();
  renderMethodology();
  updateRunBadges();
}

/* -------------------------------------------------------------------------- */
/* Simulator rendering                                                         */
/* -------------------------------------------------------------------------- */

function renderSimulation(payload) {
  const scenarios = pick(payload, 'scenarios', {});
  const bestKey = pick(payload, 'best', null);
  const records = scenarioRecords(scenarios, bestKey);

  const summary = el('sim-summary');
  if (summary) {
    const best = records.find((record) => record.key === bestKey);
    const note = String(pick(payload, 'note', '') || '');
    const items = [
      {
        label: 'Best scenario',
        value: best ? `${best.key} · ${best.name}` : String(bestKey || '—'),
        sub: best ? `Gross margin ${fmtL(best.totalMargin)}` : 'No scenario selected',
      },
      {
        label: 'Break-even sales / employee',
        value: fmtL(pick(payload, 'break_even_sales_per_employee')),
        sub: 'Monthly sales needed to cover one hire',
      },
      {
        label: 'Payback period',
        value: (() => {
        const months = asNumber(pick(payload, 'payback_months'));
        return months === null ? 'Not reached' : `${fmtNum(months, 1)} months`;
      })(),
        sub: 'Time to recover the hiring cost',
      },
    ];

    summary.innerHTML = items
      .map(
        (item) => `
        <article class="stat">
          <span class="stat__label">${esc(item.label)}</span>
          <span class="stat__value stat__value--text">${esc(item.value)}</span>
          <span class="stat__sub">${esc(item.sub)}</span>
        </article>`,
      )
      .join('');

    // Replace any previous note instead of appending a new one, so
    // repeated simulator runs never stack callouts.
    const scope = summary.parentElement || document;
    scope.querySelectorAll('.sim-note').forEach((node) => node.remove());

    if (note) {
      summary.insertAdjacentHTML(
        'afterend',
        `<div class="callout sim-note"><svg class="icon" aria-hidden="true"><use href="#i-info"></use></svg><div>${esc(
          note,
        )}</div></div>`,
      );
    }
  }

  renderScenarioCards(
    el('sim-cards'),
    records,
    asNumber(pick(state.simulation, 'assumptions.horizon')),
  );

  const deltaRows = records.map((record) => [
    `<span class="key-badge">${esc(record.key)}</span>`,
    esc(record.name),
    esc(fmtL(record.totalMargin)),
    esc(fmtL(record.avgMargin)),
    `<span class="delta ${deltaClass(record.delta)}">${esc(fmtSigned(record.delta))} L</span>`,
    esc(fmtFlexiblePct(record.probPositive)),
  ]);

  renderTable(el('sim-delta-table'), {
    head: [
      { label: 'Key' },
      { label: 'Scenario' },
      { label: 'Gross margin', num: true },
      { label: 'Avg monthly margin', num: true },
      { label: 'Delta vs best', num: true },
      { label: 'P(margin > 0)', num: true },
    ],
    rows: deltaRows,
  });

  charts.renderChart('chart-sim-cumulative', charts.buildScenarioCumulativeChart, {
    scenarios: records,
  });
}

/* -------------------------------------------------------------------------- */
/* Chrome: run badges, report link, global events                             */
/* -------------------------------------------------------------------------- */

function updateRunBadges() {
  const runId = pick(state.analysis, 'run_id', null);
  if (!runId) return;
  [
    'overview-run-id',
    'forecast-run-id',
    'scenarios-run-id',
    'workforce-run-id',
    'sensitivity-run-id',
    'methodology-run-id',
  ].forEach((id) => {
    const node = el(id);
    if (!node) return;
    node.textContent = `run ${String(runId).slice(0, 8)}`;
    show(node, true);
  });
}

function syncReportLink() {
  const link = el('report-link');
  const runId = pick(state.analysis, 'run_id', null);
  if (!link) return;
  if (runId) {
    link.setAttribute('href', api.reportUrl(runId));
    link.setAttribute('aria-disabled', 'false');
    link.classList.remove('is-disabled');
  } else {
    link.setAttribute('href', '#');
    link.setAttribute('aria-disabled', 'true');
    link.classList.add('is-disabled');
  }
}

/**
 * Download the PDF report.
 *
 * The server renders the report on demand, so the first click has to ask
 * for it to be built before the browser can fetch the file.
 */
async function downloadReport(event) {
  if (event) event.preventDefault();
  const link = el('report-link');
  const runId = pick(state.analysis, 'run_id', null);
  if (!link || !runId) {
    toast('No report', 'Run an analysis first.', 'error');
    return;
  }

  const previous = link.getAttribute('href');
  link.setAttribute('aria-disabled', 'true');
  link.classList.add('is-disabled');
  toast('Preparing report', 'Rendering the PDF and charts on the server…', 'info');

  try {
    await api.buildReport(runId);
    window.location.href = api.reportUrl(runId);
    toast('Report ready', 'The PDF report has been downloaded.', 'success');
  } catch (error) {
    const message = error && error.message ? error.message : 'The report failed.';
    toast('Report failed', message, 'error');
  } finally {
    link.setAttribute('href', previous || api.reportUrl(runId));
    link.setAttribute('aria-disabled', 'false');
    link.classList.remove('is-disabled');
  }
}

function initGlobalEvents() {
  const retry = el('retry-btn');
  if (retry) {
    retry.addEventListener('click', () => {
      toast('Retrying', 'Reconnecting to the analysis API…', 'info', 2200);
      loadAnalysis({ silent: true });
    });
  }

  const reportLink = el('report-link');
  if (reportLink) {
    reportLink.addEventListener('click', (event) => {
      if (reportLink.getAttribute('aria-disabled') === 'true') {
        toast('No report yet', 'Run an analysis to generate the PDF report.', 'error');
        return;
      }
      void downloadReport(event);
    });
  }

  const paramsForm = el('params-form');
  if (paramsForm) {
    paramsForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const submit = el('params-submit');
      const note = el('params-note');
      if (submit) submit.disabled = true;
      if (note) note.textContent = 'Running the analysis pipeline…';
      hideError();
      try {
        const analysis = await api.createAnalysis(readParams());
        state.analysis = analysis;
        await loadSensitivity();
        populateDatasetSelect();
        syncParamsFromAnalysis();
        renderAll();
        syncReportLink();
        toast(
          'Analysis updated',
          'The pipeline re-ran with the new assumptions.',
          'success',
        );
        if (simulator) {
          suppressSimulationErrors = true;
          await simulator.run({ auto: true });
          suppressSimulationErrors = false;
        }
      } catch (error) {
        const message = error && error.message ? error.message : 'The re-run failed.';
        showError('Re-run failed', message);
        toast('Re-run failed', message, 'error');
      } finally {
        if (submit) submit.disabled = false;
        if (note) {
          note.textContent =
            'Re-running executes the full forecasting & financial pipeline on the server.';
        }
      }
    });
  }

  const downloadButton = el('download-scenarios');
  if (downloadButton) {
    downloadButton.addEventListener('click', downloadScenarioCsv);
  }

  window.addEventListener('unhandledrejection', (event) => {
    const reason = event && event.reason;
    const message = reason && reason.message ? reason.message : 'Unexpected error.';
    if (suppressSimulationErrors) return;
    toast('Unexpected error', message, 'error');
  });

  window.addEventListener('error', (event) => {
    const message = event && event.message ? event.message : 'Unexpected error.';
    toast('Unexpected error', message, 'error');
  });
}

/* -------------------------------------------------------------------------- */
/* Bootstrap                                                                   */
/* -------------------------------------------------------------------------- */

function bootstrap() {
  try {
    initTheme();
    initNav();
    initGlobalEvents();
    startHealthPolling();

    simulator = createSimulator({
      getRunId: () => pick(state.analysis, 'run_id', null),
      onResults: (payload) => {
        state.simulation = payload;
        renderSimulation(payload);
        if (!suppressSimulationErrors) {
          toast('Simulation updated', 'Scenario results refreshed from the server.', 'success');
        }
      },
      onStatus: (status) => {
        if (status && status.state === 'error' && !suppressSimulationErrors) {
          toast('Simulation failed', status.message || 'The API is unreachable.', 'error');
        }
      },
      onError: (error) => {
        if (suppressSimulationErrors) return;
        toast(
          'Simulation failed',
          error && error.message ? error.message : 'The API is unreachable.',
          'error',
        );
      },
    });
    simulator.init();
    simulator.setEnabled(false);

    loadAnalysis({ silent: true });
  } catch (error) {
    showError(
      'Dashboard failed to start',
      error && error.message ? error.message : 'An unexpected error occurred.',
    );
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrap);
} else {
  bootstrap();
}
