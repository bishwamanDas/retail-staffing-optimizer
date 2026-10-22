/**
 * Store Staffing Analyzer — what-if simulator.
 *
 * Owns the simulator controls, keeps the readouts in sync with the sliders and
 * drives `POST /api/v1/simulate`. Releasing a slider schedules a debounced
 * auto-run; the manual button forces an immediate run.
 */

import { api } from './api.js';

const AUTO_RUN_DELAY = 450;

const CONTROL_IDS = {
  salary: 'sim-salary',
  commission: 'sim-commission',
  horizon: 'sim-horizon',
  salesShock: 'sim-shock',
};

const OUTPUT_IDS = {
  salary: 'sim-salary-out',
  commission: 'sim-commission-out',
  horizon: 'sim-horizon-out',
  salesShock: 'sim-shock-out',
};

/** Maps a control key onto the matching key of the simulator state. */
const STATE_KEYS = {
  salary: 'salary',
  commission: 'commission',
  horizon: 'horizon',
  salesShock: 'sales_shock_pct',
};

const BADGE_ID = 'sim-badge';
const STATUS_ID = 'sim-status';
const FORM_ID = 'sim-form';

/**
 * Read the current simulator state from the DOM.
 *
 * Every value is clamped to the *input's own* min/max attributes, so the
 * sandbox never silently rewrites a value the UI itself would reject.
 */
export function readSimulatorState() {
  const salary = document.getElementById(CONTROL_IDS.salary);
  const commission = document.getElementById(CONTROL_IDS.commission);
  const horizon = document.getElementById(CONTROL_IDS.horizon);
  const shock = document.getElementById(CONTROL_IDS.salesShock);
  const checked = document.querySelector('input[name="decision"]:checked');

  const bound = (input, fallback) => {
    if (!input) return { min: 0, max: Number.POSITIVE_INFINITY };
    const min = Number(input.min);
    const max = Number(input.max);
    return {
      min: Number.isFinite(min) ? min : 0,
      max: Number.isFinite(max) ? max : Number.POSITIVE_INFINITY,
      fallback,
    };
  };

  const salaryBounds = bound(salary, 3);
  const commissionBounds = bound(commission, 0.05);
  const horizonBounds = bound(horizon, 12);
  const shockBounds = bound(shock, 0);

  return {
    salary: clampNumber(
      salary ? salary.value : salaryBounds.fallback,
      salaryBounds.min,
      salaryBounds.max,
    ),
    commission: clampNumber(
      commission ? commission.value : commissionBounds.fallback,
      commissionBounds.min,
      commissionBounds.max,
    ),
    horizon: Math.round(
      clampNumber(
        horizon ? horizon.value : horizonBounds.fallback,
        horizonBounds.min,
        horizonBounds.max,
      ),
    ),
    sales_shock_pct: clampNumber(
      shock ? shock.value : shockBounds.fallback,
      shockBounds.min,
      shockBounds.max,
    ),
    decision: checked && checked.value ? checked.value : 'auto',
  };
}

/** Parse + clamp a numeric input value. */
function clampNumber(value, min, max) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return min;
  return Math.min(max, Math.max(min, parsed));
}

/** Paint the filled portion of a range input using a CSS custom property. */
function paintSlider(input) {
  if (!input) return;
  const min = Number(input.min);
  const max = Number(input.max);
  const value = Number(input.value);
  if (!Number.isFinite(min) || !Number.isFinite(max) || max === min) return;
  const pct = ((value - min) / (max - min)) * 100;
  input.style.setProperty('--fill', `${Math.max(0, Math.min(100, pct))}%`);
}

/** Human readable label for a slider readout. */
function formatReadout(name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  switch (name) {
    case 'salary':
      return `${value.toFixed(2)} L`;
    case 'commission':
      return `${(value * 100).toFixed(1)}%`;
    case 'horizon':
      return `${value} months`;
    case 'salesShock':
      return `${value > 0 ? '+' : ''}${value.toFixed(0)}%`;
    default:
      return String(value);
  }
}

/**
 * Create a simulator controller.
 *
 * @param {object}   options
 * @param {Function} options.getRunId   returns the active analysis run id
 * @param {Function} options.onResults  called with the simulation payload
 * @param {Function} options.onStatus   called with { state, message }
 * @param {Function} options.onError    called with an Error
 */
export function createSimulator(options = {}) {
  const {
    getRunId = () => null,
    onResults = () => {},
    onStatus = () => {},
    onError = () => {},
  } = options;

  let timer = null;
  let running = false;
  let queued = false;
  let inFlight = 0;
  let enabled = true;

  const form = document.getElementById(FORM_ID);
  const badge = document.getElementById(BADGE_ID);
  const status = document.getElementById(STATUS_ID);

  function setBadge(text, tone) {
    if (!badge) return;
    badge.textContent = text;
    badge.className = `badge badge--${tone || 'neutral'}`;
  }

  function setStatus(message) {
    if (status) status.textContent = message;
  }

  function syncOutputs() {
    const state = readSimulatorState();
    Object.keys(CONTROL_IDS).forEach((name) => {
      const input = document.getElementById(CONTROL_IDS[name]);
      const output = document.getElementById(OUTPUT_IDS[name]);
      paintSlider(input);
      const value = state[STATE_KEYS[name]];
      if (output) output.textContent = formatReadout(name, value);
    });
    syncRangeHints();
    return state;
  }

  /** Derive the range hints from the sliders themselves. */
  function syncRangeHints() {
    const salary = document.getElementById(CONTROL_IDS.salary);
    const horizon = document.getElementById(CONTROL_IDS.horizon);
    const salaryHint = document.getElementById('sim-salary-hint');
    const horizonHint = document.getElementById('sim-horizon-hint');

    if (salary && salaryHint) {
      const min = Number(salary.min);
      const max = Number(salary.max);
      if (Number.isFinite(min) && Number.isFinite(max)) {
        salaryHint.textContent = `Range ${min} – ${max} L per employee per month`;
      }
    }
    if (horizon && horizonHint) {
      const min = Number(horizon.min);
      const max = Number(horizon.max);
      if (Number.isFinite(min) && Number.isFinite(max)) {
        horizonHint.textContent = `${min} – ${max} months of simulation`;
      }
    }
  }

  function scheduleAutoRun() {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      run({ auto: true });
    }, AUTO_RUN_DELAY);
  }

  /**
   * Execute a simulation against the API.
   *
   * @param {object} meta `{ auto: boolean }`
   */
  async function run(meta = {}) {
    const runId = getRunId();
    if (!runId) {
      setBadge('no run', 'warning');
      setStatus('Load an analysis first — the simulator needs a run to perturb.');
      onStatus({ state: 'idle', message: 'No analysis run available.' });
      return null;
    }

    // A request is already in flight: remember that newer values are
    // waiting and re-run once the current one settles, rather than
    // dropping the change on the floor.
    if (running) {
      queued = true;
      return null;
    }

    const state = syncOutputs();
    running = true;
    inFlight += 1;
    setBadge('running', 'accent');
    setStatus(meta.auto ? 'Recomputing scenarios…' : 'Running simulation…');
    onStatus({ state: 'running', message: 'Simulation in progress.' });

    try {
      const payload = await api.simulate({
        run_id: runId,
        salary: state.salary,
        commission: state.commission,
        horizon: state.horizon,
        decision: state.decision,
        sales_shock_pct: state.sales_shock_pct,
      });

      running = false;
      setBadge('ready', 'positive');
      // Report the horizon the server actually used, not the one asked
      // for, so the label never disagrees with the scenario cards.
      const echoed = pickAssumptions(payload);
      const usedHorizon = (echoed && echoed.horizon) || state.horizon;
      setStatus(
        `Simulation complete · ${usedHorizon}-month horizon · ${state.decision} strategy`,
      );
      onStatus({
        state: 'ready',
        message: 'Simulation updated.',
        assumptions: echoed,
      });
      onResults(payload);
      return payload;
    } catch (error) {
      running = false;
      setBadge('failed', 'critical');
      setStatus(
        error && error.message
          ? `Simulation failed: ${error.message}`
          : 'Simulation failed — the API is unreachable.',
      );
      onStatus({ state: 'error', message: 'Simulation failed.' });
      onError(error);
      return null;
    } finally {
      inFlight = Math.max(0, inFlight - 1);
      void inFlight;
      // A change arrived while this request was running: run again with
      // the newest values instead of leaving the panel on stale numbers.
      if (queued && !running) {
        queued = false;
        void run({ auto: true });
      }
    }
  }

  /** Wire up the DOM. Safe to call once, after the DOM is parsed. */
  function init() {
    syncOutputs();

    Object.keys(CONTROL_IDS).forEach((name) => {
      const input = document.getElementById(CONTROL_IDS[name]);
      if (!input) return;
      input.addEventListener('input', () => {
        syncOutputs();
      });
      input.addEventListener('change', () => {
        syncOutputs();
        if (enabled) scheduleAutoRun();
      });
    });

    const decision = document.getElementById('sim-decision');
    if (decision) {
      decision.addEventListener('change', () => {
        if (enabled) scheduleAutoRun();
      });
    }

    if (form) {
      form.addEventListener('submit', (event) => {
        event.preventDefault();
        if (timer) {
          clearTimeout(timer);
          timer = null;
        }
        run({ auto: false });
      });
    }

    setBadge('idle', 'neutral');
    setStatus('Adjust a slider to re-run automatically.');
  }

  return {
    init,
    run,
    getState: readSimulatorState,
    syncOutputs,
    /**
     * Seed the sandbox from the assumptions of the loaded analysis, so
     * the simulator perturbs the *same* scenario the dashboard is showing
     * rather than an unrelated set of defaults.
     *
     * @param {object} assumptions the analysis `assumptions` block
     */
    seed(assumptions) {
      if (!assumptions || typeof assumptions !== 'object') return;
      const set = (id, value, fallback) => {
        const node = document.getElementById(id);
        const parsed = Number(value);
        if (!node || !Number.isFinite(parsed)) return;
        const min = Number(node.min);
        const max = Number(node.max);
        let next = parsed;
        if (Number.isFinite(min)) next = Math.max(min, next);
        if (Number.isFinite(max)) next = Math.min(max, next);
        if (fallback !== undefined && !Number.isFinite(parsed)) next = fallback;
        node.value = String(next);
      };
      set(CONTROL_IDS.salary, assumptions.salary);
      set(CONTROL_IDS.commission, assumptions.commission);
      set(CONTROL_IDS.horizon, assumptions.forecast_months);
      set(CONTROL_IDS.salesShock, 0);
      syncOutputs();
    },
    setEnabled(value) {
      enabled = Boolean(value);
      Object.keys(CONTROL_IDS).forEach((name) => {
        const input = document.getElementById(CONTROL_IDS[name]);
        if (input) input.disabled = !enabled;
      });
      // The decision radios are part of the sandbox and must be disabled
      // together with the sliders.
      document.querySelectorAll('#sim-decision input').forEach((input) => {
        input.disabled = !enabled;
      });
      const runButton = document.getElementById('sim-run');
      if (runButton) runButton.disabled = !enabled;
    },
    isRunning: () => running,
  };
}

/** Extract the echoed assumptions from a simulation payload, defensively. */
function pickAssumptions(payload) {
  if (!payload || typeof payload !== 'object') return null;
  const assumptions = payload.assumptions;
  if (!assumptions || typeof assumptions !== 'object') return null;
  return {
    salary: assumptions.salary ?? null,
    commission: assumptions.commission ?? null,
    horizon: assumptions.horizon ?? null,
    decision: assumptions.decision ?? null,
    sales_shock_pct: assumptions.sales_shock_pct ?? null,
  };
}

export default createSimulator;
