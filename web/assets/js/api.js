/**
 * Store Staffing Analyzer — API client.
 *
 * Thin, defensive wrapper around the FastAPI service mounted at `/api/v1`.
 * Every network call returns a promise that either resolves with parsed JSON
 * or rejects with an `ApiError`. Callers are expected to attach a `.catch()`.
 */

/** Base path of the REST API (same origin as the static bundle). */
export const API_BASE = '/api/v1';

/** Error raised for every failed API interaction. */
export class ApiError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = options.status ?? 0;
    this.code = options.code ?? null;
    this.url = options.url ?? null;
    this.isNetworkError = options.status === 0;
  }
}

/**
 * Safely read a nested value out of an arbitrary object.
 *
 * @param {unknown} obj      source object (may be null/undefined)
 * @param {string}  path     dot separated path, e.g. `forecast.metrics` or `scenarios.A.monthly`
 * @param {*}       fallback value returned when the path cannot be resolved
 * @returns {*} resolved value or `fallback`
 */
export function pick(obj, path, fallback = null) {
  if (obj === null || obj === undefined) return fallback;
  if (typeof path !== 'string' || path.length === 0) return fallback;

  const parts = path.split('.');
  let current = obj;

  for (const rawPart of parts) {
    if (current === null || current === undefined) return fallback;
    const part = rawPart.trim();
    if (part.length === 0) return fallback;

    if (Array.isArray(current)) {
      const index = Number(part);
      if (!Number.isInteger(index) || index < 0 || index >= current.length) return fallback;
      current = current[index];
      continue;
    }

    if (typeof current === 'object') {
      if (!Object.prototype.hasOwnProperty.call(current, part)) return fallback;
      current = current[part];
      continue;
    }

    return fallback;
  }

  return current === undefined || current === null ? fallback : current;
}

/**
 * Try several candidate paths in order and return the first resolvable value.
 * Used where the API has more than one documented spelling for a concept.
 *
 * @param {unknown} obj
 * @param {string[]} paths
 * @param {*} fallback
 */
export function pickAny(obj, paths, fallback = null) {
  for (const path of paths) {
    const value = pick(obj, path, undefined);
    if (value !== undefined) return value;
  }
  return fallback;
}

/** Coerce to a finite number, or `fallback` when impossible. */
export function asNumber(value, fallback = null) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : fallback;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }
  return fallback;
}

/** Coerce to an array, or an empty array. */
export function asArray(value) {
  if (Array.isArray(value)) return value;
  if (value === null || value === undefined) return [];
  return [value];
}

/**
 * Perform a request against the API and normalise failures.
 *
 * @param {string} path      path relative to `/api/v1`
 * @param {object} options   `{ method, body, headers, signal }`
 */
async function request(path, options = {}) {
  const { method = 'GET', body = null, headers = {}, signal = null } = options;
  const url = path.startsWith('http') ? path : `${API_BASE}${path}`;

  let response;
  try {
    response = await fetch(url, {
      method,
      body,
      headers,
      signal,
      credentials: 'same-origin',
    });
  } catch (error) {
    throw new ApiError(
      `Network request to ${url} failed — the analysis service is unreachable.`,
      { status: 0, url },
    );
  }

  let payload = null;
  let rawText = '';
  try {
    rawText = await response.text();
  } catch (error) {
    rawText = '';
  }

  if (rawText) {
    try {
      payload = JSON.parse(rawText);
    } catch (error) {
      payload = rawText;
    }
  }

  if (!response.ok) {
    let detail = null;
    if (payload && typeof payload === 'object' && payload.detail) {
      detail = String(payload.detail);
    } else if (typeof payload === 'string' && payload.trim()) {
      const text = payload.trim();
      // HTML error pages (e.g. a static server answering the API path) are not
      // useful to show in the UI — fall back to a generic HTTP message.
      detail = text.startsWith('<') ? null : text.slice(0, 300);
    }
    const code =
      payload && typeof payload === 'object' && payload.code ? payload.code : null;
    throw new ApiError(
      detail || `Request failed with HTTP ${response.status}.`,
      { status: response.status, code, url },
    );
  }

  return payload;
}

/** Public, endpoint-scoped API surface. */
export const api = {
  /** `GET /api/v1/health` */
  health(options = {}) {
    return request('/health', options);
  },

  /** `GET /api/v1/datasets` */
  datasets(options = {}) {
    return request('/datasets', options);
  },

  /** `GET /api/v1/analyses` — list of cached runs. */
  runs(options = {}) {
    return request('/analyses', options);
  },

  /** `POST /api/v1/analyses` — multipart/form-data. */
  createAnalysis(params = {}, options = {}) {
    const form = new FormData();
    // Only send `dataset` when one was actually chosen: the server then
    // resolves its own default instead of the client hard-coding a name.
    if (params.dataset !== undefined && params.dataset !== null && params.dataset !== '') {
      form.append('dataset', String(params.dataset));
    }
    if (params.salary !== undefined && params.salary !== null) {
      form.append('salary', String(params.salary));
    }
    if (params.commission !== undefined && params.commission !== null) {
      form.append('commission', String(params.commission));
    }
    if (params.forecast_months !== undefined && params.forecast_months !== null) {
      form.append('forecast_months', String(params.forecast_months));
    }
    if (params.start_month !== undefined && params.start_month !== null) {
      form.append('start_month', String(params.start_month));
    }
    if (params.file) {
      form.append('file', params.file);
    }
    return request('/analyses', { method: 'POST', body: form, ...options });
  },

  /** `GET /api/v1/analyses/{run_id}` */
  analysis(runId, options = {}) {
    return request(`/analyses/${encodeURIComponent(runId)}`, options);
  },

  /** `POST /api/v1/analyses/{run_id}/report` — render the report artefacts. */
  buildReport(runId, options = {}) {
    return request(`/analyses/${encodeURIComponent(runId)}/report`, {
      method: 'POST',
      ...options,
    });
  },

  /** `GET /api/v1/analyses/{run_id}/sensitivity` */
  sensitivity(runId, options = {}) {
    return request(`/analyses/${encodeURIComponent(runId)}/sensitivity`, options);
  },

  /** `POST /api/v1/simulate` */
  simulate(payload = {}, options = {}) {
    return request('/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      ...options,
    });
  },

  /** Absolute URL of the PDF report for a run. */
  reportUrl(runId) {
    return `${API_BASE}/analyses/${encodeURIComponent(runId)}/report.pdf`;
  },

  /** Absolute URL of the HTML report for a run. */
  htmlReportUrl(runId) {
    return `${API_BASE}/analyses/${encodeURIComponent(runId)}/report.html`;
  },

  /** Absolute URL of the results CSV for a run. */
  csvUrl(runId) {
    return `${API_BASE}/analyses/${encodeURIComponent(runId)}/results.csv`;
  },

  /** Absolute URL of a server-rendered PNG chart. */
  chartUrl(runId, chartName) {
    return `${API_BASE}/analyses/${encodeURIComponent(runId)}/charts/${encodeURIComponent(
      chartName,
    )}.png`;
  },
};

export default api;
