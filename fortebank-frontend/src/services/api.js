import { normalizeJob, normalizeJobId, validateFile } from './model.js';

export function createApiService({ fetchImpl = (...args) => fetch(...args), requestTimeoutMs = 30000, basePath = '/api/v1' } = {}) {
  if (!/^\/(?!\/)[A-Za-z0-9_-]+(?:\/[A-Za-z0-9_-]+)*$/.test(basePath)) throw new Error('The API base path must be a same-origin path without a trailing slash.');
  async function request(path, options = {}) {
    const apiKey = options.apiKey?.trim();
    if (!apiKey) throw new Error('Enter your API key before uploading a video.');
    const { apiKey: _key, ...requestOptions } = options;
    const controller = new AbortController();
    let timedOut = false;
    const abort = () => controller.abort();
    if (options.signal?.aborted) controller.abort();
    else options.signal?.addEventListener('abort', abort, { once: true });
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, requestTimeoutMs);
    try {
      const response = await fetchImpl(path, { ...requestOptions, signal: controller.signal, credentials: 'same-origin', redirect: 'error', headers: { Accept: 'application/json', 'X-API-Key': apiKey } });
      let data;
      try { data = await response.json(); } catch { throw new Error('The backend returned an unreadable response. Check the Django API route and proxy configuration.'); }
      if (!response.ok) {
        const messages = {
          401: 'Backend access denied. Check the API key you entered.',
          403: 'Backend access denied. Check the API key you entered.',
          413: 'The video exceeds the backend upload size limit.',
          415: 'The backend did not accept this video format.',
          429: 'The backend is busy. Please try again shortly.'
        };
        const detail = typeof data?.detail === 'string' ? data.detail : Array.isArray(data?.file) ? data.file.join(' ') : '';
        throw new Error(messages[response.status] || detail || `The local backend returned HTTP ${response.status}. Please try again.`);
      }
      return data;
    } catch (error) {
      if (timedOut) throw new Error('The backend request timed out. A submitted job may still be running; check the local service before retrying.');
      if (controller.signal.aborted) throw new DOMException('Analysis cancelled', 'AbortError');
      if (error instanceof TypeError) throw new Error('Cannot reach the local backend. Check the Django service and frontend proxy.');
      throw error;
    } finally {
      clearTimeout(timer);
      options.signal?.removeEventListener('abort', abort);
    }
  }
  return {
    async startAnalysis(file, { signal, apiKey } = {}) {
      const error = validateFile(file);
      if (error) throw new Error(error);
      if (signal?.aborted) throw new DOMException('Analysis cancelled', 'AbortError');
      const body = new FormData();
      body.append('file', file); // Browser sets multipart boundaries. Do not set Content-Type.
      const data = await request(`${basePath}/analyze`, { method: 'POST', body, signal, apiKey });
      return { jobId: normalizeJobId(data), status: 'queued', stage: 0, violations: [] };
    },
    async getAnalysis(jobId, { signal, apiKey } = {}) {
      const data = await request(`${basePath}/jobs/${encodeURIComponent(jobId)}`, { method: 'GET', signal, apiKey });
      return normalizeJob(data, jobId);
    }
  };
}
