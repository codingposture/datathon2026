import { useCallback, useEffect, useRef, useState } from 'react';
import { createDemoService } from '../services/demo.js';
import { createApiService } from '../services/api.js';
import { runAnalysis } from '../services/run-analysis.js';

const initial = { status: 'ready', stage: -1, violations: [], source: 'local', error: '' };

export function useAnalysis(services) {
  const servicesRef = useRef(null);
  if (!servicesRef.current) servicesRef.current = services || { demo: createDemoService(), local: createApiService({ basePath: import.meta.env.VITE_API_BASE || '/api/v1' }) };
  const [analysis, setAnalysis] = useState(initial);
  const active = useRef({ version: 0, controller: null });
  const cancel = useCallback(() => {
    active.current.version++;
    active.current.controller?.abort();
    servicesRef.current.demo.reset?.();
  }, []);
  useEffect(() => () => cancel(), [cancel]);
  const reset = useCallback(() => { cancel(); setAnalysis(initial); }, [cancel]);
  const start = useCallback(async ({ file, mode, scenario, apiKey }) => {
    cancel();
    const version = active.current.version;
    const controller = new AbortController();
    active.current.controller = controller;
    setAnalysis({ ...initial, source: mode, status: mode === 'local' ? 'uploading' : 'queued', stage: 0 });
    try {
      await runAnalysis({
        service: servicesRef.current[mode], file, scenario, apiKey, signal: controller.signal,
        pollIntervalMs: mode === 'demo' ? 350 : 2000,
        onUpdate(result) {
          if (active.current.version === version) setAnalysis({ ...result, source: mode });
        }
      });
    } catch (error) {
      if (active.current.version === version && error.name !== 'AbortError') setAnalysis(previous => ({ ...initial, jobId: previous.jobId, source: mode, status: 'failed', error: error.message }));
    } finally {
      if (active.current.version === version) {
        active.current.controller = null;
        servicesRef.current.demo.reset?.();
      }
    }
  }, [cancel]);
  return { analysis, start, reset, busy: ['uploading', 'queued', 'processing'].includes(analysis.status) };
}
