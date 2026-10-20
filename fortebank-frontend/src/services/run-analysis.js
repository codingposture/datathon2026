export function abortableDelay(ms, signal) {
  return new Promise((resolve, reject) => {
    const cancel = () => { clearTimeout(timer); reject(new DOMException('Analysis cancelled', 'AbortError')); };
    const timer = setTimeout(() => { signal?.removeEventListener('abort', cancel); resolve(); }, ms);
    if (signal?.aborted) cancel();
    else signal?.addEventListener('abort', cancel, { once: true });
  });
}

// One whole-job deadline, including submission, matches the case's five-minute limit.
export async function runAnalysis({ service, file, scenario, apiKey, signal, onUpdate, pollIntervalMs = 2000, timeoutMs = 300000 }) {
  const controller = new AbortController();
  let timedOut = false;
  const cancel = () => controller.abort();
  if (signal?.aborted) controller.abort();
  else signal?.addEventListener('abort', cancel, { once: true });
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const job = await service.startAnalysis(file, { scenario, apiKey, signal: controller.signal });
    if (controller.signal.aborted) throw new DOMException('Analysis cancelled', 'AbortError');
    onUpdate({ jobId: job.jobId, status: 'queued', stage: 0, violations: [] });
    while (true) {
      if (controller.signal.aborted) throw new DOMException('Analysis cancelled', 'AbortError');
      const result = await service.getAnalysis(job.jobId, { apiKey, signal: controller.signal });
      if (controller.signal.aborted) throw new DOMException('Analysis cancelled', 'AbortError');
      onUpdate(result);
      if (result.status === 'completed' || result.status === 'failed') return result;
      await abortableDelay(pollIntervalMs, controller.signal);
    }
  } catch (error) {
    if (timedOut) throw new Error('Analysis exceeded the five-minute deadline. The local backend job may still be running.');
    throw error;
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener('abort', cancel);
  }
}
