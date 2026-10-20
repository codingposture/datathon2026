import { describe, expect, it, vi } from 'vitest';
import { formatTimestamp, normalizeJob, safeReferenceUrl, validateFile } from '../src/services/model.js';
import { createDemoService } from '../src/services/demo.js';
import { createApiService } from '../src/services/api.js';
import { runAnalysis } from '../src/services/run-analysis.js';

export const done = () => ({ job_id: 'job-1', status: 'done', verdict: 'violation', violations: [{ type: 'Synthetic event', timecode_sec: 44.2, description: 'Fictional visible action for a test.', evidence_frames: [{ timestamp_sec: 44.2, image_base64: '/9j/2Q==' }, { timestamp_sec: 45, image_base64: '/9j/2Q==' }] }] });
const json = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });

describe('case contract', () => {
  it('validates synthetic metadata and accepts uppercase MOV', () => {
    expect(validateFile({ name: 'synthetic.MOV', size: 12 })).toBe('');
    for (const file of [null, { name: 'synthetic.mp4', size: 12 }, { name: 'synthetic.mov.exe', size: 12 }, { name: 'synthetic.mov', size: 0 }]) expect(validateFile(file)).not.toBe('');
  });
  it('formats event timestamps including hour boundaries', () => {
    for (const [value, expected] of [[44.2, '00:44'], [83, '01:23'], [3599, '59:59'], [3661, '01:01:01'], [-1, '00:00'], [Infinity, '00:00']]) expect(formatTimestamp(value)).toBe(expected);
  });
  it('normalizes the exact case wire format and preserves fractional timestamps', () => {
    const result = normalizeJob(done(), 'job-1');
    expect(result.status).toBe('completed');
    expect(result.violations[0].timestampSeconds).toBe(44.2);
    expect(result.violations[0].frames).toHaveLength(2);
    expect(result.violations[0].frames[0].src).toBe('data:image/jpeg;base64,/9j/2Q==');
  });
  it('handles processing, failed, and no-violation responses', () => {
    expect(normalizeJob({ job_id: 'job-1', status: 'processing' }).stage).toBe(1);
    expect(normalizeJob({ job_id: 'job-1', status: 'failed', error: 'Synthetic failure' }).error).toBe('Synthetic failure');
    expect(normalizeJob({ job_id: 'job-1', status: 'done', verdict: 'no_violation', violations: [] }).violations).toEqual([]);
  });
  it('rejects mismatched jobs, contradictory verdicts, and malformed timestamps', () => {
    expect(() => normalizeJob(done(), 'different')).toThrow(/does not match/);
    const contradiction = done(); contradiction.verdict = 'no_violation';
    expect(() => normalizeJob(contradiction)).toThrow(/contradicts/);
    const invalid = done(); invalid.violations[0].timecode_sec = -1;
    expect(() => normalizeJob(invalid)).toThrow(/timecode_sec/);
    expect(() => normalizeJob({ job_id: 'job-1', status: 'unknown' })).toThrow(/status/);
  });
  it('keeps findings reviewable when evidence is absent or invalid', () => {
    const result = done(); result.violations[0].evidence_frames = [];
    expect(normalizeJob(result).violations[0].frames[0].src).toBe('');
    result.violations[0].evidence_frames = [{ image_base64: 'https://external.example/image.jpg' }];
    expect(normalizeJob(result).violations[0].frames[0].src).toBe('');
  });
  it('supports optional rule references without inventing a citation', () => {
    expect(normalizeJob(done()).violations[0].ruleReference).toBeNull();
    const result = done(); result.violations[0].rule_reference = { document_title: 'Synthetic test document', section: 'Test section', url: 'javascript:alert(1)' };
    expect(normalizeJob(result).violations[0].ruleReference).toEqual({ documentTitle: 'Synthetic test document', section: 'Test section', url: null });
    expect(safeReferenceUrl('https://adilet.zan.kz/')).toBe('https://adilet.zan.kz/');
  });
});

describe('Django adapter with mocked fetch only', () => {
  it('posts multipart field file, encodes job IDs, and sends the entered key on both requests', async () => {
    const fetchImpl = vi.fn().mockResolvedValueOnce(json({ job_id: 'job/1' })).mockResolvedValueOnce(json({ job_id: 'job/1', status: 'processing' }));
    const service = createApiService({ fetchImpl });
    const file = new File(['synthetic test bytes'], 'synthetic.MOV');
    const job = await service.startAnalysis(file, { apiKey: 'test-key' });
    expect(job.jobId).toBe('job/1');
    const [url, options] = fetchImpl.mock.calls[0];
    expect(url).toBe('/api/v1/analyze');
    expect(options.method).toBe('POST');
    expect(options.body.get('file').name).toBe('synthetic.MOV');
    expect(options.headers).toEqual({ Accept: 'application/json', 'X-API-Key': 'test-key' });
    expect(options.headers).not.toHaveProperty('Content-Type');
    await service.getAnalysis(job.jobId, { apiKey: 'test-key' });
    expect(fetchImpl.mock.calls[1][1].headers['X-API-Key']).toBe('test-key');
    expect(fetchImpl.mock.calls[1][0]).toBe('/api/v1/jobs/job%2F1');
  });
  it.each([[401, /access denied/], [403, /access denied/], [413, /size limit/], [415, /format/], [429, /busy/], [503, /Synthetic server detail/]])('explains HTTP %s', async (status, message) => {
    const service = createApiService({ fetchImpl: vi.fn().mockResolvedValue(json({ detail: 'Synthetic server detail' }, status)) });
    await expect(service.getAnalysis('job-1', { apiKey: 'test-key' })).rejects.toThrow(message);
  });
  it('reports unreadable JSON and network errors', async () => {
    await expect(createApiService({ fetchImpl: vi.fn().mockResolvedValue(new Response('<html>wrong route</html>')) }).getAnalysis('job-1', { apiKey: 'test-key' })).rejects.toThrow(/unreadable/);
    await expect(createApiService({ fetchImpl: vi.fn().mockRejectedValue(new TypeError('network')) }).getAnalysis('job-1', { apiKey: 'test-key' })).rejects.toThrow(/Cannot reach/);
  });
  it('requires a key before sending any request', async () => {
    const fetchImpl = vi.fn();
    const service = createApiService({ fetchImpl });
    await expect(service.startAnalysis(new File(['synthetic'], 'test.mov'))).rejects.toThrow(/Enter your API key/);
    await expect(service.getAnalysis('job-1', { apiKey: '  ' })).rejects.toThrow(/Enter your API key/);
    expect(fetchImpl).not.toHaveBeenCalled();
  });
  it('rejects invalid selections without making a request', async () => {
    const fetchImpl = vi.fn();
    await expect(createApiService({ fetchImpl }).startAnalysis(new File(['synthetic'], 'test.txt'))).rejects.toThrow(/\.mov/);
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(() => createApiService({ basePath: 'https://external.example' })).toThrow(/same-origin/);
    expect(() => createApiService({ basePath: '/api/v1/' })).toThrow(/trailing slash/);
  });
  it('aborts a timed-out request', async () => {
    vi.useFakeTimers();
    const fetchImpl = vi.fn((_url, { signal }) => new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('Cancelled', 'AbortError')))));
    const outcome = createApiService({ fetchImpl, requestTimeoutMs: 100 }).getAnalysis('job-1', { apiKey: 'test-key' }).catch(error => error);
    await vi.advanceTimersByTimeAsync(101);
    expect((await outcome).message).toMatch(/timed out/);
  });
});

describe('local demo and job lifetime', () => {
  it('runs every demo scenario without reading content or calling fetch', async () => {
    let now = 0;
    const service = createDemoService({ now: () => now });
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    const file = { name: 'synthetic.MOV', size: 12 };
    for (const property of ['arrayBuffer', 'text', 'stream', 'slice', 'path']) Object.defineProperty(file, property, { get() { throw new Error('File contents were accessed'); } });
    for (const scenario of ['findings', 'empty', 'error', 'missing-evidence']) {
      now = 0;
      const job = await service.startAnalysis(file, { scenario });
      expect((await service.getAnalysis(job.jobId)).status).toBe('queued');
      now = 1200; expect((await service.getAnalysis(job.jobId)).stage).toBe(1);
      now = 5000;
      const result = await service.getAnalysis(job.jobId);
      expect(result.status).toBe(scenario === 'error' ? 'failed' : 'completed');
      if (scenario === 'empty') expect(result.violations).toHaveLength(0);
      if (scenario === 'missing-evidence') expect(result.violations[0].frames[0].src).toBe('');
    }
    expect(fetchSpy).not.toHaveBeenCalled();
    service.reset();
    await expect(service.getAnalysis('demo-1')).rejects.toThrow(/no longer available/);
  });
  it('stops polling at the five-minute deadline', async () => {
    vi.useFakeTimers();
    const service = { startAnalysis: vi.fn().mockResolvedValue({ jobId: 'job-1' }), getAnalysis: vi.fn().mockResolvedValue({ status: 'processing' }) };
    const onUpdate = vi.fn();
    const outcome = runAnalysis({ service, file: null, onUpdate }).catch(error => error);
    await vi.advanceTimersByTimeAsync(300001);
    expect((await outcome).message).toMatch(/five-minute deadline/);
    const calls = service.getAnalysis.mock.calls.length;
    await vi.advanceTimersByTimeAsync(10000);
    expect(service.getAnalysis).toHaveBeenCalledTimes(calls);
  });
});
