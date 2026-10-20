// The UI uses camelCase. The DRF wire contract is normalized at one boundary.
export function validateFile(file) {
  if (!file || typeof file.name !== 'string' || !/\.mov$/i.test(file.name)) return 'Please choose a QuickTime video with a .mov extension.';
  if (!Number.isFinite(file.size) || file.size <= 0) return 'This file is empty. Please choose a non-empty .mov file.';
  return '';
}

export function formatTimestamp(seconds) {
  const value = Math.floor(Number(seconds));
  if (!Number.isFinite(value) || value < 0) return '00:00';
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const remainder = String(value % 60).padStart(2, '0');
  return hours > 0 ? `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${remainder}` : `${String(minutes).padStart(2, '0')}:${remainder}`;
}

export function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

const contractError = message => new Error(`Unexpected backend response: ${message}`);
const validSeconds = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;

export function normalizeJobId(data) {
  if (!data || typeof data.job_id !== 'string' || !data.job_id.trim()) throw contractError('a non-empty job_id is required.');
  return data.job_id;
}

export function safeReferenceUrl(url) {
  try {
    const parsed = new URL(url);
    return ['https:', 'http:'].includes(parsed.protocol) ? parsed.href : null;
  } catch { return null; }
}

function normalizeFrame(frame, eventTime, title) {
  // Only raw JPEG base64 from the case contract is accepted, never an arbitrary URL.
  const encoded = frame?.image_base64;
  const validImage = typeof encoded === 'string' && encoded.startsWith('/9j/') && /^[A-Za-z0-9+/]+={0,2}$/.test(encoded);
  return {
    timestampSeconds: validSeconds(frame?.timestamp_sec) ? frame.timestamp_sec : eventTime,
    src: validImage ? `data:image/jpeg;base64,${encoded}` : '',
    alt: `Evidence frame for ${title}`
  };
}

export function normalizeJob(data, expectedJobId) {
  const jobId = normalizeJobId(data);
  if (expectedJobId && jobId !== expectedJobId) throw contractError('job_id does not match the requested analysis.');
  if (data.status === 'processing') return { jobId, status: 'processing', stage: 1, violations: [] };
  if (data.status === 'failed') return { jobId, status: 'failed', error: typeof data.error === 'string' ? data.error : typeof data.detail === 'string' ? data.detail : 'The local AI service could not process this video.', violations: [] };
  if (data.status !== 'done') throw contractError('status must be processing, done, or failed.');
  if (!['violation', 'no_violation'].includes(data.verdict) || !Array.isArray(data.violations)) throw contractError('a valid verdict and violations array are required.');
  if ((data.verdict === 'no_violation') !== (data.violations.length === 0)) throw contractError('the verdict contradicts the findings.');
  const violations = data.violations.map((item, index) => {
    if (!item || typeof item.type !== 'string' || !item.type.trim() || typeof item.description !== 'string' || !item.description.trim() || !validSeconds(item.timecode_sec)) throw contractError(`finding ${index + 1} needs a type, description, and non-negative numeric timecode_sec.`);
    if (item.evidence_frames != null && !Array.isArray(item.evidence_frames)) throw contractError('evidence_frames must be an array.');
    if (item.evidence_frames?.length > 3) throw contractError('a finding can contain at most three evidence frames.');
    const rule = item.rule_reference;
    return {
      id: `event-${index + 1}`,
      title: item.type,
      description: item.description,
      timestampSeconds: item.timecode_sec,
      frames: item.evidence_frames?.length ? item.evidence_frames.map(frame => normalizeFrame(frame, item.timecode_sec, item.type)) : [{ src: '', timestampSeconds: item.timecode_sec, alt: 'Evidence not supplied' }],
      // The narrative requires sources, but section 10 does not define a wire field.
      // This optional extension must be agreed with the Django team.
      ruleReference: rule && typeof rule.document_title === 'string' && rule.document_title.trim() ? {
        documentTitle: rule.document_title,
        section: typeof rule.section === 'string' ? rule.section : '',
        url: safeReferenceUrl(rule.url)
      } : null
    };
  });
  return { jobId, status: 'completed', verdict: data.verdict, violations };
}
