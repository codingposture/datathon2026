import { validateFile } from './model.js';

function sampleViolations() {
  const examples = [
    ['Restricted area entry', 'A fictional person enters the marked staff-only zone.', 83, 'entry'],
    ['Unattended item', 'A fictional bag is left beside the lobby seating area.', 167, 'item'],
    ['Emergency exit obstruction', 'A fictional object blocks the marked emergency exit.', 254, 'exit']
  ];
  return examples.map(([title, description, timestampSeconds, asset], index) => ({
    id: `event-${index + 1}`, title, description, timestampSeconds, ruleReference: null,
    frames: [{ src: `/assets/evidence-${asset}.svg`, timestampSeconds, alt: `Fictional schematic illustration: ${title}` }]
  }));
}

export function createDemoService({ now = () => Date.now() } = {}) {
  const jobs = new Map();
  let nextId = 0;
  return {
    async startAnalysis(file, { scenario = 'findings', signal } = {}) {
      if (signal?.aborted) throw new DOMException('Analysis cancelled', 'AbortError');
      if (file != null) { const error = validateFile(file); if (error) throw new Error(error); }
      if (!['findings', 'empty', 'error', 'missing-evidence'].includes(scenario)) throw new Error('Unknown demo scenario.');
      const jobId = `demo-${++nextId}`;
      jobs.set(jobId, { startedAt: now(), scenario }); // No file object or contents are stored.
      return { jobId, status: 'queued', stage: 0, violations: [] };
    },
    async getAnalysis(jobId, { signal } = {}) {
      if (signal?.aborted) throw new DOMException('Analysis cancelled', 'AbortError');
      const job = jobs.get(jobId);
      if (!job) throw new Error('This analysis is no longer available. Start a new demo.');
      const elapsed = now() - job.startedAt;
      if (elapsed < 1200) return { jobId, status: 'queued', stage: 0, violations: [] };
      if (elapsed < 3200) return { jobId, status: 'processing', stage: 1, violations: [] };
      if (job.scenario === 'error') return { jobId, status: 'failed', error: 'Simulated processing error. Try again or choose another demo scenario.', violations: [] };
      if (elapsed < 4800) return { jobId, status: 'processing', stage: 2, violations: [] };
      const violations = job.scenario === 'empty' ? [] : sampleViolations();
      if (job.scenario === 'missing-evidence') violations[0].frames[0].src = '';
      return { jobId, status: 'completed', verdict: violations.length ? 'violation' : 'no_violation', violations };
    },
    reset() { jobs.clear(); }
  };
}
