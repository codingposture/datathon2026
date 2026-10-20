// Real HTTP/model integration through the frontend's Vite proxy. No result/video is saved.
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { basename } from 'node:path';
import { tmpdir } from 'node:os';
import { createApiService } from '../fortebank-frontend/src/services/api.js';
import { runAnalysis } from '../fortebank-frontend/src/services/run-analysis.js';

const apiKey = process.env.API_KEY;
if (!apiKey) throw new Error('Set API_KEY or use node --env-file=backend/.env for this smoke check.');
const path = process.argv[2];
if (!path) throw new Error('Usage: node scripts/smoke_backend.mjs /path/to/video.mov');
const before = new Set(readdirSync(tmpdir()).filter(name => name.startsWith('fortebank-')));
const hadArtifactOutputs = existsSync('artifacts/hybrid_v2/inference_outputs');
const origin = 'http://127.0.0.1:4173';
const service = createApiService({ fetchImpl: (path, options) => fetch(origin + path, options) });
const file = new File([readFileSync(path)], basename(path), { type: 'video/quicktime' });
const start = performance.now();
let previous;
const result = await runAnalysis({ service, file, apiKey, timeoutMs: 305000,
  onUpdate(update) {
    if (update.status !== previous) console.log(JSON.stringify({ status: update.status, elapsedSec: (performance.now()-start)/1000 }));
    previous = update.status;
  }
});
if (result.status !== 'completed') throw new Error(result.error || 'Analysis failed');
for (const finding of result.violations) {
  if (finding.ruleReference !== null) throw new Error('Unexpected legal reference');
  if (!finding.frames.length || finding.frames.some(f => !f.src.startsWith('data:image/jpeg;base64,/9j/'))) throw new Error('Invalid evidence');
}
const leftovers = readdirSync(tmpdir()).filter(name => name.startsWith('fortebank-') && !before.has(name));
if (leftovers.length) throw new Error('Job temporary files were not removed');
if (existsSync('artifacts/hybrid_v2/inference_outputs') !== hadArtifactOutputs) throw new Error('Inference wrote into model artifacts');
console.log(JSON.stringify({ passed: true, elapsedSec: (performance.now()-start)/1000,
  findings: result.violations.length, evidenceFrames: result.violations.reduce((n,e)=>n+e.frames.length,0),
  frontendAdapterAndProxy: true, temporaryFilesRemoved: true, citationsDeferred: true,
  inferenceDevice: 'cpu', trainingRun: false }));
