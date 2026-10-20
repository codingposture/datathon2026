import React from 'react';

function getStatus(analysis, mode) {
  const demo = analysis.source === 'demo';
  if (analysis.status === 'uploading') return ['Uploading', 'Sending video to local Django', 'Waiting for the backend to accept the analysis job.', 'active'];
  if (analysis.status === 'queued') return ['Queued', demo ? 'Preparing your demo' : 'Video accepted', demo ? 'Creating a fictional analysis job. No video contents are accessed.' : 'Waiting for the local model to start.', 'active'];
  if (analysis.status === 'processing') return ['Processing', demo ? analysis.stage === 2 ? 'Preparing evidence' : 'Analyzing activity' : 'Local AI analysis in progress', demo ? analysis.stage === 2 ? 'Matching fictional findings to illustrated evidence.' : 'Simulating event detection using prepared example data.' : 'Waiting for the local model. Analysis can take up to five minutes.', 'active'];
  if (analysis.status === 'failed') return ['Failed', 'Analysis could not finish', analysis.error || 'The analysis failed. Please try again.', 'error'];
  if (analysis.status === 'completed') return ['Complete', demo ? 'Demo analysis complete' : 'Analysis complete', `${analysis.violations.length} ${demo ? 'fictional ' : 'possible '}finding${analysis.violations.length === 1 ? '' : 's'} ready for review.${demo ? ' No video was read or uploaded.' : ' Human review is required.'}`, 'success'];
  return ['Ready', 'Ready when you are', mode === 'demo' ? 'Select a .mov file or try the sample to explore the workflow.' : 'Select a .mov file, then upload it to your local Django service.', 'neutral'];
}

export default function ProcessingPanel({ analysis, mode, busy, scenario, setScenario }) {
  const [badge, title, description, tone] = getStatus(analysis, mode);
  const complete = analysis.status === 'completed';
  const stage = analysis.status === 'failed' ? -1 : analysis.stage;
  const stages = mode === 'demo' ? [['Prepare analysis', 'Create a demo analysis job'], ['Analyze activity', 'Simulate incident detection'], ['Prepare evidence', 'Organize fictional findings']] : [['Submit video', 'Send to the local Django API'], ['Analyze activity', 'Run the local AI model'], ['Prepare evidence', 'Receive results and JPEG frames']];
  return <div className="panel processing-panel">
    <div className="panel-heading"><h2><span className="step-number">02</span>Processing status</h2><span id="status-badge" className={`badge badge-${tone}`}>{badge}</span></div>
    <div id="processing-message" className="processing-message" role="status" aria-live="polite" aria-atomic="true"><strong>{title}</strong><p>{description}</p></div>
    <ol className="processing-steps" aria-label="Analysis stages">{stages.map(([label, detail], index) => {
      const done = complete || index < stage;
      const active = !complete && index === stage;
      return <li key={label} className={done ? 'stage-complete' : active ? 'stage-active' : ''} aria-current={active ? 'step' : undefined}><span className="stage-marker">{done ? '✓' : index + 1}</span><div><strong>{label}</strong><span>{detail}</span></div><span className="stage-state">{done ? 'Done' : active ? 'In progress' : 'Waiting'}</span></li>;
    })}</ol>
    <div className="scenario-control">{mode === 'demo' ? <><label htmlFor="demo-scenario">Demo scenario</label><select id="demo-scenario" disabled={busy} value={scenario} onChange={event => setScenario(event.target.value)}><option value="findings">Example violations</option><option value="empty">No violations</option><option value="error">Processing error</option><option value="missing-evidence">Unavailable evidence</option></select></> : <><span>Processing deadline</span><strong className="deadline-label">5 minutes · local AI</strong></>}</div>
    {analysis.jobId && <p className="job-id">Job ID: <span>{analysis.jobId}</span></p>}
  </div>;
}
