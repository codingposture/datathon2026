import React, { useState } from 'react';
import Icon from './Icon.jsx';
import { formatTimestamp } from '../services/model.js';

function Evidence({ finding, demo }) {
  const [frameIndex, setFrameIndex] = useState(0);
  const [failedSrc, setFailedSrc] = useState(null);
  const frame = finding.frames[frameIndex];
  const unavailable = !frame.src || failedSrc === frame.src;
  const rule = finding.ruleReference;
  return <div className="evidence-column">
    <div className="evidence-heading"><span className="column-label">EVIDENCE PREVIEW</span><span id="evidence-time" className="timestamp">{formatTimestamp(frame.timestampSeconds)}</span></div>
    <div className="evidence-frame">
      {!unavailable && <img key={frame.src} id="evidence-image" src={frame.src} alt={frame.alt} onError={() => setFailedSrc(frame.src)}/>}
      {unavailable && <div id="evidence-fallback" className="evidence-fallback"><Icon name="image"/><strong>Evidence image unavailable</strong><span>The finding is still available for review.</span></div>}
      <span className="illustration-label">{demo ? 'ILLUSTRATED DEMO · NOT REAL FOOTAGE' : 'LOCAL AI EVIDENCE · HUMAN REVIEW REQUIRED'}</span>
    </div>
    {finding.frames.length > 1 && <div className="frame-navigation" aria-label="Evidence frames">{finding.frames.map((item, index) => <button key={index} type="button" className="frame-button" aria-pressed={index === frameIndex} onClick={() => setFrameIndex(index)}>Frame {index + 1} <span>{formatTimestamp(item.timestampSeconds)}</span></button>)}</div>}
    <div className="evidence-caption"><div aria-live="polite" aria-atomic="true"><h3 id="evidence-title">{finding.title}</h3><p>{finding.description}</p></div><span className="badge badge-demo">{demo ? 'Demo finding' : 'Possible violation'}</span></div>
    <div className="rule-reference"><span className="column-label">RULE &amp; NORMATIVE SOURCE</span>{rule ? <><strong>{rule.documentTitle}</strong>{rule.section && <p>{rule.section}</p>}{rule.url && <a href={rule.url} target="_blank" rel="noopener noreferrer">Open source <span aria-hidden="true">↗</span></a>}</> : <p>{demo ? 'Illustrative finding only. No legal rule is claimed in this demo.' : 'Rule reference not provided by the backend. Request the document title and relevant section before drawing a conclusion.'}</p>}</div>
  </div>;
}

export default function ResultsPanel({ analysis, busy, onReset }) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const violations = analysis.violations || [];
  const selected = violations[Math.min(selectedIndex, Math.max(0, violations.length - 1))];
  const completed = analysis.status === 'completed';
  const demo = analysis.source === 'demo';
  let title = 'Your findings will appear here';
  let description = 'Start an analysis to explore incident timestamps and evidence.';
  if (busy) { title = demo ? 'Demo analysis in progress' : 'Local AI analysis in progress'; description = 'Your findings will appear when processing finishes.'; }
  if (analysis.status === 'failed') { title = 'No results available'; description = analysis.error; }
  if (completed && !violations.length) { title = 'No violation detected'; description = demo ? 'The fictional analysis completed with no findings. Choose another scenario to explore evidence review.' : 'The model reported no visually observable signs of a violation. This result is an aid for human review.'; }
  return <section className="panel results-panel" aria-labelledby="results-heading" aria-busy={busy}>
    <div className="panel-heading results-heading"><div className="results-title"><h2 id="results-heading"><span className="step-number">03</span>Analysis results</h2><span id="results-count" className="count-badge">{violations.length} finding{violations.length === 1 ? '' : 's'}</span></div><button id="reset-button" className="text-button reset-button" type="button" onClick={onReset}>Reset analysis <span aria-hidden="true">↺</span></button></div>
    {completed && <div className={`verdict-strip ${violations.length ? 'verdict-review' : 'verdict-clear'}`} role="status"><Icon name={violations.length ? 'alert' : 'shield'}/><strong>{analysis.verdict === 'violation' ? 'Signs of possible violations detected' : 'No violation detected'}</strong><span>{demo ? 'Fictional demo result' : 'AI-assisted assessment · Human review required'}</span></div>}
    {!selected ? <div id="results-empty" className="empty-state"><span className="empty-icon"><Icon name="grid"/></span><h3>{title}</h3><p>{description}</p><span className="empty-caption">A clear timeline. Evidence in context.</span></div> : <div id="results-content" className="results-grid">
      <div className="findings-column"><div className="column-label">DETECTED EVENTS <span>{demo ? 'FICTIONAL DATA' : 'LOCAL AI RESULTS'}</span></div><div className="violation-list" aria-label="Possible violation findings">{violations.map((finding, index) => <button key={finding.id} type="button" className="violation-card" aria-pressed={finding.id === selected.id} aria-controls="evidence-title" onClick={() => setSelectedIndex(index)}><span className="finding-icon"><Icon name="alert"/></span><div className="finding-details"><strong>{finding.title}</strong><p>{finding.description}</p><div className="finding-meta"><span className="finding-time"><Icon name="clock"/>{formatTimestamp(finding.timestampSeconds)}</span><span>View evidence</span></div></div><Icon name="chevron" className="icon finding-arrow"/></button>)}</div><p className="review-note"><Icon name="alert"/>AI findings require human review.</p></div>
      <Evidence key={selected.id} finding={selected} demo={demo}/>
    </div>}
  </section>;
}
