import React, { useState } from 'react';
import UploadPanel from './components/UploadPanel.jsx';
import ProcessingPanel from './components/ProcessingPanel.jsx';
import ResultsPanel from './components/ResultsPanel.jsx';
import { useAnalysis } from './hooks/useAnalysis.js';

export default function App({ services }) {
  const [mode, setMode] = useState('local');
  const [apiKey, setApiKey] = useState('');
  const [file, setFile] = useState(null);
  const [scenario, setScenario] = useState('findings');
  const [resetKey, setResetKey] = useState(0);
  const { analysis, start, reset: resetJob, busy } = useAnalysis(services);
  function reset() { resetJob(); setFile(null); setScenario('findings'); setResetKey(value => value + 1); }
  function changeMode(value) { reset(); setMode(value); }
  function chooseFile(nextFile) { resetJob(); setFile(nextFile); }
  function sample() { setMode('demo'); start({ file: null, mode: 'demo', scenario }); }
  return <>
  <a className="skip-link" href="#main">Skip to analysis</a>
  <svg className="icon-library" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <symbol id="i-grid" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></symbol>
    <symbol id="i-upload" viewBox="0 0 24 24"><path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5"/></symbol>
    <symbol id="i-shield" viewBox="0 0 24 24"><path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z"/><path d="m8 12 3 3 5-6"/></symbol>
    <symbol id="i-play" viewBox="0 0 24 24"><path d="m9 5 11 7-11 7V5Z"/></symbol>
    <symbol id="i-file" viewBox="0 0 24 24"><path d="M14 3H5v18h14V8l-5-5Zm0 0v5h5M8 13h8M8 17h5"/></symbol>
    <symbol id="i-clock" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></symbol>
    <symbol id="i-chevron" viewBox="0 0 24 24"><path d="m9 5 7 7-7 7"/></symbol>
    <symbol id="i-check" viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></symbol>
    <symbol id="i-alert" viewBox="0 0 24 24"><path d="m12 3 10 18H2L12 3Zm0 6v5m0 3v1"/></symbol>
    <symbol id="i-image" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8" cy="8" r="1.5"/><path d="m3 17 5-5 4 4 4-6 5 7"/></symbol>
  </svg>

  <aside className="sidebar" aria-label="Workspace navigation">
    <a className="brand" href="/" aria-label="ForteBank Datathon home"><img src="/assets/mark.svg" width="34" height="34" alt=""/><span>forte<span className="brand-period">.</span></span></a>
    <div className="project-label">DATATHON WORKSPACE</div>
    <nav aria-label="Main"><a className="nav-link active" href="#main" aria-current="page"><svg className="icon" aria-hidden="true"><use href="#i-grid"/></svg>Video analysis</a></nav>
    <div className="sidebar-note"><span className="note-icon"><svg className="icon" aria-hidden="true"><use href="#i-shield"/></svg></span><strong>Built for a safer space</strong><p>Turn observations into clear, reviewable insights.</p></div>
    <div className="sidebar-footer"><span className="avatar">DT</span><div><strong>Datathon team</strong><span>React frontend · Local AI ready</span></div></div>
  </aside>

  <div className="workspace">
    <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <strong>Video analysis</strong></div><div className="local-status"><span className="status-dot"/><span className="workspace-label">Local workspace</span><span className="topbar-divider"/><label className="sr-only" htmlFor="analysis-mode">Analysis mode</label><select id="analysis-mode" className="mode-select" disabled={busy} value={mode} onChange={event => changeMode(event.target.value)}><option value="demo">Demo mode</option><option value="local">Local AI · Django</option></select></div></header>
    <main id="main" tabIndex="-1">
      <noscript><p className="demo-notice">Enable JavaScript in your browser to run this local demonstration.</p></noscript>
      <div className="page-heading"><div><p className="eyebrow">FORTEBANK DATATHON</p><h1>Surveillance analysis</h1><p className="subtitle">From footage to findings. A clearer view of every incident.</p></div><span className="outline-badge"><svg className="icon" aria-hidden="true"><use href="#i-shield"/></svg>Analysis workspace</span></div>

      <div className="demo-notice"><span className="demo-symbol"><svg className="icon" aria-hidden="true"><use href="#i-shield"/></svg></span><div><strong>{mode === 'demo' ? 'Demo mode — no video is uploaded' : 'Local AI mode — upload to your Django service'}</strong><p>{mode === 'demo' ? 'Processing is simulated. All findings and evidence illustrations are fictional.' : 'Enter your API key, select a video, then upload it for analysis. Results come from the local backend.'}</p></div><span className="demo-tag">{mode === 'demo' ? 'DEMO' : 'LOCAL AI'}</span></div>

      <section className="setup-grid" aria-label="Prepare an analysis"><UploadPanel apiKey={apiKey} setApiKey={value => { reset(); setApiKey(value); }} mode={mode} file={file} setFile={chooseFile} busy={busy} resetKey={resetKey} onAnalyze={() => start({ file, mode, scenario, apiKey })} onSample={sample}/><ProcessingPanel analysis={analysis} mode={mode} busy={busy} scenario={scenario} setScenario={setScenario}/></section>

      <ResultsPanel key={`${resetKey}-${analysis.jobId || 'idle'}`} analysis={analysis} busy={busy} onReset={reset}/>
      <footer className="page-footer"><span>ForteBank Datathon <span aria-hidden="true">·</span> Surveillance analysis prototype</span><span>Independent team project. Not an official bank product.</span></footer>
    </main>
  </div>
</>;
}
