import React, { useEffect, useRef, useState } from 'react';
import Icon from './Icon.jsx';
import { formatFileSize, validateFile } from '../services/model.js';

export default function UploadPanel({ apiKey, setApiKey, mode, file, setFile, busy, onAnalyze, onSample, resetKey }) {
  const locked = mode === 'local' && !apiKey.trim();
  const input = useRef(null);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  useEffect(() => { setError(''); setDragging(false); if (input.current) input.current.value = ''; }, [resetKey]);
  useEffect(() => {
    const prevent = event => event.preventDefault();
    window.addEventListener('dragover', prevent);
    window.addEventListener('drop', prevent);
    return () => { window.removeEventListener('dragover', prevent); window.removeEventListener('drop', prevent); };
  }, []);
  function choose(nextFile) {
    if (busy || locked) return;
    const message = nextFile ? validateFile(nextFile) : '';
    setError(message);
    setFile(message ? null : nextFile);
    input.current.value = '';
  }
  function drop(event) {
    event.preventDefault(); setDragging(false);
    if (busy || locked) return;
    if (event.dataTransfer.files.length !== 1) {
      choose(null); setError('Please choose one .mov file at a time.');
    } else choose(event.dataTransfer.files[0]);
  }
  return <div className="panel upload-panel">
    <div className="panel-heading"><h2><span className="step-number">01</span>Select a video</h2><span className="muted small">QuickTime · .mov</span></div>
    {mode === 'local' && <div className="api-key-control">
      <label htmlFor="api-key">API key</label>
      <input id="api-key" type="password" autoComplete="off" spellCheck={false} value={apiKey} disabled={busy} onChange={event => setApiKey(event.target.value)} aria-describedby="api-key-help" placeholder="Enter your backend API key"/>
      <p id="api-key-help">Required before selecting a video. The backend checks this key on upload and every status request. Kept only in memory until you reload or clear it.</p>
    </div>}
    <div id="drop-zone" className={`drop-zone${dragging ? ' dragging' : ''}`} onDrop={drop} onDragOver={event => { event.preventDefault(); if (!busy && !locked) setDragging(true); }} onDragLeave={() => setDragging(false)}>
      <span className="upload-symbol"><Icon name="upload"/></span>
      <h3>Drop your .mov file here</h3><p>or choose a file from your computer</p>
      <label className="button button-secondary file-picker" htmlFor="video-file">Choose video<input ref={input} id="video-file" type="file" accept=".mov" disabled={busy || locked} aria-invalid={Boolean(error)} aria-describedby="file-help file-error" onChange={event => { if (event.target.files[0]) choose(event.target.files[0]); }}/></label>
      <p id="file-help" className="file-help">{mode === 'demo' ? 'Demo checks the filename and size only. Video contents stay unread.' : 'Selection stays local. Only Upload & analyze sends the file to Django.'}</p>
    </div>
    {file && <div className="selected-file"><span className="file-symbol"><Icon name="file"/></span><div><strong>{file.name}</strong><span>{formatFileSize(file.size)} · {mode === 'demo' ? 'Metadata only · No upload' : 'Ready for local analysis'}</span></div><button className="text-button" type="button" disabled={busy} onClick={() => { choose(null); input.current.focus(); }}>Remove</button></div>}
    <p id="file-error" className="error-text" role="alert" hidden={!error}>{error}</p>
    <div className="upload-actions"><button id="analyze-button" className="button button-primary" type="button" disabled={busy || locked || !file} onClick={onAnalyze}><Icon name={mode === 'demo' ? 'play' : 'upload'}/>{mode === 'demo' ? 'Analyze in demo' : 'Upload & analyze'}</button><span className="action-divider">or</span><button id="sample-button" className="text-button" type="button" disabled={busy} onClick={onSample}>Try sample analysis <span aria-hidden="true">↗</span></button></div>
  </div>;
}
