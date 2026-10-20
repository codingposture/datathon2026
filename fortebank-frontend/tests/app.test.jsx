import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import App from '../src/App.jsx';
import { createApiService } from '../src/services/api.js';
import { createDemoService } from '../src/services/demo.js';

const response = data => new Response(JSON.stringify(data), { headers: { 'Content-Type': 'application/json' } });
function wireResult() {
  return { job_id: 'job-1', status: 'done', verdict: 'violation', violations: [{
    type: 'Synthetic cash-handling event', timecode_sec: 44.2, description: 'Fictional action used only in a test.',
    evidence_frames: [{ timestamp_sec: 44.2, image_base64: '/9j/2Q==' }, { timestamp_sec: 45, image_base64: '/9j/2Q==' }],
    rule_reference: { document_title: 'Synthetic test source', section: 'Test section', url: 'https://adilet.zan.kz/' }
  }] };
}

it('runs the sample in React with zero backend requests', async () => {
  vi.useFakeTimers();
  const fetchSpy = vi.spyOn(globalThis, 'fetch');
  render(<App/>);
  fireEvent.click(screen.getByRole('button', { name: /Try sample analysis/ }));
  expect(screen.getByLabelText('Analysis mode')).toBeDisabled();
  await act(() => vi.advanceTimersByTimeAsync(5000));
  expect(screen.getByText('3 findings')).toBeInTheDocument();
  expect(screen.getByText('Signs of possible violations detected')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: /Unattended item/ }));
  expect(screen.getByRole('heading', { name: 'Unattended item' })).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Choose video'), { target: { files: [new File(['synthetic bytes'], 'new-selection.MOV')] } });
  expect(screen.queryByRole('heading', { name: 'Unattended item' })).not.toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Your findings will appear here' })).toBeInTheDocument();
  expect(fetchSpy).not.toHaveBeenCalled();
});

it.each([['empty', 'No violation detected'], ['error', 'No results available'], ['missing-evidence', 'Evidence image unavailable']])('supports the %s scenario', async (scenario, expected) => {
  vi.useFakeTimers();
  render(<App/>);
  fireEvent.change(screen.getByLabelText('Analysis mode'), { target: { value: 'demo' } });
  fireEvent.change(screen.getByLabelText('Demo scenario'), { target: { value: scenario } });
  fireEvent.click(screen.getByRole('button', { name: /Try sample analysis/ }));
  await act(() => vi.advanceTimersByTimeAsync(5000));
  expect(screen.getAllByText(expected).length).toBeGreaterThan(0);
  expect(screen.getByRole('button', { name: /Try sample analysis/ })).toBeEnabled();
});

it('validates synthetic selections without reading or uploading them', () => {
  const fetchSpy = vi.spyOn(globalThis, 'fetch');
  render(<App/>);
  expect(screen.getByLabelText('Analysis mode')).toHaveValue('local');
  const input = screen.getByLabelText('Choose video');
  expect(input).toBeDisabled();
  fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'test-key' } });
  expect(input).toBeEnabled();
  fireEvent.change(input, { target: { files: [new File(['synthetic bytes'], 'synthetic.txt')] } });
  expect(screen.getByRole('alert')).toHaveTextContent('.mov');
  expect(screen.getByRole('button', { name: 'Upload & analyze' })).toBeDisabled();
  fireEvent.change(input, { target: { files: [new File(['synthetic bytes'], 'synthetic.MOV')] } });
  expect(screen.getByText('synthetic.MOV')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Upload & analyze' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: 'Remove' }));
  expect(screen.queryByText('synthetic.MOV')).not.toBeInTheDocument();
  expect(fetchSpy).not.toHaveBeenCalled();
});

it('ignores a late response after reset and aborts its request signal', async () => {
  let resolve;
  const demo = { reset: vi.fn(), startAnalysis: vi.fn().mockResolvedValue({ jobId: 'old' }), getAnalysis: vi.fn(() => new Promise(done => { resolve = done; })) };
  render(<App services={{ demo, local: {} }}/>);
  fireEvent.click(screen.getByRole('button', { name: /Try sample analysis/ }));
  await waitFor(() => expect(demo.getAnalysis).toHaveBeenCalled());
  fireEvent.click(screen.getByRole('button', { name: /Reset analysis/ }));
  expect(demo.getAnalysis.mock.calls[0][1].signal.aborted).toBe(true);
  await act(async () => resolve({ jobId: 'old', status: 'completed', verdict: 'no_violation', violations: [] }));
  expect(screen.getByRole('heading', { name: 'Your findings will appear here' })).toBeInTheDocument();
  expect(screen.queryByText('Demo analysis complete')).not.toBeInTheDocument();
});

it('submits only on an explicit local-mode click and reviews multiple backend frames', async () => {
  const fetchImpl = vi.fn().mockResolvedValueOnce(response({ job_id: 'job-1' })).mockResolvedValueOnce(response(wireResult()));
  render(<App services={{ demo: createDemoService(), local: createApiService({ fetchImpl }) }}/>);
  expect(screen.getByLabelText('Analysis mode')).toHaveValue('local');
  fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'test-key' } });
  fireEvent.change(screen.getByLabelText('Choose video'), { target: { files: [new File(['synthetic bytes'], 'synthetic.MOV')] } });
  expect(fetchImpl).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Upload & analyze' }));
  await screen.findByRole('heading', { name: 'Synthetic cash-handling event' });
  expect(fetchImpl).toHaveBeenCalledTimes(2);
  for (const [, options] of fetchImpl.mock.calls) expect(options.headers['X-API-Key']).toBe('test-key');
  expect(screen.getByText('Synthetic test source')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: /Frame 2/ }));
  expect(document.getElementById('evidence-time')).toHaveTextContent('00:45');
  fireEvent.error(screen.getByRole('img', { name: 'Evidence frame for Synthetic cash-handling event' }));
  expect(screen.getByText('Evidence image unavailable')).toBeInTheDocument();
  expect(screen.getByText('Human review is required.', { exact: false })).toBeInTheDocument();
});

it('cleans up polling on unmount', async () => {
  vi.useFakeTimers();
  const demo = { reset: vi.fn(), startAnalysis: vi.fn().mockResolvedValue({ jobId: 'job-1' }), getAnalysis: vi.fn().mockResolvedValue({ status: 'processing', stage: 1, violations: [] }) };
  const view = render(<App services={{ demo, local: {} }}/>);
  fireEvent.click(screen.getByRole('button', { name: /Try sample analysis/ }));
  await act(() => vi.advanceTimersByTimeAsync(10));
  view.unmount();
  const calls = demo.getAnalysis.mock.calls.length;
  await act(() => vi.advanceTimersByTimeAsync(10000));
  expect(demo.getAnalysis).toHaveBeenCalledTimes(calls);
});

it('shows backend key rejection without substituting demo findings', async () => {
  const fetchImpl = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'Missing or invalid X-API-Key.' }), { status: 403 }));
  render(<App services={{ demo: createDemoService(), local: createApiService({ fetchImpl }) }}/>);
  fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'incorrect-key' } });
  fireEvent.change(screen.getByLabelText('Choose video'), { target: { files: [new File(['synthetic'], 'test.mov')] } });
  fireEvent.click(screen.getByRole('button', { name: 'Upload & analyze' }));
  expect((await screen.findAllByText(/Check the API key you entered/)).length).toBeGreaterThan(0);
  expect(fetchImpl).toHaveBeenCalledTimes(1);
  fireEvent.change(screen.getByLabelText('API key'), { target: { value: '' } });
  expect(screen.getByLabelText('Choose video')).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Upload & analyze' })).toBeDisabled();
  expect(screen.queryByText('test.mov')).not.toBeInTheDocument();
});
