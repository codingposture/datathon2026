# ForteBank Datathon React frontend

The original burgundy dashboard, rewritten in **React with Vite**, for a **Django REST Framework backend running a local AI model**. The sidebar, layout, stylesheet, icons, and fictional evidence illustrations are preserved. Case-specific additions include verdicts, multiple JPEG evidence frames, and rule references.

The frontend starts in **Local AI · Django** mode. Enter the `API_KEY` configured in
`backend/.env` before choosing a video. Uploading and polling send that entered key as
`X-API-Key`; Django validates it. The key stays in React memory and is not saved in browser
storage. A reload clears it. Wrong keys show an access-denied error; results are never
replaced with demo findings.

The backend in [../backend/](../backend/README.md) returns the verdict, categories,
timestamps, descriptions and JPEG evidence displayed by the frontend. Temporary files
and 15-minute RAM results are used; legal citations remain deferred.
Demo mode is optional and uses clearly marked fictional results without backend requests.

## 1. Extract and start the frontend

1. Extract the ZIP to a normal folder. Do not run the project inside the ZIP.
2. Install Node.js 24.15 or newer, which includes npm.
3. Open a terminal inside the extracted `fortebank-frontend` folder. You should see `package.json` there.
4. Install the dependencies:

```powershell
npm install
```

5. Start the website:

```powershell
npm run dev
```

6. Open **http://127.0.0.1:4173/** on the same computer.

Keep the terminal open while using the website. Press Ctrl+C to stop it before moving the folder. If port 4173 is already in use, stop the other copy of this project's server first; the configured server intentionally does not switch to another port.

Demo mode works without Django, an API key, or a video. Double-clicking `index.html` does not run a React app. The computer needs internet access to download dependencies initially; the running frontend does not use a CDN or cloud AI.

The original dependency lockfile, `pnpm-lock.yaml`, is included. `npm install` is supported and creates its own `package-lock.json` on the teammate's computer; npm does not consume the pnpm lockfile. For a dependency install matching that existing lockfile exactly, use pnpm 11.25 with `pnpm install --frozen-lockfile`, then `pnpm dev`.

## 2. Explore without a video

1. Select **Demo mode** in **Analysis mode**.
2. Click **Try sample analysis**. About five seconds later, three fictional findings appear.
3. Select a finding to see its timestamp and illustrated evidence.
4. Use **Demo scenario** to try no findings, processing failure, and unavailable evidence.
5. Click **Reset analysis** to clear the page and cancel frontend polling. Late responses cannot restore cleared results.

The old example event categories remain illustrative placeholders; they are not confirmed cash-discipline rules or claims about the model's capabilities.

## 3. Connect the local Django backend

The frontend already implements the case's exact API:

| Action | Request |
| --- | --- |
| Submit video | `POST /api/v1/analyze`, multipart field **file** |
| Poll job | `GET /api/v1/jobs/{job_id}` |
| Backend authentication | `X-API-Key`, entered by the user and checked by Django |

Ask your backend teammates for the local Django origin and API key. Copy the example configuration:

```powershell
Copy-Item .env.example .env.local
```

Edit `.env.local` locally:

```dotenv
DJANGO_ORIGIN=http://127.0.0.1:8000
```

Restart `npm run dev` after changing the backend origin. Local AI mode is the default.
Enter the backend's `API_KEY` in the password field, then choose a `.mov` and click
**Upload & analyze**. Selecting a file alone never uploads it. Django checks the key on
submission and each poll. Changing or clearing the key clears the current file/results.
**Try sample analysis** explicitly switches to demo mode.

The Vite proxy forwards the entered key unchanged to loopback Django. It never substitutes
an environment key. `DJANGO_API_KEY` in old frontend configuration is no longer used.
Do not put keys in `VITE_` variables or source code. The browser holds only the key the user
enters, for the current page session.

See [the backend contract](docs/BACKEND_CONTRACT.md) for exact JSON examples, DRF setup requirements, rule-source extensions, and production integration. The local backend now uses the existing hybrid_v2 model and has a separate real-model HTTP integration check through this proxy and adapter.

## 4. Understand the code

| File or folder | Purpose |
| --- | --- |
| `src/App.jsx` | Preserved page shell, selected mode, and selected file |
| `src/components/` | Upload controls, processing stages, findings, and evidence |
| `src/hooks/useAnalysis.js` | React job state, reset, cancellation, and stale-response protection |
| `src/services/api.js` | Real Django HTTP adapter; no automatic submission |
| `src/services/model.js` | Input checks, timestamps, and backend response normalization |
| `src/services/demo.js` | Fictional results with no file reads or API calls |
| `src/services/run-analysis.js` | Polling and the five-minute overall deadline |
| `styles.css` | Original dashboard styling with small additions for case requirements |
| `public/assets/` | Original local SVG illustrations and mark |
| `vite.config.js` | React tooling and server-side local API proxy |

React state replaces manual DOM updates. Components receive data through props. The service boundary maps the case's snake_case JSON to the UI's camelCase objects, keeping Django-specific parsing out of display components.

## 5. Test and build

```powershell
npm test
npm run build
npm run preview
```

`npm run build` creates `dist/`, containing the local React bundle and assets. It requires no CDN, external font, analytics, or cloud AI service at runtime. `npm run preview` serves only the built frontend; it does **not** provide the development Django proxy. Use demo mode there, or serve `dist/` through your production server with the backend integration described in the contract guide.

The automated tests use synthetic File objects and mocked fetch responses, never actual videos. They cover exact endpoints and multipart fields, JPEG frame normalization, verdict consistency, optional sources, API failures, timeouts, privacy in demo mode, React interactions, multiple frames, reset during an active request, and unmount cleanup.

For visual checks, run the demo and inspect desktop/mobile widths, keyboard navigation, and image fallbacks. Development loads only local app assets plus Vite's local hot-reload connection; demo processing makes no backend requests.

Current integration verification: **29 frontend tests passed**, **19 backend tests passed**, and the production build passed. Earlier browser verification: React demo/evidence selection and local-mode switching worked in the browser, and layouts at 320, 390, 768, and 1280 pixels had no horizontal overflow. The desktop layout was also inspected at 1440 pixels. The final browser check reported no warnings/errors and loaded the illustrated evidence successfully. Preview screenshots are excluded from this source-only handoff.

## Case analysis and limits

[Case analysis](docs/CASE_ANALYSIS.md) maps the supplied DOCX to this implementation. The case requires a complete deployed service and a jury API; this repository provides the local frontend and integration contract. A Django server, model, validated rules, jury key, and public deployment still belong to the full team's delivery.

Results are possible violations for human review, not legal or disciplinary conclusions. File extensions are a frontend convenience; Django must validate actual files, sizes, and model inputs. Reset stops the browser's requests, not an already-running server job. There is no history, face identification, audio processing, cloud inference, or persistent browser storage.

## Source-only handoff

This archive contains the React source, HTML, CSS, JavaScript, package manifest, original dependency lockfile, local fictional SVG assets, configuration, tests, and integration documentation. It deliberately excludes `node_modules`, `dist`, build outputs, logs, development-server artifacts, real `.env` files, API keys, videos, extracted frames, datasets, screenshots, and the original case DOCX. Dependencies are not preinstalled in this archive.

The standalone frontend folder does not include the sibling Django backend or model weights.
In this full project they live in `backend/` and `artifacts/hybrid_v2/`. Private keys remain
in ignored local configuration files. The optional rule-reference field is documented in the contract.
