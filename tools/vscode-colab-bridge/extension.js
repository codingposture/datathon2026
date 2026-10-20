const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

function activate(context) {
  const root = '/home/codingposture/Projects/Datathon';
  const spool = path.join(root, 'runs', 'colab_bridge');
  fs.mkdirSync(spool, {recursive: true, mode: 0o700});
  const active = new Set();
  context.subscriptions.push(vscode.window.registerUriHandler({
    async handleUri(uri) {
      const id = new URLSearchParams(uri.query).get('id');
      if (!id || !/^[a-zA-Z0-9_-]{1,80}$/.test(id) || active.has(id)) return;
      const reply = path.join(spool, id + '.reply.json');
      if (fs.existsSync(reply)) return;
      active.add(id);
      const requestPath = path.join(spool, id + '.request.json');
      const state = {id, status: 'running', outputs: []};
      const persist = () => {
        fs.writeFileSync(reply + '.tmp', JSON.stringify(state, null, 2), {mode: 0o600});
        fs.renameSync(reply + '.tmp', reply);
      };
      try {
        const request = JSON.parse(fs.readFileSync(requestPath, 'utf8'));
        const ext = vscode.extensions.getExtension('ms-toolsai.jupyter');
        const api = await ext.activate();
        state.notebooks = [];
        let selected;
        for (const document of vscode.workspace.notebookDocuments) {
          if (request.notebookUri && document.uri.toString() !== request.notebookUri) continue;
          const kernel = await api.kernels.getKernel(document.uri);
          state.notebooks.push({uri: document.uri.toString(), kernel: Boolean(kernel),
            status: kernel?.status, language: kernel?.language});
          if (kernel && !selected) selected = {kernel, uri: document.uri};
        }
        if (request.action === 'inspect') {
          state.status = 'done'; persist(); return;
        }
        if (request.action === 'transfer') {
          const remote = vscode.Uri.parse(request.remoteUri);
          if (remote.scheme !== 'colab') throw new Error('Only colab filesystem transfers are allowed');
          const local = path.resolve(root, request.localPath);
          if (!local.startsWith(root + path.sep)) throw new Error('Local path must remain in the Datathon workspace');
          if (request.direction === 'upload') {
            const bytes = fs.readFileSync(local);
            await vscode.workspace.fs.writeFile(remote, bytes);
            state.bytes = bytes.length;
            state.sha256 = crypto.createHash('sha256').update(bytes).digest('hex');
          } else if (request.direction === 'download') {
            const bytes = await vscode.workspace.fs.readFile(remote);
            fs.mkdirSync(path.dirname(local), {recursive: true});
            fs.writeFileSync(local, bytes);
            state.bytes = bytes.length;
            state.sha256 = crypto.createHash('sha256').update(bytes).digest('hex');
          } else throw new Error('Unknown transfer direction');
          state.status = 'done'; persist(); return;
        }
        if (request.action !== 'execute' || !selected) throw new Error('No user-started notebook kernel available');
        state.notebookUri = selected.uri.toString();
        persist();
        const token = new vscode.CancellationTokenSource();
        try {
          for await (const output of selected.kernel.executeCode(request.code, token.token)) {
            for (const item of output.items) {
              if (item.mime.startsWith('image/')) continue;
              const text = Buffer.from(item.data).toString('utf8');
              state.outputs.push({mime: item.mime, text});
              if (item.mime === 'application/vnd.code.notebook.error') state.kernelError = true;
            }
            persist();
          }
        } finally { token.dispose(); }
        state.status = state.kernelError ? 'failed' : 'done';
      } catch (error) {
        state.status = 'failed';
        state.error = String(error.message || error);
      } finally { persist(); active.delete(id); }
    }
  }));
}
exports.activate = activate;
