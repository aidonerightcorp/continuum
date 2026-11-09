// Continuum Web Dashboard Surface — Paper & Ink Light Palette (#f5f2ea / #141413)
(function registerContinuumDashboard() {
  const PALETTE = {
    bg: '#f5f2ea',
    card: '#faf8f3',
    ink: '#141413',
    muted: '#4a4843',
    border: '#cfc8b8',
    sage: '#2f5d50',
    ochre: '#8c5a2b'
  };

  function renderContinuumPanel(container, statusPayload) {
    if (!container) return;
    container.textContent = '';
    const root = document.createElement('section');
    root.style.backgroundColor = PALETTE.bg;
    root.style.color = PALETTE.ink;
    root.style.colorScheme = 'only light';
    root.style.border = '1px solid ' + PALETTE.border;
    root.style.borderRadius = '8px';
    root.style.padding = '16px';
    root.style.fontFamily = 'ui-sans-serif, system-ui, sans-serif';

    const heading = document.createElement('h2');
    heading.textContent = 'Continuum v1.3.0 — OmniNode-Encased Self-Improving Agent Engine';
    heading.style.margin = '0 0 6px 0';
    heading.style.fontSize = '18px';
    root.appendChild(heading);

    const sub = document.createElement('p');
    const eCount = statusPayload && statusPayload.engine_count ? statusPayload.engine_count : 30;
    const sCount = statusPayload && statusPayload.sub_engine_count ? statusPayload.sub_engine_count : 112;
    const dims = statusPayload && statusPayload.hypergrid_total_dimensions ? statusPayload.hypergrid_total_dimensions : 172;
    const targets = statusPayload && Array.isArray(statusPayload.supported_targets) ? statusPayload.supported_targets.length : 9;
    sub.textContent = eCount + ' Functional Engines • ' + sCount + ' Sub-Engines • 18 Tools • ' + dims + '-Axis TT-Cross HyperGrid • ' + targets + ' Portable Runtimes • OmniNode >> // | @ Shapes';
    sub.style.margin = '0 0 12px 0';
    sub.style.color = PALETTE.muted;
    sub.style.fontSize = '13px';
    root.appendChild(sub);

    const omni = (statusPayload && statusPayload.omninode) ? statusPayload.omninode : {};
    const lockOk = omni.lockfile_verified !== false;
    const lockSha = omni.lockfile_sha256 ? String(omni.lockfile_sha256).slice(0, 16) : 'verified-cas';
    const cellSha = omni.active_grid1_cell ? String(omni.active_grid1_cell).slice(0, 16) : 'grid1-ready';
    const omniCard = document.createElement('div');
    omniCard.style.backgroundColor = PALETTE.card;
    omniCard.style.border = '1px solid ' + PALETTE.border;
    omniCard.style.borderRadius = '6px';
    omniCard.style.padding = '12px';
    omniCard.style.marginBottom = '10px';
    omniCard.textContent = 'OmniNode CARD Lockfile: ' + (lockOk ? 'VERIFIED' : 'UNVERIFIED') + ' (' + lockSha + ') | Active grid/1 Cell SHA-256: ' + cellSha + ' | Operators: >> // | @';
    root.appendChild(omniCard);

    const card = document.createElement('div');
    card.style.backgroundColor = PALETTE.card;
    card.style.border = '1px solid ' + PALETTE.border;
    card.style.borderRadius = '6px';
    card.style.padding = '12px';
    const tasks = statusPayload && statusPayload.persisted_tasks ? statusPayload.persisted_tasks : 0;
    const verifs = statusPayload && statusPayload.persisted_verifications ? statusPayload.persisted_verifications : 0;
    const atts = statusPayload && statusPayload.persisted_attractors ? statusPayload.persisted_attractors : 0;
    card.textContent = 'SQLite WAL Read-Only Snapshot — Tasks: ' + tasks + ' | Independent Verifications: ' + verifs + ' | Saved Memories: ' + atts;
    root.appendChild(card);

    container.appendChild(root);
  }

  if (typeof window !== 'undefined' && typeof window.registerDashboardPlugin === 'function') {
    window.registerDashboardPlugin('continuum', {
      mount(el) {
        renderContinuumPanel(el, null);
        if (typeof window.fetch === 'function') {
          window.fetch('/api/plugins/continuum/status')
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (data) { if (data) renderContinuumPanel(el, data); })
            .catch(function () {});
        }
      }
    });
  }
})();
