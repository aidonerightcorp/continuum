import { atom, cn, host, ListRow, SegmentedControl, Tip, ToggleRow, useValue } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

const PAPER_INK_STYLE = {
  backgroundColor: '#f5f2ea',
  color: '#141413',
  colorScheme: 'only light',
  border: '1px solid #cfc8b8',
  borderRadius: '8px',
  padding: '14px',
  fontFamily: 'ui-sans-serif, system-ui, sans-serif'
}

const CARD_STYLE = {
  backgroundColor: '#faf8f3',
  color: '#141413',
  border: '1px solid #cfc8b8',
  borderRadius: '6px',
  padding: '10px',
  marginTop: '8px'
}

const DEFAULTS = {
  showStatusChip: true,
  embeddingSlice: '64'
}

const INITIAL_TELEMETRY = {
  version: '1.3.0',
  engine_count: 30,
  sub_engine_count: 112,
  hypergrid_total_dimensions: 172,
  supported_targets: ['rust', 'glsl', 'js', 'python', 'sql', 'c99', 'wasm', 'wat', 'wgsl'],
  matryoshka_spaces: ['task2vec', 'config2vec', 'weight2vec', 'trace2vec'],
  matryoshka_prefix_dims: [16, 32, 64, 128],
  omninode: {
    encased: true,
    lockfile_verified: true,
    lockfile_sha256: 'verified-cas-lockfile',
    total_cards: 78,
    grid1_format: 'grid/1',
    active_grid1_cell: 'canonical-grid1-sha256',
    supported_operators: ['@', '>>', '|', '//']
  },
  persisted_tasks: 0,
  persisted_verifications: 0,
  persisted_attractors: 0
}

function ContinuumStatusChip({ prefs, telemetry }) {
  const gateway = useValue(host.state.gateway)
  const { showStatusChip, embeddingSlice } = useValue(prefs)
  const snap = useValue(telemetry)
  const omni = snap.omninode || {}
  const lockLabel = omni.lockfile_verified ? 'CAS✓' : 'CAS!'

  if (!showStatusChip) {
    return null
  }

  return jsx(Tip, {
    label: `Continuum v${snap.version} • ${snap.engine_count} engines / ${snap.sub_engine_count} sub-engines • OmniNode ${lockLabel} • gateway: ${gateway}`,
    children: jsxs('span', {
      className: cn('inline-flex h-full items-center gap-1.5 px-2 text-[0.6875rem]'),
      style: {
        backgroundColor: '#f5f2ea',
        color: '#141413',
        colorScheme: 'only light',
        borderLeft: '1px solid #cfc8b8'
      },
      children: [
        jsx('span', { 'aria-hidden': true, children: '◈' }),
        jsx('span', { children: `Continuum ${snap.engine_count}x${snap.sub_engine_count} · ${lockLabel} · ${embeddingSlice}d` })
      ]
    })
  })
}

function ContinuumInspectorPane({ prefs, telemetry, onRefresh }) {
  const { embeddingSlice } = useValue(prefs)
  const snap = useValue(telemetry)
  const targets = snap.supported_targets || []
  const omni = snap.omninode || {}
  const lockHash = String(omni.lockfile_sha256 || '').slice(0, 16) || 'pending'
  const cellHash = String(omni.active_grid1_cell || '').slice(0, 16) || 'pending'
  const ops = (omni.supported_operators || ['@', '>>', '|', '//']).join(' ')

  return jsxs('div', {
    style: PAPER_INK_STYLE,
    children: [
      jsxs('div', {
        style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
        children: [
          jsxs('div', {
            children: [
              jsx('div', {
                style: { fontSize: '15px', fontWeight: 700, color: '#141413' },
                children: 'Continuum — OmniNode-Encased Self-Improving Agent Engine'
              }),
              jsx('div', {
                style: { fontSize: '12px', color: '#4a4843', marginTop: '2px' },
                children: `v${snap.version} • 8 Layers • ${snap.engine_count} Engines • ${snap.sub_engine_count} Sub-Engines • 18 Tools • ${snap.hypergrid_total_dimensions}-Axis TT-Cross Grid`
              })
            ]
          }),
          jsx('button', {
            type: 'button',
            onClick: onRefresh,
            style: {
              backgroundColor: '#ebe6da',
              color: '#141413',
              border: '1px solid #cfc8b8',
              borderRadius: '4px',
              padding: '4px 10px',
              fontSize: '12px',
              cursor: 'pointer'
            },
            children: 'Sync State'
          })
        ]
      }),
      jsxs('div', {
        style: CARD_STYLE,
        children: [
          jsx('div', {
            style: { fontSize: '12px', fontWeight: 600, color: '#8c5a2b' },
            children: 'OmniNode CARD Lockfile & Canonical grid/1 Cell Receipt'
          }),
          jsx('div', {
            style: { fontSize: '12px', color: '#141413', marginTop: '4px' },
            children: `CARD Lockfile: ${omni.lockfile_verified ? 'VERIFIED' : 'UNVERIFIED'} (${lockHash}) • Cards: ${omni.total_cards || 0} • Active grid/1 Cell: ${cellHash} • Operators: ${ops}`
          })
        ]
      }),
      jsxs('div', {
        style: CARD_STYLE,
        children: [
          jsx('div', {
            style: { fontSize: '12px', fontWeight: 600, color: '#2f5d50' },
            children: 'Latent Mind SAE Probing, Nullspace Steering & 9 Portable Runtimes'
          }),
          jsx('div', {
            style: { fontSize: '12px', color: '#141413', marginTop: '4px' },
            children: `Spaces: ${(snap.matryoshka_spaces || []).join(' · ')} | Slice: ${embeddingSlice}d | Targets (${targets.length}): ${targets.join(', ')}`
          })
        ]
      }),
      jsxs('div', {
        style: CARD_STYLE,
        children: [
          jsx('div', {
            style: { fontSize: '12px', fontWeight: 600, color: '#354f6b' },
            children: 'Profile-Scoped SQLite WAL State (Read-Only)'
          }),
          jsx('div', {
            style: { fontSize: '12px', color: '#141413', marginTop: '4px' },
            children: `Leased Tasks: ${snap.persisted_tasks} • Verified Records: ${snap.persisted_verifications} • Saved Memories: ${snap.persisted_attractors}`
          })
        ]
      })
    ]
  })
}

export default {
  id: 'continuum',
  name: 'Continuum',
  description: 'OmniNode-encased unified self-improving agent engine for Hermes: 30 cybernetic engines (112 sub-engines), 18 tools, >> // | @ shape composition, grid/1 cell receipts, SAE/ELK probing, Nullspace steering, and 9-target skill compilation.',
  register(ctx) {
    const prefs = atom({ ...DEFAULTS, ...ctx.storage.get('prefs', {}) })
    const telemetry = atom(INITIAL_TELEMETRY)

    const updatePrefs = patch => {
      prefs.set({ ...prefs.get(), ...patch })
      ctx.storage.set('prefs', prefs.get())
    }

    const refreshTelemetry = () => {
      if (typeof ctx.rest !== 'function') {
        return
      }
      Promise.resolve(ctx.rest('/status'))
        .then(data => {
          if (data && typeof data === 'object') {
            telemetry.set({ ...telemetry.get(), ...data })
          }
        })
        .catch(() => {})
    }

    refreshTelemetry()

    ctx.register({
      id: 'continuum-chip',
      area: 'statusBar.right',
      order: 105,
      render: () => jsx(ContinuumStatusChip, { prefs, telemetry })
    })

    ctx.register({
      id: 'continuum-pane',
      area: 'panes',
      title: 'Continuum Engine',
      icon: 'cpu',
      render: () => jsx(ContinuumInspectorPane, { prefs, telemetry, onRefresh: refreshTelemetry })
    })

    ctx.registerSettingsPage?.({
      id: 'settings',
      title: 'Continuum',
      icon: 'cpu',
      render: function GeneralSettings() {
        const { showStatusChip, embeddingSlice } = useValue(prefs)
        return jsxs('div', {
          style: { colorScheme: 'only light' },
          children: [
            jsx(ToggleRow, {
              checked: showStatusChip,
              description: 'Display live 30x112 OmniNode lockfile and embedding slice indicator in the status bar.',
              label: 'Show Continuum status chip',
              onChange: enabled => updatePrefs({ showStatusChip: enabled })
            }),
            jsx(ListRow, {
              action: jsx(SegmentedControl, {
                onChange: value => updatePrefs({ embeddingSlice: value }),
                options: [
                  { id: '16', label: '16d' },
                  { id: '32', label: '32d' },
                  { id: '64', label: '64d' },
                  { id: '128', label: '128d' }
                ],
                value: embeddingSlice
              }),
              description: 'Select default nested embedding slice width for fast retrieval and inspection.',
              title: 'Default embedding slice width'
            })
          ]
        })
      }
    })
  }
}
