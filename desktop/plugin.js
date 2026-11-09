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
  version: '1.0.0',
  engine_count: 20,
  sub_engine_count: 72,
  hypergrid_total_dimensions: 100,
  matryoshka_spaces: ['task2vec', 'config2vec', 'weight2vec', 'trace2vec'],
  matryoshka_prefix_dims: [16, 32, 64, 128],
  persisted_tasks: 0,
  persisted_verifications: 0,
  persisted_attractors: 0
}

function ContinuumStatusChip({ prefs, telemetry }) {
  const gateway = useValue(host.state.gateway)
  const { showStatusChip, embeddingSlice } = useValue(prefs)
  const snap = useValue(telemetry)

  if (!showStatusChip) {
    return null
  }

  return jsx(Tip, {
    label: `Continuum v${snap.version} • ${snap.engine_count} engines / ${snap.sub_engine_count} sub-engines • gateway: ${gateway}`,
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
        jsx('span', { children: `Continuum ${snap.engine_count}E · ${embeddingSlice}d` })
      ]
    })
  })
}

function ContinuumInspectorPane({ prefs, telemetry, onRefresh }) {
  const { embeddingSlice } = useValue(prefs)
  const snap = useValue(telemetry)

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
                children: 'Continuum — Unified Self-Improving Agent Engine'
              }),
              jsx('div', {
                style: { fontSize: '12px', color: '#4a4843', marginTop: '2px' },
                children: `v${snap.version} • 5 Layers • ${snap.engine_count} Engines • ${snap.sub_engine_count} Sub-Engines • ${snap.hypergrid_total_dimensions} Tunable Parameters`
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
            style: { fontSize: '12px', fontWeight: 600, color: '#2f5d50' },
            children: 'Shared 4-Space Nested Embeddings'
          }),
          jsx('div', {
            style: { fontSize: '12px', color: '#141413', marginTop: '4px' },
            children: `Spaces: ${(snap.matryoshka_spaces || []).join(' · ')} | Active Slice: ${embeddingSlice}d of [${(snap.matryoshka_prefix_dims || []).join(', ')}]`
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
  description: 'Unified self-improving agent engine for Hermes: context compression, local episodic memory, verified subtask routing, adaptive tuning, and 7-target skill compilation.',
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
              description: 'Display live 20-engine and embedding slice indicator in the status bar.',
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
