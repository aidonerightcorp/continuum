// Continuum in the Hermes desktop app: a status chip and a pane that show what the right-sizing loop chose and
// how it went. Reads this plugin's own backend route /api/plugins/continuum/rightsize (read-only).
// It shows counts, choices and outcomes: no Task text and no reply text.
import { atom, cn, ListRow, Tip, ToggleRow, useValue } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

const VERSION = '2.0.0'
const REFRESH_MS = 5000
const DEFAULTS = { showStatusChip: true }
const EMPTY = { present: false, turns: 0, scored: 0, escalated: 0, by_model: [], by_mode: [], recent: [], policy: {} }
const CHEAPER_MODES = ['exploit', 'explore']

const num = x => Number(x || 0).toLocaleString()
const pct = x => (x === null || x === undefined ? 'no outcome yet' : `${Math.round(100 * x)}%`)
const cheaperTurns = snap =>
  (snap.by_mode || []).filter(m => CHEAPER_MODES.includes(m.mode)).reduce((sum, m) => sum + m.turns, 0)

function outcome(row) {
  if (row.score === null || row.score === undefined) {
    return 'open'
  }
  return `${row.score >= 0.5 ? 'good' : 'not good'} · ${row.source || ''}${row.weight >= 0.5 ? '' : ' (weak)'}`
}

function ContinuumStatusChip({ prefs, records }) {
  const { showStatusChip } = useValue(prefs)
  const snap = useValue(records)

  if (!showStatusChip) {
    return null
  }

  const cheaper = cheaperTurns(snap)

  return jsx(Tip, {
    label: `Continuum ${VERSION}: ${num(snap.turns)} turns recorded, ${num(cheaper)} sent to a cheaper model, ${num(snap.escalated)} returned to your model`,
    children: jsxs('span', {
      className: cn('inline-flex h-full items-center gap-1.5 px-2 text-[0.6875rem]'),
      children: [
        jsx('span', { 'aria-hidden': true, children: '◈' }),
        jsx('span', { children: `Continuum ${num(cheaper)}/${num(snap.turns)}` })
      ]
    })
  })
}

function Row({ label, value }) {
  return jsxs('div', {
    className: 'flex items-baseline justify-between gap-3',
    children: [
      jsx('span', { className: 'text-(--ui-text-tertiary)', children: label }),
      jsx('span', { className: 'tabular-nums', children: value })
    ]
  })
}

function ContinuumPane({ records, onRefresh }) {
  const snap = useValue(records)
  const policy = snap.policy || {}

  if (!snap.present || !snap.turns) {
    return jsxs('div', {
      className: 'flex h-full flex-col gap-2 p-3 text-sm',
      children: [
        jsx('div', { className: 'font-medium', children: 'Continuum' }),
        jsx('div', {
          className: 'text-(--ui-text-tertiary)',
          children:
            'No turns are recorded yet. Use Hermes as usual. To let Continuum send easy Tasks to a cheaper model, set small_model in the plugin settings and accept llm.model_override.'
        })
      ]
    })
  }

  return jsxs('div', {
    className: 'flex h-full flex-col gap-3 overflow-y-auto p-3 text-sm',
    children: [
      jsxs('div', {
        className: 'flex items-center justify-between',
        children: [
          jsx('div', { className: 'font-medium', children: 'Continuum: right-sizing' }),
          jsx('button', {
            type: 'button',
            className: 'text-xs text-(--ui-text-tertiary) underline',
            onClick: onRefresh,
            children: 'Refresh'
          })
        ]
      }),
      jsxs('div', {
        className: 'flex flex-col gap-1',
        children: [
          jsx(Row, { label: 'Turns recorded', value: num(snap.turns) }),
          jsx(Row, { label: 'With an outcome', value: num(snap.scored) }),
          jsx(Row, { label: 'Sent to a cheaper model', value: num(cheaperTurns(snap)) }),
          jsx(Row, { label: 'Returned to your model inside the turn', value: num(snap.escalated) }),
          jsx(Row, { label: 'Checks by your own model', value: `${num(snap.checks)} (${num(snap.checks_failed)} FAIL)` }),
          jsx(Row, {
            label: 'Policy',
            value: `floor ${policy.floor || 'default'} · evidence ${policy.min_evidence || 'default'}`
          })
        ]
      }),
      jsx('div', { className: 'font-medium', children: 'By model' }),
      jsx('div', {
        className: 'flex flex-col gap-1',
        children: (snap.by_model || []).map(m =>
          jsx(Row, {
            label: m.model,
            value: `${num(m.turns)} turns · ${pct(m.pass_rate)} good · ${num((m.tok_in || 0) + (m.tok_out || 0))} tokens`
          }, m.model)
        )
      }),
      jsx('div', { className: 'font-medium', children: 'Latest turns' }),
      jsx('div', {
        className: 'flex flex-col gap-1',
        children: (snap.recent || []).map((r, i) =>
          jsx(Row, {
            label: `${r.model || ''} · ${r.mode || ''}${r.escalated ? ' → your model' : ''}`,
            value: outcome(r)
          }, i)
        )
      })
    ]
  })
}

export default {
  id: 'continuum',
  name: 'Continuum',
  description:
    'Hermes sizes itself to each Task: see what Continuum chose for each turn, how it went, and what it learned.',
  register(ctx) {
    const prefs = atom({ ...DEFAULTS, ...ctx.storage.get('prefs', {}) })
    const records = atom(EMPTY)

    const updatePrefs = patch => {
      prefs.set({ ...prefs.get(), ...patch })
      ctx.storage.set('prefs', prefs.get())
    }

    const refresh = () => {
      if (typeof ctx.rest !== 'function') {
        return
      }
      Promise.resolve(ctx.rest('/rightsize'))
        .then(data => {
          if (data && typeof data === 'object') {
            records.set({ ...EMPTY, ...data })
          }
        })
        .catch(() => {})
    }

    refresh()
    ctx.setInterval(refresh, REFRESH_MS) // a scoped timer: Hermes clears it with the plugin

    ctx.register({
      id: 'continuum-chip',
      area: 'statusBar.right',
      order: 105,
      render: () => jsx(ContinuumStatusChip, { prefs, records })
    })

    ctx.register({
      id: 'continuum-pane',
      area: 'panes',
      title: 'Continuum',
      icon: 'cpu',
      data: { placement: 'right', width: '320px' },
      render: () => jsx(ContinuumPane, { records, onRefresh: refresh })
    })

    ctx.registerSettingsPage?.({
      id: 'settings',
      title: 'Continuum',
      icon: 'cpu',
      render: function GeneralSettings() {
        const { showStatusChip } = useValue(prefs)
        return jsxs('div', {
          children: [
            jsx(ToggleRow, {
              checked: showStatusChip,
              description: 'Show how many turns Continuum sent to a cheaper model in the status bar.',
              label: 'Show Continuum status chip',
              onChange: enabled => updatePrefs({ showStatusChip: enabled })
            }),
            jsx(ListRow, {
              description:
                'Model, tool and check settings live in config.yaml under plugins.entries.continuum.settings. Run /continuum in a session for the full status.',
              title: 'Right-sizing settings'
            })
          ]
        })
      }
    })
  }
}
