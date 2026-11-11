// Continuum dashboard tab. Reads /api/plugins/continuum/rightsize (read-only) and shows what the
// right-sizing loop chose and how it went. It shows counts, choices and outcomes: no Task text, no reply text.
(function () {
  "use strict";

  var SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK || !window.__HERMES_PLUGINS__) return;
  var React = SDK.React;
  var h = React.createElement;
  var C = SDK.components;
  var useState = SDK.hooks.useState;
  var useEffect = SDK.hooks.useEffect;
  var REFRESH_MS = 5000;

  function pct(x) {
    return x === null || x === undefined ? "no outcome yet" : Math.round(100 * x) + "%";
  }
  function num(x) {
    return Number(x || 0).toLocaleString();
  }
  function ago(ts) {
    return ts && SDK.utils && SDK.utils.timeAgo ? SDK.utils.timeAgo(ts) : "";
  }
  function count(list, key, name) {
    var hit = (list || []).filter(function (r) { return r[key] === name; })[0];
    return hit ? hit.turns : 0;
  }

  function Tile(props) {
    return h(C.Card, null,
      h(C.CardContent, { className: "p-4" },
        h("div", { className: "text-2xl font-semibold tabular-nums" }, props.value),
        h("div", { className: "text-sm text-muted-foreground" }, props.label)));
  }

  function Section(props) {
    return h(C.Card, null,
      h(C.CardHeader, null, h(C.CardTitle, { className: "text-base" }, props.title)),
      h(C.CardContent, null, props.children));
  }

  function Table(props) {
    return h("div", { className: "overflow-x-auto" },
      h("table", { className: "w-full text-sm" },
        h("thead", null, h("tr", { className: "text-left text-muted-foreground" },
          props.head.map(function (t, i) { return h("th", { key: i, className: "py-1 pr-4 font-medium" }, t); }))),
        h("tbody", null, props.rows.map(function (cells, i) {
          return h("tr", { key: i, className: "border-t border-border" },
            cells.map(function (c, j) { return h("td", { key: j, className: "py-1 pr-4 align-top tabular-nums" }, c); }));
        }))));
  }

  function outcome(r) {
    if (r.score === null || r.score === undefined) return "open";
    return (r.score >= 0.5 ? "good" : "not good") + " · " + (r.source || "") + (r.weight >= 0.5 ? "" : " (weak)");
  }

  function ContinuumPage() {
    var state = useState(null);
    var data = state[0];
    var setData = state[1];
    var errState = useState("");
    var err = errState[0];
    var setErr = errState[1];

    useEffect(function () {
      var live = true;
      function load() {
        SDK.fetchJSON("/api/plugins/continuum/rightsize")
          .then(function (d) { if (live) { setData(d); setErr(""); } })
          .catch(function (e) { if (live) setErr(String(e && e.message ? e.message : e)); });
      }
      load();
      var timer = setInterval(load, REFRESH_MS);
      return function () { live = false; clearInterval(timer); };
    }, []);

    if (err && !data) {
      return h(Section, { title: "Continuum" }, h("p", { className: "text-sm text-muted-foreground" },
        "The record file could not be read: " + err));
    }
    if (!data) {
      return h(Section, { title: "Continuum" }, h("p", { className: "text-sm text-muted-foreground" }, "Loading…"));
    }
    if (!data.present || !data.turns) {
      return h(Section, { title: "Continuum: Hermes sizes itself to each Task" },
        h("p", { className: "text-sm text-muted-foreground" },
          "No turns are recorded yet. Use Hermes as usual; every turn is fingerprinted and recorded on this machine. " +
          "To let Continuum send easy Tasks to a cheaper model, set plugins.entries.continuum.settings.small_model " +
          "and accept llm.model_override when you run `hermes plugins enable continuum`."));
    }

    var cheaper = count(data.by_mode, "mode", "exploit") + count(data.by_mode, "mode", "explore");
    var policy = data.policy || {};

    return h("div", { className: "flex flex-col gap-4" },
      h("div", null,
        h("h2", { className: "text-lg font-semibold" }, "Continuum: Hermes sizes itself to each Task"),
        h("p", { className: "text-sm text-muted-foreground" },
          "Counts, choices and outcomes from the record file on this machine. No Task text is shown here.")),
      h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "1rem" } },
        h(Tile, { value: num(data.turns), label: "turns recorded" }),
        h(Tile, { value: num(data.scored), label: "turns with an outcome" }),
        h(Tile, { value: num(cheaper), label: "turns sent to a cheaper model" }),
        h(Tile, { value: num(data.escalated), label: "returned to your model inside the turn" })),
      h(Section, { title: "By model" },
        h(Table, {
          head: ["Model", "Turns", "Good outcomes", "Evidence", "Tokens in", "Tokens out", "Seconds"],
          rows: (data.by_model || []).map(function (m) {
            return [m.model, num(m.turns), pct(m.pass_rate), Number(m.evidence || 0).toFixed(1), num(m.tok_in),
              num(m.tok_out), Number(m.seconds || 0).toFixed(0)];
          })
        })),
      h("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-4" },
        h(Section, { title: "How turns were chosen" },
          h(Table, { head: ["Choice", "Turns"], rows: (data.by_mode || []).map(function (m) { return [m.mode, num(m.turns)]; }) })),
        h(Section, { title: "Where outcomes came from" },
          h(Table, { head: ["Signal", "Turns"], rows: (data.by_source || []).map(function (m) { return [m.source, num(m.turns)]; }) }),
          data.checks ? h("p", { className: "text-sm text-muted-foreground mt-2" },
            "Checks by your own model: " + num(data.checks) + " (" + num(data.checks_failed) + " said FAIL).") : null)),
      h(Section, { title: "Policy" },
        h("p", { className: "text-sm" },
          "Floor " + (policy.floor || "default") + " · evidence needed " + (policy.min_evidence || "default") +
          " · similarity preset " + (policy.preset || "default") +
          (data.tool_tokens_saved ? " · tool list sizing removed about " + num(data.tool_tokens_saved) + " prompt tokens" : "") +
          (data.imported ? " · " + num(data.imported) + " imported records" : "")),
        (data.policy_log || []).length ? h(Table, {
          head: ["When", "Event", "Detail"],
          rows: data.policy_log.map(function (p) {
            return [ago(p.ts), p.event, h("span", { className: "font-mono text-xs break-all" }, JSON.stringify(p.detail).slice(0, 240))];
          })
        }) : h("p", { className: "text-sm text-muted-foreground" }, "No policy change yet.")),
      h(Section, { title: "Latest turns" },
        h(Table, {
          head: ["When", "Model", "Choice", "Outcome", "Tokens", "Why"],
          rows: (data.recent || []).map(function (r) {
            return [ago(r.ts), r.model || "", (r.mode || "") + (r.escalated ? " → your model" : ""), outcome(r),
              num((r.tok_in || 0) + (r.tok_out || 0)), h("span", { className: "text-muted-foreground" }, r.reason || "")];
          })
        })));
  }

  window.__HERMES_PLUGINS__.register("continuum", ContinuumPage);
})();
