/* MLRecall frontend - plain JavaScript, no build step. */
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const LEVEL_NAMES = ["fragile", "settling", "solid", "strong"];
const DIFF = { 1: "easier to retain", 2: "medium to retain", 3: "harder to retain" };
let user = null, token = localStorage.getItem("mlrecall_token"), topicsCache = [], openGroups = new Set();

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = "Bearer " + token;
  const res = await fetch("/api" + path, { ...opts, headers, body: opts.body ? JSON.stringify(opts.body) : undefined });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || "Something went wrong.");
  return data;
}

function renderMath(el) {
  if (window.renderMathInElement) {
    window.renderMathInElement(el, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }], throwOnError: false });
  } else { setTimeout(() => window.renderMathInElement && renderMath(el), 600); }
}

/* a ring that fills as memory strengthens (the app's one signature element) */
function ring(fraction, kind = "green", size = 22) {
  const r = 8, c = 2 * Math.PI * r, color = kind === "blue" ? "#2F6FA8" : "#3F8F5F";
  const arc = Math.max(0, Math.min(1, fraction)) * c;
  return `<svg class="ring" width="${size}" height="${size}" viewBox="0 0 22 22" aria-hidden="true">
    <circle cx="11" cy="11" r="${r}" fill="none" stroke="#DCE3E8" stroke-width="3"/>
    <circle cx="11" cy="11" r="${r}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round"
      stroke-dasharray="${arc} ${c}" transform="rotate(-90 11 11)"/></svg>`;
}
function statusRing(t, size = 22) {
  if (t.status === "quizzed") return ring((t.level + 1) / 4, "green", size);
  if (t.status === "read") return ring(0.12, "blue", size);
  return ring(0, "blue", size);
}
const pct = (x) => (x == null ? "-" : Math.round(x * 100) + "%");

/* ---------- login ---------- */
function signIn(data) {
  token = data.token; localStorage.setItem("mlrecall_token", token);
  user = data; start();
}
async function boot() {
  if (token) {
    try { user = await api("/me"); return start(); } catch (e) { localStorage.removeItem("mlrecall_token"); token = null; user = null; }
  }
  $("#login").hidden = false;
}
$("#login-btn").onclick = async () => {
  try { signIn(await api("/login", { method: "POST", body: { username: $("#username").value, password: $("#password").value } })); }
  catch (e) { $("#login-error").textContent = e.message; }
};
$("#password").addEventListener("keydown", e => { if (e.key === "Enter") $("#login-btn").click(); });
$("#demo-btn").onclick = async () => {
  try { signIn(await api("/dev/seed_demo", { method: "POST" })); }
  catch (e) { $("#login-error").textContent = "The demo account is only available when the app is started with python run.py."; }
};

async function start() {
  $("#login").hidden = true; $("#app").hidden = false;
  await refreshSidebar();
  window.addEventListener("hashchange", route);
  route();
}

/* ---------- sidebar ---------- */
async function refreshSidebar() {
  user = await api("/me");
  $("#who").innerHTML = esc(user.username) + (user.is_simulated ? '<span class="sim-badge">simulated learner</span>' : "");
  $("#today-label").textContent = "Today: " + user.today + (user.day_offset ? ` (clock +${user.day_offset} d)` : "");
  topicsCache = await api("/topics");
  const active = location.hash.split("/")[2];
  const html = topicsCache.map(g => {
    const done = g.topics.filter(t => t.status === "quizzed").length;
    const open = openGroups.has(g.group) || g.topics.some(t => t.id === active);
    return `<div><button class="group-title" data-g="${esc(g.group)}" aria-expanded="${open}">
      <span>${esc(g.group)}</span><span class="count">${done}/${g.topics.length}</span></button>
      <div class="topic-links" ${open ? "" : "hidden"}>${g.topics.map(t =>
      `<a class="topic-link ${t.id === active ? "active" : ""}" href="#/topic/${t.id}" title="${esc(t.label + " - " + t.summary)}">
          ${statusRing(t)}<span>${esc(t.title)}</span>${t.due_today ? '<span class="due">due</span>' : ""}</a>`).join("")}</div></div>`;
  }).join("");
  $("#topic-list").innerHTML = html;
  document.querySelectorAll(".group-title").forEach(b => b.onclick = () => {
    const g = b.dataset.g; openGroups.has(g) ? openGroups.delete(g) : openGroups.add(g);
    b.nextElementSibling.hidden = !openGroups.has(g); b.setAttribute("aria-expanded", openGroups.has(g));
  });
}
$("#menu-btn").onclick = () => $("#sidebar").classList.toggle("open");

/* ---------- router ---------- */
async function route() {
  $("#sidebar").classList.remove("open");
  const [, page, arg, arg2] = (location.hash || "#/today").split("/");
  document.querySelectorAll(".nav a").forEach(a => a.classList.toggle("active", a.dataset.nav === page));
  document.querySelectorAll(".topic-link").forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#/topic/" + arg));
  const main = $("#main");
  try {
    if (page === "topic") await viewTopic(arg, main);
    else if (page === "revise") await viewRevise(arg, main);
    else if (page === "quiz") await viewQuiz(arg, main, arg2 ? Number(arg2) : null);
    else if (page === "progress") await viewProgress(main);
    else if (page === "settings") await viewSettings(main);
    else await viewToday(main);
  } catch (e) { main.innerHTML = `<div class="panel info"><h3>Something needs attention</h3><p>${esc(e.message)}</p></div>`; }
  window.scrollTo(0, 0); main.focus({ preventScroll: true });
}

/* ---------- Today's recall (home dashboard) ---------- */
async function viewToday(main) {
  let r, p;
  try { [r, p] = await Promise.all([api("/recall/today"), api("/progress")]); }
  catch (e) { main.innerHTML = `<h1>Today's recall</h1><div class="panel info"><h3>The agent needs training first</h3><p>${esc(e.message)}</p><p class="note">Open a terminal in the project folder, run the command above, then refresh this page.</p></div>`; return; }
  const first = topicsCache[0].topics[0];
  const recent = p.history.slice(0, 3).map(h => `${esc(h.topic)} ${esc(h.score)}`).join(", ") || "none yet";
  const stats = `<div class="stats">
      <div class="stat"><b>${p.total_topics}</b><span>topics available</span></div>
      <div class="stat"><b>${p.read} / ${p.total_topics - p.read}</b><span>read / not yet read</span></div>
      <div class="stat"><b>${p.quizzed}</b><span>topics with quiz attempts</span></div>
      <div class="stat"><b>${p.due_today}</b><span>due for revision today</span></div>
      <div class="stat"><b>${r.K}</b><span>daily revision limit</span></div></div>
      <p class="note">Recent quiz scores: ${recent}.</p>`;
  const rows = r.items.map(it => {
    const done = it.status === "completed", skipped = it.status === "skipped";
    const text = done ? "Reviewed today. It will return when it is due again." : skipped ? "Skipped today. It keeps aging, so it will rank higher tomorrow." : esc(it.reason);
    const actions = done || skipped ? `<span class="chip ${done ? "green" : ""}">${done ? "done" : "skipped"}</span>`
      : `<span style="display:flex;flex-wrap:wrap;gap:.4rem"><a class="btn primary" href="#/revise/${it.topic_id}">Start revision</a>
         <a class="btn quiet" href="#/topic/${it.topic_id}">Open lesson</a><button class="btn quiet" data-skip="${it.topic_id}">Skip</button></span>`;
    return `<div class="recall-row ${done ? "done" : ""}">${ring(done ? 1 : it.est_recall, "green", 34)}
      <div><p class="title">${esc(it.title)}</p><p class="why">${text}</p></div>${actions}</div>`;
  }).join("");
  refreshSidebar();            // today's list now exists, so the "due" badges can show
  main.innerHTML = `<h1>Today's recall</h1>
    <p class="lead">Up to ${r.K} topics the agent picked for ${esc(r.today)}, chosen from the quizzes you have taken.</p>${stats}
    ${r.items.length ? `<div class="recall-list">${rows}</div><p class="note">Remaining capacity today: ${r.remaining} of ${r.K}. The ring shows the model's estimated recall, which is an estimate, not your score.</p>` : ""}
    ${r.message ? `<div class="panel info"><h3>${r.items.length ? "Note" : "Nothing to review yet"}</h3><p>${esc(r.message)}</p>${r.items.length ? "" : `<a class="btn green" href="#/topic/${first.id}">Start with ${esc(first.title)}</a>`}</div>` : ""}
    <div class="panel"><h3>How the agent chooses</h3><p class="note">Each quiz tells the agent how much you remembered and when. A reinforcement-learning index then ranks the topics you have studied, favouring those where a review now protects the most memory, and shows the top ${r.K}. Skipping a topic does not change what the agent knows about it; the topic keeps aging.</p></div>`;
  document.querySelectorAll("[data-skip]").forEach(b => b.onclick = async () => { await api(`/recall/${b.dataset.skip}/skip`, { method: "POST" }); viewToday(main); });
}

/* ---------- focused revision ---------- */
async function viewRevise(id, main) {
  const r = await api(`/topics/${id}/revision/start`, { method: "POST" });
  main.innerHTML = `<div class="meta"><span class="chip">Revision</span></div><h1>${esc(r.title)}</h1>
    <p class="lead">A quick refresher of the key ideas and formulas. Then answer a short quiz from memory: the quiz is what counts as your review.</p>
    <article class="lesson" id="lesson">${r.summary_html}</article>
    <div class="actions"><a class="btn green" href="#/quiz/${id}/${r.attempt_id}">I'm ready, start the quiz</a><a class="btn quiet" href="#/today">Back to today's list</a></div>`;
  renderMath($("#lesson"));
}

/* ---------- topic page ---------- */
async function viewTopic(id, main) {
  const t = await api("/topics/" + id);
  const prereq = t.prerequisites.length ? "Builds on: " + t.prerequisites.map(p => `<a href="#/topic/${p.id}">${esc(p.title)}</a>`).join(", ") : "No prerequisites";
  const state = t.status === "quizzed"
    ? `<span class="chip green">Memory ${LEVEL_NAMES[t.level]} - last quiz ${esc(t.last_score)} on ${esc(t.last_quiz)} - estimated recall ${pct(t.est_recall)}</span>`
    : t.status === "read" ? '<span class="chip">Read - take the quiz to set your starting point</span>' : "";
  main.innerHTML = `<div class="meta"><span class="chip">${esc(t.group)}</span><span class="chip">${DIFF[t.difficulty]}</span><span>${prereq}</span></div>
    <h1>${esc(t.title)}</h1><p class="lead">${esc(t.summary)}</p>${state}
    <article class="lesson" id="lesson">${t.lesson_html}</article>
    <div class="actions">
      <button class="btn quiet" id="read-btn" ${t.status !== "not_started" ? "disabled" : ""}>${t.status === "not_started" ? "Mark as read" : "Marked as read"}</button>
      <a class="btn green" id="quiz-btn" href="#/quiz/${t.id}" ${t.status === "not_started" ? 'aria-disabled="true" style="opacity:.45;pointer-events:none"' : ""}>${t.status === "quizzed" ? "Take a recall quiz" : "Take the quiz"}</a>
      <span class="note">${t.status === "not_started" ? "Read the lesson, then mark it as read to unlock the quiz." : t.n_questions_in_bank + " questions in this topic's bank; each quiz uses 5."}</span>
    </div>`;
  renderMath($("#lesson"));
  $("#read-btn").onclick = async () => { await api(`/topics/${id}/read`, { method: "POST" }); await refreshSidebar(); viewTopic(id, main); };
}

/* ---------- quiz ---------- */
async function viewQuiz(id, main, attemptId = null) {
  const q = await api(`/topics/${id}/quiz`);
  const intro = { first: "This first quiz sets your starting point for this topic.", review: "A recall check. Answer from memory, without reopening the lesson.", retake: "You already took a quiz today, so this is practice. It will not change your schedule." }[q.kind];
  main.innerHTML = `<div class="meta"><span class="chip">${esc(q.kind === "review" ? "Recall quiz" : q.kind === "first" ? "First quiz" : "Practice")}</span></div>
    <h1>${esc(q.topic)}</h1><p class="lead">${intro}</p><form id="quiz">${q.questions.map((qq, i) => `<fieldset class="q" style="border:1px solid var(--line)">
      <legend class="note">Question ${i + 1} of ${q.questions.length}</legend><p class="stem">${esc(qq.stem)}</p>
      ${qq.options.map((o, j) => `<label class="opt"><input type="radio" name="q${i}" value="${j}"><span>${esc(o)}</span></label>`).join("")}</fieldset>`).join("")}
    <div class="actions"><button class="btn primary" id="submit" disabled>Submit answers</button><a class="btn quiet" href="#/topic/${id}">Back to the lesson</a></div></form>`;
  renderMath(main);
  const form = $("#quiz"), submit = $("#submit");
  form.onchange = () => { submit.disabled = q.questions.some((_, i) => !form.querySelector(`input[name=q${i}]:checked`)); };
  form.onsubmit = async (e) => {
    e.preventDefault(); submit.disabled = true;
    const answers = q.questions.map((_, i) => Number(form.querySelector(`input[name=q${i}]:checked`).value));
    const res = await api(`/topics/${id}/quiz`, { method: "POST", body: { question_ids: q.questions.map(x => x.id), answers, attempt_id: attemptId } });
    await refreshSidebar(); showResult(id, res, main);
  };
}

function showResult(id, r, main) {
  const msg = r.kind === "first" ? `Starting point set: memory ${LEVEL_NAMES[r.level_after]}. The agent will bring this topic back for review from tomorrow.`
    : r.kind === "retake" ? "Saved as practice. Your schedule is unchanged."
      : r.level_after > r.level_before ? "Strong recall. The topic moved up, so it will come back later than before."
        : r.level_after < r.level_before ? "This one slipped. It moved down and will come back sooner."
          : r.n_correct >= 4 ? "Strong recall. This topic is already at its highest level." : "This one slipped, and it was already at its lowest level, so it will return soon.";
  main.innerHTML = `<h1>${esc(r.topic.title)}</h1><div class="score">${r.n_correct} / ${r.n_questions}</div><p class="lead">${esc(msg)}</p>
    ${r.review.map((q, i) => `<div class="q"><p class="stem">${i + 1}. ${esc(q.stem)}</p>${q.options.map((o, j) =>
    `<div class="opt ${j === q.correct_answer ? "right" : j === q.your_answer ? "wrong" : ""}"><span>${"ABCD"[j]}.</span><span>${esc(o)}${j === q.correct_answer ? " (correct)" : j === q.your_answer ? " (your answer)" : ""}</span></div>`).join("")}
      <div class="why-box">${esc(q.why)}</div></div>`).join("")}
    <div class="actions"><a class="btn primary" href="#/today">Back to today's recall</a><a class="btn quiet" href="#/topic/${id}">Reread the lesson</a></div>`;
  renderMath(main);
}

/* ---------- progress ---------- */
async function viewProgress(main) {
  const p = await api("/progress");
  const rows = p.topics.filter(t => t.status !== "not_started").map(t => `<tr><td>${statusRing(t, 18)} <a href="#/topic/${t.id}">${esc(t.title)}</a></td>
    <td>${esc(t.label)}</td><td>${t.status === "quizzed" ? LEVEL_NAMES[t.level] : "-"}</td><td>${esc(t.last_score || "-")}</td><td>${t.days_since_quiz ?? "-"}</td><td>${pct(t.est_recall)}</td></tr>`).join("");
  main.innerHTML = `<h1>Progress</h1><p class="lead">What you did is shown separately from what the model estimates.</p>
    <div class="stats"><div class="stat"><b>${p.read}/${p.total_topics}</b><span>topics read</span></div><div class="stat"><b>${p.quizzed}</b><span>topics quizzed</span></div>
      <div class="stat"><b>${pct(p.avg_last_score)}</b><span>average last quiz score (observed)</span></div>
      <div class="stat est"><b>${pct(p.avg_est_recall)}</b><span>average estimated recall today (estimate)</span></div>
      <div class="stat"><b>${p.recommended_done}/${p.recommended_total}</b><span>agent picks completed</span></div></div>
    ${rows ? `<h2>Your topics</h2><table class="data"><thead><tr><th>Topic</th><th>Status</th><th>Memory level</th><th>Last quiz</th><th>Days ago</th><th>Est. recall</th></tr></thead><tbody>${rows}</tbody></table>` : '<div class="panel info"><h3>Nothing yet</h3><p>Open a topic from the sidebar, read it and take its quiz.</p></div>'}
    ${p.history.length > 1 ? `<h2>Score trend</h2>${trendSvg(p.history)}<p class="note">Each point is one quiz, oldest on the left. These are observed scores.</p>` : ""}
    ${p.revision_history.length ? `<h2>Revision history</h2><table class="data"><thead><tr><th>Date</th><th>Topic</th><th>Status</th></tr></thead><tbody>${p.revision_history.map(h => `<tr><td>${esc(h.day)}</td><td>${esc(h.topic)}</td><td>${esc(h.status)}</td></tr>`).join("")}</tbody></table>` : ""}
    ${p.history.length ? `<h2>Quiz history</h2><table class="data"><thead><tr><th>Date</th><th>Time</th><th>Topic</th><th>Type</th><th>Score</th></tr></thead><tbody>${p.history.map(h => `<tr><td>${esc(h.day)}</td><td>${esc((h.at || "").slice(11, 16))}</td><td>${esc(h.topic)}</td><td>${esc(h.kind)}</td><td>${esc(h.score)}</td></tr>`).join("")}</tbody></table>` : ""}`;
}

function trendSvg(history) {
  const pts = history.slice().reverse().map(h => { const [a, b] = h.score.split("/").map(Number); return a / b; });
  const w = 560, h = 120, x = i => 20 + i * (w - 40) / Math.max(1, pts.length - 1), y = v => h - 15 - v * (h - 30);
  const line = pts.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" aria-label="Quiz score trend" style="max-width:${w}px;background:#fff;border:1px solid var(--line);border-radius:10px">
    <line x1="20" x2="${w - 20}" y1="${y(0.8)}" y2="${y(0.8)}" stroke="#DCE3E8" stroke-dasharray="4 4"/>
    <polyline points="${line}" fill="none" stroke="#2F6FA8" stroke-width="2.5"/>
    ${pts.map((v, i) => `<circle cx="${x(i)}" cy="${y(v)}" r="3.5" fill="#3F8F5F"/>`).join("")}</svg>`;
}

/* ---------- settings ---------- */
async function viewSettings(main) {
  main.innerHTML = `<h1>Settings</h1>
    <div class="panel"><h3>Daily review budget</h3><p class="note">How many topics the agent may ask you to review each day.</p>
      <div class="slider-row"><input type="range" id="k" min="1" max="15" value="${user.K}"><b id="kv">${user.K}</b><button class="btn primary" id="save-k">Save</button></div></div>
    <div class="panel"><h3>Simulated clock</h3><p class="note">Spaced repetition runs over days. To try it without waiting, move this account's clock forward. Today is ${esc(user.today)}.</p>
      <button class="btn quiet" data-adv="1">Advance 1 day</button> <button class="btn quiet" data-adv="7">Advance 7 days</button></div>
    <div class="panel"><h3>About your account</h3><p class="note">Learner speed estimate: ${user.speed_estimate} (1.0 is typical; below 1 means you forget faster). It starts at 1.0 and updates as you take review quizzes.${user.is_simulated ? " This account holds simulated data." : ""}</p>
      <button class="btn quiet" id="logout">Switch user</button></div>`;
  $("#k").oninput = e => $("#kv").textContent = e.target.value;
  $("#save-k").onclick = async () => { user = await api("/settings", { method: "POST", body: { K: Number($("#k").value) } }); $("#save-k").textContent = "Saved"; };
  document.querySelectorAll("[data-adv]").forEach(b => b.onclick = async () => { await api("/dev/advance_day", { method: "POST", body: { days: Number(b.dataset.adv) } }); await refreshSidebar(); viewSettings(main); });
  $("#logout").onclick = async () => { try { await api("/logout", { method: "POST" }); } catch (e) { } localStorage.removeItem("mlrecall_token"); location.hash = ""; location.reload(); };
}

boot();