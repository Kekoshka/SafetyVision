// SafetyVision — веб-интерфейс. Без внешних библиотек: работает в закрытом контуре.
// Все запросы относительные (api/...), поэтому интерфейс работает и за прокси (Jupyter и т.п.).
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(path + ': ' + r.status);
  return r.json();
};
const fmtTime = (ms) => new Date(ms).toLocaleString('ru-RU');
const fmtDur = (e) => (e.ts_end ? ((e.ts_end - e.ts_start) / 1000).toFixed(0) + ' с' : 'идёт');
const STATUS = { new: 'Новое', confirmed: 'Подтверждено', false: 'Ложное', forwarded: 'Передано мастеру' };
const SEV = { critical: 'Критично', high: 'Высокая', warning: 'Предупреждение' };

let META = { types: {}, zone_kinds: {} };
let CAMS = [];
let lastId = 0;
let feed = [];

// ------------------------------------------------------------- вкладки
document.querySelectorAll('nav button').forEach((b) =>
  b.addEventListener('click', () => {
    document.querySelectorAll('nav button').forEach((x) => x.classList.toggle('active', x === b));
    document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === 'tab-' + b.dataset.tab));
    if (b.dataset.tab === 'journal') loadJournal();
    if (b.dataset.tab === 'stats') loadStats();
    if (b.dataset.tab === 'perf') loadPerf();
    if (b.dataset.tab === 'zones') zonesOpen();
  })
);

// ------------------------------------------------------------- звук
let audioCtx;
function beep(critical) {
  if (!$('#soundOn').checked) return;
  try {
    audioCtx = audioCtx || new AudioContext();
    const n = critical ? 3 : 1;
    for (let i = 0; i < n; i++) {
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.frequency.value = critical ? 880 : 660;
      o.connect(g); g.connect(audioCtx.destination);
      const t = audioCtx.currentTime + i * 0.25;
      g.gain.setValueAtTime(0.15, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
      o.start(t); o.stop(t + 0.2);
    }
  } catch (e) { /* браузер запретил звук до первого клика */ }
}

// ------------------------------------------------------------- мониторинг
function buildGrid() {
  $('#camGrid').innerHTML = CAMS.map((c) => `
    <div class="cam" id="cam-${esc(c.id)}">
      <img data-cam="${esc(c.id)}" alt="${esc(c.name)}">
      <div class="cap"><b>${esc(c.name)}</b><span class="st" id="st-${esc(c.id)}"></span></div>
    </div>`).join('') || '<p class="muted">Камеры не настроены (config.json → cameras)</p>';
  document.querySelectorAll('#camGrid img').forEach(liveLoop);
}

function liveLoop(img) {
  const cam = img.dataset.cam;
  const next = () => setTimeout(() => { img.src = `api/cameras/${cam}/live.jpg?t=${Date.now()}`; }, 120);
  img.onload = next;
  img.onerror = () => setTimeout(next, 1000);
  next();
}

async function refreshCams() {
  try {
    CAMS = await api('api/cameras');
    for (const c of CAMS) {
      const st = $(`#st-${c.id}`);
      if (st) st.textContent = c.state === 'running' ? `${c.fps.toFixed(1)} FPS · ${Math.round(c.latency_ms)} мс · людей ${c.persons}` : c.state + (c.error ? ': ' + c.error : '');
      const box = $(`#cam-${c.id}`);
      if (box) box.classList.toggle('alarm', (c.active || []).some((a) => (META.types[a.type] || {}).severity === 'critical'));
    }
    if ($('#tab-perf').classList.contains('active')) renderPerf();
  } catch (e) { console.warn(e); }
}

function evCard(e, fresh) {
  return `<div class="ev sev-${e.severity} ${fresh ? 'fresh' : ''}" data-id="${e.id}">
    <img src="api/events/${e.id}/snapshot.jpg" loading="lazy">
    <div><div class="t">${esc(e.title)}</div>
      <div class="s">${esc(e.camera_name)}${e.zone ? ' · ' + esc(e.zone) : ''}</div>
      <div class="s">${fmtTime(e.ts_alert)} · <span class="badge ${e.status}">${STATUS[e.status] || e.status}</span></div></div>
  </div>`;
}

function renderFeed(freshIds = new Set()) {
  $('#feed').innerHTML = feed.map((e) => evCard(e, freshIds.has(e.id))).join('') || '<p class="muted">Пока нарушений нет</p>';
}

async function pollEvents() {
  try {
    const evs = await api(`api/events?since_id=${lastId}&limit=50`);
    if (evs.length) {
      const fresh = new Set(evs.map((e) => e.id));
      const firstLoad = lastId === 0;
      lastId = Math.max(lastId, ...evs.map((e) => e.id));
      feed = [...evs, ...feed].slice(0, 40);
      renderFeed(firstLoad ? new Set() : fresh);
      if (!firstLoad) beep(evs.some((e) => e.severity === 'critical'));
    }
    const nw = await api('api/events?status=new&limit=10000');
    const pill = $('#newCount');
    pill.textContent = `${nw.length} новых`;
    pill.classList.toggle('hot', nw.some((e) => e.severity === 'critical'));
  } catch (e) { console.warn(e); }
}

document.addEventListener('click', (ev) => {
  const card = ev.target.closest('[data-id]');
  if (card && !ev.target.closest('.modal')) openEvent(+card.dataset.id);
});

// ------------------------------------------------------------- карточка события
let curEvent = null, clipTimer = null;
async function openEvent(id) {
  curEvent = await api(`api/events/${id}`);
  const e = curEvent;
  $('#mTitle').textContent = e.title;
  $('#mMeta').innerHTML = [
    ['Камера', e.camera_name], ['Время', fmtTime(e.ts_alert)], ['Начало', fmtTime(e.ts_start)],
    ['Длительность', fmtDur(e)], ['Важность', SEV[e.severity] || e.severity], ['Зона', e.zone || '—'],
    ['Детали', e.detail || '—'], ['Трек', e.track_id >= 0 ? '#' + e.track_id : '—'],
    ['Статус', `<span class="badge ${e.status}">${STATUS[e.status]}</span>`], ['Событие', '#' + e.id],
  ].map(([k, v]) => `<div><b>${k}</b>${k === 'Статус' ? v : esc(v)}</div>`).join('');
  $('#mComment').value = e.comment || '';
  showSnap();
  $('#modal').classList.remove('hidden');
}
function showSnap() { clearInterval(clipTimer); $('#mImg').src = `api/events/${curEvent.id}/snapshot.jpg`; }
function showClip() {
  clearInterval(clipTimer);
  let k = 0;
  const n = Math.max(1, curEvent.clip_frames);
  const step = () => { $('#mImg').src = `api/events/${curEvent.id}/clip_${String(k).padStart(2, '0')}.jpg`; k = (k + 1) % n; };
  step();
  clipTimer = setInterval(step, 250);
}
$('#mSnap').onclick = showSnap;
$('#mClip').onclick = showClip;
$('#mClose').onclick = () => { clearInterval(clipTimer); $('#modal').classList.add('hidden'); };
$('#modal').addEventListener('click', (e) => { if (e.target.id === 'modal') $('#mClose').onclick(); });
document.querySelectorAll('.info [data-st]').forEach((b) =>
  b.addEventListener('click', async () => {
    await api(`api/events/${curEvent.id}/status`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: b.dataset.st, comment: $('#mComment').value }),
    });
    const upd = await api(`api/events/${curEvent.id}`);
    feed = feed.map((x) => (x.id === upd.id ? upd : x));
    renderFeed();
    if ($('#tab-journal').classList.contains('active')) loadJournal();
    openEvent(upd.id);
  })
);

// ------------------------------------------------------------- журнал
function journalQuery() {
  const p = new URLSearchParams();
  const v = (id) => $(id).value;
  if (v('#fType')) p.set('type', v('#fType'));
  if (v('#fCamera')) p.set('camera', v('#fCamera'));
  if (v('#fSeverity')) p.set('severity', v('#fSeverity'));
  if (v('#fStatus')) p.set('status', v('#fStatus'));
  if (v('#fZone')) p.set('zone', v('#fZone'));
  if (v('#fFrom')) p.set('from', new Date(v('#fFrom')).getTime());
  if (v('#fTo')) p.set('to', new Date(v('#fTo')).getTime());
  return p;
}
async function loadJournal() {
  const p = journalQuery();
  $('#csvLink').href = 'api/export.csv?' + p.toString();
  p.set('limit', 500);
  const evs = await api('api/events?' + p.toString());
  $('#journalBody').innerHTML = evs.map((e) => `<tr data-id="${e.id}">
    <td><img src="api/events/${e.id}/snapshot.jpg" loading="lazy"></td>
    <td>${fmtTime(e.ts_alert)}</td><td>${esc(e.camera_name)}</td>
    <td><span class="sevdot ${e.severity}"></span>${esc(e.title)}</td>
    <td>${esc(e.zone || '')}${e.zone && e.detail ? ' · ' : ''}${esc(e.detail || '')}</td>
    <td>${fmtDur(e)}</td><td><span class="badge ${e.status}">${STATUS[e.status]}</span></td></tr>`).join('')
    || '<tr><td colspan="7" class="muted">Нет событий по фильтру</td></tr>';
}
$('#fApply').onclick = loadJournal;

// ------------------------------------------------------------- зоны
const Z = { cam: null, zones: [], pts: [], img: new Image() };
function zonesOpen() {
  const sel = $('#zCamera');
  if (!sel.options.length) {
    sel.innerHTML = CAMS.map((c) => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join('');
    sel.onchange = zonesLoad;
  }
  zonesLoad();
}
function zonesLoad() {
  Z.cam = CAMS.find((c) => c.id === $('#zCamera').value) || CAMS[0];
  if (!Z.cam) return;
  Z.zones = JSON.parse(JSON.stringify(Z.cam.zones || []));
  Z.pts = [];
  Z.img.onload = zDraw;
  Z.img.src = `api/cameras/${Z.cam.id}/frame.jpg?t=${Date.now()}`;
  zList();
}
$('#zReload').onclick = () => { Z.img.src = `api/cameras/${Z.cam.id}/frame.jpg?t=${Date.now()}`; };
const ZCOL = { danger: '#ff4d4f', fence: '#ff9f1a', under_load: '#d14dff' };
function zDraw() {
  const c = $('#zCanvas');
  c.width = Z.img.naturalWidth || 1280;
  c.height = Z.img.naturalHeight || 720;
  const g = c.getContext('2d');
  g.drawImage(Z.img, 0, 0);
  const poly = (pts, col, closed) => {
    if (!pts.length) return;
    g.beginPath();
    pts.forEach(([x, y], i) => (i ? g.lineTo(x * c.width, y * c.height) : g.moveTo(x * c.width, y * c.height)));
    if (closed) { g.closePath(); g.fillStyle = col + '40'; g.fill(); }
    g.strokeStyle = col; g.lineWidth = Math.max(2, c.width / 500); g.stroke();
    pts.forEach(([x, y]) => { g.fillStyle = col; g.fillRect(x * c.width - 4, y * c.height - 4, 8, 8); });
  };
  Z.zones.forEach((z, i) => {
    poly(z.points, ZCOL[z.kind] || '#f5d547', true);
    if (z.points.length) {
      g.font = `${Math.max(14, c.width / 60)}px sans-serif`; g.fillStyle = '#fff';
      g.fillText(`Z${i + 1} ${z.name}`, z.points[0][0] * c.width + 6, z.points[0][1] * c.height - 6);
    }
  });
  poly(Z.pts, '#7cc242', false);
}
$('#zCanvas').addEventListener('click', (e) => {
  const r = e.target.getBoundingClientRect();
  Z.pts.push([+((e.clientX - r.left) / r.width).toFixed(4), +((e.clientY - r.top) / r.height).toFixed(4)]);
  zDraw();
});
$('#zUndo').onclick = () => { Z.pts.pop(); zDraw(); };
$('#zClose').onclick = () => {
  if (Z.pts.length < 3) { $('#zMsg').textContent = 'Нужно минимум 3 точки'; return; }
  const z = { id: 'z' + Date.now().toString(36), name: $('#zName').value || 'Зона ' + (Z.zones.length + 1), kind: $('#zKind').value, points: Z.pts };
  if ($('#zFrom').value) z.active_from = $('#zFrom').value;
  if ($('#zTo').value) z.active_to = $('#zTo').value;
  Z.zones.push(z);
  Z.pts = [];
  $('#zName').value = '';
  zList(); zDraw();
  $('#zMsg').textContent = 'Зона добавлена — не забудьте сохранить';
};
function zList() {
  $('#zList').innerHTML = Z.zones.map((z, i) => `<div class="zitem"><span>Z${i + 1} · ${esc(z.name)} · ${esc(META.zone_kinds[z.kind] || z.kind)}${z.active_from || z.active_to ? ` · ${z.active_from || '00:00'}–${z.active_to || '24:00'}` : ''}</span>
    <button data-del="${i}">✕</button></div>`).join('') || '<p class="muted">Зон нет</p>';
  document.querySelectorAll('[data-del]').forEach((b) => (b.onclick = () => { Z.zones.splice(+b.dataset.del, 1); zList(); zDraw(); }));
}
$('#zSave').onclick = async () => {
  try {
    await api(`api/cameras/${Z.cam.id}/zones`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ zones: Z.zones }) });
    $('#zMsg').textContent = 'Сохранено, правила уже применяются';
    await refreshCams();
  } catch (e) { $('#zMsg').textContent = 'Ошибка: ' + e.message; }
};

// ------------------------------------------------------------- статистика
function bars(el, obj, labelOf = (k) => k) {
  const entries = Object.entries(obj).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map((e) => e[1]));
  $(el).innerHTML = entries.map(([k, v]) => `<div class="bar"><span>${esc(labelOf(k))}</span><div class="track"><div class="fill" style="width:${(100 * v) / max}%"></div></div><b>${v}</b></div>`).join('') || '<p class="muted">Нет данных</p>';
}
async function loadStats() {
  const s = await api('api/stats');
  const title = (k) => (s.types[k] || {}).title || k;
  let conf = 0, fals = 0;
  Object.values(s.precision).forEach((p) => { conf += p.confirmed; fals += p.false; });
  const todayLocal = Object.entries(s.by_day).find(([d]) => d === new Date().toLocaleDateString('sv-SE'));
  $('#statCards').innerHTML = [
    [s.total, 'Всего нарушений'],
    [todayLocal ? todayLocal[1] : 0, 'Сегодня'],
    [conf + fals ? Math.round((100 * conf) / (conf + fals)) + '%' : '—', `Точность по отметкам (${conf + fals} оценено)`],
    [(s.mean_alert_delay_ms / 1000).toFixed(1) + ' с', 'Среднее время до оповещения'],
  ].map(([v, l]) => `<div class="card"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
  bars('#chType', s.by_type, title);
  bars('#chCam', s.by_camera);
  bars('#chDay', s.by_day);
  const mh = Math.max(1, ...s.by_hour);
  $('#chHour').innerHTML = s.by_hour.map((v, h) => `<div class="col" title="${v}"><div style="height:${(100 * v) / mh}%"></div>${h}</div>`).join('');
  $('#chPrec').innerHTML = Object.entries(s.precision).map(([k, p]) => `<div class="bar"><span>${esc(title(k))}</span><div class="track"><div class="fill" style="width:${100 * p.precision}%"></div></div><b>${Math.round(100 * p.precision)}%</b></div>`).join('') || '<p class="muted">Операторы ещё не отмечали события</p>';
}

// ------------------------------------------------------------- производительность
function renderPerf() {
  $('#perfBody').innerHTML = CAMS.map((c) => `<tr><td>${esc(c.name)}</td><td>${esc(c.state)}</td>
    <td>${c.fps.toFixed(1)} / ${c.process_fps_target}</td><td>${Math.round(c.latency_ms)}</td>
    <td>${Object.entries(c.models).map(([k, m]) => `${esc(k)} (${esc(m.backend)}): ${m.pre_ms.toFixed(1)} / <b>${m.infer_ms.toFixed(1)}</b> / ${m.post_ms.toFixed(1)}${m.net_ms > 0.5 ? ` · сеть ${m.net_ms.toFixed(0)}` : ''}${m.tpu_ms > 0.05 ? ` · <b class="tpu">TPU ${m.tpu_ms.toFixed(1)}</b>` : ''}`).join('<br>')}</td>
    <td>${c.rules_ms.toFixed(2)}</td><td>${c.persons} / ${c.vehicles} / ${c.objects}</td></tr>`).join('');
}
async function loadPerf() {
  renderPerf();
  const b = await api('api/bench');
  $('#benchBody').innerHTML = b.map((r) => `<tr><td>${esc(r.model_name)}</td><td>${esc(r.backend)}</td><td>${r.streams}</td>
    <td>${r.infer_ms.mean.toFixed(1)} / ${r.infer_ms.p95.toFixed(1)}</td>
    <td>${r.server ? `${r.server.mean_ms.toFixed(1)} / ${r.server.p95_ms.toFixed(1)} (${esc(r.server.runtime)})` : '—'}</td>
    <td>${r.server && r.server.tpu_ms_mean > 0 ? `<b class="tpu">${r.server.tpu_ms_mean.toFixed(1)}</b>` : (r.tpu_ms && r.tpu_ms.mean > 0 ? `<b class="tpu">${r.tpu_ms.mean.toFixed(1)}</b>` : '—')}</td>
    <td>${r.net_ms ? r.net_ms.mean.toFixed(1) : '—'}</td><td>${r.total_ms.mean.toFixed(1)}</td>
    <td><b>${r.throughput_fps.toFixed(1)}</b></td><td>${esc(r.time)}</td></tr>`).join('')
    || '<tr><td colspan="10" class="muted">Запустите: ./safety_service bench --model pose --image кадр.jpg</td></tr>';
}

// ------------------------------------------------------------- старт
(async function init() {
  META = await api('api/meta');
  $('#fType').innerHTML += Object.entries(META.types).map(([k, v]) => `<option value="${k}">${esc(v.title)}</option>`).join('');
  $('#zKind').innerHTML = Object.entries(META.zone_kinds).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join('');
  CAMS = await api('api/cameras');
  $('#fCamera').innerHTML += CAMS.map((c) => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join('');
  buildGrid();
  refreshCams();
  setInterval(refreshCams, 2000);
  pollEvents();
  setInterval(pollEvents, 1000);
})();
