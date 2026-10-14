// Karen's 3D interface: wires the Three.js scene and the glass HUD to the
// local Python bridge (Server-Sent Events in, small JSON POSTs out).
import { KarenScene } from './scene.js';

const token = new URLSearchParams(location.search).get('t') || '';
const $ = (id) => document.getElementById(id);
const MODE_COLORS = {
  booting: '79, 216, 255', ready: '79, 216, 255', listening: '255, 181, 71', thinking: '155, 140, 255',
  searching: '61, 220, 176', speaking: '98, 227, 255', error: '255, 92, 122',
};
const MODE_LABELS = {
  booting: 'INITIALIZING', ready: 'STANDING BY', listening: 'LISTENING', thinking: 'THINKING',
  searching: 'SEARCHING THE WEB', speaking: 'SPEAKING', error: 'ATTENTION',
};
const BOOT_MIN_MS = 2800;
const startedAt = performance.now();

const state = {
  mode: 'booting', ready: false, booted: false, busy: false,
  reply: '', cpu: Array(60).fill(0), voice: false, wake: false, voiceAvailable: false,
};

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const params = new URLSearchParams(location.search);
let scene = null;
try {
  // Diagnostics: ?scene=0 disables 3D, ?fps=N caps the idle frame rate.
  if (params.get('scene') !== '0') {
    scene = new KarenScene($('scene'), { reducedMotion, idleFps: Number(params.get('fps')) || 24 });
    scene.start();
  }
} catch (error) {
  console.error('3D scene unavailable', error);  // The HUD still works without WebGL.
}

// -- bridge ---------------------------------------------------------------

async function post(action, body = {}) {
  try {
    const response = await fetch(`/api/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Karen-Token': token },
      body: JSON.stringify(body),
    });
    return await response.json();
  } catch {
    return { ok: false };
  }
}

const events = new EventSource(`/events?t=${encodeURIComponent(token)}`);
const on = (name, handler) => events.addEventListener(name, (event) => handler(JSON.parse(event.data || '{}')));

on('snapshot', (data) => {
  $('boot-log').innerHTML = '';
  (data.boot || []).forEach(addBootLine);
  $('history-list').innerHTML = '';
  (data.history || []).forEach(addHistory);
  setModel(data.model);
  setReminder(data.reminder);
  setToggles(data);
  if (data.reduced_motion) document.body.classList.add('reduced-motion');
  if (data.ready) {
    state.ready = true;
    const last = [...(data.history || [])].reverse().find((entry) => entry.speaker === 'KAREN');
    finishBoot(last ? last.text : '');
  }
  setMode(data.mode || 'booting');
});
on('boot', (data) => addBootLine(data.line));
on('ready', (data) => { state.ready = true; setToggles(data); finishBoot(data.greeting); });
on('mode', (data) => setMode(data.mode));
on('heard', (data) => { showHeard(data.text); state.reply = ''; renderReply('', true); });
on('chunk', (data) => { state.reply += data.text; renderReply(state.reply); });
on('reply', (data) => { renderReply(data.text, !data.text.startsWith(stripSources(state.reply).body)); state.reply = ''; });
on('log', addHistory);
on('toast', (data) => toast(data.text, data.seconds));
on('telemetry', setTelemetry);
on('level', (data) => scene && scene.setLevel(data.value));
on('model', (data) => setModel(data.status));
on('toggles', setToggles);
on('reminder', (data) => setReminder(data.next));

// -- boot -----------------------------------------------------------------

function addBootLine(line) {
  const item = document.createElement('li');
  item.textContent = line;
  $('boot-log').appendChild(item);
  const count = $('boot-log').children.length;
  $('boot-bar').firstElementChild.style.width = `${Math.min(100, count * 18)}%`;
}

function finishBoot(greeting) {
  if (state.booted) return;
  const wait = Math.max(0, BOOT_MIN_MS - (performance.now() - startedAt));
  setTimeout(() => {
    state.booted = true;
    $('boot-bar').firstElementChild.style.width = '100%';
    document.body.classList.add('booted');
    for (const id of ['input', 'send']) $(id).disabled = false;
    if (greeting) renderReply(greeting, true);
    $('input').focus();
  }, wait);
}

// -- mode & captions ------------------------------------------------------

function setMode(mode) {
  state.mode = mode;
  state.busy = ['thinking', 'searching', 'listening'].includes(mode);
  document.body.dataset.mode = mode;
  document.documentElement.style.setProperty('--accent-rgb', MODE_COLORS[mode] || MODE_COLORS.ready);
  document.documentElement.style.setProperty('--accent', `rgb(${MODE_COLORS[mode] || MODE_COLORS.ready})`);
  $('status-text').textContent = MODE_LABELS[mode] || mode.toUpperCase();
  $('f-state').textContent = MODE_LABELS[mode] || mode.toUpperCase();
  $('talk').disabled = state.busy || !state.ready;
  $('send').disabled = state.busy || !state.ready;
  if (scene) scene.setMode(mode);
  if (mode === 'listening') renderReply('Listening. Go ahead.', true);
}

function showHeard(text) {
  $('heard').textContent = text;
  $('heard').classList.toggle('show', Boolean(text));
}

function stripSources(text) {
  const match = text.match(/\n\s*Sources?:\s*(.+)$/s);
  return match ? { body: text.slice(0, match.index), sources: match[1].trim() } : { body: text, sources: '' };
}

let renderedText = '';
function renderReply(text, reset = false) {
  const { body, sources } = stripSources(text || '');
  const reply = $('reply');
  if (reset || !body.startsWith(renderedText)) {
    reply.innerHTML = '';
    renderedText = '';
  }
  // Append only the new part as fading words (smooth with streaming answers).
  const addition = body.slice(renderedText.length);
  const tokens = addition.split(/(\s+)/);
  let delay = 0;
  for (const piece of tokens) {
    if (!piece) continue;
    if (/^\s+$/.test(piece)) { reply.appendChild(document.createTextNode(piece.includes('\n') ? '\n' : ' ')); continue; }
    const word = document.createElement('span');
    word.className = 'word';
    word.textContent = piece;
    word.style.animationDelay = `${Math.min(delay, 900)}ms`;
    delay += reset ? 22 : 8;
    reply.appendChild(word);
  }
  renderedText = body;
  reply.style.whiteSpace = 'pre-line';
  $('sources').textContent = sources ? `SOURCES · ${sources.toUpperCase()}` : '';
  $('sources').classList.toggle('show', Boolean(sources));
}

// -- panels ---------------------------------------------------------------

function setTelemetry({ cpu, ram }) {
  $('cpu-value').textContent = `${Math.round(cpu)}%`;
  $('ram-value').textContent = `${Math.round(ram)}%`;
  $('cpu-bar').style.transform = `scaleX(${cpu / 100})`;
  $('ram-bar').style.transform = `scaleX(${ram / 100})`;
  state.cpu.push(cpu);
  state.cpu.shift();
  drawSpark();
}

function drawSpark() {
  const canvas = $('spark');
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  const accent = MODE_COLORS[state.mode] || MODE_COLORS.ready;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  for (let y = 0; y <= height; y += height / 4) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
  const points = state.cpu.map((value, index) => [index * (width / (state.cpu.length - 1)), height - 4 - (value / 100) * (height - 8)]);
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, `rgba(${accent}, 0.35)`);
  gradient.addColorStop(1, `rgba(${accent}, 0)`);
  ctx.beginPath();
  points.forEach(([x, y], index) => (index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.lineTo(width, height); ctx.lineTo(0, height); ctx.closePath();
  ctx.fillStyle = gradient; ctx.fill();
  ctx.beginPath();
  points.forEach(([x, y], index) => (index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.strokeStyle = `rgb(${accent})`; ctx.lineWidth = 1.6; ctx.shadowColor = `rgb(${accent})`; ctx.shadowBlur = 8;
  ctx.stroke(); ctx.shadowBlur = 0;
}

function setModel(status) { if (status) $('model-status').textContent = status; }
function setReminder(next) { $('f-reminder').textContent = next || 'NONE'; }

function setToggles(data) {
  if ('voice' in data) state.voice = data.voice;
  if ('wake' in data) state.wake = data.wake;
  if ('voice_available' in data) state.voiceAvailable = data.voice_available;
  $('t-voice').classList.toggle('on', state.voice);
  $('t-voice').querySelector('b').textContent = state.voice ? 'ON' : 'OFF';
  $('t-wake').classList.toggle('on', state.wake);
  $('t-wake').querySelector('b').textContent = state.wake ? 'ON' : 'OFF';
  $('f-voice').textContent = state.voice ? 'ENABLED' : state.voiceAvailable ? 'MUTED' : 'UNAVAILABLE';
  $('f-wake').textContent = state.wake ? "LISTENING FOR 'HEY JARVIS'" : 'OFF';
}

function addHistory(entry) {
  const item = document.createElement('div');
  item.className = `entry ${entry.speaker === 'YOU' ? 'you' : ''}`;
  const who = document.createElement('div');
  who.className = 'who';
  who.textContent = `${entry.speaker} · ${entry.time || ''}`;
  const text = document.createElement('div');
  text.className = 'text';
  text.textContent = entry.text;
  item.append(who, text);
  const list = $('history-list');
  list.appendChild(item);
  while (list.children.length > 200) list.firstChild.remove();
  list.scrollTop = list.scrollHeight;
}

function toast(text, seconds = 4) {
  const element = document.createElement('div');
  element.className = 'toast';
  element.textContent = text;
  $('toasts').appendChild(element);
  setTimeout(() => { element.classList.add('out'); setTimeout(() => element.remove(), 450); }, seconds * 1000);
}

function tickClock() {
  const now = new Date();
  $('clock-time').textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
  $('clock-date').textContent = now.toLocaleDateString([], { weekday: 'long', day: '2-digit', month: 'long', year: 'numeric' }).toUpperCase();
  const seconds = Math.floor((performance.now() - startedAt) / 1000);
  const pad = (value) => String(value).padStart(2, '0');
  $('f-uptime').textContent = `${pad(Math.floor(seconds / 3600))}:${pad(Math.floor((seconds % 3600) / 60))}:${pad(seconds % 60)}`;
}
tickClock();
setInterval(tickClock, 1000);

// -- input ----------------------------------------------------------------

$('composer').addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = $('input').value.trim();
  if (!text || state.busy || !state.ready) return;
  $('input').value = '';
  await post('send', { text });
});
$('talk').addEventListener('click', () => post('talk'));
$('stop').addEventListener('click', () => post('stop'));
document.querySelectorAll('[data-toggle]').forEach((button) => {
  button.addEventListener('click', () => post('toggle', { name: button.dataset.toggle }));
});
$('t-history').addEventListener('click', () => toggleHistory());

function toggleHistory(force) {
  const open = force ?? !document.body.classList.contains('history-open');
  document.body.classList.toggle('history-open', open);
  $('t-history').classList.toggle('on', open);
  if (open) $('history-list').scrollTop = $('history-list').scrollHeight;
}

window.addEventListener('keydown', (event) => {
  if (event.ctrlKey && event.code === 'Space') { event.preventDefault(); if (!state.busy) post('talk'); }
  else if (event.ctrlKey && event.key.toLowerCase() === 'h') { event.preventDefault(); toggleHistory(); }
  else if (event.ctrlKey && event.key.toLowerCase() === 'q') { event.preventDefault(); post('quit').finally(() => window.close()); }
  else if (event.key === 'Escape') {
    if (document.body.classList.contains('history-open')) toggleHistory(false);
    else post('stop');
  } else if (!event.ctrlKey && !event.altKey && event.key.length === 1 && document.activeElement !== $('input') && state.booted) {
    $('input').focus();  // Start typing anywhere.
  }
});
