"""Build a self-contained HTML audio annotator for TTS cache clips.

Input : one or more .m4a files.
Output: a single standalone .html file that embeds every clip (audio +
        pre-computed waveform peaks) and provides:
          - waveform visualisation with a time ruler
          - play / pause, click-to-seek, a synced playhead
          - drag-select a region → label it (noise / silence / …)
          - an export box with all annotations to paste back for analysis

The waveform peaks are computed here in Python (afconvert → samples) and
embedded as JSON, so the page never depends on the browser decoding AAC
for display — playback uses the native <audio> element.

Usage (from server/):
    python3 scripts/audio_annotator.py data/tts_cache/<hash>.m4a ...
    python3 scripts/audio_annotator.py --out scripts/output/annotate.html <files...>
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import soundfile as sf  # noqa: E402

PX_PER_SEC = 12          # waveform-column resolution
MIN_COLS, MAX_COLS = 600, 4000


def decode(path: Path) -> tuple[np.ndarray, int]:
    """m4a → mono float32 samples via afconvert (auto-detects container)."""
    with tempfile.TemporaryDirectory(prefix="annot-") as td:
        wav = Path(td) / "o.wav"
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16", str(path), str(wav)],
            check=True, capture_output=True, timeout=60,
        )
        data, sr = sf.read(wav, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def compute_peaks(y: np.ndarray, n_cols: int) -> list[list[float]]:
    """Per-column [min, max] of the waveform — the standard peak view."""
    edges = np.linspace(0, len(y), n_cols + 1).astype(int)
    peaks: list[list[float]] = []
    for i in range(n_cols):
        seg = y[edges[i]:edges[i + 1]]
        if seg.size:
            peaks.append([round(float(seg.min()), 3), round(float(seg.max()), 3)])
        else:
            peaks.append([0.0, 0.0])
    return peaks


def build_clip(path: Path) -> dict:
    y, sr = decode(path)
    dur = len(y) / sr if sr else 0.0
    n_cols = min(MAX_COLS, max(MIN_COLS, int(dur * PX_PER_SEC)))
    peaks = compute_peaks(y, n_cols) if len(y) else [[0.0, 0.0]]
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "name": path.name,
        "duration": round(dur, 3),
        "peaks": peaks,
        "audio": "data:audio/mp4;base64," + b64,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help=".m4a clip path(s)")
    ap.add_argument("--out", default="scripts/output/audio_annotator.html")
    args = ap.parse_args()

    clips = []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"skip (not found): {p}")
            continue
        try:
            clips.append(build_clip(p))
            print(f"  embedded {p.name}  ({clips[-1]['duration']:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {p.name}: {exc}")
    if not clips:
        raise SystemExit("no clips embedded")

    html = HTML_TEMPLATE.replace("/*__CLIPS__*/", json.dumps(clips))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    mb = out.stat().st_size / 1e6
    print(f"\nwrote {out}  ({len(clips)} clip(s), {mb:.1f} MB)")
    print(f"open it in a browser:  open {out}")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>TTS 音频标注</title>
<style>
  body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:18px;
       background:#1c1f24;color:#e6e6e6;}
  h2{margin:0 0 12px;font-size:17px;}
  .row{margin:10px 0;display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
  select,button{font-size:13px;padding:4px 8px;background:#2c313a;color:#e6e6e6;
       border:1px solid #444;border-radius:5px;cursor:pointer;}
  button:hover{background:#3a4150;}
  button:disabled{opacity:.4;cursor:default;}
  canvas{background:#12141a;border:1px solid #333;border-radius:6px;
       display:block;width:100%;cursor:crosshair;}
  #meta{color:#8aa0c0;font-size:13px;}
  #cur,#selInfo{font-variant-numeric:tabular-nums;font-size:13px;color:#9fb;}
  table{border-collapse:collapse;margin-top:8px;font-size:13px;}
  th,td{border:1px solid #3a3f48;padding:3px 8px;text-align:left;}
  th{background:#2c313a;}
  textarea{width:100%;box-sizing:border-box;background:#12141a;color:#cde;
       border:1px solid #444;border-radius:6px;font-family:ui-monospace,monospace;
       font-size:12px;padding:8px;}
  .hint{color:#7d8794;font-size:12px;margin:4px 0;}
  .tag-noise{color:#ff8a7a;} .tag-silence{color:#7ab8ff;}
  .tag-garbled{color:#ffd166;} .tag-other{color:#c0c0c0;}
</style>
</head>
<body>
<h2>TTS 音频缺陷标注</h2>

<div class="row">
  <label>音频：</label>
  <select id="clipSel"></select>
  <span id="meta"></span>
</div>

<canvas id="wave" height="280"></canvas>

<div class="row">
  <button id="play">▶ 播放</button>
  <button id="playSel" disabled>▶ 播放选区</button>
  <span id="cur">0.00 / 0.00 s</span>
  <span id="selInfo"></span>
</div>
<div class="row">
  <label>标签：</label>
  <select id="label">
    <option value="noise">noise 杂音</option>
    <option value="silence">silence 静音</option>
    <option value="garbled">garbled 乱码</option>
    <option value="other">other 其它</option>
  </select>
  <button id="addAnn" disabled>+ 把选区加为标注</button>
  <span class="hint">拖动波形选择区间 · 单击波形跳转 · 空格播放/暂停</span>
</div>

<audio id="audio"></audio>

<table id="annTable">
  <thead><tr><th>#</th><th>起 (s)</th><th>止 (s)</th><th>时长</th>
  <th>标签</th><th>试听</th><th>删除</th></tr></thead>
  <tbody></tbody>
</table>

<h3 style="margin-bottom:4px;">标注结果（复制粘贴给 Claude）</h3>
<div class="row" style="margin-top:0;">
  <button id="copy">复制全部</button>
  <span id="copied" class="hint"></span>
</div>
<textarea id="export" rows="12" readonly></textarea>

<script>
const CLIPS = /*__CLIPS__*/;
const LABEL_COLORS = {noise:'#ff8a7a',silence:'#7ab8ff',garbled:'#ffd166',other:'#c0c0c0'};

const cv=document.getElementById('wave'), ctx=cv.getContext('2d');
const audio=document.getElementById('audio');
const clipSel=document.getElementById('clipSel');
const labelSel=document.getElementById('label');
const exportBox=document.getElementById('export');

// annotations keyed by clip index → [{start,end,label}]
const anns={};
let ci=0;                 // current clip index
let sel=null;             // {t0,t1} pending selection in seconds
let drag=null;            // {x0} during a drag
let playSelEnd=null;      // when set, pause playback at this time

CLIPS.forEach((c,i)=>{
  anns[i]=[];
  const o=document.createElement('option');
  o.value=i; o.textContent=`${i+1}. ${c.name}  (${c.duration.toFixed(1)}s)`;
  clipSel.appendChild(o);
});

function clip(){return CLIPS[ci];}
function dur(){return clip().duration;}
function W(){return cv.width;}
function x2t(x){return Math.max(0,Math.min(dur(), x/W()*dur()));}
function t2x(t){return t/dur()*W();}

function sizeCanvas(){
  // fixed display width = container width; backing store matches for crispness
  const w=Math.max(700, document.body.clientWidth-36);
  cv.width=w; cv.style.width=w+'px';
}

function tickStep(d){
  for(const s of [0.5,1,2,5,10,20,30,60]) if(d/s<=14) return s;
  return 120;
}

function draw(){
  const w=W(), h=cv.height, ruler=46, wf=h-ruler, mid=wf/2;
  ctx.clearRect(0,0,w,h);
  // waveform
  const pk=clip().peaks, n=pk.length;
  ctx.strokeStyle='#5b8fd6'; ctx.lineWidth=1;
  ctx.beginPath();
  for(let i=0;i<n;i++){
    const x=Math.round(i/n*w)+0.5;
    ctx.moveTo(x, mid - pk[i][1]*mid*0.95);
    ctx.lineTo(x, mid - pk[i][0]*mid*0.95);
  }
  ctx.stroke();
  ctx.strokeStyle='#333'; ctx.beginPath();
  ctx.moveTo(0,mid); ctx.lineTo(w,mid); ctx.stroke();
  // committed annotations
  for(const a of anns[ci]){
    const x0=t2x(a.start), x1=t2x(a.end);
    ctx.fillStyle=(LABEL_COLORS[a.label]||'#ccc')+'33';
    ctx.fillRect(x0,0,x1-x0,wf);
    ctx.strokeStyle=LABEL_COLORS[a.label]||'#ccc';
    ctx.strokeRect(x0+0.5,0.5,x1-x0,wf-1);
  }
  // pending selection
  if(sel){
    const x0=t2x(Math.min(sel.t0,sel.t1)), x1=t2x(Math.max(sel.t0,sel.t1));
    ctx.fillStyle='#ffffff22';
    ctx.fillRect(x0,0,x1-x0,wf);
    ctx.strokeStyle='#fff'; ctx.setLineDash([4,3]);
    ctx.strokeRect(x0+0.5,0.5,x1-x0,wf-1); ctx.setLineDash([]);
  }
  // ruler
  ctx.fillStyle='#181b21'; ctx.fillRect(0,wf,w,ruler);
  ctx.strokeStyle='#445'; ctx.fillStyle='#9aa7b8';
  ctx.font='11px ui-monospace,monospace'; ctx.textAlign='center';
  const step=tickStep(dur());
  for(let t=0;t<=dur()+1e-6;t+=step){
    const x=t2x(t);
    ctx.beginPath(); ctx.moveTo(x,wf); ctx.lineTo(x,wf+6); ctx.stroke();
    ctx.fillText(t.toFixed(step<1?1:0)+'s', x, wf+20);
  }
  // playhead
  const x=t2x(audio.currentTime||0);
  ctx.strokeStyle='#39ff88'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,wf); ctx.stroke();
  ctx.lineWidth=1;
}

function fmt(t){return t.toFixed(2);}

function refreshTable(){
  const tb=document.querySelector('#annTable tbody');
  tb.innerHTML='';
  anns[ci].sort((a,b)=>a.start-b.start);
  anns[ci].forEach((a,i)=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${i+1}</td><td>${fmt(a.start)}</td><td>${fmt(a.end)}</td>`+
      `<td>${fmt(a.end-a.start)}</td>`+
      `<td class="tag-${a.label}">${a.label}</td>`+
      `<td><button data-p="${i}">▶</button></td>`+
      `<td><button data-d="${i}">×</button></td>`;
    tb.appendChild(tr);
  });
  tb.querySelectorAll('button[data-p]').forEach(b=>b.onclick=()=>{
    const a=anns[ci][+b.dataset.p]; playRegion(a.start,a.end);});
  tb.querySelectorAll('button[data-d]').forEach(b=>b.onclick=()=>{
    anns[ci].splice(+b.dataset.d,1); refreshTable(); draw(); updateExport();});
  updateExport();
}

function updateExport(){
  let out=[];
  CLIPS.forEach((c,i)=>{
    if(!anns[i].length) return;
    out.push('# '+c.name);
    anns[i].slice().sort((a,b)=>a.start-b.start).forEach(a=>{
      out.push(`${fmt(a.start)}-${fmt(a.end)} ${a.label}`);
    });
    out.push('');
  });
  exportBox.value=out.join('\n').trim() ||
    '（还没有标注。拖动波形选择缺陷区间，选标签，点"+把选区加为标注"。）';
}

function loadClip(i){
  ci=i; sel=null; playSelEnd=null;
  audio.src=clip().audio; audio.load();
  document.getElementById('meta').textContent=
    `${clip().name} · ${dur().toFixed(2)}s`;
  document.getElementById('addAnn').disabled=true;
  document.getElementById('playSel').disabled=true;
  document.getElementById('selInfo').textContent='';
  sizeCanvas(); draw(); refreshTable();
}

// --- playback ---
function tick(){
  document.getElementById('cur').textContent=
    `${(audio.currentTime||0).toFixed(2)} / ${dur().toFixed(2)} s`;
  if(playSelEnd!=null && audio.currentTime>=playSelEnd){
    audio.pause(); playSelEnd=null;
  }
  draw();
  if(!audio.paused) requestAnimationFrame(tick);
}
function playRegion(t0,t1){
  audio.currentTime=t0; playSelEnd=t1;
  audio.play();
}
document.getElementById('play').onclick=()=>{
  if(audio.paused){playSelEnd=null; audio.play();} else audio.pause();
};
document.getElementById('playSel').onclick=()=>{
  if(sel) playRegion(Math.min(sel.t0,sel.t1),Math.max(sel.t0,sel.t1));
};
audio.onplay=()=>{document.getElementById('play').textContent='⏸ 暂停';
  requestAnimationFrame(tick);};
audio.onpause=()=>{document.getElementById('play').textContent='▶ 播放'; tick();};
audio.onended=()=>{document.getElementById('play').textContent='▶ 播放';};

// --- mouse: drag-select / click-seek ---
function evt2x(e){const r=cv.getBoundingClientRect();return e.clientX-r.left;}
cv.addEventListener('pointerdown',e=>{
  drag={x0:evt2x(e)}; cv.setPointerCapture(e.pointerId);
});
cv.addEventListener('pointermove',e=>{
  if(!drag) return;
  const x=evt2x(e);
  if(Math.abs(x-drag.x0)>3){
    sel={t0:x2t(drag.x0), t1:x2t(x)};
    const a=Math.min(sel.t0,sel.t1), b=Math.max(sel.t0,sel.t1);
    document.getElementById('selInfo').textContent=
      `选区 ${a.toFixed(2)}–${b.toFixed(2)}s (${(b-a).toFixed(2)}s)`;
    document.getElementById('addAnn').disabled=false;
    document.getElementById('playSel').disabled=false;
    draw();
  }
});
cv.addEventListener('pointerup',e=>{
  if(!drag) return;
  const x=evt2x(e);
  if(Math.abs(x-drag.x0)<=3){              // a click → seek
    audio.currentTime=x2t(x); draw();
  }
  drag=null;
});
document.getElementById('addAnn').onclick=()=>{
  if(!sel) return;
  anns[ci].push({start:Math.min(sel.t0,sel.t1),
                 end:Math.max(sel.t0,sel.t1), label:labelSel.value});
  sel=null;
  document.getElementById('addAnn').disabled=true;
  document.getElementById('playSel').disabled=true;
  document.getElementById('selInfo').textContent='';
  refreshTable(); draw();
};

clipSel.onchange=()=>loadClip(+clipSel.value);
document.getElementById('copy').onclick=()=>{
  exportBox.select(); navigator.clipboard.writeText(exportBox.value);
  document.getElementById('copied').textContent='已复制 ✓';
  setTimeout(()=>document.getElementById('copied').textContent='',1500);
};
window.addEventListener('keydown',e=>{
  if(e.code==='Space'){e.preventDefault();document.getElementById('play').click();}
});
window.addEventListener('resize',()=>{sizeCanvas();draw();});

loadClip(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
