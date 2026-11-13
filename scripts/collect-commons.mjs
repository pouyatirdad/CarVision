// Collect freely licensed rear/front photos per vehicle class from Wikimedia Commons (research test set).
// usage: node scripts/collect-commons.mjs scripts/data/commons-classes.json <outDir>
// Downloads 800 px thumbnails of files whose name/category says rear (or front), keeps only CC/PD licences and
// writes <outDir>/manifest.json with title, page URL, licence and author of every file (attribution). Resumable.
// Images are NOT committed to the repository.
// classes.json: { "<classId>": ["Category A", "Category B"], ... }
import fs from 'node:fs';
import path from 'node:path';

const [, , classFile, outDir] = process.argv;
const classes = JSON.parse(fs.readFileSync(classFile, 'utf8'));
const UA = 'CarVision-research/0.1 (static web app; building a small evaluation set)';
const API = 'https://commons.wikimedia.org/w/api.php';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const REAR = /\b(rear|back|heck|hinten|arri[eè]re|achter|posterior[e]?|trasera|traseira|r[uü]ckansicht|rearview|rear-view|backside|tail)\b|heckansicht|_r\.|\(r\)/i;
const FRONT = /\b(front|vorne|avant|voor|anterior[e]?|delantera|dianteira|frontansicht|frontal)\b|frontansicht/i;
const SKIP_FILE = /interior|innen|dashboard|cockpit|engine|motor|seat|wheel|trunk|kofferraum|logo|badge|emblem|lamp|light|detail|instrument|steering|\.(svg|pdf|tif|webm|ogv)$/i;
const SKIP_CAT = /interior|engine|in competition|police|taxi|rall|nascar|racing|toys|model cars|advertis|crash|wreck|accident|tuning|modified|concept|details|logos|emblems|lights|drawings|scale models|people with|in art|limousine|ambulance/i;

async function api(params) {
  const url = `${API}?${new URLSearchParams({ format: 'json', formatversion: '2', ...params })}`;
  for (let i = 0; i < 8; i++) {
    try {
      const r = await fetch(url, { headers: { 'User-Agent': UA } });
      if (r.ok) return r.json();
      console.error('HTTP', r.status, 'retry', i);
    } catch (e) {
      console.error('fetch error', e.message);
    }
    await sleep(5000 * (i + 1));
  }
  console.error('giving up', url);
  return {};
}

async function members(cat, type) {
  const out = [];
  let cont = {};
  do {
    const j = await api({ action: 'query', list: 'categorymembers', cmtitle: `Category:${cat}`, cmtype: type, cmlimit: '500', ...cont });
    out.push(...(j.query?.categorymembers ?? []).map((m) => m.title));
    cont = j.continue ?? null;
    await sleep(400);
  } while (cont);
  return out;
}

async function walk(cat, depth, seen, files, trail) {
  if (seen.has(cat)) return;
  seen.add(cat);
  for (const f of await members(cat, 'file')) if (!files.has(f)) files.set(f, trail.concat(cat));
  if (depth <= 0) return;
  for (const sub of await members(cat, 'subcat')) {
    const name = sub.replace(/^Category:/, '');
    if (SKIP_CAT.test(name)) continue;
    await walk(name, depth - 1, seen, files, trail.concat(cat));
  }
}

async function info(titles) {
  const j = await api({ action: 'query', prop: 'imageinfo', titles: titles.join('|'), iiprop: 'url|extmetadata|size', iiurlwidth: '800' });
  return j.query?.pages ?? [];
}

const strip = (s) => String(s ?? '').replace(/<[^>]+>/g, '').trim();
const mpath = path.join(outDir, 'manifest.json');
const manifest = fs.existsSync(mpath) ? JSON.parse(fs.readFileSync(mpath, 'utf8')) : [];
const done = new Set(manifest.map((m) => m.cls));
for (const [cls, cats] of Object.entries(classes)) {
  if (done.has(cls)) continue;
  const files = new Map();
  const seen = new Set();
  for (const c of cats) await walk(c, 2, seen, files, []);
  const pick = { rear: [], front: [] };
  for (const [title, trail] of files) {
    if (SKIP_FILE.test(title)) continue;
    const hay = `${title} ${trail.join(' ')}`;
    const rear = REAR.test(hay);
    const front = FRONT.test(title) || /front/i.test(trail.at(-1) ?? '');
    if (rear && !front) pick.rear.push(title);
    else if (front && !rear) pick.front.push(title);
  }
  console.log(`${cls}: ${files.size} files, rear-named ${pick.rear.length}, front-named ${pick.front.length}`);
  for (const view of ['rear', 'front']) {
    const list = pick[view].slice(0, view === 'rear' ? 40 : 8);
    for (let i = 0; i < list.length; i += 20) {
      const pages = await info(list.slice(i, i + 20));
      for (const p of pages) {
        const ii = p.imageinfo?.[0];
        if (!ii?.thumburl) continue;
        const m = ii.extmetadata ?? {};
        const lic = strip(m.LicenseShortName?.value);
        if (!/^(CC|Public domain|PD|GFDL|CC0)/i.test(lic) && !/cc-by|cc0|public domain/i.test(lic)) continue;
        const dir = path.join(outDir, view, cls);
        fs.mkdirSync(dir, { recursive: true });
        const name = `${String(manifest.length).padStart(4, '0')}.jpg`;
        let r = null;
        for (let k = 0; k < 5 && !(r && r.ok); k++) { try { r = await fetch(ii.thumburl, { headers: { 'User-Agent': UA } }); } catch { r = null; } if (!(r && r.ok)) await sleep(4000 * (k + 1)); }
        if (!(r && r.ok)) continue;
        fs.writeFileSync(path.join(dir, name), Buffer.from(await r.arrayBuffer()));
        manifest.push({ file: `${view}/${cls}/${name}`, view, cls, title: p.title, page: ii.descriptionurl, license: lic, artist: strip(m.Artist?.value).slice(0, 120) });
        await sleep(300);
      }
    }
  }
  fs.writeFileSync(path.join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 1));
}
console.log(`total ${manifest.length}`);
