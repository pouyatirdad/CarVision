// Loads models lazily, shows real download progress, caches the bytes (Cache API) so later visits work
// offline, reuses one ONNX Runtime session per model, and releases sessions on request.

import * as ort from 'onnxruntime-web';
import { CACHE_NAME, MODELS, MODEL_BASE_URL } from '../config.js';

// The WASM runtime is resolved by the bundler (ort.bundle.min.mjs references it via new URL(..., import.meta.url)),
// so it is emitted into dist/assets with the correct base path automatically.
// Multi-threaded WASM needs cross-origin isolation (SharedArrayBuffer), which GitHub Pages cannot send.
ort.env.wasm.numThreads =
  typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated
    ? Math.min(4, (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2)
    : 1;

export class ModelLoadError extends Error {
  constructor(message, { model, cause } = {}) {
    super(message);
    this.name = 'ModelLoadError';
    this.model = model;
    this.cause = cause;
  }
}

export async function detectWebGPU() {
  try {
    if (typeof navigator === 'undefined' || !navigator.gpu) return false;
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

let pruned = false;

async function openCache() {
  try {
    const cache = typeof caches !== 'undefined' ? await caches.open(CACHE_NAME) : null;
    if (cache && !pruned) {
      pruned = true;
      // Drop cached models that are no longer part of this version (renamed/replaced files) to free space.
      const base = globalThis.location?.href ?? 'http://localhost/';
      const current = new Set(Object.values(MODELS).map((m) => new URL(MODEL_BASE_URL + m.file, base).href));
      for (const req of await cache.keys()) if (!current.has(req.url)) await cache.delete(req);
    }
    return cache;
  } catch {
    return null; // private mode / blocked storage: just download every time
  }
}

/** Download with streaming progress; resolves to a Uint8Array. Throws on truncated downloads. */
async function download(url, approxBytes, onProgress, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status} while downloading ${url}`);
  const header = Number(res.headers.get('content-length'));
  const encoded = res.headers.get('content-encoding');
  // With transport compression the Content-Length is the compressed size; don't use it as the total.
  const total = header && !encoded ? header : approxBytes;
  if (!res.body) {
    const buf = new Uint8Array(await res.arrayBuffer());
    onProgress?.(buf.length, buf.length);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(Math.min(loaded, total), total);
  }
  if (header && !encoded && loaded !== header) {
    throw new Error(`Download interrupted (${loaded} of ${header} bytes).`);
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  onProgress?.(loaded, loaded);
  return out;
}

export class ModelManager {
  /**
   * @param {(event: {model:string, phase:'download'|'init'|'ready', loaded?:number, total?:number,
   *        fromCache?:boolean, provider?:string}) => void} [onEvent]
   */
  constructor(onEvent) {
    this.onEvent = onEvent || (() => {});
    this.pending = new Map(); // key -> Promise<LoadedModel>
    this.webgpu = undefined;
    this.forceWasm = false;
  }

  /** forceWasm: skip WebGPU entirely (used by ?backend=wasm to exercise / force the CPU fallback). */
  configure({ forceWasm } = {}) {
    if (typeof forceWasm === 'boolean') this.forceWasm = forceWasm;
  }

  async capabilities() {
    if (this.webgpu === undefined) this.webgpu = await detectWebGPU();
    return { webgpu: this.webgpu, wasmThreads: ort.env.wasm.numThreads, crossOriginIsolated: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated };
  }

  /** Returns { session, provider, inputName, outputName, fromCache, bytes }; concurrent callers share one load. */
  get(key) {
    if (!this.pending.has(key)) {
      const p = this.#load(key).catch((e) => {
        this.pending.delete(key); // allow a retry after a failure (e.g. a dropped connection)
        throw e;
      });
      this.pending.set(key, p);
    }
    return this.pending.get(key);
  }

  isLoaded(key) {
    return this.pending.has(key);
  }

  async #load(key) {
    const cfg = MODELS[key];
    if (!cfg) throw new ModelLoadError(`Unknown model "${key}".`, { model: key });
    const url = MODEL_BASE_URL + cfg.file;
    const emit = (e) => this.onEvent({ model: key, ...e });

    let bytes;
    let fromCache = false;
    try {
      const cache = await openCache();
      const hit = cache ? await cache.match(url) : null;
      if (hit) {
        bytes = new Uint8Array(await hit.arrayBuffer());
        fromCache = true;
        emit({ phase: 'download', loaded: bytes.length, total: bytes.length, fromCache: true });
      } else {
        // One automatic retry for flaky connections.
        for (let attempt = 0; ; attempt++) {
          try {
            bytes = await download(url, cfg.approxBytes, (loaded, total) => emit({ phase: 'download', loaded, total, fromCache: false }));
            break;
          } catch (e) {
            if (attempt >= 1) throw e;
            await new Promise((r) => setTimeout(r, 1000));
          }
        }
        try {
          await cache?.put(url, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } }));
        } catch {
          /* quota exceeded etc.: not fatal */
        }
      }
    } catch (e) {
      throw new ModelLoadError(
        `The "${cfg.title}" model (${(cfg.approxBytes / 1e6).toFixed(0)} MB) could not be downloaded. Check your connection and try again.`,
        { model: key, cause: e },
      );
    }

    emit({ phase: 'init' });
    const caps = await this.capabilities();
    const order = cfg.providers.filter((p) => p !== 'webgpu' || (caps.webgpu && !this.forceWasm));
    if (!order.includes('wasm')) order.push('wasm');

    let lastError;
    for (const provider of order) {
      try {
        const session = await ort.InferenceSession.create(bytes, {
          executionProviders: [provider],
          graphOptimizationLevel: 'all',
        });
        const loaded = { key, session, provider, inputName: session.inputNames[0], outputName: session.outputNames[0], fromCache, bytes: bytes.length };
        if (provider === 'webgpu') await this.#warmup(loaded, cfg); // also surfaces run-time-only WebGPU failures
        emit({ phase: 'ready', provider });
        return loaded;
      } catch (e) {
        lastError = e;
        console.warn(`[CarVision] ${key}: ${provider} failed, trying next provider`, e);
      }
    }
    throw new ModelLoadError(
      `The "${cfg.title}" model could not be initialised in this browser (${lastError?.message ?? 'unknown error'}). Try a recent Chrome, Edge, Firefox or Safari.`,
      { model: key, cause: lastError },
    );
  }

  async #warmup(loaded, cfg) {
    const dims = cfg.key === 'iranian' ? [1, 3, cfg.inputHeight, cfg.inputWidth] : [1, 3, cfg.inputSize, cfg.inputSize];
    const n = dims.reduce((a, b) => a * b, 1);
    const out = await loaded.session.run({ [loaded.inputName]: new ort.Tensor('float32', new Float32Array(n), dims) });
    const data = out[loaded.outputName].data;
    if (!data || data.length === 0 || Number.isNaN(data[0])) throw new Error('WebGPU warm-up produced invalid output');
  }

  /** Run a model on a prepared CHW float32 buffer. Returns the output Float32Array. */
  async run(key, data, dims) {
    const m = await this.get(key);
    // ORT-Web sessions reject overlapping run() calls ("Session already started"); a new upload can start while the
    // previous image is still being processed, so runs on one session are queued.
    const previous = m.queue ?? Promise.resolve();
    const job = previous.catch(() => {}).then(() => m.session.run({ [m.inputName]: new ort.Tensor('float32', data, dims) }));
    m.queue = job;
    const out = await job;
    return { output: out[m.outputName].data, provider: m.provider };
  }

  async release(key) {
    const p = this.pending.get(key);
    if (!p) return;
    this.pending.delete(key);
    try {
      const m = await p;
      await m.session.release();
    } catch {
      /* never loaded */
    }
  }

  async releaseAll() {
    await Promise.all([...this.pending.keys()].map((k) => this.release(k)));
  }
}
