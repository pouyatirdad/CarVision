// Main-thread facade over the inference worker. Falls back to running the same pipeline on the main
// thread when module workers / OffscreenCanvas are unavailable (older browsers).

export class RecognitionClient {
  /** @param {{disableWorker?: boolean}} [opts] disableWorker forces the main-thread fallback (also used for testing it) */
  constructor({ disableWorker = false } = {}) {
    this.disableWorker = disableWorker;
    this.nextId = 1;
    this.pending = new Map(); // id -> {resolve, reject, onEvent}
    this.worker = null;
    this.local = null; // main-thread fallback: { recognizer, bitmap }
    this.mode = 'worker';
  }

  #ensure() {
    // Memoised: concurrent first calls must share ONE worker (otherwise early messages go to a discarded one).
    return (this.starting ??= this.#start());
  }

  async #start() {
    if (this.worker || this.local) return;
    const workerOk = !this.disableWorker && typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined';
    if (workerOk) {
      try {
        const worker = new Worker(new URL('../workers/inference.worker.js', import.meta.url), { type: 'module' });
        await new Promise((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error('worker start timeout')), 15000);
          worker.onmessage = (e) => {
            if (e.data.type === 'ready') {
              clearTimeout(timer);
              resolve();
            }
          };
          worker.onerror = (e) => {
            clearTimeout(timer);
            reject(new Error(e.message || 'worker failed to start'));
          };
        });
        worker.onmessage = (e) => this.#onMessage(e.data);
        worker.onerror = (e) => console.error('[CarVision] worker error', e.message);
        this.worker = worker;
        return;
      } catch (e) {
        console.warn('[CarVision] Web Worker unavailable, running on the main thread.', e);
      }
    }
    const { CarRecognizer } = await import('./car-recognition.js');
    this.mode = 'main-thread';
    this.local = {
      recognizer: new CarRecognizer((event) => this.#activeOnEvent?.(event)),
      bitmap: null,
    };
  }

  #activeOnEvent = null;

  #onMessage(msg) {
    const p = this.pending.get(msg.id);
    if (!p) return;
    if (msg.type === 'event') p.onEvent?.(msg.event);
    else if (msg.type === 'result') {
      this.pending.delete(msg.id);
      p.resolve(msg.result);
    } else if (msg.type === 'error') {
      this.pending.delete(msg.id);
      p.reject(Object.assign(new Error(msg.error.message), { name: msg.error.name, model: msg.error.model }));
    }
  }

  async #call(type, payload, onEvent, transfer = []) {
    await this.#ensure();
    if (this.local) {
      this.#activeOnEvent = onEvent;
      const l = this.local;
      switch (type) {
        case 'capabilities':
          return l.recognizer.capabilities();
        case 'configure':
          l.recognizer.configure(payload);
          return true;
        case 'setImage':
          l.bitmap?.close();
          l.bitmap = payload.bitmap;
          return true;
        case 'recognize':
          return l.recognizer.recognize(l.bitmap, payload.options);
        case 'clearImage':
          l.bitmap?.close();
          l.bitmap = null;
          return true;
        case 'releaseModels':
          return l.recognizer.releaseAll();
      }
    }
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onEvent });
      this.worker.postMessage({ id, type, payload }, transfer);
    });
  }

  capabilities() {
    return this.#call('capabilities');
  }

  configure(options) {
    return this.#call('configure', options);
  }

  /** Hands the decoded image to the worker (ownership of the ImageBitmap is transferred). */
  setImage(bitmap) {
    return this.#call('setImage', { bitmap }, null, [bitmap]);
  }

  recognize(options, onEvent) {
    return this.#call('recognize', { options }, onEvent);
  }

  clearImage() {
    return this.#call('clearImage');
  }

  /** Free the ONNX Runtime sessions (and their memory). They are re-created lazily on the next recognition. */
  releaseModels() {
    return this.#call('releaseModels');
  }
}
