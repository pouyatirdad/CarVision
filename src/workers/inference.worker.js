// Runs the whole recognition pipeline off the main thread so the UI never blocks during model
// download, session creation or inference.
import { CarRecognizer } from '../ai/car-recognition.js';

let currentId = null;
const recognizer = new CarRecognizer((event) => self.postMessage({ id: currentId, type: 'event', event }));
let bitmap = null;

self.onmessage = async (e) => {
  const { id, type, payload } = e.data;
  currentId = id;
  try {
    let result;
    switch (type) {
      case 'capabilities':
        result = await recognizer.capabilities();
        break;
      case 'configure':
        recognizer.configure(payload);
        result = true;
        break;
      case 'setImage':
        bitmap?.close();
        bitmap = payload.bitmap;
        result = true;
        break;
      case 'recognize':
        if (!bitmap) throw new Error('No image has been loaded.');
        result = await recognizer.recognize(bitmap, payload.options);
        break;
      case 'clearImage':
        bitmap?.close();
        bitmap = null;
        result = true;
        break;
      case 'releaseModels':
        await recognizer.releaseAll();
        result = true;
        break;
      default:
        throw new Error(`Unknown message type: ${type}`);
    }
    self.postMessage({ id, type: 'result', result });
  } catch (err) {
    self.postMessage({ id, type: 'error', error: { message: err?.message || String(err), name: err?.name, model: err?.model } });
  }
};

self.postMessage({ type: 'ready' });
