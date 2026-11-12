// Image -> tensor-data preprocessing using only browser-native canvas APIs.
// Works in a Worker (OffscreenCanvas) and on the main thread (HTMLCanvasElement fallback).

function createCanvas(width, height) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  return c;
}

function getContext(canvas) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  return ctx;
}

/**
 * Letterbox for the detector: scale to fit `size`x`size` keeping the aspect ratio, pad with 114 grey at the
 * bottom/right (YOLOX convention), output CHW float32 in BGR order with raw 0-255 values (no normalisation).
 */
export function letterboxBGR(bitmap, size) {
  const scale = Math.min(size / bitmap.width, size / bitmap.height);
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = createCanvas(size, size);
  const ctx = getContext(canvas);
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, size, size);
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    out[i] = data[p + 2]; // B
    out[plane + i] = data[p + 1]; // G
    out[2 * plane + i] = data[p]; // R
  }
  return { data: out, scale };
}

/**
 * Crop `box` ({x1,y1,x2,y2} in bitmap pixels) and resize it (no aspect-ratio preservation - the classifiers
 * were trained that way) to `width`x`height`; output CHW float32 RGB normalised with (v/255 - mean) / std.
 */
export function cropToRGBTensorData(bitmap, box, { width, height, mean, std }) {
  const sx = Math.max(0, Math.floor(box.x1));
  const sy = Math.max(0, Math.floor(box.y1));
  const sw = Math.max(1, Math.min(bitmap.width, Math.ceil(box.x2)) - sx);
  const sh = Math.max(1, Math.min(bitmap.height, Math.ceil(box.y2)) - sy);
  const canvas = createCanvas(width, height);
  const ctx = getContext(canvas);
  ctx.fillStyle = '#fff'; // transparent PNG regions become white instead of black
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);
  const plane = width * height;
  const out = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    out[i] = (data[p] / 255 - mean[0]) / std[0];
    out[plane + i] = (data[p + 1] / 255 - mean[1]) / std[1];
    out[2 * plane + i] = (data[p + 2] / 255 - mean[2]) / std[2];
  }
  return out;
}

export const IMAGENET = { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };
export const HALF = { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] };
