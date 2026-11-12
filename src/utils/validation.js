import { LIMITS } from '../config.js';
import { formatBytes } from './formatting.js';

/** Sniff the real image format from the first bytes (the MIME type / extension can lie). */
export function sniffFormat(bytes) {
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'JPEG';
  if (bytes.length >= 8 && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) return 'PNG';
  if (
    bytes.length >= 12 &&
    String.fromCharCode(...bytes.slice(0, 4)) === 'RIFF' &&
    String.fromCharCode(...bytes.slice(8, 12)) === 'WEBP'
  ) {
    return 'WEBP';
  }
  return null;
}

/**
 * Read pixel dimensions from the file header without decoding the image, so absurdly large images can be
 * refused before the browser tries to allocate memory for them. Returns null if it cannot be determined.
 */
export function readDimensions(bytes, format) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  try {
    if (format === 'PNG') return { width: dv.getUint32(16), height: dv.getUint32(20) };
    if (format === 'JPEG') {
      let i = 2;
      while (i + 9 < bytes.length) {
        if (bytes[i] !== 0xff) {
          i++;
          continue;
        }
        const marker = bytes[i + 1];
        if (marker === 0xff) {
          i++;
          continue;
        }
        if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
          return { height: dv.getUint16(i + 5), width: dv.getUint16(i + 7) };
        }
        if (marker === 0xd8 || (marker >= 0xd0 && marker <= 0xd7) || marker === 0x01) {
          i += 2;
        } else {
          i += 2 + dv.getUint16(i + 2);
        }
      }
      return null;
    }
    if (format === 'WEBP') {
      const chunk = String.fromCharCode(...bytes.slice(12, 16));
      if (chunk === 'VP8 ') return { width: dv.getUint16(26, true) & 0x3fff, height: dv.getUint16(28, true) & 0x3fff };
      if (chunk === 'VP8L') {
        const b = dv.getUint32(21, true);
        return { width: (b & 0x3fff) + 1, height: ((b >> 14) & 0x3fff) + 1 };
      }
      if (chunk === 'VP8X') {
        return {
          width: 1 + (bytes[24] | (bytes[25] << 8) | (bytes[26] << 16)),
          height: 1 + (bytes[27] | (bytes[28] << 8) | (bytes[29] << 16)),
        };
      }
    }
  } catch {
    /* fall through */
  }
  return null;
}

/**
 * Validate a user-selected file. Returns { ok: true, format, dimensions } or { ok: false, code, message }.
 */
export async function validateImageFile(file) {
  if (!file) return { ok: false, code: 'no-file', message: 'No file was selected.' };
  if (file.size === 0) return { ok: false, code: 'empty', message: 'This file is empty.' };
  if (file.size > LIMITS.maxFileBytes) {
    return {
      ok: false,
      code: 'too-large',
      message: `This file is ${formatBytes(file.size)}. The maximum allowed size is ${formatBytes(LIMITS.maxFileBytes)}.`,
    };
  }

  const ext = (file.name.split('.').pop() || '').toLowerCase();
  const typeOk = LIMITS.acceptedTypes.includes(file.type);
  const extOk = LIMITS.acceptedExtensions.includes(ext);
  if (!typeOk && !(file.type === '' && extOk)) {
    return { ok: false, code: 'unsupported-type', message: 'Unsupported file type. Please choose a JPG, PNG or WEBP image.' };
  }

  const head = new Uint8Array(await file.slice(0, 64 * 1024).arrayBuffer());
  const format = sniffFormat(head);
  if (!format) {
    return { ok: false, code: 'not-an-image', message: 'This file does not look like a valid JPG, PNG or WEBP image.' };
  }

  const dimensions = readDimensions(head, format);
  if (dimensions) {
    if (dimensions.width * dimensions.height > LIMITS.maxPixels) {
      return {
        ok: false,
        code: 'too-many-pixels',
        message: `This image is ${dimensions.width}×${dimensions.height} pixels, which is too large to process in a browser. Please use a smaller image.`,
      };
    }
    if (Math.min(dimensions.width, dimensions.height) < LIMITS.minSidePx) {
      return { ok: false, code: 'too-small', message: `This image is too small (minimum ${LIMITS.minSidePx} pixels on the shorter side).` };
    }
  }
  return { ok: true, format, dimensions };
}
