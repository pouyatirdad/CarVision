import test from 'node:test';
import assert from 'node:assert/strict';
import { readDimensions, sniffFormat, validateImageFile } from '../src/utils/validation.js';

const png = (w, h) => {
  const b = new Uint8Array(33);
  b.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], 0);
  const dv = new DataView(b.buffer);
  dv.setUint32(16, w);
  dv.setUint32(20, h);
  return b;
};

test('sniffFormat recognises JPEG/PNG/WEBP and rejects others', () => {
  assert.equal(sniffFormat(new Uint8Array([0xff, 0xd8, 0xff, 0xe0])), 'JPEG');
  assert.equal(sniffFormat(png(1, 1)), 'PNG');
  const webp = new Uint8Array(12);
  webp.set([0x52, 0x49, 0x46, 0x46], 0);
  webp.set([0x57, 0x45, 0x42, 0x50], 8);
  assert.equal(sniffFormat(webp), 'WEBP');
  assert.equal(sniffFormat(new TextEncoder().encode('GIF89a......')), null);
});

test('readDimensions reads PNG headers', () => {
  assert.deepEqual(readDimensions(png(640, 480), 'PNG'), { width: 640, height: 480 });
});

test('validateImageFile rejects wrong type, empty, oversized and fake images', async () => {
  const mk = (bytes, name, type) => new File([bytes], name, { type });
  assert.equal((await validateImageFile(mk(new Uint8Array(0), 'a.png', 'image/png'))).code, 'empty');
  assert.equal((await validateImageFile(mk(new Uint8Array(10), 'a.gif', 'image/gif'))).code, 'unsupported-type');
  assert.equal((await validateImageFile(mk(new Uint8Array(100), 'fake.jpg', 'image/jpeg'))).code, 'not-an-image');
  assert.equal((await validateImageFile(mk(new Uint8Array(16 * 1024 * 1024), 'big.jpg', 'image/jpeg'))).code, 'too-large');
  assert.equal((await validateImageFile(mk(png(20000, 20000), 'huge.png', 'image/png'))).code, 'too-many-pixels');
  assert.equal((await validateImageFile(mk(png(800, 600), 'ok.png', 'image/png'))).ok, true);
});
