import { MODELS } from '../config.js';
import { formatBytes } from '../utils/formatting.js';

const STATUS_TEXT = {
  detect: 'Looking for vehicles in the photo…',
  'classify-international': 'Identifying the vehicle (international model)…',
  'classify-iranian': 'Checking Iranian-market models…',
};

/** Real progress only: determinate while downloading, indeterminate while the model/inference is running. */
export class Progress {
  constructor({ root, label, bar, detail }) {
    this.root = root;
    this.label = label;
    this.bar = bar;
    this.detail = detail;
  }

  show(text = 'Working…') {
    this.root.hidden = false;
    this.label.textContent = text;
    this.detail.textContent = '';
    this.bar.removeAttribute('value'); // indeterminate
  }

  hide() {
    this.root.hidden = true;
  }

  /** Feed events coming from the recognition pipeline. */
  handle(event) {
    if (event.type === 'status') {
      this.label.textContent = STATUS_TEXT[event.phase] ?? 'Working…';
      this.detail.textContent = '';
      this.bar.removeAttribute('value');
    } else if (event.type === 'model') {
      const cfg = MODELS[event.model];
      if (event.phase === 'download') {
        if (event.fromCache) {
          this.label.textContent = `Loading ${cfg.title} from cache…`;
          this.detail.textContent = '';
          this.bar.removeAttribute('value');
        } else {
          this.label.textContent = `Downloading ${cfg.title} (first use only)…`;
          this.bar.max = 100;
          this.bar.value = event.total ? Math.round((event.loaded / event.total) * 100) : 0;
          this.detail.textContent = `${formatBytes(event.loaded)} of ${formatBytes(event.total)}`;
        }
      } else if (event.phase === 'init') {
        this.label.textContent = `Starting ${cfg.title}…`;
        this.detail.textContent = '';
        this.bar.removeAttribute('value');
      }
    }
  }
}
