const KEY = 'carvision-theme';

function stored() {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function initTheme(button) {
  const apply = (theme) => {
    document.documentElement.dataset.theme = theme;
    button.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  };
  apply(document.documentElement.dataset.theme || stored() || 'dark');
  button.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    apply(next);
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* storage blocked: theme just won't persist */
    }
  });
}
