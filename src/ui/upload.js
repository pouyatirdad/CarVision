// Drag & drop + file picker wiring. Calls onFile(file) with the chosen File.
export function initUpload({ dropzone, fileInput, cameraInput, browseBtn, cameraBtn, onFile }) {
  const pick = (input) => () => {
    input.value = ''; // allow re-selecting the same file
    input.click();
  };
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    pick(fileInput)();
  });
  cameraBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    pick(cameraInput)();
  });
  dropzone.addEventListener('click', pick(fileInput));
  dropzone.addEventListener('keydown', (e) => {
    if (e.target === dropzone && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      pick(fileInput)();
    }
  });

  for (const input of [fileInput, cameraInput]) {
    input.addEventListener('change', () => {
      const file = input.files?.[0];
      if (file) onFile(file);
    });
  }

  const over = (on) => dropzone.classList.toggle('is-over', on);
  for (const type of ['dragenter', 'dragover']) {
    dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      over(true);
    });
  }
  for (const type of ['dragleave', 'drop']) {
    dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      over(false);
    });
  }
  dropzone.addEventListener('drop', (e) => {
    const files = [...(e.dataTransfer?.files ?? [])];
    if (files.length) onFile(files[0]);
  });
  // Dropping outside the zone must not navigate the tab to the image.
  for (const type of ['dragover', 'drop']) window.addEventListener(type, (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    if (!dropzone.contains(e.target)) {
      const files = [...(e.dataTransfer?.files ?? [])];
      if (files.length) onFile(files[0]);
    }
  });
}
