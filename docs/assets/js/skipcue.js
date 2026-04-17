(function () {
  const sections = Array.from(document.querySelectorAll('.pub-section[id]'));
  sections.forEach((sec, i) => {
    const next = sections[i + 1];
    if (!next) return;
    const nextHeading = next.querySelector('h2');
    const label = nextHeading ? nextHeading.textContent : 'Next section';
    const cue = document.createElement('div');
    cue.className = 'skip-cue';
    const btn = document.createElement('button');
    btn.textContent = 'Skip to: ' + label + ' \u2193';
    btn.addEventListener('click', () => {
      next.scrollIntoView({ behavior: 'smooth' });
    });
    cue.appendChild(btn);
    sec.appendChild(cue);
  });
})();
