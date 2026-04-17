(function () {
  const sidebar = document.getElementById('toc-sidebar');
  if (!sidebar) return;

  const headings = Array.from(document.querySelectorAll('.pub-content h2[id]'));
  if (headings.length === 0) return;

  const ul = document.createElement('ul');
  headings.forEach(h => {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent;
    a.dataset.target = h.id;
    li.appendChild(a);
    ul.appendChild(li);
  });
  sidebar.appendChild(ul);

  const links = Array.from(ul.querySelectorAll('a'));
  const obs = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        links.forEach(a => a.classList.remove('active'));
        const active = links.find(a => a.dataset.target === entry.target.id);
        if (active) active.classList.add('active');
      }
    });
  }, { rootMargin: '-20% 0px -70% 0px' });

  headings.forEach(h => obs.observe(h));
})();
