(function () {
  const base = document.querySelector('meta[name="baseurl"]')?.content || '/lazy-rudder-paper';
  fetch(base + '/_data/lean_status.json')
    .then(r => r.json())
    .then(data => {
      const counts = document.getElementById('lean-counts');
      if (counts) {
        counts.innerHTML =
          `<strong>${data.counts.proven}</strong> proven &nbsp;&middot;&nbsp; ` +
          `<strong>${data.counts.deferred}</strong> deferred &nbsp;&middot;&nbsp; ` +
          `<strong>${data.counts.stub}</strong> stubs`;
      }

      const tbody = document.getElementById('lean-tbody');
      if (!tbody) return;
      tbody.innerHTML = '';
      data.theorems.forEach(t => {
        const tr = document.createElement('tr');
        const tdName = document.createElement('td');
        tdName.style.cssText = 'font-family:monospace;font-size:.85rem;padding:.35rem .5rem;';
        tdName.textContent = t.name;
        const tdStatus = document.createElement('td');
        tdStatus.style.cssText = 'padding:.35rem .5rem;';
        const badge = document.createElement('span');
        badge.className = 'lean-badge ' + t.status;
        badge.textContent = t.status;
        tdStatus.appendChild(badge);
        const tdKind = document.createElement('td');
        tdKind.style.cssText = 'color:var(--fg-muted);font-size:.85rem;padding:.35rem .5rem;';
        tdKind.textContent = t.kind;
        tr.appendChild(tdName);
        tr.appendChild(tdStatus);
        tr.appendChild(tdKind);
        tbody.appendChild(tr);
      });
    })
    .catch(err => {
      const el = document.getElementById('lean-counts');
      if (el) el.textContent = 'Could not load theorem data.';
      console.error('lean:', err);
    });
})();
