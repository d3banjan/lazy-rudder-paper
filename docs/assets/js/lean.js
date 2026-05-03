(function () {
  const base = document.querySelector('meta[name="baseurl"]')?.content || '/lazy-rudder-paper';
  fetch(base + '/assets/data/lean_status.json')
    .then(r => r.json())
    .then(data => {
      const counts = document.getElementById('lean-counts');
      if (counts) {
        const c = data.counts;
        const parts = [
          `<strong>${c.proven}</strong> proven`,
          `<strong>${c.partial ?? 0}</strong> partial`,
          `<strong>${c.deferred}</strong> deferred`,
          `<strong>${c.paper_facing_sorry ?? 0}</strong> paper-facing sorry`,
          `<strong>${c.stub}</strong> stubs`,
        ];
        counts.innerHTML = parts.join(' &nbsp;&middot;&nbsp; ');
      }

      const tbody = document.getElementById('lean-tbody');
      if (!tbody) return;
      tbody.innerHTML = '';
      data.theorems.forEach(t => {
        const tr = document.createElement('tr');
        const tdName = document.createElement('td');
        tdName.style.cssText = 'font-family:monospace;font-size:.85rem;padding:.35rem .5rem;';
        if (t.source_url) {
          const a = document.createElement('a');
          a.href = t.source_url;
          a.textContent = t.name;
          a.title = `View Lean source (line ${t.line})`;
          a.target = '_blank';
          a.rel = 'noopener';
          tdName.appendChild(a);
        } else {
          tdName.textContent = t.name;
        }
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
