(function () {
  const el = document.getElementById('widget-figE');
  if (!el || typeof d3 === 'undefined') return;

  const base = document.querySelector('meta[name="baseurl"]')?.content || '/lazy-rudder-paper';
  fetch(base + '/_data/modules.json')
    .then(r => r.json())
    .then(data => render(el, data))
    .catch(err => { el.textContent = 'Chart unavailable.'; console.error('figE:', err); });

  function render(container, data) {
    const ncols = 2;
    const cellW = Math.min((container.clientWidth || 640) / ncols, 360);
    const cellH = 180;
    const margin = { top: 28, right: 10, bottom: 36, left: 46 };
    const iw = cellW - margin.left - margin.right;
    const ih = cellH - margin.top - margin.bottom;
    const nrows = Math.ceil(data.modules.length / ncols);
    const totalH = nrows * cellH + 24;

    const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${cellW * ncols} ${totalH}`)
      .attr('width', '100%');

    const tooltip = d3.select('body').append('div').attr('class', 'widget-tooltip')
      .style('opacity', 0).style('position', 'absolute');

    data.modules.forEach((mod, mi) => {
      const row = Math.floor(mi / ncols), col = mi % ncols;
      const gx = col * cellW + margin.left;
      const gy = row * cellH + margin.top;
      const g = svg.append('g').attr('transform', `translate(${gx},${gy})`);

      const vals = data.runs.flatMap(r => (r.by_module[mod] || []).map(l => l.bonus_R_k5));
      if (!vals.length) return;

      const nLayers = data.runs[0].by_module[mod]?.length || 0;
      const x = d3.scaleLinear().domain([0, Math.max(nLayers - 1, 1)]).range([0, iw]);
      const y = d3.scaleLinear().domain([0, d3.max(vals) * 1.1]).range([ih, 0]);

      g.append('g').attr('transform', `translate(0,${ih})`)
        .call(d3.axisBottom(x).ticks(Math.min(nLayers, 6)).tickFormat(d3.format('d')));
      g.append('g').call(d3.axisLeft(y).ticks(3));

      g.append('text').attr('x', iw / 2).attr('y', -10)
        .attr('text-anchor', 'middle').style('font-size', '11px')
        .attr('fill', 'currentColor')
        .text(data.module_labels[mod] || mod);

      const line = d3.line().x((_, i) => x(i)).y(d => y(d.bonus_R_k5)).curve(d3.curveMonotoneX);

      data.runs.forEach(run => {
        const layers = run.by_module[mod] || [];
        if (!layers.length) return;
        g.append('path').datum(layers)
          .attr('fill', 'none').attr('stroke', run.color).attr('stroke-width', 1.8)
          .attr('d', line);
        g.selectAll(null).data(layers).join('circle')
          .attr('r', 3.5)
          .attr('cx', (_, i) => x(i)).attr('cy', d => y(d.bonus_R_k5))
          .attr('fill', run.color).style('cursor', 'pointer')
          .on('mouseover', (ev, d) => {
            tooltip.transition().duration(80).style('opacity', 1);
            tooltip.html(`<strong>${run.label} \u2014 ${data.module_labels[mod] || mod}</strong><br>Layer ${d.layer}<br>bonus_R: ${d.bonus_R_k5.toFixed(2)}\xd7`)
              .style('left', (ev.pageX + 10) + 'px').style('top', (ev.pageY - 28) + 'px');
          })
          .on('mouseleave', () => tooltip.transition().duration(150).style('opacity', 0));
      });
    });

    // Global legend
    const legend = svg.append('g').attr('transform', `translate(${cellW * ncols - 130}, 8)`);
    data.runs.forEach((run, i) => {
      const lg = legend.append('g').attr('transform', `translate(${i * 75}, 0)`);
      lg.append('line').attr('x2', 14).attr('stroke', run.color).attr('stroke-width', 2);
      lg.append('text').attr('x', 18).attr('dy', '0.35em').style('font-size', '10px')
        .attr('fill', 'currentColor').text(run.label);
    });
  }
})();
