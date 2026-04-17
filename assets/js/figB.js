(function () {
  const el = document.getElementById('widget-figB');
  if (!el || typeof d3 === 'undefined') return;

  const base = document.querySelector('meta[name="baseurl"]')?.content || '/lazy-rudder-paper';
  fetch(base + '/_data/bonus_r.json')
    .then(r => r.json())
    .then(data => render(el, data))
    .catch(err => { el.textContent = 'Chart unavailable.'; console.error('figB:', err); });

  function render(container, data) {
    const W = Math.min(container.clientWidth || 640, 720), H = 320;
    const margin = { top: 20, right: 130, bottom: 50, left: 55 };
    const iw = W - margin.left - margin.right;
    const ih = H - margin.top - margin.bottom;

    const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%');
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const allVals = data.runs.flatMap(r => r.per_layer.map(l => l.bonus_R_k5));
    const x = d3.scaleLinear().domain([0, data.n_layers - 1]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, d3.max(allVals) * 1.1]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(data.n_layers).tickFormat(d3.format('d')))
      .append('text').attr('x', iw / 2).attr('y', 38)
      .attr('fill', 'currentColor').attr('text-anchor', 'middle')
      .style('font-size', '12px').text('Layer');

    g.append('g').call(d3.axisLeft(y).ticks(5))
      .append('text').attr('transform', 'rotate(-90)').attr('x', -ih / 2).attr('y', -42)
      .attr('fill', 'currentColor').attr('text-anchor', 'middle')
      .style('font-size', '12px').text('bonus_R (k=5)');

    // Random baseline at 1x
    g.append('line').attr('x1', 0).attr('x2', iw)
      .attr('y1', y(1)).attr('y2', y(1))
      .attr('stroke', '#9ca3af').attr('stroke-dasharray', '3,3').attr('stroke-width', 1);
    g.append('text').attr('x', 4).attr('y', y(1) - 4)
      .style('font-size', '10px').attr('fill', '#9ca3af').text('random baseline (1\xd7)');

    const line = d3.line().x((_, i) => x(i)).y(d => y(d.bonus_R_k5)).curve(d3.curveMonotoneX);

    const tooltip = d3.select('body').append('div').attr('class', 'widget-tooltip')
      .style('opacity', 0).style('position', 'absolute');

    data.runs.forEach(run => {
      g.append('path').datum(run.per_layer)
        .attr('fill', 'none').attr('stroke', run.color).attr('stroke-width', 2)
        .attr('d', line);

      g.selectAll(null).data(run.per_layer).join('circle')
        .attr('r', 4)
        .attr('cx', (_, i) => x(i)).attr('cy', d => y(d.bonus_R_k5))
        .attr('fill', run.color).style('cursor', 'pointer')
        .on('mouseover', (ev, d) => {
          tooltip.transition().duration(80).style('opacity', 1);
          tooltip.html(`<strong>${run.label}</strong><br>Layer ${d.layer}<br>bonus_R: ${d.bonus_R_k5.toFixed(2)}\xd7<br>srank: ${d.srank.toFixed(2)}`)
            .style('left', (ev.pageX + 10) + 'px').style('top', (ev.pageY - 28) + 'px');
        })
        .on('mouseleave', () => tooltip.transition().duration(150).style('opacity', 0));
    });

    // Legend
    const legend = svg.append('g').attr('transform', `translate(${W - 125}, ${margin.top})`);
    data.runs.forEach((run, i) => {
      const lg = legend.append('g').attr('transform', `translate(0, ${i * 20})`);
      lg.append('line').attr('x2', 16).attr('stroke', run.color).attr('stroke-width', 2);
      lg.append('text').attr('x', 20).attr('dy', '0.35em').style('font-size', '11px')
        .attr('fill', 'currentColor').text(run.label);
    });
  }
})();
