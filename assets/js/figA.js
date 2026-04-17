(function () {
  const el = document.getElementById('widget-figA');
  if (!el || typeof d3 === 'undefined') return;

  const base = document.querySelector('meta[name="baseurl"]')?.content || '/lazy-rudder-paper';
  fetch(base + '/_data/srank.json')
    .then(r => r.json())
    .then(data => render(el, data))
    .catch(err => { el.textContent = 'Chart unavailable.'; console.error('figA:', err); });

  function render(container, data) {
    const W = Math.min(container.clientWidth || 640, 720), H = 340;
    const margin = { top: 20, right: 30, bottom: 50, left: 60 };
    const iw = W - margin.left - margin.right;
    const ih = H - margin.top - margin.bottom;

    const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('width', '100%');

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const models = data.models;
    const dValues = models.map(m => m.d_model);
    const x = d3.scaleLog().domain([Math.min(...dValues) * 0.7, Math.max(...dValues) * 1.3]).range([0, iw]);
    const allSrank = models.flatMap(m => [m.srank_dpo, m.srank_clm].filter(v => v !== null));
    const y = d3.scaleLinear().domain([0, Math.max(...allSrank) * 1.2]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).tickValues(dValues).tickFormat(d => d + ''))
      .append('text').attr('x', iw / 2).attr('y', 40)
      .attr('fill', 'currentColor').attr('text-anchor', 'middle')
      .style('font-size', '12px').text('d_model');

    g.append('g').call(d3.axisLeft(y).ticks(5))
      .append('text').attr('transform', 'rotate(-90)').attr('x', -ih / 2).attr('y', -45)
      .attr('fill', 'currentColor').attr('text-anchor', 'middle')
      .style('font-size', '12px').text('stable rank');

    // Floor reference line
    g.append('line')
      .attr('x1', 0).attr('x2', iw)
      .attr('y1', y(3.6)).attr('y2', y(3.6))
      .attr('stroke', '#9ca3af').attr('stroke-dasharray', '4,3').attr('stroke-width', 1);
    g.append('text').attr('x', iw - 4).attr('y', y(3.6) - 5)
      .attr('text-anchor', 'end').attr('fill', '#9ca3af')
      .style('font-size', '11px').text('srank \u2248 3.6 floor');

    const tooltip = d3.select('body').append('div').attr('class', 'widget-tooltip')
      .style('opacity', 0).style('position', 'absolute');

    const showTip = (event, d, label, val) => {
      tooltip.transition().duration(100).style('opacity', 1);
      tooltip.html(`<strong>${d.name}</strong><br>${label}: ${val.toFixed(2)}`)
        .style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 28) + 'px');
    };
    const hideTip = () => tooltip.transition().duration(200).style('opacity', 0);

    // DPO dots
    g.selectAll('.dot-dpo').data(models).join('circle')
      .attr('class', 'dot-dpo').attr('r', 7)
      .attr('cx', d => x(d.d_model)).attr('cy', d => y(d.srank_dpo))
      .attr('fill', '#4e79a7').attr('stroke', '#fff').attr('stroke-width', 1.5)
      .style('cursor', 'pointer')
      .on('mouseover', (ev, d) => showTip(ev, d, 'DPO srank', d.srank_dpo))
      .on('mouseleave', hideTip);

    // CLM dots
    const clmModels = models.filter(d => d.srank_clm !== null);
    if (clmModels.length) {
      g.selectAll('.dot-clm').data(clmModels).join('circle')
        .attr('class', 'dot-clm').attr('r', 7)
        .attr('cx', d => x(d.d_model)).attr('cy', d => y(d.srank_clm))
        .attr('fill', '#76b7b2').attr('stroke', '#fff').attr('stroke-width', 1.5)
        .style('cursor', 'pointer')
        .on('mouseover', (ev, d) => showTip(ev, d, 'CLM srank', d.srank_clm))
        .on('mouseleave', hideTip);
    }

    // Model labels
    g.selectAll('.label').data(models).join('text')
      .attr('class', 'label').attr('x', d => x(d.d_model)).attr('y', d => y(d.srank_dpo) - 12)
      .attr('text-anchor', 'middle').style('font-size', '11px').attr('fill', 'currentColor')
      .text(d => d.name);

    // Legend
    const legend = svg.append('g').attr('transform', `translate(${margin.left + 8}, ${H - 16})`);
    [['#4e79a7', 'DPO'], ['#76b7b2', 'CLM']].forEach(([c, l], i) => {
      const lg = legend.append('g').attr('transform', `translate(${i * 80}, 0)`);
      lg.append('circle').attr('r', 5).attr('fill', c);
      lg.append('text').attr('x', 9).attr('dy', '0.35em').style('font-size', '11px')
        .attr('fill', 'currentColor').text(l);
    });
  }
})();
