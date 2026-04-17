(function () {
  const container = document.getElementById('explore-chart');
  if (!container || typeof d3 === 'undefined') return;

  const base = document.querySelector('meta[name="baseurl"]')?.content || '/lazy-rudder-paper';
  fetch(base + '/assets/data/explore.json')
    .then(r => r.json())
    .then(data => init(container, data))
    .catch(err => { container.textContent = 'Chart unavailable.'; console.error('explore:', err); });

  function init(root, data) {
    // ── State ────────────────────────────────────────────────────────────────
    const state = {
      models: new Set(data.models.map(m => m.name)),
      metric: 'srank',           // 'srank' | 'bonus_R_k5'
      objective: 'dpo',          // 'dpo' | 'clm' | 'both'
      layerAxis: 'absolute',     // 'absolute' | 'fractional'
    };

    // ── Controls ─────────────────────────────────────────────────────────────
    const controls = document.createElement('div');
    controls.className = 'explore-controls';

    // Model checkboxes
    const modelGroup = makeGroup('Models');
    data.models.forEach(m => {
      const lbl = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = true;
      cb.addEventListener('change', () => {
        if (cb.checked) state.models.add(m.name);
        else state.models.delete(m.name);
        redraw();
      });
      const dot = document.createElement('span');
      dot.style.cssText = `display:inline-block;width:10px;height:10px;border-radius:50%;background:${m.color};flex-shrink:0`;
      lbl.appendChild(cb);
      lbl.appendChild(dot);
      lbl.appendChild(document.createTextNode(' ' + m.name));
      modelGroup.appendChild(lbl);
    });
    controls.appendChild(modelGroup);

    // Metric radio
    const metricGroup = makeGroup('Metric');
    [['srank', 'Stable rank'], ['bonus_R_k5', 'bonus_R (k=5)']].forEach(([val, label]) => {
      const lbl = makeRadio('explore-metric', val, label, val === state.metric, () => {
        state.metric = val;
        redraw();
      });
      metricGroup.appendChild(lbl);
    });
    controls.appendChild(metricGroup);

    // Objective radio
    const objGroup = makeGroup('Objective');
    [['dpo', 'DPO'], ['clm', 'CLM'], ['both', 'Both']].forEach(([val, label]) => {
      const lbl = makeRadio('explore-obj', val, label, val === state.objective, () => {
        state.objective = val;
        redraw();
      });
      objGroup.appendChild(lbl);
    });
    controls.appendChild(objGroup);

    // Layer axis radio
    const axisGroup = makeGroup('Layer axis');
    [['absolute', 'Absolute index'], ['fractional', 'Fractional depth (0→1)']].forEach(([val, label]) => {
      const lbl = makeRadio('explore-axis', val, label, val === state.layerAxis, () => {
        state.layerAxis = val;
        redraw();
      });
      axisGroup.appendChild(lbl);
    });
    controls.appendChild(axisGroup);

    root.appendChild(controls);

    // Note line
    const note = document.createElement('p');
    note.className = 'explore-note';
    root.appendChild(note);

    // Chart container
    const chartDiv = document.createElement('div');
    chartDiv.style.width = '100%';
    root.appendChild(chartDiv);

    const tooltip = d3.select('body').append('div').attr('class', 'widget-tooltip')
      .style('opacity', 0).style('position', 'absolute').style('pointer-events', 'none');

    // ── Draw ─────────────────────────────────────────────────────────────────
    function redraw() {
      d3.select(chartDiv).selectAll('*').remove();

      const W = Math.min(root.clientWidth || 700, 720);
      const H = 360;
      const margin = { top: 22, right: 20, bottom: 52, left: 58 };
      const iw = W - margin.left - margin.right;
      const ih = H - margin.top - margin.bottom;

      const svg = d3.select(chartDiv).append('svg')
        .attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%');
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      // Collect traces
      const traces = [];
      const skippedModels = [];

      data.models.forEach(m => {
        if (!state.models.has(m.name)) return;

        const objectives = state.objective === 'both'
          ? [['dpo', false], ['clm', true]]
          : [[state.objective, false]];

        objectives.forEach(([obj, dashed]) => {
          const src = m[obj];
          if (!src) {
            // Only note skip when CLM is explicitly requested
            if (obj === 'clm') skippedModels.push(m.name);
            return;
          }
          traces.push({
            model: m.name,
            color: m.color,
            dashed,
            obj,
            points: src.per_layer,
            n_layers: m.n_layers,
          });
        });
      });

      // Update note
      note.textContent = skippedModels.length
        ? `No CLM data for: ${skippedModels.join(', ')} (DPO adapters only).`
        : '';

      if (!traces.length) {
        g.append('text').attr('x', iw / 2).attr('y', ih / 2)
          .attr('text-anchor', 'middle').attr('fill', 'currentColor')
          .style('font-size', '13px').text('No models selected.');
        return;
      }

      // Scales
      const xKey = state.layerAxis === 'fractional' ? 'layer_frac' : 'layer';
      const yKey = state.metric;

      const allX = traces.flatMap(t => t.points.map(p => p[xKey]));
      const allY = traces.flatMap(t => t.points.map(p => p[yKey]));
      const xDomain = [d3.min(allX), d3.max(allX)];
      const yMax = d3.max(allY) * 1.12;

      const x = d3.scaleLinear().domain(xDomain).range([0, iw]);
      const y = d3.scaleLinear().domain([0, yMax]).range([ih, 0]);

      // Axes
      const xTickCount = state.layerAxis === 'fractional' ? 6 : Math.max(...traces.map(t => t.n_layers));
      const xAxis = state.layerAxis === 'fractional'
        ? d3.axisBottom(x).ticks(6).tickFormat(d3.format('.1f'))
        : d3.axisBottom(x).ticks(xTickCount).tickFormat(d3.format('d'));

      g.append('g').attr('transform', `translate(0,${ih})`).call(xAxis)
        .append('text').attr('x', iw / 2).attr('y', 40)
        .attr('fill', 'currentColor').attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text(state.layerAxis === 'fractional' ? 'Depth (fraction of model)' : 'Layer');

      g.append('g').call(d3.axisLeft(y).ticks(5))
        .append('text').attr('transform', 'rotate(-90)').attr('x', -ih / 2).attr('y', -44)
        .attr('fill', 'currentColor').attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text(state.metric === 'srank' ? 'Stable rank' : 'bonus_R (k=5)');

      // Lines + dots
      const line = d3.line()
        .x(p => x(p[xKey])).y(p => y(p[yKey]))
        .curve(d3.curveMonotoneX);

      traces.forEach(tr => {
        g.append('path').datum(tr.points)
          .attr('fill', 'none')
          .attr('stroke', tr.color)
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', tr.dashed ? '5,3' : null)
          .attr('d', line);

        g.selectAll(null).data(tr.points).join('circle')
          .attr('r', 3.5)
          .attr('cx', p => x(p[xKey]))
          .attr('cy', p => y(p[yKey]))
          .attr('fill', tr.color)
          .attr('opacity', 0.85)
          .style('cursor', 'pointer')
          .on('mouseover', (ev, d) => {
            const val = d[yKey];
            const xVal = state.layerAxis === 'fractional'
              ? `depth ${d.layer_frac.toFixed(2)}`
              : `layer ${d.layer}`;
            const metricLabel = state.metric === 'srank' ? 'srank' : 'bonus_R (k=5)';
            const objLabel = tr.obj.toUpperCase();
            tooltip.transition().duration(60).style('opacity', 1);
            tooltip.html(
              `<strong>${tr.model} ${objLabel}</strong><br>${xVal}<br>${metricLabel}: ${val.toFixed(3)}`
            ).style('left', (ev.pageX + 12) + 'px').style('top', (ev.pageY - 32) + 'px');
          })
          .on('mouseleave', () => tooltip.transition().duration(120).style('opacity', 0));
      });

      // Legend (inline, right side of chart area if space, else bottom)
      const legendX = iw - 4;
      const legendAnchor = 'end';
      const visibleModels = [...new Set(traces.map(t => t.model))];
      visibleModels.forEach((name, i) => {
        const m = data.models.find(m => m.name === name);
        const lg = g.append('g').attr('transform', `translate(${legendX}, ${i * 18})`);
        lg.append('line').attr('x1', -24).attr('x2', -6)
          .attr('stroke', m.color).attr('stroke-width', 2);
        lg.append('circle').attr('cx', -15).attr('cy', 0).attr('r', 3).attr('fill', m.color);
        lg.append('text').attr('x', 0).attr('dy', '0.35em')
          .style('font-size', '11px').attr('fill', 'currentColor')
          .attr('text-anchor', legendAnchor).text(name);
      });

      // Dashed/solid legend when objective=both
      if (state.objective === 'both') {
        const ly = visibleModels.length * 18 + 8;
        [['DPO', false], ['CLM', true]].forEach(([label, dashed], i) => {
          const lg = g.append('g').attr('transform', `translate(${legendX}, ${ly + i * 18})`);
          lg.append('line').attr('x1', -24).attr('x2', -6)
            .attr('stroke', '#555').attr('stroke-width', 1.5)
            .attr('stroke-dasharray', dashed ? '5,3' : null);
          lg.append('text').attr('x', 0).attr('dy', '0.35em')
            .style('font-size', '11px').attr('fill', 'currentColor')
            .attr('text-anchor', legendAnchor).text(label);
        });
      }
    }

    redraw();
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────
  function makeGroup(labelText) {
    const div = document.createElement('div');
    div.className = 'control-group';
    const span = document.createElement('span');
    span.className = 'control-label';
    span.textContent = labelText;
    div.appendChild(span);
    return div;
  }

  function makeRadio(name, value, labelText, checked, onChange) {
    const lbl = document.createElement('label');
    const inp = document.createElement('input');
    inp.type = 'radio';
    inp.name = name;
    inp.value = value;
    inp.checked = checked;
    inp.addEventListener('change', () => { if (inp.checked) onChange(); });
    lbl.appendChild(inp);
    lbl.appendChild(document.createTextNode(' ' + labelText));
    return lbl;
  }
})();
