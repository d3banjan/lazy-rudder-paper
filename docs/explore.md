---
layout: pub
title: "Explore Data"
page_id: explore
extra_js:
  - /assets/js/explore.js
  - /assets/js/figA.js
  - /assets/js/figB.js
---

<div class="pub-section hero" id="explore-hero">

# Explore the Data

Compare stable rank and &gamma;-rudder signal across model scales. Select any combination of models to overlay them on the same chart.

</div>

---

<div class="pub-section" id="explore-chart-section">

What changes with scale?
{:.section-label}

## Per-layer geometry across 70M &rarr; 1B

<div id="explore-chart" class="widget"></div>

<p class="aside" markdown="0">70M and 160M have DPO adapters only. 410M and 1B have both DPO and CLM. Layer counts differ: 6 / 12 / 24 / 16 layers. Use "fractional depth" axis to align them.</p>

</div>

---

<div class="pub-section" id="explore-seeds">

Seed robustness
{:.section-label}

## Does the signal hold across random seeds?

<div id="widget-figB" class="widget" aria-label="Bonus_R per layer at 1B, DPO vs CLM, two seeds"></div>

<p class="aside" markdown="0">Two independent 1B runs: seed 42 and seed 117. Independent data draws. The curves track closely &mdash; the signal is not a seed artifact.</p>

</div>
