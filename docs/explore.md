---
layout: pub
title: "Explore Data"
page_id: explore
extra_js:
  - /assets/js/figA.js
  - /assets/js/figB.js
  - /assets/js/figE.js
---

<div class="pub-section" id="explore-intro">

## Explore the Data

All charts from the paper, interactive. Hover any point for exact values.

</div>

---

<div class="pub-section" id="explore-srank">

## Srank floor across model scales

<div id="widget-figA" class="widget"></div>

Source: `spectral_overlap_gamma_petri` + `spectral_autopsy` + `spectral_overlap_gamma_1b_seed117`

</div>

---

<div class="pub-section" id="explore-gamma">

## γ bonus_R per layer — Pythia-1B

<div id="widget-figB" class="widget"></div>

Four traces: DPO and CLM at two independent seeds (42 and 117). Random baseline shown as dashed reference.

</div>

---

<div class="pub-section" id="explore-modules">

## γ across adapter modules — Pythia-410M

<div id="widget-figE" class="widget"></div>

Two traces (DPO, CLM) across four LoRA target modules.

</div>
