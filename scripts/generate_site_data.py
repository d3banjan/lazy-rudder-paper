#!/usr/bin/env python3
"""Generate widget-ready JSONs for GitHub Pages microsite from paper results.

Reads empirical JSON outputs and Lean source, emits 4 flat JSONs to docs/_data/:
  - srank.json: per-model stable rank floor (scatter plot)
  - bonus_r.json: per-layer bonus_R curves for 1B traces (line plot)
  - modules.json: per-module bonus_R by (module x layer) (small multiples)
  - lean_status.json: Lean theorem proof status (dashboard)
"""

import json
import re
from pathlib import Path
from statistics import mean


def build_srank():
    """Extract stable rank floor for each model size.

    Returns:
        dict with schema srank_v1, containing per-model srank_dpo and srank_clm.
    """
    srank_data = {
        "_schema": "srank_floor_v1",
        "_source_files": [
            "results/spectral_overlap_gamma_petri/results.json",
            "results/spectral_overlap_gamma/results.json",
            "results/spectral_overlap_gamma_1b_seed117/results.json",
        ],
        "models": [],
    }

    # 70M and 160M from petri
    petri_path = Path("results/spectral_overlap_gamma_petri/results.json")
    with open(petri_path) as f:
        petri = json.load(f)

    srank_data["models"].append(
        {
            "name": "70M",
            "d_model": 512,
            "srank_dpo": round(petri["pythia-70m"]["avg"]["srank"], 2),
            "srank_clm": None,
        }
    )

    srank_data["models"].append(
        {
            "name": "160M",
            "d_model": 768,
            "srank_dpo": round(petri["pythia-160m"]["avg"]["srank"], 2),
            "srank_clm": None,
        }
    )

    # 410M from spectral_overlap_gamma — adapter srank_delta (NOT base weight stable_rank)
    # spectral_autopsy/results.json contains base weight stable_rank (avg ~5.2) which is
    # irrelevant here. The adapter srank lives in spectral_overlap_gamma per-layer srank_delta.
    gamma_path = Path("results/spectral_overlap_gamma/results.json")
    with open(gamma_path) as f:
        gamma = json.load(f)

    dpo_r128 = gamma["runs"]["v2_dpo_r128"]["per_layer"]
    dpo_srank = round(mean(layer["srank_delta"] for layer in dpo_r128), 2)

    clm_r128 = gamma["runs"]["v3_clm_r128"]["per_layer"]
    clm_srank = round(mean(layer["srank_delta"] for layer in clm_r128), 2)

    srank_data["models"].append(
        {
            "name": "410M",
            "d_model": 1024,
            "srank_dpo": dpo_srank,
            "srank_clm": clm_srank,
        }
    )

    # 1B from seed117
    seed117_path = Path("results/spectral_overlap_gamma_1b_seed117/results.json")
    with open(seed117_path) as f:
        seed117 = json.load(f)

    # Use v2_dpo_r128_1b_s42 and v3_clm_r128_1b_s42
    runs = seed117["runs"]

    dpo_1b_per_layer = runs["v2_dpo_r128_1b_s42"]["per_layer"]
    dpo_1b_srank = round(mean(layer["srank_delta"] for layer in dpo_1b_per_layer), 2)

    clm_1b_per_layer = runs["v3_clm_r128_1b_s42"]["per_layer"]
    clm_1b_srank = round(mean(layer["srank_delta"] for layer in clm_1b_per_layer), 2)

    srank_data["models"].append(
        {
            "name": "1B",
            "d_model": 2048,
            "srank_dpo": dpo_1b_srank,
            "srank_clm": clm_1b_srank,
        }
    )

    return srank_data


def build_bonus_r():
    """Extract per-layer bonus_R k5 curves for 1B traces.

    Returns:
        dict with schema bonus_r_v1, containing 4 traces (dpo/clm x seed42/seed117).
    """
    seed117_path = Path("results/spectral_overlap_gamma_1b_seed117/results.json")
    with open(seed117_path) as f:
        seed117 = json.load(f)

    d_model = seed117["d_in"]
    n_layers = seed117["n_layers"]

    # Get random baseline k5
    random_baseline = seed117["random_baseline"]
    random_baseline_k5 = round(random_baseline["k5"]["p_right"], 5)

    runs_config = [
        ("v2_dpo_r128_1b_s42", "dpo_s42", "DPO seed 42", "#4e79a7"),
        ("v3_clm_r128_1b_s42", "clm_s42", "CLM seed 42", "#76b7b2"),
        ("v3_dpo_r128_1b_s117", "dpo_s117", "DPO seed 117", "#f28e2b"),
        ("v4_clm_r128_1b_s117", "clm_s117", "CLM seed 117", "#e15759"),
    ]

    bonus_r_data = {
        "_schema": "bonus_r_v1",
        "_source_files": ["results/spectral_overlap_gamma_1b_seed117/results.json"],
        "model": "pythia-1b",
        "n_layers": n_layers,
        "d_model": d_model,
        "random_baseline_k5": random_baseline_k5,
        "runs": [],
    }

    for run_key, run_id, label, color in runs_config:
        if run_key not in seed117["runs"]:
            print(f"Warning: {run_key} not found in runs")
            continue

        run_data = seed117["runs"][run_key]
        per_layer = run_data["per_layer"]

        trace = {
            "id": run_id,
            "label": label,
            "color": color,
            "per_layer": [],
        }

        for layer_data in per_layer:
            trace["per_layer"].append(
                {
                    "layer": layer_data["layer"],
                    "bonus_R_k5": round(layer_data["k5"]["bonus_right"], 4),
                    "srank": round(layer_data["srank_delta"], 3),
                }
            )

        bonus_r_data["runs"].append(trace)

    return bonus_r_data


def build_modules():
    """Extract per-module bonus_R by layer.

    Returns:
        dict with schema modules_v1, containing per-module per-layer data.
    """
    modules_path = Path("results/spectral_overlap_gamma_modules/results.json")
    with open(modules_path) as f:
        modules_json = json.load(f)

    # Module mappings
    module_names = [
        "attention.query_key_value",
        "attention.dense",
        "mlp.dense_h_to_4h",
        "mlp.dense_4h_to_h",
    ]
    module_labels = {
        "attention.query_key_value": "QKV projection",
        "attention.dense": "Output projection",
        "mlp.dense_h_to_4h": "MLP up",
        "mlp.dense_4h_to_h": "MLP down",
    }

    runs_config = [
        ("410m_dpo", "dpo", "DPO", "#4e79a7"),
        ("410m_clm", "clm", "CLM", "#76b7b2"),
    ]

    modules_data = {
        "_schema": "modules_v1",
        "_source_files": ["results/spectral_overlap_gamma_modules/results.json"],
        "modules": module_names,
        "module_labels": module_labels,
        "runs": [],
    }

    for run_key, run_id, label, color in runs_config:
        if run_key not in modules_json["runs"]:
            print(f"Warning: {run_key} not found in modules runs")
            continue

        run_data = modules_json["runs"][run_key]
        by_module = {}

        for module_name in module_names:
            if module_name not in run_data["modules"]:
                print(f"Warning: {module_name} not found in {run_key}")
                by_module[module_name] = []
                continue

            module_per_layer = run_data["modules"][module_name]["per_layer"]
            by_module[module_name] = [
                {
                    "layer": layer_data["layer"],
                    "bonus_R_k5": round(layer_data["bonus_R_k5"], 4),
                }
                for layer_data in module_per_layer
            ]

        modules_data["runs"].append(
            {
                "id": run_id,
                "label": label,
                "color": color,
                "by_module": by_module,
            }
        )

    return modules_data


def parse_lean_theorems(lean_path: Path) -> list[dict]:
    """Parse Lean file for theorem/lemma/def with sorry status.

    Args:
        lean_path: Path to SubspaceOverlap.lean

    Returns:
        List of dicts with name, kind, and status (proven/deferred/stub).
    """
    with open(lean_path) as f:
        content = f.read()

    theorems = []

    # Pattern: (theorem|lemma|def) name ... := body
    # Look for patterns with := on same or following line
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r"^(theorem|lemma|def)\s+(\w+)", line)

        if not match:
            i += 1
            continue

        kind = match.group(1)
        name = match.group(2)

        # Collect lines until we find := or by
        full_text = line
        j = i + 1
        while j < len(lines) and not re.search(r":=|by", lines[j]):
            full_text += "\n" + lines[j]
            j += 1

        # Add the line with := or by
        if j < len(lines):
            full_text += "\n" + lines[j]

            # Check next line too for sorry
            if j + 1 < len(lines):
                full_text += "\n" + lines[j + 1]

            # Determine status
            has_sorry = "sorry" in full_text
            is_stub = re.search(r"True\s*:=\s*sorry", full_text)

            if is_stub:
                status = "stub"
            elif has_sorry:
                status = "deferred"
            else:
                status = "proven"

            theorems.append(
                {
                    "name": name,
                    "kind": kind,
                    "status": status,
                }
            )

        i = j + 1

    return theorems


def _read_lean_count_from_tex(tex_path: Path, macro_name: str) -> int:
    """Extract a \\newcommand{\\<macro_name>}{<value>} integer from lean_status.tex."""
    with open(tex_path) as f:
        content = f.read()
    m = re.search(r"\\newcommand\{\\%s\}\{(\d+)\}" % re.escape(macro_name), content)
    if m:
        return int(m.group(1))
    raise ValueError(f"Macro \\{macro_name} not found in {tex_path}")


def build_lean_status():
    """Read Lean proof status from manuscript/lean_status.tex (authoritative source).

    lean/SubspaceOverlap.lean is a thin import wrapper; the actual theorems live
    in LeanMining/NeuralGeometry/SubspaceOverlap.lean (monorepo root).  Parsing
    the wrapper returns 0 theorems.  manuscript/lean_status.tex is auto-generated
    from the canonical Lean file via `make lean-status` and is committed — it is
    the single authoritative source for the site dashboard.

    Returns:
        dict with schema lean_status_v1, containing theorem counts and details.
    """
    tex_path = Path("manuscript/lean_status.tex")

    proven_count = _read_lean_count_from_tex(tex_path, "leanProvenCount")
    deferred_count = _read_lean_count_from_tex(tex_path, "leanInProofSorryCount")
    stub_count = _read_lean_count_from_tex(tex_path, "leanStubCount")
    partial_count = _read_lean_count_from_tex(tex_path, "leanPartialCount")
    paper_facing_sorry = _read_lean_count_from_tex(tex_path, "leanPaperFacingSorryCount")

    # Build per-theorem list from the \thmStatOf* macros (order matches the tex table)
    STATUS_MAP = {
        "\\proofCheck": "proven",
        "\\proofSorry": "deferred",
        "\\proofStub": "stub",
        "\\proofPartial": "partial",
    }
    with open(tex_path) as f:
        tex_content = f.read()
    theorems = []
    for m in re.finditer(
        r"\\newcommand\{\\thmStatOf([A-Za-z]+)\}\{(\\proof[A-Za-z]+)\}", tex_content
    ):
        camel_name = m.group(1)
        status_tex = m.group(2)
        # Convert CamelCase back to snake_case heuristically (lowercase at uppercase boundary)
        snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", camel_name).lower()
        status = STATUS_MAP.get(status_tex, status_tex)
        theorems.append({"name": snake_name, "kind": "theorem", "status": status})

    return {
        "_schema": "lean_status_v1",
        "_source_files": ["manuscript/lean_status.tex"],
        "counts": {
            "proven": proven_count,
            "deferred": deferred_count,
            "stub": stub_count,
            "partial": partial_count,
            "paper_facing_sorry": paper_facing_sorry,
        },
        "theorems": theorems,
    }


def build_explore():
    """Build per-layer geometry data for the 4-model interactive playground.

    Returns:
        dict with schema explore_v1, one entry per model (70M/160M/410M/1B)
        containing per-layer srank and bonus_R_k5 for each available objective.
    """
    models = []

    # ── 70M ──────────────────────────────────────────────────────────────────
    petri_path = Path("results/spectral_overlap_gamma_petri/results.json")
    with open(petri_path) as f:
        petri = json.load(f)

    for key, name, d_model, color in [
        ("pythia-70m", "70M", 512, "#4e79a7"),
        ("pythia-160m", "160M", 768, "#f28e2b"),
    ]:
        m = petri[key]
        n_layers = m["n_layers"]
        per_layer = []
        for entry in m["per_layer"]:
            lyr = entry["layer"]
            per_layer.append(
                {
                    "layer": lyr,
                    "layer_frac": round(lyr / max(n_layers - 1, 1), 4),
                    "srank": round(entry["srank"], 4),
                    "bonus_R_k5": round(entry["bonus_R_k5"], 4),
                }
            )
        models.append(
            {
                "name": name,
                "d_model": d_model,
                "n_layers": n_layers,
                "color": color,
                "dpo": {"per_layer": per_layer},
                "clm": None,
            }
        )

    # ── 410M ─────────────────────────────────────────────────────────────────
    # srank comes from spectral_autopsy (per-module list; average per layer).
    # bonus_R_k5 comes from spectral_overlap_gamma (per-layer srank_delta / k5).
    autopsy_path = Path("results/spectral_autopsy/results.json")
    with open(autopsy_path) as f:
        autopsy = json.load(f)

    gamma_path = Path("results/spectral_overlap_gamma/results.json")
    with open(gamma_path) as f:
        gamma = json.load(f)

    def _410m_per_layer(autopsy_run_key, gamma_run_key):
        autopsy_entries = autopsy[autopsy_run_key]
        gamma_entries = gamma["runs"][gamma_run_key]["per_layer"]
        n = len(gamma_entries)

        # Build srank lookup: layer -> mean stable_rank across modules
        from collections import defaultdict

        srank_by_layer = defaultdict(list)
        for e in autopsy_entries:
            srank_by_layer[e["layer"]].append(e["stable_rank"])
        avg_srank = {lyr: mean(vals) for lyr, vals in srank_by_layer.items()}

        result = []
        for g_entry in gamma_entries:
            lyr = g_entry["layer"]
            result.append(
                {
                    "layer": lyr,
                    "layer_frac": round(lyr / max(n - 1, 1), 4),
                    "srank": round(avg_srank.get(lyr, g_entry["srank_delta"]), 4),
                    "bonus_R_k5": round(g_entry["k5"]["bonus_right"], 4),
                }
            )
        return result

    models.append(
        {
            "name": "410M",
            "d_model": 1024,
            "n_layers": 24,
            "color": "#e15759",
            "dpo": {"per_layer": _410m_per_layer("v2_dpo_r128", "v2_dpo_r128")},
            "clm": {"per_layer": _410m_per_layer("v3_clm_r128", "v3_clm_r128")},
        }
    )

    # ── 1B ───────────────────────────────────────────────────────────────────
    seed117_path = Path("results/spectral_overlap_gamma_1b_seed117/results.json")
    with open(seed117_path) as f:
        seed117 = json.load(f)

    n_layers_1b = seed117["n_layers"]

    def _1b_per_layer(run_key):
        entries = seed117["runs"][run_key]["per_layer"]
        n = len(entries)
        result = []
        for e in entries:
            lyr = e["layer"]
            result.append(
                {
                    "layer": lyr,
                    "layer_frac": round(lyr / max(n - 1, 1), 4),
                    "srank": round(e["srank_delta"], 4),
                    "bonus_R_k5": round(e["k5"]["bonus_right"], 4),
                }
            )
        return result

    models.append(
        {
            "name": "1B",
            "d_model": 2048,
            "n_layers": n_layers_1b,
            "color": "#76b7b2",
            "dpo": {"per_layer": _1b_per_layer("v2_dpo_r128_1b_s42")},
            "clm": {"per_layer": _1b_per_layer("v3_clm_r128_1b_s42")},
        }
    )

    return {
        "_schema": "explore_v1",
        "_source_files": [
            "results/spectral_overlap_gamma_petri/results.json",
            "results/spectral_autopsy/results.json",
            "results/spectral_overlap_gamma/results.json",
            "results/spectral_overlap_gamma_1b_seed117/results.json",
        ],
        "models": models,
    }


def main():
    """Generate all 4 site data JSONs."""
    output_dir = Path("docs/assets/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate srank
    print("Generating srank.json...")
    srank_data = build_srank()
    with open(output_dir / "srank.json", "w") as f:
        json.dump(srank_data, f, indent=2)
    print(f"  {len(srank_data['models'])} models")

    # Generate bonus_r
    print("Generating bonus_r.json...")
    bonus_r_data = build_bonus_r()
    with open(output_dir / "bonus_r.json", "w") as f:
        json.dump(bonus_r_data, f, indent=2)
    print(f"  {len(bonus_r_data['runs'])} traces")

    # Generate modules
    print("Generating modules.json...")
    modules_data = build_modules()
    with open(output_dir / "modules.json", "w") as f:
        json.dump(modules_data, f, indent=2)
    print(f"  {len(modules_data['runs'])} runs")

    # Generate lean_status — write to both assets/data/ (consumed by JS) and
    # docs/_data/ (consumed by Jekyll templates as site.data.lean_status).
    print("Generating lean_status.json...")
    lean_status_data = build_lean_status()
    with open(output_dir / "lean_status.json", "w") as f:
        json.dump(lean_status_data, f, indent=2)
    jekyll_data_dir = Path("docs/_data")
    jekyll_data_dir.mkdir(parents=True, exist_ok=True)
    with open(jekyll_data_dir / "lean_status.json", "w") as f:
        json.dump(lean_status_data, f, indent=2)
    counts = lean_status_data["counts"]
    print(
        f"  {counts['proven']} proven, "
        f"{counts['deferred']} deferred, "
        f"{counts['stub']} stub"
    )

    # Generate explore
    print("Generating explore.json...")
    explore_data = build_explore()
    with open(output_dir / "explore.json", "w") as f:
        json.dump(explore_data, f, indent=2)
    print(f"  {len(explore_data['models'])} models")

    print("\nDone!")


if __name__ == "__main__":
    main()
