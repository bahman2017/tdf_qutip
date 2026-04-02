#!/usr/bin/env python3
"""
Scan ``outputs/*.png`` and regenerate ``docs/FIGURES_INDEX.md``.

Run from ``tdf_qutip`` root::

    PYTHONPATH=. python scripts/generate_docs_index.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"
TARGET = DOCS / "FIGURES_INDEX.md"

# One-line hints when filename alone is ambiguous (optional overrides)
CAPTIONS: dict[str, str] = {
    "correlation_xx.png": "Two-qubit ⟨XX⟩: standard vs TDF τ-drive.",
    "correlation_yy.png": "Two-qubit ⟨YY⟩: standard vs TDF.",
    "correlation_zz.png": "Two-qubit ⟨ZZ⟩: standard vs TDF.",
    "chsh_tdf_vs_standard.png": "CHSH-style S(t): standard vs TDF.",
    "tau_extraction_Cxx_fit.png": "Fitted τ-model vs data (C_xx).",
    "tau_extraction_Cyy_fit.png": "Fitted τ-model vs data (C_yy).",
    "tau_extraction_CHSH_fit.png": "Fitted τ-model vs data (CHSH).",
    "tau_extraction_tau.png": "Extracted τ(t) from correlation fit.",
    "tau_identifiability_tau_overlay.png": "Multi-start τ curves (identifiability).",
    "tau_identifiability_param_histograms.png": "Parameter histograms across restarts.",
    "tau_manifold_param_space.png": "Low-loss runs in parameter PCA space.",
    "tau_manifold_tau_overlay.png": "τ(t) ensemble on degeneracy manifold.",
    "tau_manifold_embedding.png": "2D PCA embedding of τ trajectories.",
    "tau_kernel_modes.png": "Manifold mean τ and leading PCA kernel modes φ_k(t).",
    "tau_kernel_variance.png": "Explained variance of τ residual PCA.",
    "tau_kernel_sensitivity.png": "Observable RMSE vs λ along each kernel mode.",
    "tau_mode_fit_mode1.png": "ODE fit (damped vs relax) for kernel mode 1.",
    "tau_mode_fit_mode2.png": "ODE fit for kernel mode 2.",
    "tau_mode_fit_mode3.png": "ODE fit for kernel mode 3.",
    "tau_mode_fit_summary.png": "Explained variance + best ODE model per mode.",
    "hidden_spectrum_bar.png": "Effective λ_eff per mode (damped vs relaxation).",
    "hidden_spectrum_compact_fit.png": "Oscillatory λ vs n with A n² compact reference.",
    "chi_geometry_spectrum_models.png": "Oscillatory tower vs flat / offset / warped χ fits.",
    "chi_geometry_soft_mode.png": "Soft (relax) vs oscillatory λ_eff (bars).",
    "tdf_vs_colored_noise_joint_scores.png": "Joint cross-observable coupling scores (three models).",
    "tdf_vs_noise_metrics_tdf.png": "Sweep metrics vs τ_c (TDF).",
    "tdf_vs_noise_metrics_ou_colored.png": "Sweep metrics vs OU correlation time.",
    "tdf_vs_noise_metrics_pink.png": "Sweep metrics vs pink noise strength.",
    "tdf_vs_noise_coupling_tdf.png": "Cross-metric Pearson heatmap (TDF sweep).",
    "tdf_vs_noise_coupling_ou_colored.png": "Coupling heatmap (OU sweep).",
    "tdf_vs_noise_coupling_pink.png": "Coupling heatmap (pink sweep).",
}


def default_caption(name: str) -> str:
    if name in CAPTIONS:
        return CAPTIONS[name]
    stem = name.replace(".png", "").replace("_", " ")
    return f"Figure: {stem}."


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    pngs = sorted(OUTPUTS.glob("*.png")) if OUTPUTS.is_dir() else []

    lines = [
        "# Figures index",
        "",
        "PNG files under [`outputs/`](../outputs/) (regenerate locally; some may be gitignored).",
        "",
        "To refresh this list after a pipeline run:",
        "",
        "```bash",
        "cd tdf_qutip",
        "PYTHONPATH=. python3 scripts/generate_docs_index.py",
        "```",
        "",
        "---",
        "",
    ]

    if not pngs:
        lines.append("*No PNG files found in `outputs/` — run the pipeline in `docs/REPRO.md`.*")
        lines.append("")
    else:
        for p in pngs:
            rel = f"../outputs/{p.name}"
            lines.append(f"* [`{p.name}`]({rel}) — {default_caption(p.name)}")
        lines.append("")

    TARGET.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {TARGET} ({len(pngs)} figures)")


if __name__ == "__main__":
    main()
