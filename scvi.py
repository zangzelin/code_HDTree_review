from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train standard scVI on celegan and save UMAP plots."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/celegan"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reb/output/celegan_scvi")
    )
    parser.add_argument("--num-top-celltype", type=int, default=7)
    parser.add_argument("--top-genes", type=int, default=500)
    parser.add_argument("--n-latent", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_embryo_time(value: str) -> float:
    if "-" in value:
        return float(value.split("-")[1])
    if "<" in value:
        return float(value.split("<")[1]) - 50.0
    if ">" in value:
        return float(value.split(">")[1]) + 100.0
    raise ValueError(f"Unsupported embryo time format: {value}")


def matrix_variance(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        return mean_sq - np.square(mean)
    return np.var(np.asarray(matrix), axis=0)


def ensure_float32(matrix):
    if sparse.issparse(matrix):
        return matrix.astype(np.float32).tocsr()
    return np.asarray(matrix, dtype=np.float32)


def load_celegan_for_scvi(data_dir: Path, num_top_celltype: int, top_genes: int):
    import scanpy as sc

    adata = sc.read_h5ad(data_dir / "celegan.h5ad")

    label_celltype = pd.read_csv(
        data_dir / "celegan_celltype_2.tsv", sep="\t", header=None
    )[0].astype(str)
    label_embryo_time = pd.read_csv(
        data_dir / "celegan_embryo_time.tsv", sep="\t", header=None
    )[0].astype(str)

    if adata.n_obs != len(label_celltype):
        raise ValueError(
            f"Cell count mismatch: adata has {adata.n_obs}, labels have {len(label_celltype)}"
        )

    adata.obs["celltype"] = pd.Categorical(label_celltype.to_numpy())
    adata.obs["embryo_time"] = pd.Categorical(label_embryo_time.to_numpy())
    adata.obs["embryo_time_value"] = label_embryo_time.map(parse_embryo_time).to_numpy(
        dtype=np.float32
    )

    gene_var = matrix_variance(adata.X)
    top_gene_idx = np.argsort(gene_var)[-top_genes:]
    adata = adata[:, top_gene_idx].copy()

    celltype_counts = adata.obs["celltype"].value_counts().sort_values(ascending=False)
    top_celltypes = celltype_counts.head(num_top_celltype).index.tolist()
    print("Top celltypes:")
    for celltype, count in celltype_counts.head(num_top_celltype).items():
        print(f"  {celltype}: {count}")

    adata = adata[adata.obs["celltype"].isin(top_celltypes)].copy()
    adata.obs["celltype"] = pd.Categorical(adata.obs["celltype"].astype(str))

    # scVI expects count-like inputs rather than the z-score normalization used elsewhere.
    adata.X = ensure_float32(adata.X)
    adata.layers["counts"] = adata.X.copy()

    print("Filtered shape:", adata.shape)
    return adata


def save_umap_figures(adata, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    embedding = adata.obsm["X_umap"]
    x = embedding[:, 0]
    y = embedding[:, 1]

    def style_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        ax.set_facecolor("white")

    def pretty_label(label: str) -> str:
        return textwrap.fill(label.replace("_", " "), width=22)

    color_map = {
        "Ciliated_non_amphid_neuron": "#2ca02c",
        "Hypodermis": "#e377c2",
        "Ciliated_amphid_neuron": "#ff7f0e",
        "Body_wall_muscle": "#1f77b4",
        "Pharyngeal_muscle": "#9467bd",
        "unannotated": "#8c564b",
        "Seam_cell": "#d62728",
    }
    # tab:blue : #1f77b4
    # tab:orange : #ff7f0e
    # tab:green : #2ca02c
    # tab:red : #d62728
    # tab:purple : #9467bd
    # tab:brown : #8c564b
    # tab:pink : #e377c2
    # tab:gray : #7f7f7f
    # tab:olive : #bcbd22
    # tab:cyan : #17becf
    
    fallback_palette = [
        "#17becf",
        "#bcbd22",
        "#7f7f7f",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
    ]

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    observed_celltypes = adata.obs["celltype"].astype(str)
    categories = pd.Index(observed_celltypes).unique().tolist()
    fallback_idx = 0
    for celltype in categories:
        if celltype not in color_map:
            color_map[celltype] = fallback_palette[fallback_idx % len(fallback_palette)]
            fallback_idx += 1

    plot_order = [celltype for celltype in categories if celltype == "unannotated"] + [
        celltype for celltype in categories if celltype != "unannotated"
    ]
    for celltype in plot_order:
        mask = (observed_celltypes == celltype).to_numpy()
        ax.scatter(
            x[mask],
            y[mask],
            c=color_map[celltype],
            s=4,
            alpha=0.55 if celltype == "unannotated" else 0.9,
            linewidths=0,
            rasterized=True,
        )
    legend_order = [celltype for celltype in color_map if celltype in categories]
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[celltype],
            markeredgecolor="none",
            markersize=7,
            label=pretty_label(str(celltype)),
        )
        for celltype in legend_order
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        handlelength=0,
        handletextpad=0.5,
        labelspacing=0.9,
        fontsize=10,
    )
    style_axis(ax)
    fig.savefig(
        output_dir / "umap_celltype.png", dpi=220, bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    continuous = adata.obs["embryo_time_value"].to_numpy()
    vmax = np.quantile(np.abs(continuous - np.median(continuous)), 0.98)
    center = np.median(continuous)
    scatter = ax.scatter(
        x,
        y,
        c=continuous,
        s=4,
        alpha=0.9,
        linewidths=0,
        cmap="RdBu",
        vmin=center - vmax,
        vmax=center + vmax,
    )
    style_axis(ax)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(
        output_dir / "umap_embryo_time.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import scanpy as sc
    except Exception as exc:  # pragma: no cover - import guard for environment issues
        raise RuntimeError(
            "Failed to import scanpy. Please verify the scanpy/anndata environment before running this script."
        ) from exc

    try:
        import scvi
    except Exception as exc:  # pragma: no cover - import guard for missing dependency
        raise RuntimeError(
            "Failed to import scvi-tools. Install it first, for example: pip install scvi-tools"
        ) from exc

    scvi.settings.seed = args.seed
    sc.set_figure_params(dpi=150, figsize=(6, 5))

    adata = load_celegan_for_scvi(
        data_dir=args.data_dir,
        num_top_celltype=args.num_top_celltype,
        top_genes=args.top_genes,
    )
    

    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.SCVI(adata, n_latent=args.n_latent)
    model.train(max_epochs=args.max_epochs)

    adata.obsm["X_scVI"] = model.get_latent_representation()
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata)

    save_umap_figures(adata, args.output_dir)
    adata.write(args.output_dir / "celegan_scvi_umap.h5ad")
    pd.DataFrame(adata.obsm["X_scVI"], index=adata.obs_names).to_csv(
        args.output_dir / "scvi_latent.csv"
    )
    adata.obs[["celltype", "embryo_time", "embryo_time_value"]].to_csv(
        args.output_dir / "metadata.csv"
    )

    print(f"Saved scVI outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
