from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


DPG_COLORS: Dict[str, str] = {
    "gold": "#E3C800",
    "amber": "#F0A30A",
    "orange": "#FA6800",
    "olive": "#38432A",
    "brown": "#A95E19",
    "olive_light": "#97A15A",
    "sage": "#6E7C3A",
    "moss": "#596B31",
    "fern": "#4F5F34",
    "pine": "#2F4A32",
    "steel": "#6C8EA3",
    "plum": "#8A6FB3",
    "sand": "#C6AF73",
    "clay": "#C67A2D",
    "ink": "#111111",
    "charcoal": "#2E2E2E",
    "mid_gray": "#8A8A8A",
    "light_gray": "#D9D9D9",
    "soft_gray": "#ECE8DF",
    "paper": "#FAFAF7",
    "node_fill": "#F3E8C8",
    "node_muted": "#E7E0D2",
    "grid": "#D8D1C2",
    "edge": "#8C836C",
    "range_fill": "#D9D9D9",
    "range_marker": "#5F5A50",
    "success": "#6E7C3A",
    "danger": "#A95E19",
}


DPG_CLASS_PALETTE: List[str] = [
    DPG_COLORS["gold"],
    DPG_COLORS["steel"],
    DPG_COLORS["orange"],
    DPG_COLORS["fern"],
    DPG_COLORS["amber"],
    DPG_COLORS["plum"],
    DPG_COLORS["brown"],
    DPG_COLORS["olive"],
    DPG_COLORS["sage"],
    DPG_COLORS["pine"],
    DPG_COLORS["clay"],
    DPG_COLORS["moss"],
    DPG_COLORS["sand"],
    DPG_COLORS["olive_light"],
]


DPG_PREDICATE_LINE_PALETTE: List[str] = [
    DPG_COLORS["gold"],
    DPG_COLORS["olive_light"],
    DPG_COLORS["amber"],
    DPG_COLORS["sage"],
    DPG_COLORS["clay"],
    DPG_COLORS["olive"],
    DPG_COLORS["orange"],
    DPG_COLORS["pine"],
]

DPG_OLIVE_CLASS_PALETTE: List[str] = [
    DPG_COLORS["sand"],
    DPG_COLORS["olive_light"],
    DPG_COLORS["sage"],
    DPG_COLORS["moss"],
    DPG_COLORS["fern"],
    DPG_COLORS["olive"],
    DPG_COLORS["pine"],
    DPG_COLORS["steel"],
    DPG_COLORS["plum"],
    DPG_COLORS["gold"],
    DPG_COLORS["amber"],
    DPG_COLORS["clay"],
    DPG_COLORS["brown"],
    DPG_COLORS["orange"],
]

LEGACY_COLORS: Dict[str, str] = {
    "paper": "#FFFFFF",
    "ink": "#111111",
    "charcoal": "#222222",
    "mid_gray": "#808080",
    "light_gray": "#D9D9D9",
    "grid": "#D9D9D9",
    "edge": "#7A7A7A",
    "node_fill": "#DEEAF7",
    "node_muted": "#DEEAF7",
    "class_fill": "#FFC000",
    "range_fill": "#D3D3D3",
    "range_marker": "#696969",
    "success": "#2CA02C",
    "danger": "#D62728",
}


DPG_MPL_STYLE: Dict[str, Any] = {
    "figure.facecolor": DPG_COLORS["paper"],
    "axes.facecolor": DPG_COLORS["paper"],
    "savefig.facecolor": DPG_COLORS["paper"],
    "axes.edgecolor": DPG_COLORS["grid"],
    "axes.labelcolor": DPG_COLORS["charcoal"],
    "axes.titlecolor": DPG_COLORS["ink"],
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "font.family": "DejaVu Sans",
    "text.color": DPG_COLORS["charcoal"],
    "xtick.color": DPG_COLORS["charcoal"],
    "ytick.color": DPG_COLORS["charcoal"],
    "grid.color": DPG_COLORS["grid"],
    "grid.linewidth": 0.8,
    "grid.alpha": 0.55,
    "legend.frameon": True,
    "legend.facecolor": "#FFFDF7",
    "legend.edgecolor": DPG_COLORS["light_gray"],
}

LEGACY_MPL_STYLE: Dict[str, Any] = {
    "figure.facecolor": LEGACY_COLORS["paper"],
    "axes.facecolor": LEGACY_COLORS["paper"],
    "savefig.facecolor": LEGACY_COLORS["paper"],
    "axes.edgecolor": LEGACY_COLORS["grid"],
    "axes.labelcolor": LEGACY_COLORS["charcoal"],
    "axes.titlecolor": LEGACY_COLORS["ink"],
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "font.family": "DejaVu Sans",
    "text.color": LEGACY_COLORS["charcoal"],
    "xtick.color": LEGACY_COLORS["charcoal"],
    "ytick.color": LEGACY_COLORS["charcoal"],
    "grid.color": LEGACY_COLORS["grid"],
    "grid.linewidth": 0.8,
    "grid.alpha": 0.4,
    "legend.frameon": True,
    "legend.facecolor": "#FFFFFF",
    "legend.edgecolor": LEGACY_COLORS["light_gray"],
}


def brand_sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "dpg_brand_sequential",
        [
            DPG_COLORS["paper"],
            "#F2E7AD",
            DPG_COLORS["gold"],
            DPG_COLORS["amber"],
            DPG_COLORS["orange"],
        ],
    )


def edge_sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "dpg_edge_sequential",
        [
            DPG_COLORS["light_gray"],
            "#B9AF98",
            DPG_COLORS["edge"],
            DPG_COLORS["olive"],
        ],
    )


def community_cmap() -> ListedColormap:
    return ListedColormap(DPG_CLASS_PALETTE, name="dpg_community")


def class_cmap(n_classes: int) -> ListedColormap:
    return ListedColormap(discrete_palette(n_classes), name="dpg_classes")


def discrete_palette(n_colors: int) -> List[str]:
    if n_colors <= 0:
        return []
    repeats = (n_colors // len(DPG_CLASS_PALETTE)) + 1
    return (DPG_CLASS_PALETTE * repeats)[:n_colors]


def feature_color_map(features: List[str]) -> Dict[str, Any]:
    unique = list(dict.fromkeys(features))
    palette = discrete_palette(len(unique))
    return {feature: mcolors.to_rgba(color) for feature, color in zip(unique, palette)}


def predicate_line_color_map(features: List[str]) -> Dict[str, Any]:
    unique = list(dict.fromkeys(features))
    if not unique:
        return {}
    repeats = (len(unique) // len(DPG_PREDICATE_LINE_PALETTE)) + 1
    palette = (DPG_PREDICATE_LINE_PALETTE * repeats)[:len(unique)]
    return {feature: mcolors.to_rgba(color) for feature, color in zip(unique, palette)}


def _palette_values(palette: str) -> List[str]:
    name = str(palette or "default").lower()
    if name in {"default", "brand", "extended"}:
        return DPG_CLASS_PALETTE
    if name == "olive":
        return DPG_OLIVE_CLASS_PALETTE
    raise ValueError(f"Unknown palette '{palette}'. Expected one of: default, extended, olive.")


def _spaced_palette(palette_values: List[str], n_colors: int) -> List[str]:
    if n_colors <= 0:
        return []
    if not palette_values:
        return []
    if n_colors == 1:
        return [palette_values[min(len(palette_values) // 2, len(palette_values) - 1)]]

    if n_colors <= len(palette_values):
        max_index = len(palette_values) - 1
        raw_positions = [round(i * max_index / (n_colors - 1)) for i in range(n_colors)]
        deduped_positions: List[int] = []
        for pos in raw_positions:
            candidate = int(pos)
            while candidate in deduped_positions and candidate < max_index:
                candidate += 1
            while candidate in deduped_positions and candidate > 0:
                candidate -= 1
            deduped_positions.append(candidate)
        return [palette_values[idx] for idx in deduped_positions]

    repeats = (n_colors // len(palette_values)) + 1
    return (palette_values * repeats)[:n_colors]


def resolve_theme_context(theme: str = "dpg", palette: str = "default") -> Dict[str, Any]:
    theme_name = str(theme or "dpg").lower()
    palette_name = str(palette or "default").lower()

    if theme_name == "legacy":
        class_palette = _palette_values("default")
        return {
            "theme": "legacy",
            "palette": palette_name,
            "colors": LEGACY_COLORS,
            "mpl_style": LEGACY_MPL_STYLE,
            "class_palette": class_palette,
            "predicate_palette": ["#1F77B4", "#2CA02C", "#9467BD", "#17BECF"],
            "sequential_cmap": cm.Blues,
            "edge_cmap": cm.Greys,
            "community_cmap": cm.get_cmap("tab20"),
            "class_cmap": lambda n: cm.get_cmap("viridis", n),
            "feature_color_map": lambda features: {
                feature: cm.tab20(i / max(1, len(list(dict.fromkeys(features))) - 1))
                for i, feature in enumerate(list(dict.fromkeys(features)))
            },
            "predicate_line_color_map": lambda features: {
                feature: cm.tab10((i + 5) / 10)
                for i, feature in enumerate(list(dict.fromkeys(features)))
            },
        }

    if theme_name != "dpg":
        raise ValueError(f"Unknown theme '{theme}'. Expected one of: dpg, legacy.")

    class_palette = _palette_values(palette_name)
    predicate_palette = DPG_PREDICATE_LINE_PALETTE if palette_name != "olive" else [
        DPG_COLORS["gold"],
        DPG_COLORS["olive_light"],
        DPG_COLORS["amber"],
        DPG_COLORS["sage"],
        DPG_COLORS["clay"],
        DPG_COLORS["olive"],
        DPG_COLORS["sand"],
        DPG_COLORS["pine"],
    ]

    return {
        "theme": "dpg",
        "palette": palette_name,
        "colors": {
            **DPG_COLORS,
            "class_fill": DPG_COLORS["orange"],
        },
        "mpl_style": DPG_MPL_STYLE,
        "class_palette": class_palette,
        "predicate_palette": predicate_palette,
        "sequential_cmap": brand_sequential_cmap(),
        "edge_cmap": edge_sequential_cmap(),
        "community_cmap": ListedColormap(class_palette, name=f"dpg_community_{palette_name}"),
        "class_cmap": lambda n: ListedColormap(_spaced_palette(class_palette, n), name=f"dpg_classes_{palette_name}"),
        "feature_color_map": lambda features: {
            feature: mcolors.to_rgba(color)
            for feature, color in zip(
                list(dict.fromkeys(features)),
                _spaced_palette(class_palette, len(list(dict.fromkeys(features)))),
            )
        },
        "predicate_line_color_map": lambda features: {
            feature: mcolors.to_rgba(color)
            for feature, color in zip(
                list(dict.fromkeys(features)),
                (predicate_palette * (((len(list(dict.fromkeys(features))) - 1) // len(predicate_palette)) + 1))[: len(list(dict.fromkeys(features)))],
            )
        },
    }
