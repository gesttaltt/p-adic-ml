"""Shared matplotlib style and color constants for all p-adic-ml visualizations."""
import matplotlib as mpl
from matplotlib.patches import Patch

# ── Sequence-type colors ────────────────────────────────────────────────────
RATIONAL_COLOR  = '#4C72B0'   # muted blue
ALGEBRAIC_COLOR = '#C44E52'   # muted red
GENERATED_COLOR = '#55A868'   # muted green
RANDOM_COLOR    = '#8172B2'   # muted purple

# ── Per-prime colors (consistent across all plots) ──────────────────────────
_PRIME_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
]
_PRIMES_ORDERED = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def prime_color(p: int) -> str:
    """Return a consistent color for prime p."""
    try:
        return _PRIME_PALETTE[_PRIMES_ORDERED.index(p)]
    except ValueError:
        return '#17becf'

# ── Global style ────────────────────────────────────────────────────────────
def apply_style() -> None:
    """Call once per script to apply consistent rcParams project-wide."""
    mpl.rcParams.update({
        'figure.facecolor':    'white',
        'axes.facecolor':      '#fafafa',
        'axes.grid':           True,
        'grid.color':          '#e6e6e6',
        'axes.spines.top':     False,
        'axes.spines.right':   False,
        'font.family':         'sans-serif',
        'font.size':           11,
        'axes.titlesize':      12,
        'axes.titleweight':    'bold',
        'axes.labelsize':      10,
        'xtick.labelsize':     9,
        'ytick.labelsize':     9,
        'legend.fontsize':     9,
        'legend.framealpha':   0.85,
        'legend.edgecolor':    '#cccccc',
        'lines.linewidth':     1.5,
        'savefig.dpi':         150,
        'savefig.bbox':        'tight',
        'savefig.facecolor':   'white',
    })

# ── Reusable legend patches ──────────────────────────────────────────────────
def seq_type_legend_handles():
    """Matplotlib Patch handles for sequence-type legends."""
    return [
        Patch(color=RATIONAL_COLOR,  label='Rational'),
        Patch(color=ALGEBRAIC_COLOR, label='Algebraic'),
        Patch(color=GENERATED_COLOR, label='Generated'),
    ]
