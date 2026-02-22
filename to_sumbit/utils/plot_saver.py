import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple, List, Any


def sanitize(name: str) -> str:
    """Make a filename-safe slug."""
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_\-=.,]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "figure"


def _infer_name_from_fig(fig: Any) -> str:
    """Try to infer a short descriptive name from matplotlib figure titles."""
    try:
        # suptitle
        st = getattr(fig, "_suptitle", None)
        if st is not None:
            txt = getattr(st, "get_text", lambda: "")()
            if txt and txt.strip():
                return txt
    except Exception:
        pass
    try:
        axes = getattr(fig, "axes", [])
        for ax in axes:
            t = ax.get_title()
            if t and t.strip():
                return t
    except Exception:
        pass
    return "figure"


@dataclass
class PlotSaver:
    run_dir: str
    subdir: str = "figures"
    exts: Tuple[str, ...] = ("png",)
    dpi: int = 300
    idx: int = 0
    saved_files: List[str] = field(default_factory=list)

    def figures_dir(self) -> str:
        d = os.path.join(self.run_dir, self.subdir)
        os.makedirs(d, exist_ok=True)
        return d

    def save_fig(
        self,
        fig: Any,
        name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Sequence[str]:
        """Save a matplotlib figure with ordered index + sanitized name."""
        self.idx += 1

        if name is None or not str(name).strip():
            name = _infer_name_from_fig(fig)

        base = f"{self.idx:02d}__{sanitize(str(name))}"

        if tags:
            # only include provided tags (no invention)
            parts = []
            for k in sorted(tags.keys()):
                v = tags[k]
                if v is None:
                    continue
                parts.append(f"{sanitize(str(k))}={sanitize(str(v))}")
            if parts:
                base += "__" + "__".join(parts)

        out_dir = self.figures_dir()
        written = []
        for ext in self.exts:
            ext = ext.lstrip(".")
            path = os.path.join(out_dir, f"{base}.{ext}")
            if ext in ("png", "jpg", "jpeg", "tif", "tiff"):
                fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            else:
                fig.savefig(path, bbox_inches="tight")
            written.append(path)
            self.saved_files.append(path)
        return written


def patch_matplotlib_show(plt_module: Any, saver: PlotSaver, tags: Optional[Dict[str, str]] = None) -> None:
    """Monkeypatch matplotlib.pyplot.show to auto-save figures when shown."""
    if getattr(plt_module, "__PLOT_SAVER_PATCHED__", False):
        return

    original_show = plt_module.show

    def _show(*args, **kwargs):
        try:
            fig = plt_module.gcf()
            saver.save_fig(fig, name=None, tags=tags)
        except Exception:
            # never block notebook execution
            pass
        return original_show(*args, **kwargs)

    plt_module.show = _show
    plt_module.__PLOT_SAVER_PATCHED__ = True
