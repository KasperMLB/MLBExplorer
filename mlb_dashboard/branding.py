from __future__ import annotations

import base64
from pathlib import Path
from functools import lru_cache

import streamlit.components.v1 as components


def page_icon_path() -> str:
    return str(Path(__file__).resolve().parent / "assets" / "kasper.png")


@lru_cache(maxsize=1)
def _page_icon_data_uri() -> str:
    icon_path = Path(page_icon_path())
    encoded = base64.b64encode(icon_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def apply_branding_head() -> None:
    icon_href = _page_icon_data_uri()
    components.html(
        f"""
        <script>
        const iconHref = "{icon_href}";
        const head = window.parent.document.head;
        const linkDefs = [
          {{ rel: "icon", type: "image/png", sizes: "32x32" }},
          {{ rel: "shortcut icon", type: "image/png" }},
          {{ rel: "apple-touch-icon", type: "image/png", sizes: "180x180" }}
        ];

        for (const attrs of linkDefs) {{
          const selector = `link[rel="${{attrs.rel}}"]`;
          let link = head.querySelector(selector);
          if (!link) {{
            link = window.parent.document.createElement("link");
            head.appendChild(link);
          }}
          link.rel = attrs.rel;
          link.type = attrs.type;
          if (attrs.sizes) {{
            link.sizes = attrs.sizes;
          }}
          link.href = iconHref;
        }}
        </script>
        """,
        height=0,
        width=0,
    )
