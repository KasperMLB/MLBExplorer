from __future__ import annotations

import base64
from pathlib import Path
from functools import lru_cache

import streamlit as st
import streamlit.components.v1 as components


def page_icon_path() -> str:
    return str(Path(__file__).resolve().parent / "assets" / "kasperLogo.png")


def render_kasper_header() -> None:
    logo_path = page_icon_path()
    st.markdown(
        """
        <style>
        .kasper-header {
            display: flex;
            align-items: center;
            gap: 14px;
            margin: 0 0 18px 0;
        }
        .kasper-header img {
            width: 58px;
            height: 58px;
            object-fit: contain;
        }
        .kasper-header h1 {
            margin: 0;
            line-height: 1;
            font-size: 3.1rem;
            letter-spacing: 0;
            font-weight: 800;
        }
        @media (max-width: 640px) {
            .kasper-header img {
                width: 44px;
                height: 44px;
            }
            .kasper-header h1 {
                font-size: 2.35rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="kasper-header">
            <img src="{_image_data_uri(logo_path)}" alt="Kasper logo" />
            <h1>Kasper</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


@lru_cache(maxsize=4)
def _image_data_uri(path: str) -> str:
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix.lstrip('.') or 'png'}"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@lru_cache(maxsize=1)
def _page_icon_data_uri() -> str:
    return _image_data_uri(page_icon_path())


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
