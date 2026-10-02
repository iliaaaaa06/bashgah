"""Small display helpers shared by the pages."""
import html
from datetime import datetime

_FA_DIGITS = str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫")


def fa(value) -> str:
    """Render a number with Persian digits."""
    return str(value).translate(_FA_DIGITS)


def md_safe(text: str) -> str:
    """Stop Streamlit from treating $...$ in answers (e.g. prices) as LaTeX."""
    return text.replace("$", r"\$")


def esc(text) -> str:
    return html.escape(str(text or ""))


def fa_datetime(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso).astimezone()
    except ValueError:
        return iso
    return fa(dt.strftime("%Y/%m/%d %H:%M"))
