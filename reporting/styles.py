"""
SmartStock AI Analyzer — PDF Styles
Defines fonts, colors, and paragraph styles for ReportLab PDFs.
"""

from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import mm


# ── Color Palette ──
class Colors:
    PRIMARY = HexColor("#1A237E")       # Deep indigo
    SECONDARY = HexColor("#283593")     # Lighter indigo
    ACCENT = HexColor("#448AFF")        # Bright blue
    SUCCESS = HexColor("#00C853")       # Green
    WARNING = HexColor("#FFC107")       # Amber
    DANGER = HexColor("#D32F2F")        # Red
    TEXT = HexColor("#212121")          # Near black
    TEXT_LIGHT = HexColor("#757575")    # Gray
    BG_LIGHT = HexColor("#F5F5F5")     # Light gray
    BG_CARD = HexColor("#FFFFFF")      # White
    BORDER = HexColor("#E0E0E0")       # Light border


SIGNAL_COLORS = {
    "Strong Buy": Colors.SUCCESS,
    "Buy": HexColor("#4CAF50"),
    "Hold": Colors.WARNING,
    "Sell": HexColor("#FF5722"),
    "Strong Sell": Colors.DANGER,
}


def get_styles() -> dict[str, ParagraphStyle]:
    """Return custom PDF paragraph styles."""
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "CustomTitle",
            parent=base["Title"],
            fontSize=28,
            textColor=Colors.PRIMARY,
            spaceAfter=6 * mm,
            alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "CustomSubtitle",
            parent=base["Normal"],
            fontSize=14,
            textColor=Colors.TEXT_LIGHT,
            spaceAfter=8 * mm,
            alignment=TA_CENTER,
        ),
        "heading": ParagraphStyle(
            "CustomHeading",
            parent=base["Heading1"],
            fontSize=18,
            textColor=Colors.PRIMARY,
            spaceBefore=6 * mm,
            spaceAfter=3 * mm,
            borderWidth=0,
            borderPadding=0,
        ),
        "subheading": ParagraphStyle(
            "CustomSubheading",
            parent=base["Heading2"],
            fontSize=14,
            textColor=Colors.SECONDARY,
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "CustomBody",
            parent=base["Normal"],
            fontSize=10,
            textColor=Colors.TEXT,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=2 * mm,
        ),
        "body_bold": ParagraphStyle(
            "CustomBodyBold",
            parent=base["Normal"],
            fontSize=10,
            textColor=Colors.TEXT,
            leading=14,
            fontName="Helvetica-Bold",
        ),
        "small": ParagraphStyle(
            "CustomSmall",
            parent=base["Normal"],
            fontSize=8,
            textColor=Colors.TEXT_LIGHT,
            leading=10,
        ),
        "signal": ParagraphStyle(
            "CustomSignal",
            parent=base["Normal"],
            fontSize=20,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            spaceAfter=3 * mm,
        ),
    }

    return styles
