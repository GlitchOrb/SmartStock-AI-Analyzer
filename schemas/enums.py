"""
SmartStock AI Analyzer — Enum Definitions
"""

from enum import Enum


class ReportDepth(str, Enum):
    """Controls the number of Gemini API calls per analysis."""
    QUICK = "Quick"          # 2 Gemini calls
    STANDARD = "Standard"    # 4 Gemini calls
    DEEP = "Deep"            # 6 Gemini calls


class Signal(str, Enum):
    """Investment signal strength."""
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"


class SentimentLabel(str, Enum):
    """News sentiment classification."""
    VERY_POSITIVE = "Very Positive"
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"
    VERY_NEGATIVE = "Very Negative"


class Sector(str, Enum):
    """Market sectors for classification."""
    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare"
    FINANCE = "Finance"
    ENERGY = "Energy"
    CONSUMER = "Consumer"
    INDUSTRIAL = "Industrial"
    REAL_ESTATE = "Real Estate"
    UTILITIES = "Utilities"
    MATERIALS = "Materials"
    TELECOM = "Telecom"
    OTHER = "Other"
