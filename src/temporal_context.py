"""Temporal context utilities for fact-checking pipeline.

This module provides utilities for temporal awareness in the fact-checking
system, ensuring that the LLM has access to current date/time information
for proper temporal reasoning.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def get_current_datetime() -> str:
    """Get current datetime formatted for use in prompts.
    
    Returns:
        Formatted datetime string in both English and Vietnamese.
        
    Example:
        "January 7, 2026 at 22:16 (ICT/Vietnam Time) | Ngày 7 tháng 1 năm 2026, 22:16 (Giờ Việt Nam)"
    """
    now = datetime.now()
    
    # English format
    english_date = now.strftime("%B %d, %Y at %H:%M")
    
    # Vietnamese format
    vietnamese_months = [
        "", "tháng 1", "tháng 2", "tháng 3", "tháng 4", "tháng 5", "tháng 6",
        "tháng 7", "tháng 8", "tháng 9", "tháng 10", "tháng 11", "tháng 12"
    ]
    vietnamese_date = f"Ngày {now.day} {vietnamese_months[now.month]} năm {now.year}, {now.strftime('%H:%M')}"
    
    return f"{english_date} (ICT/Vietnam Time) | {vietnamese_date} (Giờ Việt Nam)"


def get_temporal_context_prompt() -> str:
    """Get full temporal context prompt section for LLM.
    
    Returns:
        Multi-line string with current datetime and temporal reasoning guidelines.
    """
    current_dt = get_current_datetime()
    
    return f"""
CURRENT DATE AND TIME: {current_dt}

CRITICAL - Temporal Reasoning for Fact-Checking:
- The current date above is YOUR REFERENCE for "now" or "present"
- If evidence mentions a date AFTER the current date → FUTURE (do not use to refute current claims)
- If evidence mentions a date BEFORE the current date → PAST/CURRENT (can be used)

EXAMPLE (Based on current date above):
- Evidence says "từ 12/6/2025" → If this date is BEFORE current date → This is NOW IN EFFECT
- Evidence says "sẽ có hiệu lực từ 2030" → 2030 is AFTER current date → FUTURE POLICY, ignore for current claims

For claims about "hiện nay", "hiện tại", or current state:
- ONLY use evidence about what IS TRUE NOW, not what WILL BE true later
- If you only find future-dated evidence, verdict should be "Not Enough Info" for current state claims
- Always clarify in your explanation which date context the evidence refers to
"""


def format_timestamp_for_vietnamese(dt: Optional[datetime] = None) -> str:
    """Format a datetime for Vietnamese locale.
    
    Args:
        dt: Datetime to format, defaults to current time.
        
    Returns:
        Vietnamese formatted date string.
    """
    if dt is None:
        dt = datetime.now()
    
    vietnamese_months = [
        "", "tháng 1", "tháng 2", "tháng 3", "tháng 4", "tháng 5", "tháng 6",
        "tháng 7", "tháng 8", "tháng 9", "tháng 10", "tháng 11", "tháng 12"
    ]
    
    return f"Ngày {dt.day} {vietnamese_months[dt.month]} năm {dt.year}"


def is_date_in_past(date_str: str) -> Optional[bool]:
    """Check if a date string refers to a date in the past.
    
    Args:
        date_str: Date string to parse (various formats supported).
        
    Returns:
        True if date is in the past, False if in the future, None if unparseable.
    """
    from datetime import datetime
    import re
    
    now = datetime.now()
    
    # Try various date patterns
    patterns = [
        # Vietnamese: "ngày 12 tháng 6 năm 2025"
        r'ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})',
        # ISO: "2025-06-12"
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        # Vietnamese short: "12/6/2025"
        r'(\d{1,2})/(\d{1,2})/(\d{4})',
        # English: "June 12, 2025"
        r'(\w+)\s+(\d{1,2}),?\s+(\d{4})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            try:
                groups = match.groups()
                
                # Vietnamese format: day, month, year
                if 'ngày' in pattern or '/' in pattern:
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                # ISO format: year, month, day
                elif '-' in pattern:
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                # English format
                else:
                    month_names = {
                        'january': 1, 'february': 2, 'march': 3, 'april': 4,
                        'may': 5, 'june': 6, 'july': 7, 'august': 8,
                        'september': 9, 'october': 10, 'november': 11, 'december': 12
                    }
                    month = month_names.get(groups[0].lower(), 1)
                    day, year = int(groups[1]), int(groups[2])
                
                parsed_date = datetime(year, month, day)
                return parsed_date < now
                
            except (ValueError, KeyError):
                continue
    
    return None


if __name__ == "__main__":
    # Test the module
    print("Current datetime:", get_current_datetime())
    print()
    print("Temporal context prompt:")
    print(get_temporal_context_prompt())
    print()
    print("Vietnamese format:", format_timestamp_for_vietnamese())
    print()
    
    # Test date checking
    test_dates = [
        "ngày 12 tháng 6 năm 2025",
        "2025-06-12",
        "12/6/2025",
        "June 12, 2025",
        "ngày 1 tháng 1 năm 2030",
    ]
    
    for date_str in test_dates:
        result = is_date_in_past(date_str)
        status = "PAST" if result else ("FUTURE" if result is False else "UNKNOWN")
        print(f"'{date_str}' is in the {status}")
