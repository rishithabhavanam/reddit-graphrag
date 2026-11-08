import re
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

def iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def quarter_range(year, quarter):
    start_month = (quarter - 1) * 3 + 1
    start = datetime(year, start_month, 1, tzinfo=timezone.utc)
    return {"start_date": iso(start), "end_date": iso(start + relativedelta(months=3)), "label": f"Q{quarter} {year}"}

def parse_comparison(text):
    periods = []
    for m in re.finditer(r"q([1-4])\s*(20\d{2})", text.lower()):
        periods.append(quarter_range(int(m.group(2)), int(m.group(1))))
    if len(periods) >= 2:
        return periods[:2]
    years = []
    for y in [int(x) for x in re.findall(r"\b(20\d{2})\b", text)]:
        if y not in years:
            years.append(y)
    if len(years) >= 2:
        return [{"start_date": f"{y}-01-01T00:00:00Z", "end_date": f"{y+1}-01-01T00:00:00Z", "label": str(y)} for y in years[:2]]
    return []

def parse_time_expression(text):
    comparison = bool(re.search(r"\b(compare|comparison|difference|differences|changed|change|between)\b", text or "", re.I))
    periods = parse_comparison(text)
    if periods:
        return {"periods": periods, "comparison": True}
    m = re.search(r"last\s+(\d+)\s+months?", text or "", re.I)
    if m:
        now = datetime.now(timezone.utc)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - relativedelta(months=int(m.group(1))-1)
        return {"periods": [{"start_date": iso(start), "end_date": iso(now + relativedelta(days=1)), "label": f"last {m.group(1)} months"}], "comparison": False}
    m = re.search(r"q([1-4])\s*(20\d{2})", (text or "").lower())
    if m:
        return {"periods": [quarter_range(int(m.group(2)), int(m.group(1)))], "comparison": comparison}
    m = re.search(r"\b(20\d{2})\b", text or "")
    if m:
        y = int(m.group(1))
        return {"periods": [{"start_date": f"{y}-01-01T00:00:00Z", "end_date": f"{y+1}-01-01T00:00:00Z", "label": str(y)}], "comparison": comparison}
    return {"periods": [], "comparison": comparison}
