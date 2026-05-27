import ast
import operator
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests


DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_WEATHER_LOCATION = "沈阳"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 100
MAX_ABS_NUMBER = 1_000_000_000_000
MAX_ABS_EXPONENT = 100

ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

TIMEZONE_ALIASES: dict[str, str] = {
    "中国": "Asia/Shanghai",
    "中国大陆": "Asia/Shanghai",
    "北京": "Asia/Shanghai",
    "北京时间": "Asia/Shanghai",
    "上海": "Asia/Shanghai",
    "香港": "Asia/Hong_Kong",
    "台湾": "Asia/Taipei",
    "台北": "Asia/Taipei",
    "日本": "Asia/Tokyo",
    "东京": "Asia/Tokyo",
    "韩国": "Asia/Seoul",
    "首尔": "Asia/Seoul",
    "新加坡": "Asia/Singapore",
    "英国": "Europe/London",
    "伦敦": "Europe/London",
    "法国": "Europe/Paris",
    "巴黎": "Europe/Paris",
    "德国": "Europe/Berlin",
    "柏林": "Europe/Berlin",
    "美国东部": "America/New_York",
    "纽约": "America/New_York",
    "华盛顿": "America/New_York",
    "美国中部": "America/Chicago",
    "芝加哥": "America/Chicago",
    "美国山区": "America/Denver",
    "丹佛": "America/Denver",
    "美国西部": "America/Los_Angeles",
    "洛杉矶": "America/Los_Angeles",
    "旧金山": "America/Los_Angeles",
    "澳大利亚": "Australia/Sydney",
    "悉尼": "Australia/Sydney",
    "俄罗斯": "Europe/Moscow",
    "莫斯科": "Europe/Moscow",
    "印度": "Asia/Kolkata",
    "新德里": "Asia/Kolkata",
    "迪拜": "Asia/Dubai",
    "utc": "UTC",
    "gmt": "UTC",
}

FIXED_TIMEZONES: dict[str, timezone] = {
    "UTC": UTC,
    "Asia/Shanghai": timezone(timedelta(hours=8), "CST"),
    "Asia/Hong_Kong": timezone(timedelta(hours=8), "HKT"),
    "Asia/Taipei": timezone(timedelta(hours=8), "CST"),
    "Asia/Tokyo": timezone(timedelta(hours=9), "JST"),
    "Asia/Seoul": timezone(timedelta(hours=9), "KST"),
    "Asia/Singapore": timezone(timedelta(hours=8), "SGT"),
    "Asia/Kolkata": timezone(timedelta(hours=5, minutes=30), "IST"),
    "Asia/Dubai": timezone(timedelta(hours=4), "GST"),
}

WEATHER_LOCATIONS: dict[str, dict[str, Any]] = {
    "沈阳": {"name": "沈阳", "latitude": 41.8057, "longitude": 123.4315, "timezone": "Asia/Shanghai"},
    "中国": {"name": "北京", "latitude": 39.9042, "longitude": 116.4074, "timezone": "Asia/Shanghai"},
    "中国大陆": {"name": "北京", "latitude": 39.9042, "longitude": 116.4074, "timezone": "Asia/Shanghai"},
    "北京": {"name": "北京", "latitude": 39.9042, "longitude": 116.4074, "timezone": "Asia/Shanghai"},
    "北京时间": {"name": "北京", "latitude": 39.9042, "longitude": 116.4074, "timezone": "Asia/Shanghai"},
    "上海": {"name": "上海", "latitude": 31.2304, "longitude": 121.4737, "timezone": "Asia/Shanghai"},
    "香港": {"name": "香港", "latitude": 22.3193, "longitude": 114.1694, "timezone": "Asia/Hong_Kong"},
    "台湾": {"name": "台北", "latitude": 25.0330, "longitude": 121.5654, "timezone": "Asia/Taipei"},
    "台北": {"name": "台北", "latitude": 25.0330, "longitude": 121.5654, "timezone": "Asia/Taipei"},
    "日本": {"name": "东京", "latitude": 35.6762, "longitude": 139.6503, "timezone": "Asia/Tokyo"},
    "东京": {"name": "东京", "latitude": 35.6762, "longitude": 139.6503, "timezone": "Asia/Tokyo"},
    "韩国": {"name": "首尔", "latitude": 37.5665, "longitude": 126.9780, "timezone": "Asia/Seoul"},
    "首尔": {"name": "首尔", "latitude": 37.5665, "longitude": 126.9780, "timezone": "Asia/Seoul"},
    "新加坡": {"name": "新加坡", "latitude": 1.3521, "longitude": 103.8198, "timezone": "Asia/Singapore"},
    "英国": {"name": "伦敦", "latitude": 51.5072, "longitude": -0.1276, "timezone": "Europe/London"},
    "伦敦": {"name": "伦敦", "latitude": 51.5072, "longitude": -0.1276, "timezone": "Europe/London"},
    "法国": {"name": "巴黎", "latitude": 48.8566, "longitude": 2.3522, "timezone": "Europe/Paris"},
    "巴黎": {"name": "巴黎", "latitude": 48.8566, "longitude": 2.3522, "timezone": "Europe/Paris"},
    "德国": {"name": "柏林", "latitude": 52.5200, "longitude": 13.4050, "timezone": "Europe/Berlin"},
    "柏林": {"name": "柏林", "latitude": 52.5200, "longitude": 13.4050, "timezone": "Europe/Berlin"},
    "美国东部": {"name": "纽约", "latitude": 40.7128, "longitude": -74.0060, "timezone": "America/New_York"},
    "纽约": {"name": "纽约", "latitude": 40.7128, "longitude": -74.0060, "timezone": "America/New_York"},
    "华盛顿": {"name": "华盛顿", "latitude": 38.9072, "longitude": -77.0369, "timezone": "America/New_York"},
    "美国中部": {"name": "芝加哥", "latitude": 41.8781, "longitude": -87.6298, "timezone": "America/Chicago"},
    "芝加哥": {"name": "芝加哥", "latitude": 41.8781, "longitude": -87.6298, "timezone": "America/Chicago"},
    "美国山区": {"name": "丹佛", "latitude": 39.7392, "longitude": -104.9903, "timezone": "America/Denver"},
    "丹佛": {"name": "丹佛", "latitude": 39.7392, "longitude": -104.9903, "timezone": "America/Denver"},
    "美国西部": {"name": "洛杉矶", "latitude": 34.0522, "longitude": -118.2437, "timezone": "America/Los_Angeles"},
    "洛杉矶": {"name": "洛杉矶", "latitude": 34.0522, "longitude": -118.2437, "timezone": "America/Los_Angeles"},
    "旧金山": {"name": "旧金山", "latitude": 37.7749, "longitude": -122.4194, "timezone": "America/Los_Angeles"},
    "澳大利亚": {"name": "悉尼", "latitude": -33.8688, "longitude": 151.2093, "timezone": "Australia/Sydney"},
    "悉尼": {"name": "悉尼", "latitude": -33.8688, "longitude": 151.2093, "timezone": "Australia/Sydney"},
    "俄罗斯": {"name": "莫斯科", "latitude": 55.7558, "longitude": 37.6173, "timezone": "Europe/Moscow"},
    "莫斯科": {"name": "莫斯科", "latitude": 55.7558, "longitude": 37.6173, "timezone": "Europe/Moscow"},
    "印度": {"name": "新德里", "latitude": 28.6139, "longitude": 77.2090, "timezone": "Asia/Kolkata"},
    "新德里": {"name": "新德里", "latitude": 28.6139, "longitude": 77.2090, "timezone": "Asia/Kolkata"},
    "迪拜": {"name": "迪拜", "latitude": 25.2048, "longitude": 55.2708, "timezone": "Asia/Dubai"},
    "utc": {"name": "伦敦", "latitude": 51.5072, "longitude": -0.1276, "timezone": "UTC"},
    "gmt": {"name": "伦敦", "latitude": 51.5072, "longitude": -0.1276, "timezone": "UTC"},
}

WEATHER_CODE_TEXT: dict[int, str] = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴强冰雹",
}


def get_current_time(
    timezone_name: str | None = None,
    location: str | None = None,
    timezone: str | None = None,
    **_: Any,
) -> dict[str, str]:
    """Get the current time for a region or timezone."""
    resolved_timezone = _resolve_timezone(timezone_name or timezone, location)
    tzinfo = _load_timezone(resolved_timezone)
    now = datetime.now(tzinfo)

    return {
        "location": location or resolved_timezone,
        "timezone": resolved_timezone,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "iso_timestamp": now.isoformat(timespec="seconds"),
        "utc_offset": now.strftime("%z"),
    }


def _resolve_timezone(timezone_name: str | None, location: str | None) -> str:
    for value in (timezone_name, location):
        if not value:
            continue

        normalized_value = value.strip()
        if not normalized_value:
            continue

        alias = TIMEZONE_ALIASES.get(normalized_value)
        if alias:
            return alias

        lower_value = normalized_value.lower()
        alias = TIMEZONE_ALIASES.get(lower_value)
        if alias:
            return alias

        if "/" in normalized_value or normalized_value.upper() == "UTC":
            return normalized_value

    return DEFAULT_TIMEZONE


def _load_timezone(timezone_name: str):
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        if timezone_name in FIXED_TIMEZONES:
            return FIXED_TIMEZONES[timezone_name]
        return FIXED_TIMEZONES[DEFAULT_TIMEZONE]


def calculate_expression(
    expression: str | None = None,
    query: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Safely calculate a basic arithmetic expression."""
    raw_expression = expression or query
    try:
        normalized_expression = _normalize_expression(raw_expression)
        result = _safe_calculate(normalized_expression)
    except (ArithmeticError, ValueError) as exc:
        return {
            "expression": raw_expression or "",
            "error": str(exc),
        }

    return {
        "expression": normalized_expression,
        "result": _normalize_number(result),
    }


def _normalize_expression(expression: str | None) -> str:
    if not expression or not expression.strip():
        raise ValueError("expression cannot be empty")

    stripped_expression = expression.strip()
    if len(stripped_expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("expression is too long")

    try:
        parsed = ast.parse(stripped_expression, mode="eval")
        _calculate_ast_node(parsed.body)
        return stripped_expression
    except (SyntaxError, ValueError):
        allowed_chars = set("0123456789+-*/%(). ")
        candidates: list[str] = []
        current: list[str] = []

        for char in stripped_expression:
            if char in allowed_chars:
                current.append(char)
                continue

            candidate = "".join(current).strip()
            if candidate:
                candidates.append(candidate)
            current = []

        candidate = "".join(current).strip()
        if candidate:
            candidates.append(candidate)

        for candidate in sorted(candidates, key=len, reverse=True):
            try:
                parsed = ast.parse(candidate, mode="eval")
                _calculate_ast_node(parsed.body)
                return candidate
            except (SyntaxError, ValueError):
                continue

    raise ValueError("expression contains unsupported syntax")


def _safe_calculate(expression: str) -> int | float:
    parsed = ast.parse(expression, mode="eval")
    node_count = sum(1 for _ in ast.walk(parsed))
    if node_count > MAX_AST_NODES:
        raise ValueError("expression is too complex")

    return _calculate_ast_node(parsed.body)


def _calculate_ast_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ValueError("only numbers are supported")
        if abs(node.value) > MAX_ABS_NUMBER:
            raise ValueError("number is too large")
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY_OPERATORS:
        operand = _calculate_ast_node(node.operand)
        return ALLOWED_UNARY_OPERATORS[type(node.op)](operand)

    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINARY_OPERATORS:
        left = _calculate_ast_node(node.left)
        right = _calculate_ast_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_ABS_EXPONENT:
            raise ValueError("exponent is too large")
        result = ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(result, int | float) and abs(result) > MAX_ABS_NUMBER:
            raise ValueError("result is too large")
        return result

    raise ValueError("expression contains unsupported syntax")


def _normalize_number(value: int | float) -> int | float:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def get_today_weather(
    location: str | None = None,
    timezone_name: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Get today's hourly weather forecast for a supported location."""
    weather_location = _resolve_weather_location(location, timezone_name)
    resolved_timezone = str(weather_location["timezone"])
    current_time = get_current_time(
        timezone_name=resolved_timezone,
        location=str(weather_location["name"]),
    )
    current_hour = datetime.fromisoformat(current_time["iso_timestamp"]).hour

    try:
        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": weather_location["latitude"],
                "longitude": weather_location["longitude"],
                "hourly": "temperature_2m,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset",
                "forecast_days": 1,
                "timezone": resolved_timezone,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return {
            "location": weather_location["name"],
            "timezone": resolved_timezone,
            "error": f"weather request failed: {exc}",
        }

    daily = data.get("daily", {})
    hourly = data.get("hourly", {})
    hourly_forecast = _format_hourly_weather(hourly, current_hour)
    current_weather = next(
        (item for item in hourly_forecast if item.get("is_current_hour")),
        None,
    )

    return {
        "location": weather_location["name"],
        "timezone": resolved_timezone,
        "date": _first_item(daily.get("time")),
        "current_time": current_time["timestamp"],
        "current_hour": current_hour,
        "current_hour_note": f"用户提问时处于 {current_hour:02d}:00 这一小时",
        "current_weather": current_weather,
        "temperature_unit": data.get("hourly_units", {}).get("temperature_2m", "°C"),
        "min_temperature": _first_item(daily.get("temperature_2m_min")),
        "max_temperature": _first_item(daily.get("temperature_2m_max")),
        "weather_status": _weather_code_to_text(_first_item(daily.get("weather_code"))),
        "sunrise": _format_minute_time(_first_item(daily.get("sunrise"))),
        "sunset": _format_minute_time(_first_item(daily.get("sunset"))),
        "hourly": hourly_forecast,
    }


def _resolve_weather_location(
    location: str | None,
    timezone_name: str | None,
) -> dict[str, Any]:
    for value in (location, timezone_name):
        if not value:
            continue

        normalized_value = value.strip()
        if not normalized_value:
            continue

        weather_location = WEATHER_LOCATIONS.get(normalized_value)
        if weather_location:
            return weather_location

        lower_value = normalized_value.lower()
        weather_location = WEATHER_LOCATIONS.get(lower_value)
        if weather_location:
            return weather_location

        timezone_location = _weather_location_by_timezone(normalized_value)
        if timezone_location:
            return timezone_location

        alias = TIMEZONE_ALIASES.get(normalized_value) or TIMEZONE_ALIASES.get(lower_value)
        timezone_location = _weather_location_by_timezone(alias)
        if timezone_location:
            return timezone_location

    return WEATHER_LOCATIONS[DEFAULT_WEATHER_LOCATION]


def _weather_location_by_timezone(timezone_name: str | None) -> dict[str, Any] | None:
    if not timezone_name:
        return None

    for weather_location in WEATHER_LOCATIONS.values():
        if weather_location["timezone"] == timezone_name:
            return weather_location

    return None


def _format_hourly_weather(hourly: dict[str, Any], current_hour: int) -> list[dict[str, Any]]:
    times = hourly.get("time") or []
    temperatures = hourly.get("temperature_2m") or []
    weather_codes = hourly.get("weather_code") or []
    hourly_forecast: list[dict[str, Any]] = []

    for index, time_value in enumerate(times[:24]):
        hour = _hour_from_time(time_value, index)
        hourly_forecast.append(
            {
                "hour": f"{hour:02d}:00-{(hour + 1):02d}:00" if hour < 23 else "23:00-24:00",
                "temperature": _item_at(temperatures, index),
                "weather_status": _weather_code_to_text(_item_at(weather_codes, index)),
                "is_current_hour": hour == current_hour,
            }
        )

    return hourly_forecast


def _hour_from_time(time_value: Any, fallback: int) -> int:
    if isinstance(time_value, str):
        try:
            return datetime.fromisoformat(time_value).hour
        except ValueError:
            pass
    return fallback


def _item_at(items: Any, index: int) -> Any:
    if isinstance(items, list) and index < len(items):
        return items[index]
    return None


def _first_item(items: Any) -> Any:
    return _item_at(items, 0)


def _weather_code_to_text(code: Any) -> str:
    if code is None:
        return "未知"
    try:
        return WEATHER_CODE_TEXT.get(int(code), "未知")
    except (TypeError, ValueError):
        return "未知"


def _format_minute_time(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).strftime("%H:%M")
    except ValueError:
        return value[:16]


TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_current_time": "获取当前时间，返回包含年月日时分秒的完整时间戳。默认返回北京时间；如果用户询问其他国家、城市或地区时间，传入 location 或 IANA timezone_name。",
}

TOOL_ARGUMENTS: dict[str, dict[str, str]] = {
    "get_current_time": {
        "location": "用户询问的国家、城市或地区名称，例如：北京、日本、纽约、伦敦。默认可省略。",
        "timezone_name": "可选 IANA 时区名，例如：Asia/Shanghai、Asia/Tokyo、America/New_York。",
    },
}

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "get_current_time": get_current_time,
}

TOOL_DESCRIPTIONS["calculate_expression"] = (
    "安全计算基础数学表达式，支持 +、-、*、/、//、%、** 和括号。"
    "用于用户询问计算、算一下、数学表达式求值等场景；不要执行代码，只传入表达式字符串。"
)
TOOL_ARGUMENTS["calculate_expression"] = {
    "expression": (
        "要计算的数学表达式，例如：(24+1)*4。只包含数字、运算符和括号；"
        "如果用户原句里有中文说明，提取其中的算式。"
    ),
}
TOOL_REGISTRY["calculate_expression"] = calculate_expression

TOOL_DESCRIPTIONS["get_today_weather"] = (
    "查询当日天气。用户询问天气、今天天气、当前天气时调用；"
    "如果没有具体城市，默认查询沈阳。返回最低温、最高温、阴晴状态、"
    "0点到23点每小时天气状态和温度、日出日落时间，并标注用户提问时所在小时。"
)
TOOL_ARGUMENTS["get_today_weather"] = {
    "location": (
        "用户询问的国家、城市或地区名称，例如：沈阳、北京、日本、纽约、伦敦。"
        "没有明确城市时省略，工具会默认沈阳。"
    ),
    "timezone_name": "可选 IANA 时区名；支持范围和时间查询一致。",
}
TOOL_REGISTRY["get_today_weather"] = get_today_weather
