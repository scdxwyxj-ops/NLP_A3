import re


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "had",
    "he",
    "her",
    "his",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "more",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "such",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "will",
    "with",
}


NEGATION_COMPARISON_TERMS = {
    "not",
    "no",
    "never",
    "without",
    "less",
    "more",
    "higher",
    "lower",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "decline",
    "declined",
    "greater",
    "smaller",
    "warmer",
    "cooler",
    "above",
    "below",
    "before",
    "after",
}


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'%-]*|\d+(?:\.\d+)?%?")
NUMBER_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?%?|\d{4}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"million|billion|trillion|percent|percentage|degree|degrees|celsius|fahrenheit|"
    r"metre|meter|metres|meters|feet|foot|gigatonnes?|tonnes?|ppm|w/m2)\b",
    flags=re.IGNORECASE,
)
ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'%-]+(?:\s+[A-Z][A-Za-z0-9'%-]+)*|[A-Z]{2,})\b"
)


def normalize_space(text):
    return re.sub(r"\s+", " ", text).strip(" \t\r\n\"'`.,;:")


def unique_keep_order(items):
    seen = set()
    out = []
    for item in items:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def tokens(text):
    return TOKEN_RE.findall(text)


def keyword_view(text, max_terms=18):
    terms = []
    for token in tokens(text):
        lower = token.lower().strip("'")
        if len(lower) < 3 or lower in STOPWORDS:
            continue
        terms.append(token)
    return " ".join(unique_keep_order(terms)[:max_terms])


def entity_quantity_view(text):
    entities = [normalize_space(match.group(0)) for match in ENTITY_RE.finditer(text)]
    quantities = [normalize_space(match.group(0)) for match in NUMBER_RE.finditer(text)]
    parts = unique_keep_order([*entities, *quantities])
    return " ".join(parts)


def number_context_view(text, window=5):
    toks = tokens(text)
    selected = []
    for idx, token in enumerate(toks):
        if NUMBER_RE.fullmatch(token):
            start = max(0, idx - window)
            end = min(len(toks), idx + window + 1)
            selected.extend(toks[start:end])
    return " ".join(unique_keep_order(selected))


def negation_comparison_view(text, window=6):
    toks = tokens(text)
    selected = []
    for idx, token in enumerate(toks):
        if token.lower() in NEGATION_COMPARISON_TERMS:
            start = max(0, idx - window)
            end = min(len(toks), idx + window + 1)
            selected.extend(toks[start:end])
    return " ".join(unique_keep_order(selected))


def clause_views(text):
    rough_parts = re.split(
        r"[,;:()]|\b(?:but|because|although|whereas|while|which|that|and that)\b",
        text,
        flags=re.IGNORECASE,
    )
    views = []
    for part in rough_parts:
        part = normalize_space(part)
        if len(part.split()) >= 4:
            views.append(part)
    return unique_keep_order(views)


def claim_query_views(claim_text, max_views=6):
    candidates = [
        normalize_space(claim_text),
        entity_quantity_view(claim_text),
        number_context_view(claim_text),
        negation_comparison_view(claim_text),
        keyword_view(claim_text),
        *clause_views(claim_text),
    ]
    views = []
    for view in unique_keep_order(normalize_space(item) for item in candidates):
        if len(view.split()) >= 2:
            views.append(view)
        if len(views) >= max_views:
            break
    if not views:
        views = [normalize_space(claim_text)]
    return views
