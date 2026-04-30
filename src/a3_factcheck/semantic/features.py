import re


NEGATION_CUES = {
    "no",
    "not",
    "never",
    "none",
    "without",
    "n't",
    "cannot",
    "can't",
    "won't",
}

COMPARISON_CUES = {
    "more than",
    "less than",
    "higher",
    "lower",
    "highest",
    "lowest",
    "increase",
    "decrease",
    "greater",
    "smaller",
    "faster",
    "slower",
}

CAUSALITY_CUES = {
    "because",
    "due to",
    "caused by",
    "cause",
    "causes",
    "result in",
    "results in",
    "lead to",
    "leads to",
    "contribute",
    "contributes",
    "affect",
    "affects",
}

RELATION_VERBS = {
    "produce",
    "produces",
    "produced",
    "emit",
    "emits",
    "emitted",
    "reduce",
    "reduces",
    "reduced",
    "increase",
    "increases",
    "increased",
    "warm",
    "warms",
    "warmed",
    "cool",
    "cools",
    "cooled",
    "absorb",
    "absorbs",
    "absorbed",
    "reflect",
    "reflects",
    "reflected",
}

DOMAIN_TERMS = {
    "CO2",
    "carbon dioxide",
    "greenhouse gas",
    "global warming",
    "climate change",
    "emissions",
    "temperature",
    "Australia",
    "anthropogenic",
}


def unique_in_order(items):
    seen = set()
    result = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def cue_matches(text, cues):
    lowered = text.lower()
    return [cue for cue in cues if cue in lowered]


def extract_entities(text):
    capitalized = re.findall(r"\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3}\b", text)
    domain = [term for term in DOMAIN_TERMS if term.lower() in text.lower()]
    return unique_in_order(domain + capitalized)


def extract_semantic_features(text):
    tokens = re.findall(r"[A-Za-z]+(?:n't)?", text.lower())
    percentages = re.findall(r"\b\d+(?:\.\d+)?\s*%", text)
    quantities = re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:ppm|ppb|°C|degrees|billion|million|tonnes|tons|percent|per cent)\b",
        text,
        flags=re.IGNORECASE,
    )
    years = re.findall(r"\b(?:18|19|20)\d{2}\b", text)
    negations = [token for token in tokens if token in NEGATION_CUES]

    return {
        "entities": extract_entities(text),
        "percentages": unique_in_order(percentages),
        "quantities": unique_in_order(quantities),
        "years": unique_in_order(years),
        "negation_cues": unique_in_order(negations + cue_matches(text, NEGATION_CUES)),
        "comparison_cues": unique_in_order(cue_matches(text, COMPARISON_CUES)),
        "causality_cues": unique_in_order(cue_matches(text, CAUSALITY_CUES)),
        "relation_verbs": unique_in_order(
            [token for token in tokens if token in RELATION_VERBS]
        ),
    }
