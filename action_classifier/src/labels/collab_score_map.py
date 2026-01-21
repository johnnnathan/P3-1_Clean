POS_STRONG = {
    "shaking hands",
    "hugging",
    "giving object",
    "pat on back",
}

POS_WEAK = {
    "cheer up",
    "hand waving",
    "clapping",
    "point to something",
    "point finger",
    "salute",
    "put palms together",
    "nod head/bow",
    # “phone call” can be social, but ambiguous; keep as weak positive
    "phone call",
}

NEG_STRONG = {
    "walking apart",
    "punch/slap",
    "kicking",
    "pushing",
    "staggering",
    "falling down",
    "chest pain",
    "back pain",
    "neck pain",
    "headache",
    "nausea/vomiting",
}

# Everything else will default to 0 (neutral/solo)
# But we can also explicitly list them for clarity (optional).
NEUTRAL = {
    "drink water", "eat meal", "brush teeth", "brush hair",
    "drop", "pick up", "throw", "sit down", "stand up",
    "reading", "writing", "tear up paper",
    "put on jacket", "take off jacket",
    "put on a shoe", "take off a shoe",
    "put on glasses", "take off glasses",
    "put on a hat/cap", "take off a hat/cap",
    "kicking something",
    "reach into pocket",
    "hopping", "jump up",
    "play with phone/tablet",
    "type on a keyboard",
    "taking a selfie",
    "check time (from watch)",
    "rub two hands",
    "shake head",
    "wipe face",
    "cross hands in front",
    "sneeze/cough",
    "fan self",
    "touch pocket",
    "walking towards",
}

def label_to_collab_score(label: str) -> int:
    """
    Returns a proxy collaboration score for an NTU60 label.
    """
    lab = label.strip().lower()

    if lab in POS_STRONG:
        return 2
    if lab in POS_WEAK:
        return 1
    if lab in NEG_STRONG:
        return -1

    # default neutral
    return 0

