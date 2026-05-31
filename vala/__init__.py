"""vala — a chess engine that baits a Maia-modeled human into mistakes.

Sister project to *rorschach*. Where rorschach uses Maia as a *negative filter*
to play sound-but-alien moves, vala uses Maia as an *opponent model* inside a
short expectimax search: it plays the move with the highest expected value
against the modeled (fallible) human, occasionally accepting a small objective
eval loss to set a trap the human is likely to walk into.

In the Tolkien hierarchy the Valar command the Maiar; here vala wraps Maia.
"""
