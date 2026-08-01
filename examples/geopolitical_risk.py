"""A broad factor, and what changes when a factor is broad.

Crop infestation is narrow: a handful of events a day worldwide, and the floor has to be tight or
disasters with heavy coverage crowd out the real matches. Geopolitical risk is the opposite. It
is a large, continuous share of the news, so the floor is looser, the series is dense, and the
interesting question moves from "did anything match" to "is this rising".

It is also signed. The poles turn a count of conflict coverage into a reading of whether the
conflicts being written about are escalating or settling, which is a different series from the
count and often moves first.

    export NOSIBLE_API_KEY=nos_sk_...
    export OPENAI_API_KEY=sk-...
    python geopolitical_risk.py
"""
import os

from semantic_factors import SemanticFactor

START_FROM = "2015-01-01"
STOP_AT = "2026-06-30"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

FACTOR = SemanticFactor(
    name="geopolitical_risk",
    anchors=[
        "Military forces attack, invade, or occupy another sovereign state's territory.",
        "Fighting expands across new fronts and reaches densely populated civilian areas.",
        "A ceasefire collapses and sustained combat operations resume between the two armies.",
        "Air strikes or naval attacks strike targets inside another country's borders.",
        "A government mobilises troops and moves heavy weapons towards a contested border.",
        "Two governments exchange threats and move visibly towards open armed confrontation.",
        "A territorial dispute worsens and both claimants reinforce their frontier garrisons.",
        "Diplomatic talks break down and each government withdraws its ambassador in protest.",
        "An organised group bombs or shoots civilians in order to cause widespread terror.",
        "Sabotage damages the pipelines, cables, or railways a state depends on.",
        "An attempted coup challenges control of a national government by force.",
        "Insurgents seize provinces and the national army withdraws from the territory."
    ],
    pole_hi=[
        "The confrontation is escalating and is likely to get significantly worse",
        "More forces are being committed and the fighting is spreading"
    ],
    pole_lo=[
        "The confrontation is de-escalating and a settlement is being negotiated",
        "Forces are withdrawing and the fighting has largely stopped"
    ],
    filters={"language": "en"},
    # Looser than the crop factor, because this construct genuinely describes a large slice of
    # the news rather than a rare event. Run calibrate() and compare the two.
    floor="p99"
)


def main() -> None:
    """
    Build the factor, chart the level against the signed pole balance, and save both.

    :return: None.
    """
    print(FACTOR.calibrate(start_from=START_FROM, stop_at=STOP_AT), flush=True)

    daily = FACTOR.compute(
        weights="netlocs",
        residualize=False,
        stabilize=True,
        start_from=START_FROM,
        stop_at=STOP_AT
    )
    os.makedirs(name=OUTPUT_DIR, exist_ok=True)
    FACTOR.to_csv(path=os.path.join(OUTPUT_DIR, "geopolitical_risk_daily.csv"))
    FACTOR.plot(
        annotate=True,
        show_polarity=True,
        path=os.path.join(OUTPUT_DIR, "geopolitical_risk.png")
    )
    print(daily.describe(), flush=True)
    print(FACTOR.monthly().sort(by="value", descending=True).head(12), flush=True)


if __name__ == "__main__":
    main()
