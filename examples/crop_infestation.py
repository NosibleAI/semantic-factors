"""Model the impact of crop infestation on US farming stocks.

Ten sentences say what an infestation is. One pole pair, expressed with two statements on each
side, says which way is worse. Everything after that is NOSIBLE.

The example is also a worked lesson in the two judgement calls a factor requires. It calibrates
before it builds, and it explains why it does not use the filters you might reach for first.

Run it inside an environment with `semantic-factors` installed:

    export NOSIBLE_API_KEY=nos_sk_...
    export OPENAI_API_KEY=sk-...
    python crop_infestation.py
"""
import os

from semantic_factors import SemanticFactor

START_FROM = "2015-01-01"
STOP_AT = "2026-06-30"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

#: Agricultural inputs, machinery and grain traders: the names an infestation should reach.
TICKERS = ["CTVA.US", "FMC.US", "ZTS.US", "MOS.US", "NTR.US", "DE.US", "ADM.US", "BG.US"]

FACTOR = SemanticFactor(
    name="crop_infestation",
    # Retrieve the most relevant events.
    anchors=[
        "Catastrophic pest explosions destroy cultivated crops, demanding immediate agricultural intervention.",
        "Massive parasite outbreaks devastate global food supplies, requiring urgent mitigation.",
        "Severe insect surges decimate agricultural yields, threatening human food security.",
        "Rapidly multiplying parasites destroy essential farmlands, forcing widespread protective action.",
        "Extreme infestations ruin crop production, forcing farmers into eradication measures.",
        "Swarming agricultural pests annihilate cultivated plants, stopping worldwide food production.",
        "Destructive locust surges constitute major infestations, wiping out vital harvests.",
        "Rampant pest explosions directly decimate fields, triggering rapid supply shortages.",
        "Uncontrolled parasite populations cause severe infestations, requiring massive control efforts.",
        "Catastrophic bug increases devastate cultivated food supplies, demanding swift responses."
    ],
    # Poles that make the indicator higher.
    pole_hi=[
        "The infestation has spread and is getting worse",
        "The pest swarm has grown by double this past week"
    ],
    # Poles that make the indicator lower.
    pole_lo=[
        "The infestation has subsided and is getting better",
        "The pest swarm has shrunk by half this past week"
    ],
    # Filter by fields on the events.
    #
    # Deliberately not {"region": "North America", "materiality": "high"}. Infestations that move
    # US agricultural equities are mostly not North American events: the 2020 desert locust
    # upsurge ran through East Africa and South Asia and moved crop prices worldwide. Adding
    # those two filters takes this factor from about four matches a day to one every forty days,
    # which is too sparse to regress against. calibrate() shows that in half a minute.
    filters={"language": "en"},
    # The most relevant one event in ten thousand. p99.9 also finds the locust plagues, but with
    # coverage weighting the day is then dominated by hurricanes and wildfires that clear the
    # looser floor while carrying far more coverage. Inspect .events and you can see it happen.
    floor="p99.99"
)


def main() -> None:
    """
    Calibrate, build, chart, and report which agricultural names are exposed to the factor.

    :return: None.
    """
    print(FACTOR.calibrate(start_from=START_FROM, stop_at=STOP_AT), flush=True)

    FACTOR.compute(
        weights="coverage",
        residualize=False,
        stabilize=True,
        start_from=START_FROM,
        stop_at=STOP_AT
    )
    os.makedirs(name=OUTPUT_DIR, exist_ok=True)
    FACTOR.to_csv(path=os.path.join(OUTPUT_DIR, "crop_infestation_daily.csv"))
    FACTOR.plot(
        annotate=True,
        path=os.path.join(OUTPUT_DIR, "crop_infestation.png")
    )
    print("the events that moved the biggest days", flush=True)
    print(FACTOR.events.sort(by="weight", descending=True).head(10), flush=True)

    FACTOR.regress(tickers=TICKERS)
    print(FACTOR.stats, flush=True)
    print("positively exposed", flush=True)
    print(FACTOR.longs, flush=True)
    print("negatively exposed", flush=True)
    print(FACTOR.shorts, flush=True)


if __name__ == "__main__":
    main()
