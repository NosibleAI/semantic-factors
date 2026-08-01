"""The Fed's Trade Policy Uncertainty index, rebuilt from NOSIBLE World.

This is the factor the library was written for, and it is here as the reference implementation:
every construct in `semantic_factors` exists because this index needed it.

The published index counts words. An article counts if it contains a trade-policy word AND an
uncertainty word, across seven newspapers, back to 1960. This reproduces it from event
embeddings, with no keyword list at all:

* **anchors** define the topic. An event is about trade policy if it is close to any of them.
* **poles** replace the paper's AND. Instead of requiring an uncertainty word, every relevant
  event is scored against both framings of the same decision, and the gap becomes its weight.
  An event framed as unresolved counts fully, one framed as settled counts for almost nothing.
* **breadth** is distinct publishers, and the denominator is its own trailing twelve-month
  average, which strips out the growth of the corpus.

That is the whole index:

    relevant(e) = max cosine(e, the anchors) >= 0.35
    polarity(e) = tanh((uncertain - certain) / 0.1)     # -1 resolved .. +1 uncertain
    w_unc(e)    = (1 + polarity(e)) / 2                 #  0 .. 1

                     sum of breadth(e) * w_unc(e) over relevant events on day t
    NOSIBLE-TPU(t) = -----------------------------------------------------------
                          trailing 12-month average of total daily breadth

The write-up used three long paragraph-length anchors and one long pole pair. This version says
the same things in short sentences and five matched pairs, one per kind of trade decision, which
is the shape the write-up's own economic-policy appendix uses. Three of the five pairs are taken
from that appendix verbatim. Shorter sentences are easier to read, easier to argue with, and let
each event read its polarity off the decision it is actually about.

    export NOSIBLE_API_KEY=nos_sk_...
    export OPENAI_API_KEY=sk-...
    python trade_policy_uncertainty.py

https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty
"""
import os

from semantic_factors import SemanticFactor

START_FROM = "2015-01-01"
STOP_AT = "2026-06-30"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

FACTOR = SemanticFactor(
    name="nosible_tpu",
    anchors=[
        "A government imposes, raises or threatens new tariffs on imported goods.",
        "A government suspends, lowers or removes tariffs it had previously imposed.",
        "Customs duties, import quotas or surcharges change on goods from another country.",
        "Two governments negotiate, sign or ratify a free trade agreement.",
        "A trade agreement is renegotiated, suspended or abandoned by one of its parties.",
        "A country imposes retaliatory tariffs in an escalating trade war.",
        "Anti-dumping measures, import barriers or export controls are announced.",
        "A country files or settles a trade complaint before the World Trade Organization."
    ],
    poles=[
        (
            [
                "The government is threatening or proposing to impose, raise, suspend or lift tariffs, but it is unclear what it will actually do."
            ],
            [
                "The government has imposed, raised, suspended or lifted tariffs, and the tariff decision has taken effect."
            ]
        ),
        (
            [
                "A trade agreement between governments is under negotiation and may or may not be concluded."
            ],
            [
                "A trade agreement between governments has been signed and concluded."
            ]
        ),
        (
            [
                "A trade war, trade dispute, sanctions or export controls could escalate, and nobody knows how far it will go."
            ],
            [
                "The trade war and trade dispute have been settled, and the sanctions and export controls are fixed and final."
            ]
        ),
        (
            [
                "An anti-dumping case or trade complaint has been filed and its outcome is still unknown."
            ],
            [
                "The anti-dumping case or trade complaint has been ruled on, and the outcome is final."
            ]
        ),
        (
            [
                "Import quotas or customs rules may change, and importers cannot plan for what happens next."
            ],
            [
                "The import quotas and customs rules have been set, and importers know exactly where they stand."
            ]
        )
    ],
    filters=None,
    floor=0.35,
    aggregate="max",
    tanh_scale=0.1
)

#: The episodes the published paper and the write-up both flag.
EPISODES = {
    "2018-03-01": "Section 232 steel and aluminium",
    "2018-06-15": "China Section 301 round one",
    "2019-05-31": "Mexico tariff threat",
    "2019-08-01": "August 2019 escalation",
    "2020-01-15": "Phase One deal signed",
    "2025-02-01": "IEEPA tariffs Canada Mexico China",
    "2025-04-02": "Liberation Day",
    "2025-05-12": "Geneva truce",
    "2025-11-05": "IEEPA Supreme Court hearing",
    "2026-02-01": "IEEPA struck down"
}


def main() -> None:
    """
    Build the index, write the series, and chart it.

    :return: None.
    """
    FACTOR.compute(
        weights="netlocs",
        residualize=False,
        stabilize=True,
        polarity_weighted=True,
        start_from=START_FROM,
        stop_at=STOP_AT
    )
    os.makedirs(name=OUTPUT_DIR, exist_ok=True)
    FACTOR.to_csv(path=os.path.join(OUTPUT_DIR, "nosible_tpu_daily.csv"))
    FACTOR.monthly().write_csv(file=os.path.join(OUTPUT_DIR, "nosible_tpu_monthly.csv"))
    FACTOR.plot(
        episodes=EPISODES,
        title="Trade Policy Uncertainty",
        subtitle="Track when tariffs and trade rules become material market risks.",
        path=os.path.join(OUTPUT_DIR, "nosible_tpu.png")
    )
    print(FACTOR.monthly().drop_nulls(subset=["value"]).sort(by="value", descending=True).head(10),
          flush=True)
    print(FACTOR.events.sort(by="weight", descending=True).head(10), flush=True)


if __name__ == "__main__":
    main()
