"""US labour action: how much of the news is workers withdrawing their labour, and which way.

A worked example of the diagnostic loop, not just a definition. The first draft of these anchors
described the *consequence* of a strike — "halting production", "disrupts services and idles
operations" — and the factor's heaviest contributor turned out to be the CrowdStrike outage,
followed by an FAA systems failure and a government shutdown. Those also stop operations. Every
anchor below names the people doing it instead, which removed the contamination entirely.

Reading .events after the first build is what caught it. The series on its own looked fine.

The poles are five matched pairs rather than one, because a strike escalates and settles in five
distinguishable ways and each pair reads its own. A ratification vote and a picket line being
lifted are both settlement, but they share almost no wording, so one generic pair catches only
whichever it happens to resemble.

    export NOSIBLE_API_KEY=nos_sk_...
    export OPENAI_API_KEY=sk-...
    python us_labour_action.py
"""
import os

from semantic_factors import SemanticFactor

START_FROM = "2015-01-01"
STOP_AT = "2026-06-30"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

FACTOR = SemanticFactor(
    name="us_labour_action",
    # Every anchor has workers or a union as its subject. Describe the act, never the disruption
    # it causes, or the factor finds IT outages and government shutdowns instead.
    anchors=[
        "Thousands of unionised workers walk off the job in a coordinated strike.",
        "A trade union votes overwhelmingly to authorise a strike against the employer.",
        "Striking workers form picket lines outside the plant and refuse to return.",
        "Union negotiators and company management fail to agree, and employees stop work.",
        "An employer locks out its unionised workforce during a contract dispute.",
        "Employees at further sites join a growing walkout organised by their union.",
        "Union members vote to reject the company's contract offer and keep striking.",
        "Striking employees demand higher wages and better staffing from their employer.",
        "The union announces that its members will extend the strike by another week.",
        "Workers end their strike and return to work after ratifying a new contract."
    ],
    # One pair per way a dispute escalates or resolves. The high side lifts the index, the low
    # side empties it, and each event reads its direction off whichever pair describes it.
    poles=[
        (
            [
                "The strike is spreading and more workers are walking out."
            ],
            [
                "The strike has ended and workers are returning to their jobs."
            ]
        ),
        (
            [
                "Talks between the union and the employer have collapsed."
            ],
            [
                "The union and the employer have reached a tentative agreement."
            ]
        ),
        (
            [
                "The union has voted to extend the stoppage for another week."
            ],
            [
                "Union members have ratified the new contract and the stoppage is over."
            ]
        ),
        (
            [
                "Picket lines are going up at more of the company's sites."
            ],
            [
                "Picket lines have come down and normal operations have resumed."
            ]
        ),
        (
            [
                "The employer is refusing to improve its offer and is preparing to hold out."
            ],
            [
                "The employer has improved its offer and the dispute has been settled."
            ]
        )
    ],
    filters={
        "event.country": "United States",
        "event.language": "en"
    },
    # The most relevant one event in a thousand. calibrate() showed p99.9 yields about 1.2 US
    # matches a day, which is dense enough to regress against.
    floor="p99.9"
)

#: Episodes worth marking. Naming them beats letting the chart pick the loudest headline of the
#: biggest day, which is often about a strike's consequences rather than the strike.
EPISODES = {
    "2018-04-01": "Teacher walkouts",
    "2018-11-01": "Google staff walkout",
    "2019-09-16": "GM strike",
    "2021-10-01": "Striketober",
    "2023-05-02": "Writers strike",
    "2023-07-13": "Actors strike",
    "2023-09-15": "UAW strikes Detroit Three",
    "2024-09-13": "Boeing machinists",
    "2024-10-01": "ILA dockworkers",
    "2026-05-16": "LIRR commuter rail strike"
}

#: Industrial, logistics and media names a labour dispute should reach.
TICKERS = ["GM.US", "F.US", "BA.US", "UPS.US", "DIS.US", "WBD.US", "NFLX.US", "LUV.US", "DAL.US"]


def main() -> None:
    """
    Build the factor, chart it, and report which names are exposed to it.

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
    FACTOR.to_csv(path=os.path.join(OUTPUT_DIR, "us_labour_action_daily.csv"))
    FACTOR.plot(
        episodes=EPISODES,
        title="US Labour Action",
        subtitle="Track when strikes and industrial disputes become material market risks.",
        path=os.path.join(OUTPUT_DIR, "us_labour_action.png")
    )
    print(FACTOR.events.sort(by="weight", descending=True).head(10), flush=True)

    FACTOR.regress(tickers=TICKERS)
    print(FACTOR.stats, flush=True)
    print(FACTOR.longs, flush=True)
    print(FACTOR.shorts, flush=True)


if __name__ == "__main__":
    main()
