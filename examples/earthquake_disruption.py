"""Build an earthquake disruption pressure factor from NOSIBLE World events.

The factor is high when earthquakes become economically and socially disruptive: casualties,
collapsed buildings, damaged infrastructure, supply-chain interruption, displacement, tsunamis,
or worsening aftershocks. It is low when tremors are remote or minor and when recovery restores
critical services.

Run it with credentials supplied only through the process environment:

    python earthquake_disruption.py
"""
import json
import os

from semantic_factors import SemanticFactor

START_FROM = "2023-01-01"
STOP_AT = "2026-07-31"
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output",
    "earthquake_disruption",
)

FACTOR = SemanticFactor(
    name="earthquake_disruption_pressure",
    anchors=[
        "A strong shallow earthquake strikes a populated city and causes buildings to collapse.",
        "An earthquake kills or injures residents and starts a large urban search-and-rescue operation.",
        "Seismic shaking damages homes, hospitals, schools, and other essential public buildings.",
        "An earthquake cuts electricity, drinking water, telecommunications, or fuel supplies.",
        "A major quake closes roads, bridges, railways, airports, or ports used by the regional economy.",
        "Factories, mines, warehouses, or industrial plants stop operating after earthquake damage.",
        "Earthquake-triggered landslides isolate communities and obstruct emergency responders.",
        "An offshore earthquake generates a tsunami warning or destructive coastal flooding.",
        "Powerful aftershocks cause further collapses and interrupt rescue and relief work.",
        "Authorities evacuate or shelter thousands of people displaced by severe earthquake damage.",
        "Insurers, governments, or businesses report large economic losses after an earthquake.",
        "International emergency teams and humanitarian supplies deploy after a catastrophic quake.",
    ],
    poles=[
        (
            [
                "The earthquake is catastrophic, with widespread casualties and structural destruction.",
                "Damage is expanding and emergency responders are overwhelmed.",
            ],
            [
                "The tremor is minor or remote, with no casualties and little or no damage.",
                "Authorities confirm that the earthquake caused no meaningful disruption.",
            ],
        ),
        (
            [
                "Strong aftershocks, landslides, or tsunami risk are making the disaster worse.",
                "Critical infrastructure remains closed and essential services are failing.",
            ],
            [
                "Aftershocks are fading and rescue operations are moving into stable recovery.",
                "Transport, utilities, schools, and businesses are reopening after the earthquake.",
            ],
        ),
    ],
    filters={"language": "en"},
    floor="p99.95",
    tanh_scale=0.10,
)


def main() -> None:
    """Calibrate, compute, export, and plot the earthquake factor."""
    api_key = os.environ["NOSIBLE_API_KEY"]
    embed_api_key = os.environ["OPENROUTER_API_KEY"]
    embed_base_url = "https://openrouter.ai/api/v1"

    calibration = FACTOR.calibrate(
        start_from=START_FROM,
        stop_at=STOP_AT,
        api_key=api_key,
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
    )
    print("calibration", flush=True)
    print(calibration.write_csv(), end="", flush=True)

    daily = FACTOR.compute(
        weights="materiality",
        stabilize=True,
        polarity_weighted=True,
        start_from=START_FROM,
        stop_at=STOP_AT,
        api_key=api_key,
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        top_events=8,
    )

    os.makedirs(name=OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "earthquake_disruption_daily.csv")
    events_path = os.path.join(OUTPUT_DIR, "earthquake_disruption_events.csv")
    diagnostics_path = os.path.join(OUTPUT_DIR, "earthquake_disruption_run.json")
    plot_path = os.path.join(OUTPUT_DIR, "earthquake_disruption.png")

    FACTOR.to_csv(path=csv_path)
    if FACTOR.events is not None:
        FACTOR.events.write_csv(file=events_path)
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        json.dump(FACTOR.diagnostics, handle, indent=2, sort_keys=True)
    FACTOR.plot(
        annotate=True,
        title="Earthquake disruption pressure",
        subtitle="Materiality-weighted global coverage · higher means more severe disruption",
        episodes={
            "2023-02-06": "Türkiye–Syria",
            "2023-09-08": "Morocco",
            "2024-01-01": "Noto Peninsula",
            "2024-04-03": "Hualien",
            "2025-03-28": "Myanmar",
        },
        path=plot_path,
    )

    print("\nrun diagnostics", flush=True)
    print(json.dumps(FACTOR.diagnostics, indent=2, sort_keys=True), flush=True)
    print("\nlargest monthly readings", flush=True)
    print(
        FACTOR.monthly()
        .drop_nulls(subset=["value"])
        .sort(by="value", descending=True)
        .head(12)
        .write_csv(),
        end="",
        flush=True,
    )
    print("\nmost semantically relevant retained events", flush=True)
    if FACTOR.events is not None:
        print(
            FACTOR.events.sort(by="score", descending=True).head(12).write_csv(),
            end="",
            flush=True,
        )
    print(f"\nwrote {daily.height} daily rows to {csv_path}", flush=True)
    print(f"wrote plot to {plot_path}", flush=True)


if __name__ == "__main__":
    main()
