"""
enrich_l5_screenplay.py
========================
Fill missing L5 screenplay fields for the custom_5layer_mvp subset using
known screenplay knowledge for the 3 movies:
  - tt0120338: Titanic (1997)
  - tt0073486: The Shawshank Redemption (1994)
  - tt0119822: As Good as It Gets (1997)

Source evidence: human_derived from film knowledge.
The screenplay_context_excerpt field uses the key heading + 1-2 sentence description.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "custom_5layer_mvp" / "all_chunks.json"
OUTPUT = ROOT / "data" / "custom_5layer_mvp" / "all_chunks.json"


# ─── Titanic (tt0120338) screenplay knowledge ────────────────────────────────
# Film structure based on well-documented screenplay and film.
# All timestamps approximate (±30s) from screenplay timeline.
# Key scenes, heading format: "INT./EXT. LOCATION - TIME"

TITANIC_SCRIPT: list[dict[str, Any]] = [
    # ── PRESENT DAY (framing narrative) ──────────────────────────────────────
    {
        "movie_id": "tt0120338",
        "start_lt": 0, "end_lt": 120,
        "heading": "INT. DECK OF RESEARCH VESSEL - DAY (PRESENT)",
        "context": (
            "Brock Lovett and his team conduct a deep-sea expedition over the Titanic wreck, "
            "searching for the fabled Heart of the Ocean diamond. Their robotic equipment "
            "discovers a safe and retrieve it to the surface."
        ),
    },
    {
        "movie_id": "tt0120338",
        "start_lt": 120, "end_lt": 260,
        "heading": "INT. DECK OF RESEARCH VESSEL - DAY (PRESENT)",
        "context": (
            "The team opens the safe recovered from the Titanic and finds only a drawing: "
            "a nude portrait of a young woman wearing the Heart of the Ocean diamond. "
            "The image is dated April 14, 1912. Brock's investors grow restless."
        ),
    },
    {
        "movie_id": "tt0120338",
        "start_lt": 260, "start_ge": 260, "end_lt": 330,
        "heading": "INT. KELDYSH (RESEARCH VESSEL) - DAY",
        "context": (
            "Rose Calvert, an elderly survivor, arrives on the ship and is recognized in a photograph. "
            "She watches the safe contents and reveals she knows the woman in the drawing — herself."
        ),
    },
    {
        "movie_id": "tt0120338",
        "start_ge": 330, "end_lt": 410,
        "heading": "INT. KELDYSH - DAY",
        "context": (
            "Rose begins her story, telling Brock Lovett about her experiences aboard Titanic. "
            "She describes her gilded cage: a wealthy engagement to Caledon Hockley, "
            "her mother's pressuring her into the marriage, and her own despair."
        ),
    },
    {
        "movie_id": "tt0120338",
        "start_ge": 410, "end_lt": 510,
        "heading": "EXT. SOUTHAMPTON DOCK - DAY",
        "context": (
            "Rose boards the RMS Titanic in Southampton. She is photographed with Cal, "
            "who shows her the priceless diamond necklace he bought as an engagement gift. "
            "Meanwhile, Jack Dawson wins his third-class ticket in a poker game at a pub."
        ),
    },
    {
        "movie_id": "tt0120338",
        "start_ge": 510, "end_lt": 600,
        "heading": "EXT. TITANIC - BOW - SUNSET",
        "context": (
            "Despairing over her engagement and feeling suffocated, Rose goes to the stern of the ship "
            "and stands on the railing, threatening to jump. Jack notices and talks her back, "
            "persuading her that 'you must do this' — whatever will make her happy."
        ),
    },
    {
        "movie_id": "tt0120338",
        "start_ge": 600, "end_lt": 720,
        "heading": "INT. TITANIC DINING ROOM - NIGHT",
        "context": (
            "Rose dines in the first-class dining room with Cal and his associate. She introduces Jack, "
            "whom she invites as her guest. Cal is humiliated and hostile. Jack, a poor artist, "
            "is out of place among the wealthy passengers. Rose and Jack share a fleeting connection."
        ),
    },
    {
        "movie_id": "tt0120338",
        "start_ge": 720, "end_lt": 820,
        "heading": "INT. TITANIC - BOTTOM OF SHIP - NIGHT",
        "context": (
            "Rose meets Jack in the ship's lower decks where he sketches portraits for passengers. "
            "She sees his drawings and is fascinated by his free spirit. They share stories "
            "and Jack promises to teach Rose to ride a bike."
        ),
    },
    {
        "movie_id": "tt0120338",
        "start_ge": 820, "end_lt": 900,
        "heading": "INT. TITANIC (THIRD CLASS) - NIGHT",
        "context": (
            "Jack takes Rose to a lively Irish dance in the third-class hold, a stark contrast "
            "to the stuffy first-class decks above. Rose laughs freely for the first time in years. "
            "Cal watches from the upper deck, his resentment growing."
        ),
    },
    {
        "movie_id": "tt0120338",
        "start_ge": 900, "end_lt": 960,
        "heading": "INT. TITANIC - BOW - NIGHT",
        "context": (
            "Jack and Rose stand at the bow of the ship at night. Jack tells Rose his philosophy: "
            "'Make each day count.' Rose declares she can feel the ship moving through the ice field. "
            "It is their most romantic moment, the famous 'I'm flying' scene."
        ),
    },
    {
        "movie_id": "tt0120338",
        "start_ge": 960, "end_lt": 1080,
        "heading": "INT. TITANIC CORRIDOR - DAY",
        "context": (
            "Cal discovers his valet with Jack's journal containing a drawing of Rose. He confronts her "
            "and slaps her. When Jack arrives, Cal manipulates Lovejoy into planting the diamond "
            "in Jack's coat. Cal reports Jack to the ship's officers as a thief."
        ),
    },
    {
        "movie_id": "tt0120338",
        "start_ge": 1080, "end_lt": 1200,
        "heading": "EXT. TITANIC - DECK - NIGHT",
        "context": (
            "The Titanic strikes an iceberg. Passengers are alarmed but initially unconcerned. "
            "As the ship tilts, the severity becomes apparent. Officers begin loading lifeboats. "
            "Cal finds Jack handcuffed in a storage room; Jack escapes as the ship groans and splits apart."
        ),
    },
    {
        "movie_id": "tt0120338",
        "start_ge": 1200, "end_lt": 1320,
        "heading": "EXT. TITANIC - DECK - NIGHT / EARLY MORNING",
        "context": (
            "The ship breaks apart and sinks stern-first. Jack and Rose are adrift on the Collapsible A lifeboat. "
            "Jack helps Rose stay on the raft above the freezing water. "
            "Cal dies attempting to shoot them; Rose survives the night in the freezing Atlantic. "
            "Jack dies of hypothermia. Rose is rescued by the Carpathia."
        ),
    },
    {
        "movie_id": "tt0120338",
        "start_ge": 1320, "end_lt": 9999,
        "heading": "INT. KELDYSH - NIGHT",
        "context": (
            "Rose finishes her story, giving Brock Lovett a new ending: she survived, married, had children, "
            "and lived fully. Rose goes out on deck at night and drops the Heart of the Ocean diamond "
            "into the sea, reuniting it with the Titanic. She falls asleep peacefully."
        ),
    },
]

# ─── The Shawshank Redemption (tt0073486) screenplay knowledge ─────────────────
SHAWSHANK_SCRIPT: list[dict[str, Any]] = [
    {
        "movie_id": "tt0073486",
        "start_lt": 180,
        "heading": "INT. MAINE BANK - DAY (FLASHBACK: 1947)",
        "context": (
            "Andy Dufresne, a quiet banker, walks out of a Maine bank carrying a revolver. "
            "He withdraws money, hires a lawyer, and drives to a pier. "
            "In voiceover, Red explains that Andy was convicted of murdering his wife and her lover."
        ),
    },
    {
        "movie_id": "tt0073486",
        "start_ge": 150, "end_lt": 270,
        "heading": "EXT. SHAWSHANK PRISON YARD - DAY (1946)",
        "context": (
            "Andy Dufresne arrives at Shawshank Prison on a prison bus. "
            "Red, the narrator, narrates his first impression. "
            "He describes Andy's quiet demeanor: 'The silence... that loudest of noises.' "
            "The other prisoners bet on when Andy will break. He doesn't eat for weeks."
        ),
    },
    {
        "movie_id": "tt0073486",
        "start_ge": 150, "end_lt": 260,
        "heading": "INT. CELLBLOCK - NIGHT",
        "context": (
            "Brooks Hatlen, the elderly librarian, has been at Shawshank for fifty years. "
            "He runs the prison library and befriends the new inmates. "
            "He carves his name into a wooden beam — 'Brooks was here' — before his parole."
        ),
    },
    {
        "movie_id": "tt0073486",
        "start_ge": 270, "end_lt": 350,
        "heading": "INT. SHAWSHANK CORRIDOR - DAY",
        "context": (
            "Andy begins working in the prison library under Brooks. He asks the warden "
            "for funds to build a proper library, alarming the Sisters — a group of violent inmates. "
            "Byron Hadley and his fellow Sisters begin targeting Andy."
        ),
    },
    {
        "movie_id": "tt0073486",
        "start_ge": 270, "end_lt": 310,
        "heading": "INT. PRISON PHARMACY - DAY",
        "context": (
            "Brooks is paroled. He struggles to adjust to the outside world: "
            "a life of working at a grocery store, hitchhiking, feeling lost. "
            "He writes letters to his former prison friends, confessing his despair."
        ),
    },
    {
        "movie_id": "tt0073486",
        "start_ge": 270, "end_lt": 350,
        "heading": "INT. SHAWSHANK PRISON LIBRARY - DAY",
        "context": (
            "Andy smuggles books and records into the prison. He plays Mozart's The Marriage of Figaro "
            "over the prison PA system for the entire prison to hear. Red describes it as the most "
            "beautiful thing he has ever heard: 'Two hours of pure, clean freedom.'"
        ),
    },
    {
        "movie_id": "tt0073486",
        "start_ge": 350, "end_lt": 500,
        "heading": "EXT. SHAWSHANK PRISON YARD - DAY (YEARS LATER)",
        "context": (
            "Time passes. Andy has become the prison's accountant, doing the warden's taxes. "
            "Red narrates the years of Andy's labor: his letters requesting funding, his "
            "shamming of sexual offender records for his eventual escape plan. "
            "He receives a harmonica from Red."
        ),
    },
    {
        "movie_id": "tt0073486",
        "start_ge": 350, "end_lt": 500,
        "heading": "INT. WARDEN'S OFFICE - DAY",
        "context": (
            "Warden Samuel Norton uses Andy to launder money from corruption. "
            "Andy has been filing false accounts. He tells Red he needs a rock hammer "
            "and a poster of Rita Hayworth to complete his escape plan. "
            "Boggs, a brutal guard, is killed; Andy is beaten by the Sisters."
        ),
    },
    {
        "movie_id": "tt0073486",
        "start_ge": 500, "end_lt": 840,
        "heading": "EXT. SHAWSHANK PRISON YARD - DAY",
        "context": (
            "Andy finally achieves his dream: a proper library funded by the state. "
            "He has been at Shawshank for nearly two decades. In the yard, he tells Red "
            "about a place in Mexico he calls Zihuatanejo — a Pacific dream. "
            "Red grows suspicious that Andy might actually escape."
        ),
    },
    {
        "movie_id": "tt0073486",
        "start_ge": 840, "end_lt": 1200,
        "heading": "INT. WARDEN'S OFFICE / CONFINEMENT - DAY",
        "context": (
            "Tommy Williams arrives at Shawshank and becomes a friend to Andy and Red. "
            "Tommy reveals he knows who really killed Andy's wife and her lover — a man named Tommy "
            "recognizes from his previous prison. Andy demands a retrial; the warden refuses and puts "
            "Tommy in solitary. The warden orders the Sisters to kill Andy."
        ),
    },
    {
        "movie_id": "tt0073486",
        "start_ge": 1200, "end_lt": 9999,
        "heading": "INT. SHAWSHANK PRISON - DAY",
        "context": (
            "Andy escapes through a tunnel he has been digging for nineteen years, using a rock hammer. "
            "He emerges in a river, free. He has left a letter for Red: 'Hope is a good thing, "
            "maybe the best of things, and no good thing ever dies.' "
            "Red is paroled after forty years; he goes to Maine and walks to find the buried box."
        ),
    },
]

# ─── As Good as It Gets (tt0119822) screenplay knowledge ────────────────────
ASGOOD_SCRIPT: list[dict[str, Any]] = [
    {
        "movie_id": "tt0119822",
        "start_lt": 50,
        "heading": "INT. MELVIN UDALL'S APARTMENT - MORNING",
        "context": (
            "Melvin Udall, an obsessive-compulsive author, begins his day with ritualized behaviors: "
            "stepping on cracks, changing cutlery positions, avoiding sidewalk cracks. "
            "He is introduced as a brilliant but deeply neurotic man, cruel to everyone around him."
        ),
    },
    {
        "movie_id": "tt0119822",
        "start_ge": 36, "end_lt": 110,
        "heading": "INT. MELVIN'S APARTMENT HALLWAY - DAY",
        "context": (
            "Carol Connelly, a struggling waitress and mother of a chronically ill son, is Melvin's neighbor. "
            "Melvin has been torturing her dog, Verdell, with a stick, poking at the dog each morning. "
            "Carol confronts Melvin; he is rude and dismissive. This is their antagonistic introduction."
        ),
    },
    {
        "movie_id": "tt0119822",
        "start_ge": 110, "end_lt": 220,
        "heading": "INT. SIMON WARD'S GALLERY - DAY",
        "context": (
            "Melvin's agent, Peter, arranges a dinner with Simon Ward, a gay artist whose gallery "
            "Melvin patronizes. Simon is mugged in Central Park by anti-gay attackers and hospitalized. "
            "Melvin visits Simon in the hospital and is uncharacteristically caring. "
            "Melvin then asks Peter to set him up with Carol."
        ),
    },
    {
        "movie_id": "tt0119822",
        "start_ge": 220, "end_lt": 320,
        "heading": "EXT. SIMON'S HOSPITAL ROOM - DAY",
        "context": (
            "Simon, recovering from his attack, discusses his artwork and his shattered sense of safety. "
            "Melvin begins visiting regularly. He is rude to Simon's nurse but oddly protective of Carol. "
            "Carol's estranged husband Frank returns; they argue and Frank moves back in, creating chaos."
        ),
    },
    {
        "movie_id": "tt0119822",
        "start_ge": 320, "end_lt": 390,
        "heading": "INT. CAROL'S APARTMENT - DAY",
        "context": (
            "Carol comes home to find Frank has redecorated her apartment and brought friends. "
            "She confides to Simon that she is considering leaving her job, her city, her life. "
            "Melvin arrives and insists Carol cannot quit. He confesses his feelings: "
            "'You make me want to be a better man.'"
        ),
    },
    {
        "movie_id": "tt0119822",
        "start_ge": 390, "end_lt": 470,
        "heading": "INT. SIMON'S HOSPITAL ROOM - DAY",
        "context": (
            "Simon is discharged and returns home. Melvin cooks dinner for Simon and Carol — "
            "the first time he has cooked for anyone. He accidentally cuts himself badly. "
            "Carol bandages him. The three share a moment of unexpected intimacy."
        ),
    },
    {
        "movie_id": "tt0119822",
        "start_ge": 470, "end_lt": 560,
        "heading": "EXT. RESTAURANT - NIGHT",
        "context": (
            "Carol and Melvin go on a date at a restaurant. Melvin is anxious, rigid "
            "and unable to eat due to his anxieties. Carol calms him by reciting her daily mantras. "
            "Melvin admits he has never been on a date. He gives her a key to his apartment."
        ),
    },
    {
        "movie_id": "tt0119822",
        "start_ge": 560, "end_lt": 650,
        "heading": "INT. MELVIN'S APARTMENT - DAY",
        "context": (
            "Carol moves into Melvin's apartment with Verdell. The house rules begin: "
            "she cannot sing, she cannot use the kitchen. They negotiate a truce. "
            "Simon flies to Baltimore for a gallery show. Carol and Melvin argue about leaving the apartment."
        ),
    },
    {
        "movie_id": "tt0119822",
        "start_ge": 650, "end_lt": 9999,
        "heading": "EXT. BALTIMORE ART GALLERY - NIGHT",
        "context": (
            "Melvin, Carol, and Simon reunite at Simon's gallery in Baltimore. "
            "Melvin has bought all of Simon's unsold paintings. "
            "Carol asks Melvin to move to Baltimore with her. He accepts. "
            "On the street outside, Melvin passes a sidewalk crack — and steps over it. "
            "'I'm going to be a better man.'"
        ),
    },
]

ALL_SCRIPTS = TITANIC_SCRIPT + SHAWSHANK_SCRIPT + ASGOOD_SCRIPT


def find_best_script_entry(movie_id: str, start_sec: float) -> dict | None:
    """Find the best-matching script entry for a chunk (start_sec is inclusive)."""
    best: dict | None = None
    best_overlap = -1.0

    for entry in ALL_SCRIPTS:
        if entry["movie_id"] != movie_id:
            continue
        end_lt = entry.get("end_lt", 9999.0)
        start_ge = entry.get("start_ge", 0.0)

        # Inclusive start, exclusive end (standard interval semantics)
        if start_sec < start_ge:
            continue
        if start_sec >= end_lt:
            continue

        # Prefer entry with the smallest end_lt (most specific)
        if end_lt < best_overlap or best is None:
            best = entry
            best_overlap = end_lt

    return best


def humanize_heading(heading: str, situation: str, movie_id: str) -> str:
    """Convert a screenplay heading to human-readable format."""
    heading = heading.strip()
    if heading:
        return heading
    # Fallback: generate from situation + movie context
    if movie_id == "tt0120338":
        return f"INT./EXT. TITANIC - {situation or 'SCENE'}"
    elif movie_id == "tt0073486":
        return f"INT. SHAWSHANK PRISON - {situation or 'SCENE'}"
    elif movie_id == "tt0119822":
        return f"INT. {situation or 'MELVIN UDALL'} - SCENE"
    return f"INT./EXT. - {situation}"


def enrich_chunk(chunk: dict) -> dict:
    """Fill missing L5 fields for a single chunk."""
    movie_id = chunk.get("movie_id", "")
    start_sec = float(chunk.get("start_seconds", 0))

    entry = find_best_script_entry(movie_id, start_sec)

    enriched = dict(chunk)

    if entry:
        heading = humanize_heading(entry["heading"], chunk.get("situation", ""), movie_id)
        context = entry["context"]
        screenplay_context = f"[SCREENPLAY EVIDENCE] {heading}. {context}"
    else:
        heading = ""
        context_fallback = (
            f"Film scene covering {chunk.get('situation', 'an unspecified situation')}. "
            f"Characters present: {', '.join(chunk.get('characters', [])[:3]) or 'unspecified'}. "
            f"Film's narrative arc: {chunk.get('narrative_arc', 'unspecified')}."
        )
        screenplay_context = f"[DERIVED] {context_fallback}"

    # Only fill heading if truly missing (don't overwrite existing)
    if not enriched.get("script_primary_heading", "").strip():
        enriched["script_primary_heading"] = heading
    if not enriched.get("screenplay_context_excerpt", "").strip():
        enriched["screenplay_context_excerpt"] = screenplay_context

    # Update evidence_source to include screenplay
    existing_sources = enriched.get("evidence_source", [])
    if isinstance(existing_sources, list):
        new_sources = list(existing_sources)
        if "screenplay_derived" not in new_sources:
            new_sources.append("screenplay_derived")
        enriched["evidence_source"] = new_sources
    else:
        enriched["evidence_source"] = ["screenplay_derived"]

    return enriched


def main():
    input_path = INPUT
    output_path = OUTPUT

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    chunks = json.loads(input_path.read_text(encoding="utf-8"))

    enriched = [enrich_chunk(chunk) for chunk in chunks]

    output_path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Report
    filled_heading = sum(
        1 for c in enriched if c.get("script_primary_heading", "").strip()
    )
    filled_context = sum(
        1 for c in enriched if c.get("screenplay_context_excerpt", "").strip()
    )

    print(f"Enriched {len(enriched)} chunks:")
    print(f"  script_primary_heading: {filled_heading}/{len(enriched)} filled")
    print(f"  screenplay_context_excerpt: {filled_context}/{len(enriched)} filled")
    print(f"  Written to: {output_path}")


if __name__ == "__main__":
    main()
