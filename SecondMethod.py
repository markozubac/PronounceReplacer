import cohere
import pandas as pd
import sys
import csv
import os
import time
import re

# ======= CONFIG =======
co = cohere.ClientV2("Your API key")
MODEL_NAME = "command-a-03-2025"

INPUT_CSV = "paragraph_chunks2.csv"
OUTPUT_CSV = "triplets_with_index_chunks_m2.csv"
BAD_DIR = "bad_form_triplets_chunks_m2"
BAD_CSV = os.path.join(BAD_DIR, "bad_triplets_chunks_m2.csv")

START_CHUNK_ID = 128176   # možeš promijeniti po potrebi
K_PREV = 2                # koliko prethodnih chunkova ubacujemo u 2. prolazu

# --- Skup zamjenica (lowercase) ---
PRONOUNS = {
    "i","me","myself","my","mine",
    "he","him","himself","his",
    "she","her","herself","hers",
    "it","itself","its",
    "they","them","themselves","themself","their","theirs",
    "who","whom","whose"
}

WORD_RE = re.compile(r"\b[\w&'’-]+\b", flags=re.UNICODE)

# ======= PROMPTS =======

def build_base_extraction_prompt(text: str) -> str:
    return f"""Extract only factual triplets from the following text in the format: "Subject"|"Relation"|"Object".
STRICT RULES:
- Each line MUST contain exactly 3 parts: subject, relation, object.
- Subject and object MUST each be 1–5 words (no long descriptions, no clauses).
- Relation MUST be 1–4 words.
- DO NOT include explanations, reasons, comparisons, or long sentences.
- If you cannot extract a valid triplet under these rules, skip it (do not generate).
- Output only valid triplets, one per sentence.

Example 1:
Input: Albert Einstein developed the theory of relativity while working in Switzerland.
Outputs:
"Albert Einstein"|"developed"|"theory of relativity"
"Albert Einstein"|"worked in"|"Switzerland"

Example 2:
Input: The Eiffel Tower in Paris was designed by Gustave Eiffel and completed in 1889.
Outputs:
"Eiffel Tower"|"located"|"Paris"
"Eiffel Tower"|"designed by"|"Gustave Eiffel"
"Eiffel Tower"|"completed"|"1889"

Example 3:
Input: Barack Obama served as the 44th president of the United States from 2009 to 2017.
Outputs:
"Barack Obama"|"served as"|"44th president"
"Barack Obama"|"president of"|"United States"
"Barack Obama"|"served from"|"2009"
"Barack Obama"|"served until"|"2017"

Example 4:
Input: Roberts & Vinter came under financial pressure after their printer went bankrupt.
Outputs:
"Roberts & Vinter"|"came under"|"financial pressure"
"Roberts & Vinter"|"impacted by"|"printer bankruptcy"

Example 5:
Input: FBI Mortgage Fraud Department came into existence.
Outputs:
"FBI Mortgage Fraud Department"|"came into"|"existence"

Example 6:
Input: Tyler Bates worked with films like "Dawn of the Dead, 300, Sucker Punch," and "John Wick." He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn.
Outputs: 
"Tyler Bates"|"known for film"|"Dawn of the Dead"
"Tyler Bates"|"known for film"|"300"
"Tyler Bates"|"known for film"|"Sucker Punch"
"Tyler Bates"|"known for film"|"John Wick"
"Tyler Bates"|"collaborated with"|"Zack Snyder"
"Tyler Bates"|"collaborated with"|"Rob Zombie"
"Tyler Bates"|"collaborated with"|"Neil Marshall"
"Tyler Bates"|"collaborated with"|"William Friedkin"
"Tyler Bates"|"collaborated with"|"Scott Derrickson"
"Tyler Bates"|"collaborated with"|"James Gunn"

Text:
{text}
"""

def build_context_extraction_prompt(current_text: str, prev_chunks: list[str]) -> str:
    """
    Prompt za 2. prolaz: koristi (do) 2 prethodna chunka + trenutni tekst kao JEDAN ulaz,
    rezolvira zamjenice na osnovu konteksta i IZVADI TRIPLETE sa eksplicitnim entitetima.
    """
    context_str = "\n\n".join(
        [f"[Prev {i+1}] {t}" for i, t in enumerate(prev_chunks)]
    ) if prev_chunks else "(no prior context)"

    return f"""Extract only factual triplets from the following text in the format: "Subject"|"Relation"|"Object". You are an information extraction system that MUST resolve pronouns using earlier context.

TASK:
- Use the EARLIER CONTEXT plus the CURRENT CHUNK to extract factual triplets from the CURRENT CHUNK in the strict format: "Subject"|"Relation"|"Object".
- When a pronoun in the CURRENT CHUNK (e.g., he/she/it/they/his/her/their/its/I/me/my...) refers to an entity introduced in the EARLIER CONTEXT or CURRENT CHUNK, you MUST replace it with the explicit named entity in the output triplets.

STRICT RULES:
- Each line MUST contain exactly 3 parts: subject, relation, object.
- Subject and object MUST each be 1–5 words (no long descriptions, no clauses).
- Relation MUST be 1–4 words.
- DO NOT include explanations, reasons, comparisons, or long sentences.
- If you cannot extract a valid triplet under these rules, skip it (do not generate).
- Output only valid triplets, one per sentence.

Example 1:
Input: Albert Einstein developed the theory of relativity while working in Switzerland.
Outputs:
"Albert Einstein"|"developed"|"theory of relativity"
"Albert Einstein"|"worked in"|"Switzerland"

Example 2:
Input: The Eiffel Tower in Paris was designed by Gustave Eiffel and completed in 1889.
Outputs:
"Eiffel Tower"|"located"|"Paris"
"Eiffel Tower"|"designed by"|"Gustave Eiffel"
"Eiffel Tower"|"completed"|"1889"

Example 3:
Input: Barack Obama served as the 44th president of the United States from 2009 to 2017.
Outputs:
"Barack Obama"|"served as"|"44th president"
"Barack Obama"|"president of"|"United States"
"Barack Obama"|"served from"|"2009"
"Barack Obama"|"served until"|"2017"

Example 4:
Input: Roberts & Vinter came under financial pressure after their printer went bankrupt.
Outputs:
"Roberts & Vinter"|"came under"|"financial pressure"
"Roberts & Vinter"|"impacted by"|"printer bankruptcy"

Example 5:
Input: FBI Mortgage Fraud Department came into existence.
Outputs:
"FBI Mortgage Fraud Department"|"came into"|"existence"

Example 6:
Input: Tyler Bates worked with films like "Dawn of the Dead, 300, Sucker Punch," and "John Wick." He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn.
Outputs: 
"Tyler Bates"|"known for film"|"Dawn of the Dead"
"Tyler Bates"|"known for film"|"300"
"Tyler Bates"|"known for film"|"Sucker Punch"
"Tyler Bates"|"known for film"|"John Wick"
"Tyler Bates"|"collaborated with"|"Zack Snyder"
"Tyler Bates"|"collaborated with"|"Rob Zombie"
"Tyler Bates"|"collaborated with"|"Neil Marshall"
"Tyler Bates"|"collaborated with"|"William Friedkin"
"Tyler Bates"|"collaborated with"|"Scott Derrickson"
"Tyler Bates"|"collaborated with"|"James Gunn"

EARLIER CONTEXT:
{context_str}

CURRENT CHUNK:
{current_text}
"""

# ======= LLM wrappers =======

def call_llm(prompt: str) -> str:
    resp = co.chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}]
    )
    out = ""
    for item in resp.message.content:
        if item.type == 'text':
            out += item.text
    return out.strip()

def generate_triplets_base(text: str) -> str:
    return call_llm(build_base_extraction_prompt(text))

def generate_triplets_with_context(current_text: str, prev_chunks: list[str]) -> str:
    return call_llm(build_context_extraction_prompt(current_text, prev_chunks))

# ======= Helpers =======

def normalize_triplet_line(line: str) -> str:
    return line.replace('" | "', '"|"').replace('" |"', '"|"').replace('"| "', '"|"')

def is_valid_triplet(parts: list[str]) -> bool:
    if len(parts) != 3:
        return False
    for p in parts:
        val = p.strip().lower()
        if val == "" or val == "null":
            return False
    return True

def entity_contains_pronoun(entity_text: str) -> bool:
    tokens = [t.lower() for t in WORD_RE.findall(entity_text)]
    return any(tok in PRONOUNS for tok in tokens)

def triplets_have_pronoun_in_SO(triplets_text: str) -> bool:
    for raw_line in triplets_text.splitlines():
        line = normalize_triplet_line(raw_line)
        parts = line.strip().strip('"').split('"|"')
        if len(parts) >= 3:
            subj, obj = parts[0], parts[2]
            if entity_contains_pronoun(subj) or entity_contains_pronoun(obj):
                return True
    return False

def get_prev_chunks_same_question(df: pd.DataFrame, idx: int, question_id, k: int = 2) -> list[str]:
    prev_chunks = []
    j = idx - 1
    while j >= 0 and len(prev_chunks) < k:
        if 'question_ID' in df.columns and df.loc[j, 'question_ID'] == question_id:
            prev_chunks.insert(0, df.loc[j, 'chunk'])  # hronološki [stariji ... noviji]
        j -= 1
    return prev_chunks

# ======= Main pipeline (Method 2) =======

def main():
    os.makedirs(BAD_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # stabilan poredak
    if 'chunk_ID' in df.columns:
        df = df.sort_values(by='chunk_ID', ascending=True).reset_index(drop=True)

    # već obrađeni (da izbjegnemo duplikate)
    processed_ids = set()
    if os.path.isfile(OUTPUT_CSV):
        try:
            existing = pd.read_csv(OUTPUT_CSV, delimiter='|', quotechar='"')
            if 'chunk_ID' in existing.columns:
                processed_ids = set(existing['chunk_ID'].tolist())
        except Exception as e:
            print(f"⚠️ Greška pri čitanju postojećih tripleta: {e}")

    file_exists = os.path.isfile(OUTPUT_CSV)
    bad_file_exists = os.path.isfile(BAD_CSV)

    with open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as out_f, \
         open(BAD_CSV, "a", encoding="utf-8", newline="") as bad_f:

        good_w = csv.writer(out_f, delimiter='|', quoting=csv.QUOTE_MINIMAL)
        bad_w = csv.writer(bad_f, delimiter='|', quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            good_w.writerow(["chunk_ID", "question_ID", "triplet"])
        if not bad_file_exists:
            bad_w.writerow(["chunk_ID", "question_ID", "bad_triplet"])

        for idx, row in df.iterrows():
            chunk_id = row['chunk_ID']
            qid = row['question_ID'] if 'question_ID' in row else None

            if chunk_id < START_CHUNK_ID:
                continue

            if chunk_id in processed_ids:
                print(f"⏭️ Skipping already processed chunk {chunk_id}")
                continue

            text = row['chunk']
            print(f"Generating triplets (base) for chunk {chunk_id}...")

            # 1) Prvi prolaz: samo trenutni chunk
            triplets = generate_triplets_base(text)

            # 2) Validacija: ako pronoun u S/O -> DRUGI PROLAZ sa ubačenim prethodnim chunkovima i drugačijim promptom
            if triplets and triplets_have_pronoun_in_SO(triplets):
                prev_chunks = get_prev_chunks_same_question(df, idx, qid, k=K_PREV)
                if prev_chunks:
                    print(f"↪️ Pronoun detected. Regenerating with {len(prev_chunks)} prior chunk(s) context for {chunk_id} ...")
                else:
                    print(f"↪️ Pronoun detected but no prior chunks for same question_ID. Regenerating without context (will behave like base).")

                triplets = generate_triplets_with_context(text, prev_chunks)

                # opcionalno: ako i poslije konteksta i dalje imamo pronoun u S/O, možemo napisati u bad
                # ali ovdje ćemo svejedno pokušati zapisati validne linije.

            # 3) Upis (razdvajamo validne i loše formatirane)
            wrote_any = False
            for line in triplets.splitlines() if triplets else []:
                clean = normalize_triplet_line(line)
                parts = clean.strip().strip('"').split('"|"')
                if is_valid_triplet(parts):
                    good_w.writerow([chunk_id, qid, line.strip()])
                    wrote_any = True
                else:
                    bad_w.writerow([chunk_id, qid, line.strip()])
                    print(f"⚠️ Skipped bad triplet at chunk {chunk_id}: {line.strip()}")

            if not wrote_any:
                # ako ništa validno — evidentiraj u bad fajlu radi praćenja
                bad_w.writerow([chunk_id, qid, (triplets or "").strip() or "(empty)"])
                print(f"⚠️ No valid triplets for chunk {chunk_id}.")

    print(f"\nSaved good triplets to {OUTPUT_CSV}")
    print(f"Saved bad triplets to {BAD_CSV}")

if __name__ == "__main__":
    main()
