import cohere
import pandas as pd
import csv
import os
import re
from typing import List, Dict

# ============== CONFIG ==============
co = cohere.ClientV2("Your API key")
MODEL_NAME = "command-a-03-2025"

INPUT_CSV = "paragraph_chunks2.csv"
OUTPUT_CSV = "triplets_with_index_chunks_m3.csv"
BAD_DIR = "bad_form_triplets_chunks_m3"
BAD_CSV = os.path.join(BAD_DIR, "bad_triplets_chunks_m3.csv")

START_CHUNK_ID = 399  # promijeni ako želiš preskočiti ranije chunkove
K_PREV = 2          # koliko prethodnih chunkova (sa istim question_ID) gledamo

# ============== LINGVO ==============
PRONOUNS = {
    "i","me","myself","my","mine",
    "he","him","himself","his",
    "she","her","herself","hers",
    "it","itself","its",
    "they","them","themselves","themself","their","theirs",
    "who","whom","whose"
}
WORD_RE = re.compile(r"\b[\w&'’-]+\b", flags=re.UNICODE)

# ============== PROMPTS ==============
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

def build_context_from_prev_triplets_prompt(current_text: str,
                                            context_triplets: List[str]) -> str:
    ctx = "\n".join(context_triplets) if context_triplets else "(no prior triplets)"
    return f"""You are an information extraction system that MUST resolve pronouns in the CURRENT CHUNK using ONLY the PRIOR TRIPLETS as context.
PRIOR TRIPLETS provide explicit entities and relations. Replace pronouns in your understanding (he/she/it/they/his/her/their/its/I/me/my...) with the most plausible explicit entity grounded in PRIOR TRIPLETS, when possible.

TASK:
- Extract factual triplets from the CURRENT CHUNK only, in the strict format: "Subject"|"Relation"|"Object".
- When a pronoun in the CURRENT CHUNK refers to an entity found in PRIOR TRIPLETS, you MUST output the explicit named entity instead of the pronoun.
- Do NOT output triplets about the prior context unless they are also asserted in the CURRENT CHUNK.

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

PRIOR TRIPLETS:
{ctx}

CURRENT CHUNK:
{current_text}
"""

# ============== LLM wrappers ==============
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

def generate_triplets_with_prev_triplets(current_text: str,
                                         context_triplets: List[str]) -> str:
    return call_llm(build_context_from_prev_triplets_prompt(current_text, context_triplets))

# ============== Helpers ==============
def normalize_triplet_line(line: str) -> str:
    return line.replace('" | "', '"|"').replace('" |"', '"|"').replace('"| "', '"|"')

def is_valid_triplet(parts: List[str]) -> bool:
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
    for raw in (triplets_text or "").splitlines():
        line = normalize_triplet_line(raw)
        parts = line.strip().strip('"').split('"|"')
        if len(parts) >= 3:
            s, o = parts[0], parts[2]
            if entity_contains_pronoun(s) or entity_contains_pronoun(o):
                return True
    return False

def get_prev_chunk_ids_same_question(df: pd.DataFrame, idx: int, question_id, k: int = 2) -> List[int]:
    ids = []
    j = idx - 1
    while j >= 0 and len(ids) < k:
        if 'question_ID' in df.columns and df.loc[j, 'question_ID'] == question_id:
            ids.insert(0, int(df.loc[j, 'chunk_ID']))  # hronološki [stariji .. noviji]
        j -= 1
    return ids

# ============== Main (Method 3) ==============
def main():
    os.makedirs(BAD_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    if 'chunk_ID' in df.columns:
        df = df.sort_values(by='chunk_ID', ascending=True).reset_index(drop=True)

    # izbjegni dupliranje upisa
    processed_ids = set()
    if os.path.isfile(OUTPUT_CSV):
        try:
            existing = pd.read_csv(OUTPUT_CSV, delimiter='|', quotechar='"')
            if 'chunk_ID' in existing.columns:
                processed_ids = set(existing['chunk_ID'].astype(int).tolist())
        except Exception as e:
            print(f"⚠️ Greška pri čitanju postojećih tripleta: {e}")

    file_exists = os.path.isfile(OUTPUT_CSV)
    bad_exists = os.path.isfile(BAD_CSV)

    # cache samo iz ove runde: {chunk_ID: [triplet_line, ...]}
    in_run_triplets: Dict[int, List[str]] = {}

    with open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as good_f, \
         open(BAD_CSV, "a", encoding="utf-8", newline="") as bad_f:

        good_w = csv.writer(good_f, delimiter='|', quoting=csv.QUOTE_MINIMAL)
        bad_w = csv.writer(bad_f, delimiter='|', quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            good_w.writerow(["chunk_ID", "question_ID", "triplet"])
        if not bad_exists:
            bad_w.writerow(["chunk_ID", "question_ID", "bad_triplet"])

        for idx, row in df.iterrows():
            chunk_id = int(row['chunk_ID'])
            if chunk_id < START_CHUNK_ID:
                continue
            if chunk_id in processed_ids:
                print(f"⏭️ Skipping already processed chunk {chunk_id}")
                continue

            qid = row['question_ID'] if 'question_ID' in df.columns else None
            text = str(row['chunk'])
            print(f"➡️ Chunk {chunk_id}: base extraction...")

            # 1) baza: samo trenutni chunk
            base_triplets = generate_triplets_base(text)

            # 2) ako postoji zamjenica u S/O -> prekini bazu i radi 2. prolaz sa kontekstom = tripleti iz prethodna 2 chunka (isključivo iz ove runde)
            if base_triplets and triplets_have_pronoun_in_SO(base_triplets):
                prev_ids = get_prev_chunk_ids_same_question(df, idx, qid, k=K_PREV)

                context_triplets: List[str] = []
                for pid in prev_ids:
                    if pid in in_run_triplets:
                        context_triplets.extend(in_run_triplets[pid])

                if context_triplets:
                    print(f"↪️ Pronoun detected. Regenerating with PRIOR TRIPLETS from {len(prev_ids)} prev chunk(s) for {chunk_id} ...")
                    final_triplets = generate_triplets_with_prev_triplets(text, context_triplets)
                else:
                    print(f"↪️ Pronoun detected but no prior triplets available in this run. Falling back to base for {chunk_id}.")
                    final_triplets = base_triplets
            else:
                final_triplets = base_triplets

            # 3) upis + punjenje in-run cache-a
            wrote_any = False
            current_good: List[str] = []

            for line in final_triplets.splitlines() if final_triplets else []:
                clean = normalize_triplet_line(line)
                parts = clean.strip().strip('"').split('"|"')
                if is_valid_triplet(parts):
                    good_w.writerow([chunk_id, qid, line.strip()])
                    wrote_any = True
                    current_good.append(line.strip())
                else:
                    bad_w.writerow([chunk_id, qid, line.strip()])
                    print(f"⚠️ Skipped bad triplet at chunk {chunk_id}: {line.strip()}")

            if not wrote_any:
                bad_w.writerow([chunk_id, qid, (final_triplets or '').strip() or "(empty)"])
                print(f"⚠️ No valid triplets for chunk {chunk_id}.")

            if current_good:
                in_run_triplets[chunk_id] = current_good

    print(f"\n✅ Saved good triplets to {OUTPUT_CSV}")
    print(f"✅ Saved bad triplets to {BAD_CSV}")

if __name__ == "__main__":
    main()
