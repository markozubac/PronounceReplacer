import cohere
import pandas as pd
import sys
import csv
import os
import time
import re

# Cohere klijent
co = cohere.ClientV2("cohere key value")

# Trial klijent
# co = cohere.ClientV2("sWQIbobLHntVOx0wtOB9IwV4S7MNrOupjvFFi5Gl")

# --- Skup zamjenica (lowercase) ---
PRONOUNS = {
    "i","me","myself","my","mine",
    "he","him","himself","his",
    "she","her","herself","hers",
    "it","itself","its",
    "they","them","themselves","themself","their","theirs",
    "who","whom","whose"
}

WORD_RE = re.compile(r"\b[\w&'’-]+\b", flags=re.UNICODE)  # tokenizacija sa granicama riječi

def generate_text(text):
    prompt = f"""Extract only factual triplets from the following text in the format: "Subject"|"Relation"|"Object".
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

Example 4:
Input: FBI Mortgage Fraud Department came into existence.
Outputs:
"FBI Mortgage Fraud Department"|"came into"|"existence"

Example 5:
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
    response = co.chat(
        model="command-a-03-2025",
        messages=[{'role': 'user', 'content': prompt}]
    )
    result = ""
    for item in response.message.content:
        if item.type == 'text':
            result += item.text
    return result.strip()

def rewrite_chunk_with_context(current_text, prev_chunks):
    """
    prev_chunks: lista [stariji, noviji] (0..n-1), samo oni koji imaju isti question_ID kao trenutni
    Vraća prepisani tekst trenutnog chunka sa rezolviranim referencama.
    """
    context_str = "\n\n".join(
        [f"[Prev {i+1}] {t}" for i, t in enumerate(prev_chunks)]
    ) if prev_chunks else "(no prior context)"

    prompt = f"""You are a precise coreference resolver.
Using ONLY the information in the earlier context, rewrite the CURRENT CHUNK so that every pronoun in the CURRENT CHUNK
(e.g., I, he, she, it, they, who/whom/whose, and possessives like his/her/their/its/my) is replaced with the explicit named entity it refers to.

Constraints:
- Rewrite ONLY the CURRENT CHUNK text (do not summarize or add info).
- Keep meaning, tense, and structure; just replace pronouns with their antecedents.
- If an antecedent is ambiguous or not present in context, leave the original word as-is.
- Output ONLY the rewritten chunk text without any labels or explanations.

EARLIER CONTEXT:
{context_str}

CURRENT CHUNK:
{current_text}
"""

    response = co.chat(
        model="command-a-03-2025",
        messages=[{'role': 'user', 'content': prompt}]
    )
    result = ""
    for item in response.message.content:
        if item.type == 'text':
            result += item.text
    return result.strip()

def is_valid_triplet(parts):
    if len(parts) != 3:
        return False
    for p in parts:
        val = p.strip().lower()
        if val == "" or val == "null":
            return False
    return True

def normalize_triplet_line(line):
    # Ujednači razmake oko delimiter-a
    return line.replace('" | "', '"|"').replace('" |"', '"|"').replace('"| "', '"|"')

def entity_contains_pronoun(entity_text):
    """Provjera da li subjekt ili objekt sadrži ijednu zamjenicu iz skupa, po riječima (sa granicom riječi)."""
    tokens = [t.lower() for t in WORD_RE.findall(entity_text)]
    return any(tok in PRONOUNS for tok in tokens)

def triplets_have_pronoun_in_SO(triplets_text):
    """True ako ijedan triplet ima zamjenicu u subjektu ili objektu."""
    for raw_line in triplets_text.splitlines():
        line = normalize_triplet_line(raw_line)
        parts = line.strip().strip('"').split('"|"')
        if len(parts) >= 3:
            subj, obj = parts[0], parts[2]
            if entity_contains_pronoun(subj) or entity_contains_pronoun(obj):
                return True
    return False

def get_prev_chunks_same_question(df, idx, question_id, k=2):
    """
    Vrati do k prethodnih chunkova koji imaju isti question_ID kao trenutni red (idx).
    Redoslijed: od starijeg ka novijem (tj. hronološki).
    """
    prev_chunks = []
    j = idx - 1
    while j >= 0 and len(prev_chunks) < k:
        if 'question_ID' in df.columns and df.loc[j, 'question_ID'] == question_id:
            # ubacujemo na početak da zadržimo poredak [stariji, ... , noviji]
            prev_chunks.insert(0, df.loc[j, 'chunk'])
        j -= 1
    return prev_chunks

# Ulazni fajl sa paragrafima
df = pd.read_csv("paragraph_chunks2.csv")

# Bitno: sortiraj po chunk_ID da bi "prethodna 2" bila određena stabilno
if 'chunk_ID' in df.columns:
    df = df.sort_values(by='chunk_ID', ascending=True).reset_index(drop=True)

start_context_id = 126083

# Izlazni fajl sa tripletima
triplets_file = "triplets_with_index_chunks.csv"

# Folder i fajl za loše formatirane triplete
bad_folder = "bad_form_triplets_chunks"
os.makedirs(bad_folder, exist_ok=True)
bad_triplets_file = os.path.join(bad_folder, "bad_triplets_chunks.csv")

# --- Proveri postojeće triplete da ne dupliraš ---
processed_ids = set()
if os.path.isfile(triplets_file):
    try:
        existing = pd.read_csv(triplets_file, delimiter='|', quotechar='"')
        if 'chunk_ID' in existing.columns:
            processed_ids = set(existing['chunk_ID'].tolist())
    except Exception as e:
        print(f"⚠️ Greška pri čitanju postojećih tripleta: {e}")

# Proveri da li fajlovi postoje
file_exists = os.path.isfile(triplets_file)
bad_file_exists = os.path.isfile(bad_triplets_file)

# Priprema CSV fajlova
with open(triplets_file, "a", encoding="utf-8", newline="") as csvfile, \
     open(bad_triplets_file, "a", encoding="utf-8", newline="") as badfile:

    writer = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)
    bad_writer = csv.writer(badfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)

    # Zaglavlja
    if not file_exists:
        writer.writerow(["chunk_ID", "question_ID", "triplet"])
    if not bad_file_exists:
        bad_writer.writerow(["chunk_ID", "question_ID", "bad_triplet"])

    # Iteracija
    for idx, row in df.iterrows():
        paragraph_id = row['chunk_ID']
        question_id = row['question_ID'] if 'question_ID' in row else None

        if paragraph_id < start_context_id:
            continue

        # preskoči ako je već obrađen (po postojećem fajlu)
        if paragraph_id in processed_ids:
            print(f"⏭️ Skipping already processed chunk {paragraph_id}")
            continue

        text = row['chunk']
        print(f"Generating triplets for chunk {paragraph_id}...")

        # 1) Prvo generiši triplete iz originalnog teksta
        triplets = generate_text(text)

        # 2) Ako ijedan triplet ima zamjenicu u subjektu/objektu -> rezolucija i regenerisanje
        if triplets and triplets_have_pronoun_in_SO(triplets):
            prev_chunks = get_prev_chunks_same_question(df, idx, question_id, k=2)

            if prev_chunks:
                print(f"↪️ Pronoun detected in chunk {paragraph_id}. Resolving with SAME-question context ({len(prev_chunks)} prev chunks)...")
            else:
                print(f"↪️ Pronoun detected in chunk {paragraph_id}, but no prior chunks with the same question_ID. Resolving without context...")

            rewritten_text = rewrite_chunk_with_context(text, prev_chunks)

            # Ako je model dao nešto smisleno, generiši triplete iz prepisanog
            if rewritten_text:
                triplets = generate_text(rewritten_text)
                print(f"✅ Re-generated triplets for chunk {paragraph_id} after pronoun resolution.")
            else:
                print(f"⚠️ Pronoun resolution returned empty for chunk {paragraph_id}. Using original triplets.")

        # 3) Upis rezultata (dobri/loši) – ista logika kao ranije
        for line in triplets.splitlines() if triplets else []:
            clean_line = normalize_triplet_line(line)
            parts = clean_line.strip().strip('"').split('"|"')
            if is_valid_triplet(parts):
                writer.writerow([paragraph_id, question_id, line.strip()])
            else:
                bad_writer.writerow([paragraph_id, question_id, line.strip()])
                print(f"⚠️ Skipped bad triplet at context {paragraph_id}: {line.strip()}")

print(f"\nSaved good triplets to {triplets_file}")
print(f"Saved bad triplets to {bad_triplets_file}")

