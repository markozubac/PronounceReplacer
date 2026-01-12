import os
import logging
import cohere
import pandas as pd
import csv
import re

logging.basicConfig(level=logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cohere klijent
co = cohere.ClientV2("api key")

# fastcoref
from fastcoref import FCoref
# Ako želiš GPU: FCoref(device="cuda")
coref_model = FCoref(device="cpu")

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


def _apply_replacement_with_clusters(text, clusters):
    """
    Robusnija zamjena: svaku klastersku zamjenicu zamijeni antecedentom,
    počevši od najdužih match-eva da se izbjegne preklapanje.
    """
    for cl in clusters:
        if not cl:
            continue
        antecedent = cl[0]
        mentions = list(dict.fromkeys(cl[1:]))  # uniq uz očuvanje redoslijeda
        mentions.sort(key=len, reverse=True)

        for mention in mentions:
            if not mention or mention == antecedent:
                continue
            # pokušaj boundary-aware regex zamjene
            pat = r'(?<!\w){}(?!\w)'.format(re.escape(mention))
            new_text, n = re.subn(pat, antecedent, text)
            if n == 0:
                # fallback na obično replace (za slučajeve sa interpunkcijom sl.)
                new_text = text.replace(mention, antecedent)
            text = new_text
    return text


def rewrite_chunk_with_context_fastcoref(current_text, prev_chunks):
    """
    Zamjena stare LLM funkcije: koristi fastcoref.
    - Spoji prethodne chunkove (isti question_ID) + trenutni chunk
    - Uradi coref nad cijelim tekstom (bolji antecedenti)
    - Vrati SAMO rezolviranu verziju trenutnog chunka
    """
    # specijalni markeri da precizno izvučemo dio trenutnog chunka
    START = "\n<<<CURRENT_START>>>"
    END = "\n<<<CURRENT_END>>>"

    context = ""
    if prev_chunks:
        context = "\n\n".join(prev_chunks)

    combined = (context + START + "\n" + current_text + END) if context \
        else (START + "\n" + current_text + END)

    preds = coref_model.predict(texts=[combined])
    clusters = preds[0].get_clusters(as_strings=True)

    resolved_all = _apply_replacement_with_clusters(combined, clusters)

    # izdvoji samo dio između markera
    try:
        start_idx = resolved_all.index(START) + len(START)
        end_idx = resolved_all.index(END)
        resolved_current = resolved_all[start_idx:end_idx].strip()
        return resolved_current
    except ValueError:
        # ako nešto pođe po zlu sa markerima, vrati originalni tekst
        return current_text.strip()


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
            prev_chunks.insert(0, df.loc[j, 'chunk'])
        j -= 1
    return prev_chunks

df = pd.read_csv("paragraph_chunks2.csv")

# Sort stabilnosti
if 'chunk_ID' in df.columns:
    df = df.sort_values(by='chunk_ID', ascending=True).reset_index(drop=True)

start_context_id = 126883

triplets_file = "triplets_with_index_chunks_fastCoref.csv"

bad_folder = "bad_form_triplets_chunks_fastCoref"
os.makedirs(bad_folder, exist_ok=True)
bad_triplets_file = os.path.join(bad_folder, "bad_triplets_chunks_fastCoref.csv")

processed_ids = set()
if os.path.isfile(triplets_file):
    try:
        existing = pd.read_csv(triplets_file, delimiter='|', quotechar='"')
        if 'chunk_ID' in existing.columns:
            processed_ids = set(existing['chunk_ID'].tolist())
    except Exception as e:
        print(f"⚠️ Greška pri čitanju postojećih tripleta: {e}")

file_exists = os.path.isfile(triplets_file)
bad_file_exists = os.path.isfile(bad_triplets_file)

with open(triplets_file, "a", encoding="utf-8", newline="") as csvfile, \
     open(bad_triplets_file, "a", encoding="utf-8", newline="") as badfile:

    writer = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)
    bad_writer = csv.writer(badfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)

    if not file_exists:
        writer.writerow(["chunk_ID", "question_ID", "triplet"])
    if not bad_file_exists:
        bad_writer.writerow(["chunk_ID", "question_ID", "bad_triplet"])

    for idx, row in df.iterrows():
        paragraph_id = row['chunk_ID']
        question_id = row['question_ID'] if 'question_ID' in row else None

        if paragraph_id < start_context_id:
            continue

        if paragraph_id in processed_ids:
            print(f"⏭️ Skipping already processed chunk {paragraph_id}")
            continue

        text = row['chunk']
        print(f"Generating triplets for chunk {paragraph_id}...")

        # 1) Generiši triplete iz ORIGINALNOG teksta
        triplets = generate_text(text)

        # 2) Ako tripleti u S/O sadrže zamjenice → uradi coref rezoluciju fastcoref + regeneriši triplete
        if triplets and triplets_have_pronoun_in_SO(triplets):
            prev_chunks = get_prev_chunks_same_question(df, idx, question_id, k=2)

            if prev_chunks:
                print(f"↪️ Pronoun detected in chunk {paragraph_id}. Resolving with SAME-question context ({len(prev_chunks)} prev chunks) via fastcoref...")
            else:
                print(f"↪️ Pronoun detected in chunk {paragraph_id}, no prior same-question chunks. Resolving with fastcoref on current chunk...")

            rewritten_text = rewrite_chunk_with_context_fastcoref(text, prev_chunks)

            if rewritten_text:
                triplets = generate_text(rewritten_text)
                print(f"✅ Re-generated triplets for chunk {paragraph_id} after fastcoref resolution.")
            else:
                print(f"⚠️ fastcoref resolution returned empty for chunk {paragraph_id}. Using original triplets.")

        # 3) Upis rezultata
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
