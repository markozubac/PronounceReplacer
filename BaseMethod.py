import cohere
import pandas as pd
import sys
import csv
import os
import time

# Cohere klijent
co = cohere.ClientV2("Your API key")

# Trial klijent
#co = cohere.ClientV2("sWQIbobLHntVOx0wtOB9IwV4S7MNrOupjvFFi5Gl")

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
    # Generisanje odgovora
    response = co.chat(
        model="command-a-03-2025",
        messages=[{'role':'user', 'content': prompt}]
    )
    # Prikupljanje teksta iz odgovora
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

# Ulazni fajl sa paragrafima
df = pd.read_csv("paragraph_chunks2.csv")

start_context_id = 1

# Izlazni fajl sa tripletima
triplets_file = "triplets_with_index_chunks.csv"

# Folder i fajl za loše formatirane triplete
bad_folder = "bad_form_triplets_chunks"
os.makedirs(bad_folder, exist_ok=True)
bad_triplets_file = os.path.join(bad_folder, "bad_triplets_chunks.csv")

# --- NOVO: Proveri postojeće triplete da ne dupliraš ---
processed_ids = set()
if os.path.isfile(triplets_file):
    existing = pd.read_csv(triplets_file, delimiter='|', quotechar='"')
    processed_ids = set(existing['chunk_ID'].tolist())

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
    for _, row in df.iterrows():
        paragraph_id = row['chunk_ID']
        question_id = row['question_ID'] if 'question_ID' in row else None

        if paragraph_id < start_context_id:
            continue

        text = row['chunk']
        print(f"Generating triplets for chunk {paragraph_id}...")
        triplets = generate_text(text)

        for line in triplets.splitlines():
            clean_line = line.replace('" | "', '"|"').replace('" |"', '"|"').replace('"| "', '"|"')
            parts = clean_line.strip().strip('"').split('"|"')
            if is_valid_triplet(parts):
                writer.writerow([paragraph_id, question_id, line.strip()])
            else:
                bad_writer.writerow([paragraph_id, question_id, line.strip()])
                print(f"⚠️ Skipped bad triplet at context {paragraph_id}: {line.strip()}")

print(f"\nSaved good triplets to {triplets_file}")
print(f"Saved bad triplets to {bad_triplets_file}")
