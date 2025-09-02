# -*- coding: utf-8 -*-
"""
Coreference resolution na CPU za kratke i duge tekstove (fastcoref + spaCy).

Priprema:
    pip install -U spacy fastcoref
    python -m spacy download en_core_web_sm

Pokretanje:
    # 1) Pokreni 10 ugrađenih primjera (bez argumenata):
    python coref_resolve_cpu.py

    # 2) Obradi vlastiti .txt fajl:
    python coref_resolve_cpu.py --in input.txt --out output.txt
"""

import argparse
import sys
from pathlib import Path
import spacy

# >>> KLJUČNI IMPORT: registruje spaCy factory pod imenom "fastcoref"
from fastcoref import spacy_component  # noqa: F401  (samo zbog side-effect registracije)

# --------- NLP pipeline ---------

def build_nlp():
    """Gradi spaCy pipeline s fastcoref komponentom (CPU default)."""
    try:
        nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)

    if "fastcoref" not in nlp.pipe_names:
        # nakon importa iznad, factory "fastcoref" postoji
        nlp.add_pipe("fastcoref")

    return nlp


# --------- Rješavanje za kraće tekstove ---------

def resolve_text(text: str, nlp=None) -> str:
    """
    Vrati razriješen tekst (zamjene zamjenica imenicama) koristeći novi API:
    spaCy pipe + doc._.resolved_text.
    """
    nlp = nlp or build_nlp()
    doc = nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
    return doc._.resolved_text


# --------- Chunking s preklapanjem za duge tekstove ---------

def chunk_and_resolve(text: str, max_chars: int = 4000, overlap_sents: int = 2) -> str:
    """
    Dijeli tekst u prozore (~max_chars) po rečenicama uz preklapanje od overlap_sents.
    Svaki prozor se rješava, a rezultati se spajaju.
    """
    ssplit = spacy.blank("en")
    ssplit.add_pipe("sentencizer")
    sents = [s.text.strip() for s in ssplit(text).sents if s.text.strip()]

    chunks, cur, total = [], [], 0
    for s in sents:
        s_len = len(s) + 1
        if cur and total + s_len > max_chars:
            chunks.append(cur[:])
            cur = cur[-overlap_sents:] if overlap_sents > 0 else []
            total = sum(len(x) + 1 for x in cur)
        cur.append(s)
        total += s_len
    if cur:
        chunks.append(cur)

    nlp = build_nlp()
    docs = nlp.pipe([" ".join(c) for c in chunks],
                    component_cfg={"fastcoref": {"resolve_text": True}})

    resolved_parts = []
    for i, doc in enumerate(docs):
        rsents = [s.text for s in ssplit(doc._.resolved_text).sents]
        if i > 0 and overlap_sents > 0:
            rsents = rsents[overlap_sents:]
        resolved_parts.append(" ".join(rsents))

    return " ".join(resolved_parts).strip()


# --------- Primjeri za test ---------

def get_examples():
    """10 engleskih primjera (2–8 rečenica) idealnih za coref testiranje."""
    return [
        "Mary met John at the office. She thanked him for the report.",
        "The server crashed at midnight. It caused several services to fail.",
        "Liam bought a camera. He loved it immediately. The store promised he could return it within 30 days.",
        "The committee reviewed the proposal. They found it compelling. Then they sent it back with minor edits.",
        "Acme Robotics acquired Nova Labs. The company said it would keep its brand. Investors expected it to grow. They applauded the decision.",
        "Sara placed the vase on the table. It wobbled because the surface was uneven. She moved it to the shelf. That solved the problem.",
        "Michael emailed Karen about the contract. He told her that the client wanted changes. She forwarded it to the legal team. They reviewed it overnight. In the morning, they approved it.",
        "The startup built a chatbot for the airline. It handled thousands of messages. Passengers said they liked it. The airline measured higher satisfaction, and it attributed the rise to the bot. Engineers monitored it during the launch. They fixed a memory leak when it appeared.",
        "The city council met to discuss the budget. The chair opened the session and welcomed a journalist. She asked them about the deficit. They explained that it had grown after the storm. The journalist recorded the answer; she published it later. When readers saw it, they shared it widely. That helped the council justify the new tax.",
        "Olivia adopted a puppy from the shelter. It was nervous at first, but it followed her everywhere. She bought a crate, and it slept in it the first night. The neighbors met the puppy, and they offered toys. When Olivia took it to the vet, the veterinarian said it looked healthy. She gave it a vaccine and scheduled another visit. Olivia posted photos online, and they got dozens of comments. That encouraged her to keep training it every day.",
    ]


def count_sents(text: str) -> int:
    """Pomoćna: prebroji rečenice radi lijepog zaglavlja po primjeru."""
    ssplit = spacy.blank("en")
    ssplit.add_pipe("sentencizer")
    return sum(1 for _ in ssplit(text).sents)


def print_examples_pretty(examples):
    """Lijepi, čitki ispis: Original vs Resolved za svaki primjer."""
    sys.stdout.reconfigure(encoding="utf-8")
    nlp = build_nlp()

    bar = "─" * 80
    for i, ex in enumerate(examples, 1):
        resolved = resolve_text(ex, nlp=nlp) if len(ex) <= 4000 else chunk_and_resolve(ex)
        n_sents = count_sents(ex)
        print(f"\n{bar}")
        print(f"Example {i}  •  {n_sents} sentence(s)")
        print(f"{bar}")
        print("Original:")
        print(ex)
        print("\nResolved:")
        print(resolved)
        print(bar)


# --------- CLI ---------

def main():
    ap = argparse.ArgumentParser(description="Coreference resolution na CPU (fastcoref + spaCy).")
    ap.add_argument("--in", dest="in_path", type=str, default=None,
                    help="Ulazni .txt (ako se ne navede, pokreću se ugrađeni primjeri).")
    ap.add_argument("--out", dest="out_path", type=str, default=None,
                    help="Putanja za izlazni .txt (ako se ne navede, ispisuje na stdout).")
    ap.add_argument("--max-chars", type=int, default=4000,
                    help="Maksimalna veličina prozora u znakovima (default 4000).")
    ap.add_argument("--overlap-sents", type=int, default=2,
                    help="Broj rečenica preklapanja (default 2).")
    args = ap.parse_args()

    # 1) Ako je --in proslijeđen -> standardni režim (jedan ulaz)
    if args.in_path:
        text = Path(args.in_path).read_text(encoding="utf-8")
        resolved = (
            resolve_text(text)
            if len(text) <= args.max_chars
            else chunk_and_resolve(text, max_chars=args.max_chars, overlap_sents=args.overlap_sents)
        )
        if args.out_path:
            Path(args.out_path).write_text(resolved, encoding="utf-8")
            print(f"[OK] Sačuvano: {args.out_path}")
        else:
            sys.stdout.reconfigure(encoding="utf-8")
            print(resolved)
        return

    # 2) Bez --in -> pokreni 10 ugrađenih primjera i ispiši lijepo
    examples = get_examples()
    print_examples_pretty(examples)


if __name__ == "__main__":
    main()
