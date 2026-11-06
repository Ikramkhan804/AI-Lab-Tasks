import csv
import importlib
import json
import re
import time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
model = None
RULES = {
    "late": "Late submission is normally penalized. Illness extensions require medical evidence before the deadline.",
    "password": "Reset a forgotten password using the recovery link. Never share recovery codes.",
    "assignment": "Upload assignments through the learning portal. Submit the final file as a PDF.",
    "contractor": "Contractors can work remotely only if the signed contract permits it. Employee rules do not apply to contractors.",
    "printer": "For blank printer pages, check ink levels and run the cleaning utility. Print a test page afterwards.",
    "employee": "Employees may work remotely with manager approval. Contractor permission depends on the signed contract.",
}
FILLER = (
    "This fictional training document applies only to its named policy category. "
    "Check the relevant source before following a procedure. "
)
LONG = " ".join(["Administrative"] + ["review"] * 54) + "."
DOCS = {
    k: " ".join(["context"] * 35) + " " + v + " " + LONG + " " + FILLER * 6
    for k, v in RULES.items()
}
assert all(150 <= len(t.split()) <= 250 for t in DOCS.values())
QUESTIONS = [
    ("What is required for illness extensions?", "late", "dev"),
    ("How do I reset my password?", "password", "dev"),
    ("Where do I upload assignments?", "assignment", "dev"),
    ("Can contractors work remotely?", "contractor", "dev"),
    ("What should I do for blank printer pages?", "printer", "test"),
    ("How can I regain account access?", "password", "test"),
    ("Where should I hand in coursework?", "assignment", "test"),
    ("What proof is needed when I was ill?", "late", "test"),
    ("What are tomorrow's flight prices?", None, "test"),
    ("Who won the football match?", None, "test"),
]
QUERIES = [
    dict(query=q, doc=d, evidence=RULES.get(d), split=s)
    for q, d, s in QUESTIONS
]
def save(name, data):
    Path(name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
def windows(text, size=40, overlap=0, offset=0):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    spans, out = list(re.finditer(r"\S+", text)), []
    for i in range(0, len(spans), size - overlap):
        j = min(i + size, len(spans))
        out.append((offset + spans[i].start(), offset + spans[j - 1].end()))
        if j == len(spans):
            break
    return out
def sentences(text, size=40):
    if size <= 0:
        raise ValueError("Invalid budget")
    out, pending, count = [], None, 0
    for m in re.finditer(r".+?(?:[.!?](?:\s+|$)|$)", text, re.S):
        a, b = m.span()
        n = len(m.group().split())
        if not n:
            continue
        if n > size:
            if pending:
                out.append(pending)
            out.extend(windows(m.group(), size, 0, a))
            pending, count = None, 0
        elif pending and count + n <= size:
            pending, count = (pending[0], b), count + n
        else:
            if pending:
                out.append(pending)
            pending, count = (a, b), n
    return out + ([pending] if pending else [])
def chunks(strategy):
    out = []
    for doc, text in DOCS.items():
        spans = sentences(text) if strategy == "sentence" else windows(text, 40, int(strategy))
        for i, (a, b) in enumerate(spans):
            out.append(dict(
                id=f"{doc}:{strategy}:{i}", doc=doc,
                start=a, end=b, text=text[a:b]
            ))
    return out
def audit(texts):
    if model is None:
        return 0
    lengths = [len(model.tokenizer(t, truncation=False)["input_ids"]) for t in texts]
    if max(lengths, default=0) > model.max_seq_length:
        raise ValueError("Model token limit exceeded")
    return max(lengths, default=0)
def search(texts, queries, backend):
    t = time.perf_counter()
    if backend == "tfidf":
        vectorizer = TfidfVectorizer()
        C = vectorizer.fit_transform(texts)
        Q = vectorizer.transform(queries)
    else:
        if model is None:
            raise ValueError("Embedding backend unavailable; install sentence-transformers")
        audit(texts + queries)
        C = model.encode(texts, normalize_embeddings=True)
        Q = model.encode(queries, normalize_embeddings=True)
    encoding = time.perf_counter() - t
    t = time.perf_counter()
    scores = Q @ C.T
    if backend == "tfidf":
        scores = scores.toarray()
    return scores, encoding, time.perf_counter() - t
def evaluate(records, items, backend, strategy):
    scores, encoding, searching = search(
        [r["text"] for r in records], [q["query"] for q in items], backend
    )
    hits, rr = [], []
    for q, row in zip(items, scores):
        good = []
        for rank, i in enumerate(np.argsort(-row, kind="stable")[:3], 1):
            r = records[i]
            relevant = q["doc"] == r["doc"] and bool(q["evidence"]) and q["evidence"] in r["text"]
            if relevant:
                good.append(rank)
            entry = dict(
                strategy=strategy, backend=backend, query=q["query"],
                split=q["split"], rank=rank, score=float(row[i]),
                relevant=bool(relevant), **r
            )
            logs.append(entry)
            print(entry)
        if q["doc"]:
            hits.append(bool(good))
            rr.append(1 / good[0] if good else 0)
    return dict(
        strategy=strategy, backend=backend, split=items[0]["split"],
        hit3=float(np.mean(hits)), mrr3=float(np.mean(rr)),
        chunks=len(records), encoding_seconds=encoding,
        search_seconds=searching
    )
if __name__ == "__main__":
    save("corpus.json", DOCS)
    save("queries.json", QUERIES)
    t = time.perf_counter()
    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
    except ModuleNotFoundError:
        sentence_transformers = None
    if sentence_transformers is not None:
        model = sentence_transformers.SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        print("Model load seconds:", time.perf_counter() - t)
    else:
        print("sentence-transformers unavailable; using TF-IDF only.")
    # Task 1
    samples = ["AI, search!", "I can't log in.", "help@example.com", "CS4944", "", "یہ ایک مثال ہے"]
    for text in samples:
        ws, rx = text.split(), re.findall(r"\b\w+\b", text, re.UNICODE)
        print(repr(text), "whitespace:", ws, len(ws), "regex:", rx, len(rx))
    if model is not None:
        for text in samples[:2]:
            tokens = model.tokenizer.tokenize(text)
            print("Model:", tokens, "count:", len(tokens))
    else:
        print("Model tokenization unavailable without sentence-transformers.")
    print(
        "Whitespace retains punctuation; regex splits contractions and emails. "
        "Normalization can erase meaningful distinctions. "
        "This regex is not a complete multilingual tokenizer."
    )
    # Two added FAQs, with synthetic sources.
    faqs = {
        "How do I reset my password?": ("Use the recovery link.", "it_help.txt"),
        "Where do I upload assignments?": ("Use the learning portal.", "assessment.txt"),
        "Can contractors work remotely?": ("Only if their contract permits it.", "contractor.txt"),
        "What should I do for blank printer pages?": ("Check ink and run cleaning.", "printer.txt"),
    }
    questions = list(faqs)
    for q in [questions[0], "password recovery", "", "volcano geology"]:
        if not q.strip():
            answer = "Please enter a question."
        elif q in faqs:
            answer = str(faqs[q])
        else:
            scores, _, _ = search(questions, [q], "tfidf")
            choices = [questions[i] for i in np.argsort(-scores[0])[:3] if scores[0, i] > 0]
            answer = "Possible matches; confirm first: " + str(choices) if choices else "No matching FAQ."
        print("FAQ:", repr(q), answer)
    print("Nonexact questions require confirmation because similarity does not prove identical intent.")
    # Task 2 tests and comparisons
    assert windows("") == sentences("") == []
    assert windows("short text") == [(0, 10)]
    assert sentences("short text") == [(0, 10)]
    try:
        windows("a b", 2, 2)
        raise AssertionError("Invalid overlap accepted")
    except ValueError:
        pass
    assert len(sentences(" ".join(["word"] * 95) + ".")) == 3
    all_chunks = {s: chunks(s) for s in ["0", "10", "sentence"]}
    for strategy, records in all_chunks.items():
        sizes = [len(r["text"].split()) for r in records]
        assert max(sizes) <= 40
        assert all(r["text"] == DOCS[r["doc"]][r["start"]:r["end"]] for r in records)
        print(
            "CHUNKS:", strategy, len(records), "sizes:", sizes,
            "repeated words:", sum(sizes) - sum(len(t.split()) for t in DOCS.values()),
            "token maximum:", audit([r["text"] for r in records])
        )
        for doc, evidence in RULES.items():
            print("Evidence coverage:", doc, any(r["doc"] == doc and evidence in r["text"] for r in records))
    print("Sentence splitting can fail on Dr., e.g., language boundaries and long sentences.")
    # Task 3: choose on development data before viewing test results.
    logs, metrics = [], []
    dev = [q for q in QUERIES if q["split"] == "dev"]
    test = [q for q in QUERIES if q["split"] == "test"]
    backends = ["tfidf"] + (["embedding"] if model is not None else [])
    for strategy, records in all_chunks.items():
        for backend in backends:
            metrics.append(evaluate(records, dev, backend, strategy))
    selected = max(metrics, key=lambda r: (r["mrr3"], r["hit3"], -r["chunks"]))
    save("selected_config.json", selected)
    for strategy, records in all_chunks.items():
        for backend in backends:
            metrics.append(evaluate(records, test, backend, strategy))
    save("metrics.json", metrics)
    with open("retrieval_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    print("SELECTED ON DEV:", selected)
    print("ALL METRICS:", metrics)
    # Show actual problematic top-ranked evidence, without inventing failures.
    cases = [r for r in logs if r["rank"] == 1 and not r["relevant"]]
    print("FAILURES / EDGE CASES:", cases[:3])
    print(
        "Nearest-neighbor ranking still returns text for unsupported queries. "
        "No universal cosine cutoff is used. Missing exceptions and conflicting "
        "employee/contractor scope require evidence inspection."
    )
    # Interactive evidence-only chatbot; unsupported queries get no asserted answer.
    records = all_chunks[selected["strategy"]]
    while True:
        try:
            q = input("\nQuestion (exit to stop): ").strip()
            if q.lower() == "exit":
                break
            if not q:
                print("Please enter a question.")
                continue
            scores, _, _ = search([r["text"] for r in records], [q], selected["backend"])
            order = np.argsort(-scores[0], kind="stable")[:3]
            for n, i in enumerate(order, 1):
                r = records[i]
                print(n, r["id"], float(scores[0, i]), r["start"], r["end"])
            choice = input("Candidate evidence only. Choose 1-3; Enter if none apply: ")
            if choice not in {"1", "2", "3"}:
                print("No answer selected. Rephrase or contact a human.")
                continue
            r = records[order[int(choice) - 1]]
            print("Unverified evidence, not an asserted answer:", r["text"])
            print("Source:", r["doc"], "characters:", r["start"], r["end"])
        except (EOFError, KeyboardInterrupt):
            break
        except ValueError as exc:
            print(exc)