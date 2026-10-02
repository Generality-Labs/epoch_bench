"""Export strict JSON from the pinned, nested JSON5 source without changing text."""
import hashlib
import json
import zipfile
from pathlib import Path

import json5

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "bench/task/dtbench"
SOURCE_COMMIT = "284fda2da5ee5059b9b7b59ef419553e1bd5a067"


def build():
    archive = DEST / "source_data.zip"
    records = []
    with zipfile.ZipFile(archive) as z:
        z.setpassword(b"onebox")
        checked = [
            {x["qid"] for x in json5.loads(z.read(f"question_checks_{who}.json"))}
            for who in ["caspar_oesterheld", "emery_cooper"]
        ]

        def flatten(node, prefix, tags, filename):
            tags = tags + node.get("tags", [])
            if "setup" in node:
                for child in node["questions"]:
                    flatten(child, prefix + node["setup"], tags, filename)
            else:
                records.append({
                    "qid": node["qid"],
                    "question_text": prefix + node["question_text"],
                    "permissible_answers": node["permissible_answers"],
                    "correct_answer": node["correct_answer"],
                    "attitude_q": node.get("attitude_q", False),
                    "tags": tags,
                    "source_file": filename,
                    "double_checked": all(node["qid"] in c for c in checked),
                })

        for name in sorted(z.namelist()):
            if name.startswith("setting") and name.endswith(".json"):
                flatten(json5.loads(z.read(name)), "", [], name)
    assert len(records) == 537
    assert len({q["qid"] for q in records}) == len(records)
    assert sum(not q["attitude_q"] for q in records) == 407
    assert all(q["double_checked"] for q in records)
    payload = {"source_commit": SOURCE_COMMIT,
               "source_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "questions": records}
    (DEST / "questions.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(records)} questions: 407 capability, 130 attitude")


if __name__ == "__main__":
    build()
