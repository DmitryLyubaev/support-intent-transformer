import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from p24.config import StudyConfig
from p24.contracts import FrozenStudy, PreparedSplit, Row
from p24.io import canonical, digest_bytes, digest_file, read_json, write_json


class PreparationError(ValueError):
    pass


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def duplicate_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def privacy_flag(text: str) -> bool:
    return bool(re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)", text))


def prepare(rows: list[Row], config: StudyConfig, dispositions: dict[str, str]) -> FrozenStudy:
    exclusions = defaultdict(list)
    accepted = []
    seen = set()
    for row in sorted(rows, key=lambda r: r.row_id):
        if row.row_id in seen:
            raise PreparationError("Duplicate source row identity")
        seen.add(row.row_id)
        if row.label not in config.labels:
            exclusions["outside_labels"].append(row.row_id)
        elif not isinstance(row.text, str) or not normalize(row.text):
            exclusions["invalid_text"].append(row.row_id)
        elif dispositions.get(row.row_id) == "exclude_identifying" or (
            privacy_flag(row.text) and dispositions.get(row.row_id) != "clear_nonidentifying"
        ):
            exclusions["privacy"].append(row.row_id)
        else:
            accepted.append(row)
    parent = list(range(len(accepted)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    texts = [duplicate_text(r.text) for r in accepted]
    grams = [{t[i : i + 5] for i in range(len(t) - 4)} if len(t) >= 20 else set() for t in texts]
    for i, text in enumerate(texts):
        for j in range(i):
            equal = text == texts[j]
            similar = False
            if (
                grams[i]
                and grams[j]
                and min(len(grams[i]), len(grams[j])) / max(len(grams[i]), len(grams[j])) >= 0.85
            ):
                overlap = len(grams[i] & grams[j])
                similar = overlap / (len(grams[i]) + len(grams[j]) - overlap) >= 0.85
            if equal or similar:
                parent[find(i)] = find(j)
    components = defaultdict(list)
    for i, row in enumerate(accepted):
        components[find(i)].append(row)
    development = defaultdict(list)
    test_rows = []
    groups = {}
    for members in components.values():
        group_id = min(r.row_id for r in members)
        if len({r.label for r in members}) > 1:
            exclusions["conflicting_labels"].extend(r.row_id for r in members)
            continue
        tests = [r for r in members if r.upstream_split == "test"]
        if tests:
            chosen = min(tests, key=lambda r: int(r.row_id.split(":")[1]))
            test_rows.append(chosen)
            groups[chosen.row_id] = group_id
            for r in members:
                if r != chosen:
                    exclusions["test_precedence"].append(r.row_id)
        else:
            unique = {}
            for r in sorted(members, key=lambda r: int(r.row_id.split(":")[1])):
                key = duplicate_text(r.text)
                if key in unique:
                    exclusions["exact_duplicate"].append(r.row_id)
                else:
                    unique[key] = r
            kept = tuple(unique.values())
            development[kept[0].label].append((group_id, kept))
            groups.update({r.row_id: group_id for r in kept})
    train_rows, validation_rows = [], []
    for label in config.labels:
        ordered = sorted(
            development[label], key=lambda g: digest_bytes(f"42|{label}|{g[0]}".encode())
        )
        selected, count = [], 0
        for group_id, members in ordered:
            if count + len(members) <= config.cap:
                selected.append((group_id, members))
                count += len(members)
            else:
                exclusions["development_cap"].extend(r.row_id for r in members)
        validation_count = 0
        for _, members in selected:
            if validation_count < math.ceil(count * 0.2):
                validation_rows.extend(members)
                validation_count += len(members)
            else:
                train_rows.extend(members)
        actual = [
            sum(r.label == label for r in s) for s in (train_rows, validation_rows, test_rows)
        ]
        if any(
            n < minimum
            for n, minimum in zip(
                actual, (config.min_train, config.min_validation, config.min_test)
            )
        ):
            raise PreparationError(
                f"Class minimum failed for {label}: train/validation/test={actual}"
            )

    def split(members):
        members = tuple(sorted(members, key=lambda r: r.row_id))
        manifest = {
            "schema_version": 1,
            "row_ids": [r.row_id for r in members],
            "groups": {r.row_id: groups[r.row_id] for r in members},
            "counts": dict(Counter(r.label for r in members)),
            "rows_digest": digest_bytes(canonical([asdict(r) for r in members])),
        }
        return PreparedSplit(members, manifest)

    splits = [split(s) for s in (train_rows, validation_rows, test_rows)]
    audit = {
        "schema_version": 1,
        "input_count": len(rows),
        "config": asdict(config),
        "exclusions": {k: sorted(v) for k, v in sorted(exclusions.items())},
        "dispositions": dict(dispositions),
    }
    identity = digest_bytes(canonical({"splits": [s.manifest for s in splits], "audit": audit}))
    return FrozenStudy(*splits, identity, audit)


def write_freeze(study: FrozenStudy, root: Path) -> Path:
    destination = root / "freeze"
    if destination.exists():
        if read_json(destination / "manifest.json")["identity"] != study.identity:
            raise PreparationError("Existing freeze identity differs; retain original")
        return destination
    destination.mkdir(parents=True)
    files = {}
    for name in ("train", "validation", "test"):
        split = getattr(study, name)
        write_json(destination / f"{name}.json", [asdict(r) for r in split.rows])
        files[f"{name}.json"] = digest_file(destination / f"{name}.json")
    write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "identity": study.identity,
            "files": files,
            "splits": {n: getattr(study, n).manifest for n in ("train", "validation", "test")},
            "audit": study.audit,
        },
    )
    return destination


def read_split(freeze: Path, name: str) -> PreparedSplit:
    manifest = read_json(freeze / "manifest.json")
    payload = {
        "splits": [manifest["splits"][n] for n in ("train", "validation", "test")],
        "audit": manifest["audit"],
    }
    if digest_bytes(canonical(payload)) != manifest["identity"]:
        raise PreparationError("Frozen metadata identity mismatch")
    path = freeze / f"{name}.json"
    actual = digest_file(path)
    if actual != manifest["files"][path.name] or actual != manifest["splits"][name]["rows_digest"]:
        raise PreparationError("Frozen rows digest mismatch")
    rows = tuple(Row(**r) for r in read_json(path))
    return PreparedSplit(rows, manifest["splits"][name])


def read_training(freeze: Path) -> tuple[PreparedSplit, PreparedSplit]:
    return read_split(freeze, "train"), read_split(freeze, "validation")
