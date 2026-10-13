"""节点 07 的确定性部分：稳定编号、六段渲染与提示词结构校验。

模型只写描述字段；段名、标签、关系标记、说话人编号、对白标签及时间由程序插入。
"""
from __future__ import annotations

import re

from . import schemas
from .media import FPS
from .util import mmss

SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
            "overall_soundscape", "non_diegetic_music")
TASK_TYPES = {"keyframe completion", "reference generation", "video editing", "video continuation",
              "audio reuse", "audio reference"}
LABEL_RE = re.compile(r"<(Subject|Picture|Video|Audio) (\d+)>")
TOKEN_RE = re.compile(r"\{\{([SD]):([a-z0-9.]+)\}\}")
FORBIDDEN_IN_MODEL_TEXT = re.compile(r"</?d>|\(S\d|\[Shot|\d\d:\d\d\.\d{3}|<scenetrans>|<cutoff>")


def allocate_labels(references: list[dict], facts: dict) -> dict:
    """各模态连续编号；Subject 按目标首次出现顺序，说话人按首次实际发声顺序，二者互不借用。"""
    counters = {"image": 0, "video": 0, "audio": 0}
    media, records = {}, []
    for ref in references:
        counters[ref["type"]] += 1
        kind = {"image": "Picture", "video": "Video", "audio": "Audio"}[ref["type"]]
        label = f"<{kind} {counters[ref['type']]}>"
        record = {"type": ref["type"], "label": label, "path": ref["path"]}
        media[label] = {"name": ref["name"], "path": ref["path"], "reference_id": ref["id"]}
        if ref["type"] == "video":
            record["include_audio"] = bool(ref.get("include_audio"))
            if record["include_audio"]:
                counters["audio"] += 1
                record["audio_label"] = f"<Audio {counters['audio']}>"
                media[record["audio_label"]] = {"name": ref["name"], "path": ref["path"],
                                                "reference_id": ref["id"], "synchronized_track_of": label}
        records.append(record)
    by_name = {v["name"]: k for k, v in media.items() if "synchronized_track_of" not in v}

    relations = [r for r in facts["relations"] if r["holds"] == "yes"]
    first_frame = {s["key"]: s["first_frame"] for s in facts["subjects"]}
    keys = sorted({r["target_subject_key"] for r in relations if r["role"] in schemas.SUBJECT_ROLES},
                  key=lambda k: (first_frame.get(k, 10 ** 9), k))
    subjects, definitions = {}, []
    for n, key in enumerate(keys, start=1):
        label = f"<Subject {n}>"
        own = [r for r in relations if r["target_subject_key"] == key and r["role"] in schemas.SUBJECT_ROLES]
        subjects[label] = {"key": key, "relation_ids": [r["relation_id"] for r in own],
                           "sources": sorted({by_name[r["reference_name"]] for r in own}),
                           "marker": _combine_markers([r["relation_marker"] for r in own]),
                           "shots": [s["shot_id"] for s in facts["shots"] if key in s["subject_keys"]]}
        definitions.append(label)
    structures = {}
    for label, info in media.items():
        own = [r for r in relations if r["reference_name"] == info["name"] and r["role"] in schemas.STRUCTURE_ROLES]
        if label.startswith("<Video") and own:
            structures[label] = {"relation_ids": [r["relation_id"] for r in own],
                                 "roles": sorted({r["role"] for r in own}),
                                 "marker": _combine_markers([r["relation_marker"] for r in own])}
            definitions.append(label)
    audios = {}
    for label, info in media.items():
        if label.startswith("<Audio"):
            check = next(c for c in facts["audio"]["reference_audio"] if c["reference_name"] == info["name"])
            audios[label] = {"role": check["role"], "marker": check["relation_marker"], "note": check["note"]}
            definitions.append(label)

    speakers: dict[str, str] = {}
    for utterance in sorted(facts["utterances"], key=lambda u: (u["start"], u["utterance_id"])):
        speakers.setdefault(utterance["speaker_key"], f"S{len(speakers) + 1}")
    subject_of = {v["key"]: k for k, v in subjects.items()}
    return {"reference_records": records, "media": media, "subjects": subjects, "video_structures": structures,
            "audios": audios, "definitions_required": definitions, "speakers": speakers,
            "speaker_subject_labels": {key: subject_of.get(key) for key in speakers},
            "allowed_labels": sorted(media) + sorted(subjects)}


def _combine_markers(markers: list[str]) -> str:
    """同一标签下多条关系取最保守的标记。"""
    order = ["weak_reference", "attribute_transfer", "partially_preserved", "fully_preserved"]
    return min(markers, key=order.index)


def task_prefix(labels: dict) -> str:
    """默认 reference generation；按真实声音关系添加 audio reference / audio reuse。"""
    types = ["reference generation"]
    markers = {a["marker"] for a in labels["audios"].values()}
    if markers & {"reference", "weak_reference"}:
        types.append("audio reference")
    if markers & {"fully_copy", "partially_copy"}:
        types.append("audio reuse")
    return "[" + " + ".join(types) + "]"


def dialogue_parts(facts: dict) -> dict[str, dict]:
    return {p["part_id"]: {**p, "utterance": u} for u in facts["utterances"] for p in u["parts"]}


def validate_draft(draft: dict, facts: dict, labels: dict) -> list[str]:
    """结构与引用校验（validate_claim_links）。引用了事实编号只表示可追溯，语义仍由审查节点核对。"""
    errors = []
    required = labels["definitions_required"]
    for field in ("definitions", "retention"):
        got = [d["label"] for d in draft[field]]
        if got != required:
            errors.append(f"{field} labels must be exactly {required} in this order, got {got}")
    for d in draft["definitions"]:
        if d["text"].lstrip().startswith("<"):
            errors.append(f"definition text for {d['label']} must not repeat the label; start with 'is' or a verb")
        for source in labels["subjects"].get(d["label"], {}).get("sources", []):
            if source not in d["text"]:
                errors.append(f"definition of {d['label']} must cite its source {source}")
    shot_ids = [s["shot_id"] for s in facts["shots"]]
    if [s["shot_id"] for s in draft["shots"]] != shot_ids:
        errors.append(f"shots must be exactly {shot_ids} in this order")
    parts = dialogue_parts(facts)
    fact_ids = set(shot_ids) | set(parts) | {u["utterance_id"] for u in facts["utterances"]} | {
        r["relation_id"] for r in facts["relations"]} | {"style", "audio", "target"}
    allowed = set(labels["allowed_labels"])

    def scan(where: str, value: str, tokens_allowed: bool) -> list[tuple[str, str]]:
        if FORBIDDEN_IN_MODEL_TEXT.search(value):
            errors.append(f"{where}: do not write <d>, (S1), [Shot N], timestamps, <scenetrans> or <cutoff>; "
                          "use the {{S:..}} / {{D:..}} tokens, the program inserts the rest")
        bad = {m.group(0) for m in LABEL_RE.finditer(value)} - allowed
        if bad:
            errors.append(f"{where}: unknown labels {sorted(bad)}; allowed {sorted(allowed)}")
        found = TOKEN_RE.findall(value)
        if found and not tokens_allowed:
            errors.append(f"{where}: dialogue tokens are only allowed inside shots")
        if "{{" in TOKEN_RE.sub("", value):
            errors.append(f"{where}: malformed token")
        return found

    for d in draft["definitions"] + draft["retention"]:
        scan(f"{d['label']}", d["text"] + " " + d.get("scope", ""), False)
    for name in ("summary", "overall_soundscape", "non_diegetic_music"):
        scan(name, draft[name]["text"], False)
        if not draft[name]["text"].strip():
            errors.append(f"{name} is empty")
    scan("style", draft["style"], False)
    for shot in draft["shots"]:
        found = scan(shot["shot_id"], shot["text"], True)
        want = sorted(pid for pid, p in parts.items() if p["shot_id"] == shot["shot_id"])
        got = sorted(i for kind, i in found if kind == "D")
        if got != want:
            errors.append(f"{shot['shot_id']}: must contain each of these dialogue tokens exactly once: "
                          f"{['{{D:%s}}' % w for w in want]}; found {got}")
        for pid in want:
            speaker_token = "{{S:%s}}" % parts[pid]["utterance"]["utterance_id"]
            if speaker_token not in shot["text"]:
                errors.append(f"{shot['shot_id']}: name the speaker with {speaker_token} before {{{{D:{pid}}}}}")
        for kind, i in found:
            if kind == "S" and i not in {u["utterance_id"] for u in facts["utterances"]}:
                errors.append(f"{shot['shot_id']}: unknown speaker token {{{{S:{i}}}}}")
    for item in (draft["definitions"] + draft["retention"] + draft["shots"]
                 + [draft["summary"], draft["overall_soundscape"], draft["non_diegetic_music"]]):
        if set(item["fact_ids"]) - fact_ids:
            errors.append(f"unknown fact_ids {sorted(set(item['fact_ids']) - fact_ids)}")
    used = set(LABEL_RE.findall(" ".join(
        [d["text"] for d in draft["definitions"] + draft["retention"] + draft["shots"]]
        + [draft["summary"]["text"], draft["overall_soundscape"]["text"], draft["non_diegetic_music"]["text"]])))
    missing = set(labels["media"]) - {f"<{kind} {n}>" for kind, n in used} - {d["label"] for d in draft["definitions"]}
    if missing:
        errors.append(f"every reference label must be used at least once; missing {sorted(missing)}")
    return errors


def _render_dialogue(part: dict) -> str:
    body = part["text"]
    if part["lead_scenetrans"]:
        body = "<scenetrans> " + body
    if part["trail_scenetrans"]:
        body += " <scenetrans>"
    if part["cutoff"]:
        body += " <cutoff>"
    return f"<d>[{part['utterance']['language']}] {body}</d>"


def render_six_sections(draft: dict, facts: dict, labels: dict) -> str:
    parts = dialogue_parts(facts)
    speakers = labels["speakers"]
    speaker_of = {u["utterance_id"]: speakers[u["speaker_key"]] for u in facts["utterances"]}
    shots = {s["shot_id"]: s for s in facts["shots"]}

    def fill(value: str) -> str:
        def replace(match):
            kind, ident = match.groups()
            return f"({speaker_of[ident]})" if kind == "S" else _render_dialogue(parts[ident])
        return " ".join(TOKEN_RE.sub(replace, value).split())

    definitions = [f"{d['label']} {fill(d['text'])}" for d in draft["definitions"]]
    retention = []
    for r in draft["retention"]:
        label = r["label"]
        if label in labels["subjects"]:
            where = ", ".join(f"[Shot {shots[s]['index']}]" for s in labels["subjects"][label]["shots"])
            retention.append(f"{label} (appears in {where}): {labels['subjects'][label]['marker']} - {fill(r['text'])}")
        elif label in labels["video_structures"]:
            retention.append(f"{label} ({fill(r['scope'])}): {labels['video_structures'][label]['marker']} - "
                             f"{fill(r['text'])}")
        else:
            retention.append(f"{label}: {labels['audios'][label]['marker']} - {fill(r['text'])}")
    lines = [fill(draft["style"])]
    assert facts["shots"][0]["start_frame"] == 0
    for shot in draft["shots"]:
        fact = shots[shot["shot_id"]]
        body = fill(shot["text"])
        if fact["index"] == 1:
            lines.append(f"[Shot 1] {body}")
        else:
            if body[:1].isupper() and not body.startswith(("I ", "I'")):
                body = body[0].lower() + body[1:]
            lines.append(f"[Shot {fact['index']}] At {mmss(fact['start_frame'] / FPS)}, {body}")
    sections = {
        "subject_definitions": "\n".join(definitions),
        "summary": f"{task_prefix(labels)} {fill(draft['summary']['text'])}",
        "retention_analysis": "\n".join(retention),
        "detailed_description": "\n".join(lines),
        "overall_soundscape": fill(draft["overall_soundscape"]["text"]),
        "non_diegetic_music": fill(draft["non_diegetic_music"]["text"]),
    }
    return "\n\n".join(f"{name}:\n{sections[name]}" for name in SECTIONS)


def split_sections(prompt: str) -> dict[str, str] | None:
    pattern = re.compile(r"^(" + "|".join(SECTIONS) + r"):\s*$", re.M)
    hits = list(pattern.finditer(prompt))
    if [h[1] for h in hits] != list(SECTIONS) or hits[0].start() != 0:
        return None
    return {h[1]: prompt[h.end():(hits[k + 1].start() if k + 1 < len(hits) else len(prompt))].strip()
            for k, h in enumerate(hits)}


def check_prompt(prompt: str, record: dict, confirmed_cut_frames: list[int], tolerance_frames: int) -> list[tuple]:
    """对最终字符串做结构校验，返回 (名称, 是否通过, 详情)。不依赖渲染时的中间对象。"""
    out = []
    total = record["target"]["num_frames"] / FPS
    refs = record["references"]
    secs = split_sections(prompt)
    out.append(("six_sections_in_order", secs is not None, ""))
    if secs is None:
        return out
    labels = set(LABEL_RE.findall(prompt))
    by_kind = {k: sorted(int(n) for t, n in labels if t == k) for k in ("Subject", "Picture", "Video", "Audio")}
    out.append(("labels_numbered_consecutively", all(v == list(range(1, len(v) + 1)) for v in by_kind.values()),
                by_kind))
    ref_labels = {r["label"] for r in refs} | {r["audio_label"] for r in refs if r.get("audio_label")}
    media_labels = {f"<{t} {n}>" for t, n in labels if t != "Subject"}
    out.append(("media_labels_map_to_references", media_labels == ref_labels,
                f"prompt={sorted(media_labels)} references={sorted(ref_labels)}"))
    definitions = secs["subject_definitions"]
    defined = set(re.findall(r"^(<(?:Subject|Video|Audio|Picture) \d+>)", definitions, re.M))
    subjects = {f"<Subject {n}>" for t, n in labels if t == "Subject"}
    audios = {f"<Audio {n}>" for t, n in labels if t == "Audio"}
    out.append(("subjects_and_audio_defined", subjects | audios <= defined,
                f"used={sorted(subjects | audios)} defined={sorted(defined)}"))
    prefix = re.match(r"\[([a-z +]+)\] \S", secs["summary"])
    types = [x.strip() for x in prefix[1].split("+")] if prefix else []
    out.append(("summary_task_types_valid", bool(types) and set(types) <= TASK_TYPES and len(types) == len(set(types))
                and (bool({"audio reference", "audio reuse"} & set(types)) == bool(audios)), types))
    ok, detail = True, []
    for line in secs["retention_analysis"].splitlines():
        match = re.match(r"(<(Subject|Picture|Video|Audio) \d+>)(?: \([^)]*\))?: ([a-z_]+) - \S", line)
        if not match:
            ok = False
            detail.append(f"unparsed: {line[:60]}")
            continue
        marks = schemas.AUDIO_RELATIONS if match[2] == "Audio" else schemas.VISUAL_RELATIONS
        ok &= match[3] in marks
        detail.append(f"{match[1]}={match[3]}")
    retained = set(re.findall(r"^(<(?:Subject|Video|Audio|Picture) \d+>)", secs["retention_analysis"], re.M))
    out.append(("retention_markers_valid_and_complete", ok and retained == defined, detail))
    out.append(("no_speaker_ids_in_retention", not re.search(r"\(S\d", secs["retention_analysis"]), ""))
    body = secs["detailed_description"]
    shots = re.findall(r"\[Shot (\d+)\](?: At (\d\d):(\d\d\.\d{3}),)?", body)
    numbers = [int(s[0]) for s in shots]
    times = [int(s[1]) * 60 + float(s[2]) for s in shots[1:] if s[1]]
    out.append(("shot_format", bool(shots) and numbers == list(range(1, len(numbers) + 1)) and not shots[0][1]
                and len(times) == len(shots) - 1 and all(a < b for a, b in zip(times, times[1:]))
                and all(0 < t < total for t in times), f"shots={numbers} times={times}"))
    out.append(("style_sentence_before_shot_1", not body.lstrip().startswith("[Shot"), ""))
    expected = [f / FPS for f in confirmed_cut_frames]
    tolerance = tolerance_frames / FPS + 0.0006            # 时间戳按毫秒书写
    out.append(("shot_times_match_confirmed_cut_frames", len(expected) == len(times) and all(
        abs(a - b) <= tolerance for a, b in zip(expected, times)), f"prompt={times} confirmed_frames={confirmed_cut_frames}"))
    blocks = re.findall(r"<d>(.*?)</d>", prompt, re.S)
    out.append(("dialogue_tags_wellformed", prompt.count("<d>") == prompt.count("</d>") == len(blocks) and all(
        re.match(r"\[[A-Z][a-zA-Z ]+\] \S", b) for b in blocks), blocks))
    # 听不清的对白整句省略；本实现不渲染“直接复用参考音频词句”的情形，所以占位符不应出现
    out.append(("no_unintelligible_placeholder", "[unclear]" not in prompt, ""))
    out.append(("dialogue_only_in_detailed_description", "<d>" not in "".join(
        secs[s] for s in SECTIONS if s != "detailed_description"), ""))
    order = list(dict.fromkeys(int(x) for group in re.findall(r"\((S\d+(?:,S\d+)*)\)", body)
                               for x in re.findall(r"\d+", group)))
    out.append(("speaker_ids_in_first_use_order", order == list(range(1, len(order) + 1)), order))
    out.append(("no_unresolved_tokens", "{{" not in prompt and "}}" not in prompt, ""))
    words = len(re.findall(r"[A-Za-z']+", body))
    out.append(("detailed_description_not_trivial", words >= 60, f"{words} words (guide: usually 350-500; "
                "accuracy comes first, so the range itself is not enforced)"))
    return out
