"""离线测试用的固定“模型”：只根据请求里出现的编号返回符合模式的内容，不理解画面。"""
import json
import re


def _images(request):
    return [b["id"] for b in request["content"] if b["type"] == "image"]


def _text(request):
    return "\n".join(b["text"] for b in request["content"] if b["type"] == "text")


def _json_after(request, marker):
    for block in request["content"]:
        if block["type"] == "text" and block["text"].startswith(marker):
            return json.loads(block["text"].split("\n", 1)[1])
    raise KeyError(marker)


def respond(node, request):
    body, ids = _text(request), _images(request)
    if node == "capability_probe":
        return {"answer": "red,blue"}
    if node == "source_visual_facts":
        frames = [i for i in ids if re.fullmatch(r"f\d{6}", i)]
        return {"cuts": [{"cut_id": c, "verdict": "confirmed"} for c in sorted(set(re.findall(r"- (c\d{6}):", body)))],
                "subjects": [{"key": "pattern", "kind": "other", "description": "a synthetic test pattern",
                              "clear_frame_ids": frames}],
                "events": [{"start_frame_id": frames[0], "end_frame_id": frames[-1], "description": "pattern plays",
                            "subject_keys": ["pattern"], "action_complete_at_end": "unknown"}],
                "visible_text": [], "quality_issues": [], "uncertainties": []}
    if node == "target_completeness":
        return {"completeness": "complete", "start_boundary": "clean", "end_boundary": "clean",
                "visual_quality": "high", "describability": "high", "information": "medium",
                "content_summary": "synthetic patterns", "evidence_frame_ids": ids[:1], "reason": "mock"}
    if node == "duplicate_review":
        return {"result": "clear", "reason": "mock", "evidence_ids": ids[:2]}
    if node == "reference_relations":
        target, refs, current = [], {}, None
        for block in request["content"]:
            if block["type"] == "text":
                head = re.match(r"reference ((?:img|vid)_[\d_]+) \(", block["text"])
                if head:
                    current = head[1]
                    refs[current] = []
                elif block["text"].startswith("target frame"):
                    current = None
            elif current is None:
                target.append(block["id"])
            else:
                refs[current].append(block["id"])
        relations = []
        for rid, frames in refs.items():
            video = rid.startswith("vid")
            relations.append({
                "reference_id": rid, "supported": "yes",
                "target_subject_or_event_id": "whole_target" if video else "pattern",
                "role": "camera_or_editing_structure" if video else "identity_and_appearance",
                "promised_attributes": ["pacing"] if video else ["pattern layout"],
                "preserved_attributes": ["pacing"] if video else ["pattern layout"], "changed_attributes": [],
                "target_evidence_ids": target[:2], "reference_evidence_ids": frames[:2],
                "relation_marker": "weak_reference" if video else "fully_preserved", "clarity": "high",
                "uncertainties": []})
        return {"target_subjects": [{"key": "pattern", "kind": "other", "description": "a synthetic test pattern",
                                     "importance": "main", "target_frame_ids": target[:2]}],
                "relations": relations}
    if node == "exported_visual_facts":
        frames = [i for i in ids if re.fullmatch(r"t\d{4}", i)]
        cuts = re.findall(r"t\d{4}", body.split("CANDIDATE CUT FRAMES", 1)[1].split("\n", 1)[0])
        shots = [{"start_frame_id": s, "framing": "full-frame graphic", "setting_and_lighting": "flat synthetic image",
                  "subjects_and_positions": "the pattern fills the frame", "action": "the pattern animates",
                  "camera_movement": "static", "visible_text": [], "subject_keys": ["pattern"],
                  "evidence_frame_ids": [s], "uncertainties": []} for s in ["t0000"] + cuts]
        checks = []
        for rel, name in re.findall(r"- (rel\d+): reference (\S+) ->", body):
            own = [i for i in ids if i == name or i.startswith(name + ":")]
            checks.append({"reference_id": rel, "holds": "yes", "note": "mock", "target_evidence_ids": frames[:1],
                           "reference_evidence_ids": own[:1]})
        return {"style": "A synthetic test-pattern video with flat graphics and hard cuts.", "shots": shots,
                "missed_cuts": [], "subjects": [{"key": "pattern", "description": "a synthetic test pattern",
                                                 "first_frame_id": "t0000"}],
                "relation_checks": checks,
                "speech_attribution": [{"utterance_id": u, "speaker": "off_screen", "speaker_description":
                                        "an off-screen voice", "on_screen": "no", "evidence_frame_ids": []}
                                       for u in re.findall(r"^- (u\d+) ", body, re.M)],
                "start_boundary": "clean", "end_boundary": "clean", "uncertainties": []}
    if node in ("prompt_writer", "prompt_repair"):
        facts = _json_after(request, "FACTS")
        table = _json_after(request, "LABEL TABLE")
        tokens = _json_after(request, "DIALOGUE TOKENS")
        required = table["labels_needing_definition_and_retention_lines_in_order"]
        every = " and ".join(sorted(table["media_labels"]))

        def define(label):
            sources = table["subject_labels"].get(label, {}).get("sources", [])
            return ("is the test pattern shown in " + " and ".join(sources) + ", with its layout and colours."
                    if sources else "supplies the pacing of hard cuts between full-frame graphics.")

        shots = []
        for shot in facts["shots"]:
            words = "the camera cuts to a full-frame synthetic graphic that keeps animating in place. " * 4
            lead = f"{required[0]} fills the frame. " if shot["index"] == 1 else ""
            talk = " ".join(f"An off-screen voice {t['speaker_token']} says in an off-screen voiceover, "
                            f"{t['dialogue_token']}" for t in tokens if t["shot_id"] == shot["shot_id"])
            shots.append({"shot_id": shot["shot_id"], "text": lead + words + talk, "fact_ids": [shot["shot_id"]]})
        return {"definitions": [{"label": x, "text": define(x), "fact_ids": []} for x in required],
                "summary": {"text": f"A {facts['target']['duration_seconds']:.3f}-second test clip using {every}.",
                            "fact_ids": ["target"]},
                "retention": [{"label": x, "scope": "cut pacing" if x.startswith("<Video") else "",
                               "text": "the defined attributes are kept.", "fact_ids": []} for x in required],
                "style": facts["style"], "shots": shots,
                "overall_soundscape": {"text": "Continuous noise-like sound is present.", "fact_ids": ["audio"]},
                "non_diegetic_music": {"text": "N/A", "fact_ids": ["audio"]}}
    if node == "semantic_auditor":
        return {"core_relation_valid": "yes", "boundaries_valid": "yes", "visual_consistent": "yes", "issues": [],
                "visual_uncertainties": [], "audio_uncertainties": ["audio was not heard"]}
    raise ValueError(node)
