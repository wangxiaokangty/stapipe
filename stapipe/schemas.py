"""各固定模型节点的输出模式。模型只能返回内容对象、枚举、候选编号和证据编号。"""
from __future__ import annotations

S = {"type": "string"}
STRINGS = {"type": "array", "items": S}
YES_NO = {"type": "string", "enum": ["yes", "no", "unknown"]}
LEVEL = {"type": "string", "enum": ["high", "medium", "low"]}
VISUAL_RELATIONS = ["fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"]
AUDIO_RELATIONS = ["fully_copy", "partially_copy", "reference", "weak_reference"]
SUBJECT_KINDS = ["person", "animal", "product", "prop", "environment", "text_or_logo", "other"]
# 身份/外观类角色定义 <Subject N>；结构类角色定义 <Video N>。
SUBJECT_ROLES = ["identity_and_appearance", "product_or_prop", "environment", "visual_style"]
STRUCTURE_ROLES = ["action_or_motion", "camera_or_editing_structure", "narrative_structure"]
AUDIO_ROLES = ["voice_timbre", "music_style", "sound_texture", "rhythm"]


def obj(properties: dict, optional=()) -> dict:
    return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": [k for k in properties if k not in optional]}


def arr(item: dict) -> dict:
    return {"type": "array", "items": item}


def enum(*values) -> dict:
    return {"type": "string", "enum": list(values)}


VISUAL_FACTS = obj({
    "cuts": arr(obj({"cut_id": S, "verdict": enum("confirmed", "rejected", "unknown")})),
    "subjects": arr(obj({"key": S, "kind": enum(*SUBJECT_KINDS), "description": S, "clear_frame_ids": STRINGS})),
    "events": arr(obj({"start_frame_id": S, "end_frame_id": S, "description": S, "subject_keys": STRINGS,
                       "action_complete_at_end": YES_NO})),
    "visible_text": arr(obj({"text": S, "frame_ids": STRINGS})),
    "quality_issues": arr(obj({"issue": enum("black", "severe_blur", "credits_or_endcard", "transition", "other"),
                               "frame_ids": STRINGS})),
    "uncertainties": STRINGS,
})

AUDIO_FACTS = obj({
    "utterances": arr(obj({"start": {"type": "number"}, "end": {"type": "number"},
                           "kind": enum("dialogue", "voiceover", "lyrics", "vocalization"), "language": S,
                           "text": S, "intelligible": enum("yes", "partly", "no")})),
    "sounds": arr(obj({"start": {"type": "number"}, "end": {"type": "number"}, "description": S})),
    "music": obj({"present": YES_NO, "description": S}),
    "pauses": arr(obj({"start": {"type": "number"}, "end": {"type": "number"}})),
    "uncertainties": STRINGS,
})

BOUNDARY = enum("clean", "mid_action", "mid_speech", "mid_transition", "unknown")
CANDIDATE_VERDICT = obj({
    "completeness": enum("complete", "incomplete", "unknown"),
    "start_boundary": BOUNDARY, "end_boundary": BOUNDARY,
    "visual_quality": LEVEL, "describability": LEVEL, "information": LEVEL,
    "content_summary": S, "evidence_frame_ids": STRINGS, "reason": S,
})

REFERENCE_RELATIONS = obj({
    "target_subjects": arr(obj({"key": S, "kind": enum(*SUBJECT_KINDS), "description": S,
                                "importance": enum("main", "secondary"), "target_frame_ids": STRINGS})),
    "relations": arr(obj({
        "reference_id": S, "supported": YES_NO,
        "target_subject_or_event_id": S,
        "role": enum(*SUBJECT_ROLES, *STRUCTURE_ROLES, "none"),
        "promised_attributes": STRINGS, "preserved_attributes": STRINGS, "changed_attributes": STRINGS,
        "target_evidence_ids": STRINGS, "reference_evidence_ids": STRINGS,
        "relation_marker": enum(*VISUAL_RELATIONS, "none"),
        "clarity": LEVEL, "uncertainties": STRINGS,
    })),
})

DUPLICATE_REVIEW = obj({"result": enum("clear", "duplicate", "unresolved"), "reason": S, "evidence_ids": STRINGS})

EXPORTED_VISUAL_FACTS = obj({
    "style": S,
    "shots": arr(obj({
        "start_frame_id": S, "framing": S, "setting_and_lighting": S, "subjects_and_positions": S, "action": S,
        "camera_movement": S, "visible_text": STRINGS, "subject_keys": STRINGS, "evidence_frame_ids": STRINGS,
        "uncertainties": STRINGS})),
    "missed_cuts": arr(obj({"last_frame_id_before": S, "first_frame_id_after": S})),
    "subjects": arr(obj({"key": S, "description": S, "first_frame_id": S})),
    "relation_checks": arr(obj({"reference_id": S, "holds": YES_NO, "note": S, "target_evidence_ids": STRINGS,
                                "reference_evidence_ids": STRINGS})),
    "speech_attribution": arr(obj({"utterance_id": S, "speaker": S, "speaker_description": S,
                                   "on_screen": YES_NO, "evidence_frame_ids": STRINGS})),
    "start_boundary": enum("clean", "problem", "unknown"), "end_boundary": enum("clean", "problem", "unknown"),
    "uncertainties": STRINGS,
})

EXPORTED_AUDIO_FACTS = obj({
    "utterances": arr(obj({"start": {"type": "number"}, "end": {"type": "number"},
                           "kind": enum("dialogue", "voiceover", "lyrics", "vocalization"), "language": S,
                           "text": S, "intelligible": enum("yes", "partly", "no"), "voice_description": S})),
    "soundscape": arr(obj({"description": S, "start": {"type": "number"}, "end": {"type": "number"}})),
    "music": obj({"present": YES_NO, "audience_only": YES_NO, "description": S}),
    "reference_audio_checks": arr(obj({"reference_id": S, "useful": YES_NO, "role": enum(*AUDIO_ROLES, "none"),
                                       "relation_marker": enum(*AUDIO_RELATIONS, "none"), "note": S})),
    "truncated_at_end": YES_NO,
    "uncertainties": arr(obj({"text": S, "start": {"type": "number"}, "end": {"type": "number"}})),
})

LINKED_TEXT = obj({"text": S, "fact_ids": STRINGS})
PROMPT_DRAFT = obj({
    "definitions": arr(obj({"label": S, "text": S, "fact_ids": STRINGS})),
    "summary": LINKED_TEXT,
    "retention": arr(obj({"label": S, "scope": S, "text": S, "fact_ids": STRINGS})),
    "style": S,
    "shots": arr(obj({"shot_id": S, "text": S, "fact_ids": STRINGS})),
    "overall_soundscape": LINKED_TEXT,
    "non_diegetic_music": LINKED_TEXT,
})

ISSUE_KINDS = ["wrong_fact", "relation_overclaim", "unconfirmed_dialogue", "format", "unsupported_detail",
               "missing_key_content"]
SEMANTIC_REVIEW = obj({
    "core_relation_valid": YES_NO, "boundaries_valid": YES_NO, "visual_consistent": YES_NO,
    "issues": arr(obj({
        "location": S, "kind": enum(*ISSUE_KINDS), "quote": S, "explanation": S,
        "evidence_frame_ids": STRINGS, "repairable_by_text": {"type": "boolean"}, "suggestion": S})),
    "visual_uncertainties": STRINGS, "audio_uncertainties": STRINGS,
})
