from fractions import Fraction

from stapipe import prompt
from stapipe.config import DEFAULTS
from stapipe.planning import canvas_geometry, plan_legal_video, wav_samples
from stapipe.references import beam_search
from stapipe.util import AttrDict, merge_spans, mmss

SEARCH = DEFAULTS["search"]
FRAME = Fraction(1001, 24000)


def test_walkthrough_target_gets_175_frames():
    plan = plan_legal_video(174 * FRAME, 5, 15, SEARCH)
    assert plan["num_frames"] == 175 and abs(plan["padding_seconds"] - 0.0344167) < 1e-6
    assert wav_samples(175) == 233333


def test_walkthrough_reference_fails_two_percent_and_passes_two_and_a_half():
    duration = 138 * FRAME                       # 5.75575 s，补到 141 帧需要 2.072%
    assert plan_legal_video(duration, 2, 15, SEARCH) is None
    assert plan_legal_video(duration, 2, 15, {**SEARCH, "max_tail_padding_ratio": 0.025})["num_frames"] == 141


def test_plan_never_truncates_or_exceeds_range():
    assert plan_legal_video(Fraction(1499, 100), 5, 15, SEARCH) is None      # 只能补到超过 15 秒
    assert plan_legal_video(Fraction(4), 5, 15, SEARCH) is None


def test_canvas_matches_example():
    assert canvas_geometry(1280, 720) == {"canvas_w": 1344, "canvas_h": 768, "scaled_w": 1344, "scaled_h": 756,
                                          "pad_x": 0, "pad_y": 6}
    portrait = canvas_geometry(720, 1280)
    assert (portrait["canvas_w"], portrait["canvas_h"]) == (768, 1344)


def test_small_helpers():
    assert mmss(29 / 24) == "00:01.208" and mmss(59.9996) == "01:00.000"
    assert merge_spans([[1.2, 2.6], [2.5, 3], [-1, 0.5], [9, 12]], 0, 10) == [[0, 0.5], [1.2, 3], [9, 10]]


def _facts():
    relation = {"holds": "yes", "promised_attributes": ["face"], "relation_marker": "fully_preserved"}
    return {
        "shots": [{"shot_id": "shot1", "index": 1, "start_frame": 0, "subject_keys": ["bag", "man"]},
                  {"shot_id": "shot2", "index": 2, "start_frame": 29, "subject_keys": ["man"]}],
        "subjects": [{"key": "man", "first_frame": 0}, {"key": "bag", "first_frame": 0},
                     {"key": "mover", "first_frame": 29}],
        "relations": [
            {**relation, "relation_id": "rel1", "reference_name": "picture_1", "role": "identity_and_appearance",
             "target_subject_key": "man"},
            {**relation, "relation_id": "rel2", "reference_name": "picture_1", "role": "product_or_prop",
             "target_subject_key": "bag", "relation_marker": "partially_preserved"},
            {**relation, "relation_id": "rel3", "reference_name": "video_1", "role": "narrative_structure",
             "target_subject_key": "whole_target", "relation_marker": "weak_reference"}],
        "utterances": [{"utterance_id": "u1", "start": 1.0, "speaker_key": "mover", "language": "English", "parts": [
            {"part_id": "u1.1", "shot_id": "shot1", "text": "Help", "lead_scenetrans": False,
             "trail_scenetrans": True, "cutoff": False},
            {"part_id": "u1.2", "shot_id": "shot2", "text": "me.", "lead_scenetrans": True,
             "trail_scenetrans": False, "cutoff": False}]}],
        "audio": {"reference_audio": []}, "style": "Live action.",
    }


REFERENCES = [{"id": "img_000409", "type": "image", "name": "picture_1", "path": "inputs/picture_1.png"},
              {"id": "vid_000062_000200", "type": "video", "name": "video_1", "path": "inputs/video_1.mp4",
               "include_audio": False}]


def _draft():
    return {
        "definitions": [{"label": "<Subject 1>", "text": "is the bag in <Picture 1>.", "fact_ids": ["rel2"]},
                        {"label": "<Subject 2>", "text": "is the man in <Picture 1>.", "fact_ids": ["rel1"]},
                        {"label": "<Video 1>", "text": "supplies the set-up and reveal pattern.", "fact_ids": ["rel3"]}],
        "summary": {"text": "A short clip with <Subject 1> and <Subject 2>.", "fact_ids": ["target"]},
        "retention": [{"label": "<Subject 1>", "scope": "", "text": "the print is kept.", "fact_ids": []},
                      {"label": "<Subject 2>", "scope": "", "text": "the face is kept.", "fact_ids": []},
                      {"label": "<Video 1>", "scope": "comic structure", "text": "only the pattern.", "fact_ids": []}],
        "style": "Realistic commercial style.",
        "shots": [{"shot_id": "shot1", "fact_ids": ["shot1", "u1.1"],
                   "text": "<Subject 2> holds <Subject 1> while a mover {{S:u1}} calls out, {{D:u1.1}}"},
                  {"shot_id": "shot2", "fact_ids": ["shot2"],
                   "text": "The camera cuts closer as the mover {{S:u1}} continues, {{D:u1.2}} carrying over."}],
        "overall_soundscape": {"text": "Outdoor ambience.", "fact_ids": ["audio"]},
        "non_diegetic_music": {"text": "N/A", "fact_ids": ["audio"]},
    }


def test_labels_follow_first_appearance_and_are_independent_of_speakers():
    labels = prompt.allocate_labels(REFERENCES, _facts())
    assert [r["label"] for r in labels["reference_records"]] == ["<Picture 1>", "<Video 1>"]
    assert "audio_label" not in labels["reference_records"][1]
    assert labels["subjects"]["<Subject 1>"]["key"] == "bag"          # 同一帧首次出现时按键名排序
    assert labels["subjects"]["<Subject 1>"]["marker"] == "partially_preserved"
    assert labels["definitions_required"] == ["<Subject 1>", "<Subject 2>", "<Video 1>"]
    assert labels["speakers"] == {"mover": "S1"} and labels["speaker_subject_labels"] == {"mover": None}


def test_render_and_structure_checks_round_trip():
    facts, draft = _facts(), _draft()
    labels = prompt.allocate_labels(REFERENCES, facts)
    assert prompt.validate_draft(draft, facts, labels) == []
    text = prompt.render_six_sections(draft, facts, labels)
    assert "<Subject 1> (appears in [Shot 1]): partially_preserved - the print is kept." in text
    assert "<Video 1> (comic structure): weak_reference - only the pattern." in text
    assert "[Shot 2] At 00:01.208, the camera cuts closer as the mover (S1) continues, " \
           "<d>[English] <scenetrans> me.</d> carrying over." in text
    assert "<d>[English] Help <scenetrans></d>" in text and text.startswith("subject_definitions:\n<Subject 1> is")
    record = {"target": {"num_frames": 175}, "references": labels["reference_records"]}
    failed = [c for c in prompt.check_prompt(text, record, [29], 1) if not c[1]]
    assert [c[0] for c in failed] == ["detailed_description_not_trivial"]
    wrong_cut = dict((c[0], c[1]) for c in prompt.check_prompt(text, record, [40], 1))
    assert wrong_cut["shot_times_match_confirmed_cut_frames"] is False


def test_unintelligible_speech_is_left_out_of_the_prompt_not_marked():
    from stapipe.build import _utterances_from_asr, _words_text

    word = lambda w, s, e, p: {"w": w, "s": s, "e": e, "p": p}
    evidence = {"available": True, "asr": {"language": "en", "segments": [
        {"start": 0.0, "end": 1.6, "text": "Thank you.", "no_speech_prob": 0.0, "avg_logprob": -0.4,
         "words": [word("Thank", 0.2, 0.9, 0.31), word("you.", 0.9, 1.6, 0.93)]},
        {"start": 3.0, "end": 4.0, "text": "Come in.", "no_speech_prob": 0.0, "avg_logprob": -0.2,
         "words": [word("Come", 3.0, 3.4, 0.95), word("in.", 3.4, 4.0, 0.97)]}]}}
    utterances, omitted, uncertain = _utterances_from_asr(evidence, AttrDict(DEFAULTS), 5.0)
    # 含低置信词的整句被省略：不保留听清的那几个词，也不产生占位符，只留下区间
    assert [(u["utterance_id"], _words_text(u["words"])) for u in utterances] == [("u1", "Come in.")]
    assert [(o["start"], o["end"]) for o in omitted] == [(0.2, 1.6)]
    assert len(uncertain) == 1 and abs(uncertain[0]["span"][0] - 0.1) < 1e-9 and abs(uncertain[0]["span"][1] - 1.7) < 1e-9


def test_placeholder_in_a_rendered_prompt_fails_the_structure_check():
    facts, draft = _facts(), _draft()
    labels = prompt.allocate_labels(REFERENCES, facts)
    text = prompt.render_six_sections(draft, facts, labels)
    record = {"target": {"num_frames": 175}, "references": labels["reference_records"]}
    assert dict((c[0], c[1]) for c in prompt.check_prompt(text, record, [29], 1))["no_unintelligible_placeholder"]
    marked = dict((c[0], c[1]) for c in prompt.check_prompt(text.replace("me.", "[unclear]"), record, [29], 1))
    assert marked["no_unintelligible_placeholder"] is False


def test_draft_validation_rejects_program_owned_markup_and_missing_tokens():
    facts, draft = _facts(), _draft()
    labels = prompt.allocate_labels(REFERENCES, facts)
    draft["shots"][0]["text"] = "The mover (S1) says <d>[English] Help</d> to <Subject 9>."
    errors = " ".join(prompt.validate_draft(draft, facts, labels))
    assert "do not write <d>" in errors and "unknown labels" in errors and "exactly once" in errors


def test_beam_search_prefers_coverage_and_drops_useless_references():
    cfg = AttrDict(DEFAULTS)

    def ref(ident, kind, key, role, interval, clarity="high"):
        rel = {"target_subject_or_event_id": key, "role": role, "clarity": clarity,
               "target_evidence_ids": ["a", "b"], "reference_evidence_ids": ["c", "d"]}
        return {"id": ident, "type": kind, "source_interval": interval, "relations": [rel],
                "plan": {"duration_seconds": 5.0}}

    valid = [ref("img_1", "image", "man", "identity_and_appearance", [10, 11]),
             ref("img_2", "image", "man", "identity_and_appearance", [20, 21], "low"),
             ref("img_3", "image", "bag", "product_or_prop", [30, 31]),
             ref("vid_1", "video", "whole", "narrative_structure", [100, 220])]
    subjects = [{"key": "man", "importance": "main"}, {"key": "bag", "importance": "main"}]
    best = beam_search(valid, subjects, cfg)[0]
    assert [r["id"] for r in best["refs"]] == ["img_1", "img_3", "vid_1"]
    assert best["metrics"]["layout_preferred"] and best["metrics"]["redundant_references"] == 0
