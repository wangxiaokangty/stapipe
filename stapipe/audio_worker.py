"""在 ASR 环境中运行的独立工作进程（不导入 stapipe 包）。

  <asr_python> audio_worker.py --media X --ffmpeg F --out out.json --asr-model M [--tags]

输出 Whisper 的分段/逐词转录，以及可选的 CLAP 候选声音类别。两者都只是证据，不是真值。
"""
import argparse
import json
import subprocess

import numpy as np

TAG_LABELS = ("speech|female speech|male speech|singing|music|background music score|piano|strings|"
              "drums and beat|electronic music|rock guitar|rain|wind|room tone silence|footsteps|door|"
              "crowd chatter|laughter|applause|whoosh sound effect|impact sound effect|engine|birds chirping|traffic")


def load(ffmpeg, media, sr):
    raw = subprocess.run([ffmpeg, "-v", "error", "-i", media, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def run_asr(audio, model):
    import mlx_whisper

    result = mlx_whisper.transcribe(audio, path_or_hf_repo=model, word_timestamps=True,
                                    condition_on_previous_text=False, temperature=0.0)
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 3), "end": round(seg["end"], 3), "text": seg["text"].strip(),
            "avg_logprob": round(seg.get("avg_logprob", 0), 3),
            "no_speech_prob": round(seg.get("no_speech_prob", 0), 3),
            "words": [{"w": w["word"].strip(), "s": round(w["start"], 3), "e": round(w["end"], 3),
                       "p": round(w.get("probability", 0), 3)} for w in seg.get("words", [])]})
    return {"language": result.get("language"), "segments": segments}


def run_tags(audio, sr, window=2.0, top=4):
    import torch
    from transformers import ClapModel, ClapProcessor

    labels = TAG_LABELS.split("|")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    out, w = [], int(window * sr)
    with torch.no_grad():
        te = model.get_text_features(**proc(text=[f"the sound of {x}" for x in labels], return_tensors="pt",
                                            padding=True))
        te = te if torch.is_tensor(te) else te.pooler_output
        te = te / te.norm(dim=-1, keepdim=True)
        for i in range(0, max(1, len(audio) - w // 3), w):
            seg = audio[i:i + w]
            ae = model.get_audio_features(**proc(audio=[seg], sampling_rate=sr, return_tensors="pt"))
            ae = ae if torch.is_tensor(ae) else ae.pooler_output
            ae = ae / ae.norm(dim=-1, keepdim=True)
            p = (100 * ae @ te.T).softmax(-1)[0].numpy()
            order = np.argsort(p)[::-1][:top]
            out.append({"start": round(i / sr, 3), "end": round((i + len(seg)) / sr, 3),
                        "labels": [[labels[k], round(float(p[k]), 3)] for k in order]})
    return {"model": "laion/clap-htsat-unfused", "label_set": labels, "windows": out,
            "note": "scores are a softmax over this label set only; hints, not probabilities of presence"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr-model", required=True)
    ap.add_argument("--tags", action="store_true")
    a = ap.parse_args()
    result = {"asr_model": a.asr_model, "asr": run_asr(load(a.ffmpeg, a.media, 16000), a.asr_model), "tags": None,
              "tags_error": None}
    if a.tags:
        try:
            result["tags"] = run_tags(load(a.ffmpeg, a.media, 48000), 48000)
        except Exception as error:  # noqa: BLE001 声音类别只是辅助证据，失败不阻塞转录
            result["tags_error"] = f"{type(error).__name__}: {error}"[:300]
    with open(a.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
