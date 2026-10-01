import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from retrieval import service
from retrieval.query import _translate
from retrieval.config import FusionConfig
from eval.gt import trake_score
from retrieval.trake import latency_stats

def load_synth_dev(path="data/synth_dev.json"):
    if not os.path.exists(path):
        # Fallback for kaggle
        path = "/kaggle/working/data/synth_dev.json"
        if not os.path.exists(path):
            # Another fallback
            path = "synth_dev.json"
            if not os.path.exists(path):
                # Try eval dir
                path = "eval/trake_ground_truth.json"
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Could not find synth_dev.json dataset. Checked data/synth_dev.json, /kaggle/working/data/synth_dev.json, synth_dev.json and eval/trake_ground_truth.json")
    
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
        
    out = []
    for q in raw:
        n = int(q.get("n_events", 1))
        events = []
        for k in range(1, n + 1):
            if f"event{k}_s" in q and f"event{k}_e" in q:
                events.append((int(q[f"event{k}_s"]), int(q[f"event{k}_e"])))
        
        out.append({
            "id": q.get("sample_id", q.get("id", "unknown")),
            "query": q.get("query_text", q.get("query", "")),
            "video_id": q.get("video_id", ""),
            "n_events": n,
            "events": events,
        })
    return sorted(out, key=lambda x: x["id"])

import re
_NUM = re.compile(r"\(\s*(\d+)\s*\)")
def split_events(text, n_events):
    parts = _NUM.split(text)
    if len(parts) >= 3:
        ev = [parts[i + 1].strip(" ,.;:") for i in range(1, len(parts) - 1, 2)]
        ev = [e for e in ev if e]
        if len(ev) >= n_events:
            return ev[:n_events]
    chunks = [c.strip(" ,.;:") for c in re.split(r"[,;.]", text) if c.strip()]
    return chunks[-n_events:] if len(chunks) >= n_events else chunks

# LLM latency estimate (ms) — measured once during cache generation
# Claude Haiku typically responds in 300–800ms for this prompt
LLM_LATENCY_MS = 500.0

def evaluate_config(config_name, flags, qset, s):
    eng = s["engine"]
    store = s["store"]
    
    # Apply feature flags to config
    for k, v in flags.items():
        setattr(eng.cfg, k, v)
        
    # Reset latency stats
    latency_stats['1-best'] = []
    latency_stats['k-best'] = []
    
    tot_r_auto = 0.0
    tot_final_auto = 0.0
    tot_r_oracle = 0.0
    tot_final_oracle = 0.0
    
    # Split tracking
    kis_auto = 0.0
    kis_oracle = 0.0
    n_kis = 0
    trake_auto = 0.0
    trake_oracle = 0.0
    n_trake = 0
    
    from retrieval.autofill import autofill
    from eval.gt import final_score
    from retrieval.query import decompose
    
    # Run evaluation
    for q in tqdm(qset, desc=f"Evaluating {config_name}"):
        if flags.get("use_llm_plan", False):
            # Use decompose() — reads from pre-computed cache
            dq = decompose(query_vi=q["query"], use_llm=True, kind="generic_chain")
            ev_en = dq.clauses_en if dq.clauses_en else [dq.query_en or q["query"]]
            query_en_fused = dq.query_en or q["query"]
            query_vi = dq.query_vi or q["query"]
        else:
            # Baseline: no LLM, no translation
            ev_en = [q["query"]] * q["n_events"]
            query_en_fused = q["query"]
            query_vi = q["query"]
            
        if q["n_events"] == 1:
            # KIS/QA evaluation using Autofill
            wide = eng.search(
                query_en=query_en_fused, 
                query_vi=query_vi, 
                frame_topk=3000, 
                kind="generic_chain",
                weights=getattr(eng.cfg, "weights", None)
            )
            cands = []
            for v in wide.get("videos", []):
                vi = v["video_info"]
                for j, fi in enumerate(vi["lst_keyframe_idxs"]):
                    cands.append({
                        "video_id": v["video_id"], 
                        "frame_idx": int(fi), 
                        "score": (vi.get("lst_scores") or [None])[j]
                    })
            
            # Generate 100 rows using the selected autofill method
            out = autofill(store, manual=[], candidates=cands, config=eng.cfg, target=100)
            
            fs, rank = final_score(out, q["video_id"], q["events"][0][0], q["events"][0][1])
            r = 1.0 if rank is not None else 0.0
            
            # True Oracle calculation for KIS
            oracle_rank = None
            for i, c in enumerate(cands):
                if c["video_id"] == q["video_id"] and q["events"][0][0] <= c["frame_idx"] <= q["events"][0][1]:
                    oracle_rank = i + 1
                    break
                    
            if oracle_rank is not None:
                r_oracle = 1.0
                fs_oracle = sum(1 for k in (1, 5, 20, 50, 100) if k >= oracle_rank) / 5 * 1.0
            else:
                r_oracle = 0.0
                fs_oracle = 0.0
            
            tot_r_auto += r
            tot_final_auto += fs
            tot_r_oracle += r_oracle
            tot_final_oracle += fs_oracle
            
            kis_auto += fs
            kis_oracle += fs_oracle
            n_kis += 1
            
        else:
            # TRAKE evaluation using DP
            fused, _, _ = eng.rank_videos(
                query_en_fused, 
                query_vi, 
                kind="generic_chain",
                weights=getattr(eng.cfg, "weights", None)
            )
            cands = [v for v, _ in fused if v in store.video_slice][:eng.cfg.video_topn]
            
            aligned = eng.align_videos(ev_en, cands)
            rank_dp = next((i + 1 for i, x in enumerate(aligned) if x["video_id"] == q["video_id"]), None)
            
            mine = next((x for x in aligned if x["video_id"] == q["video_id"]), None)
            
            # 1. Automatic Score (Top-1 DP)
            r_auto = 0.0
            if mine:
                r_auto = trake_score(mine["matched_frames"] if "matched_frames" in mine
                                else [m["frame_idx"] for m in mine["matched"]],
                                q["events"])
            tot_r_auto += r_auto
            
            if rank_dp:
                fs_auto = sum(1 for k in (1, 5, 20, 50, 100) if k >= rank_dp) / 5 * r_auto
            else:
                fs_auto = 0.0
            tot_final_auto += fs_auto
            
            # 2. Oracle Score (Top-K DP)
            r_oracle = r_auto
            fs_oracle = fs_auto
            if mine and "k_best_filled" in mine and eng.cfg.dp_method == "k-best":
                if config_name == "5. Full System (Ours)":
                    print(f"\n[DEBUG TRAKE {q['id']}] len(k_best_filled): {len(mine['k_best_filled'])}")
                    for k_idx, p in enumerate(mine['k_best_filled'][:3]):
                        print(f"  path {k_idx}: {p}")
                r_scores = [trake_score(filled_path, q["events"]) for filled_path in mine["k_best_filled"]]
                r_oracle = max(r_scores)
                if rank_dp:
                    fs_oracle = sum(1 for k in (1, 5, 20, 50, 100) if k >= rank_dp) / 5 * r_oracle
            tot_r_oracle += r_oracle
            tot_final_oracle += fs_oracle
            
            trake_auto += fs_auto
            trake_oracle += fs_oracle
            n_trake += 1
        
    n = len(qset)
    avg_fs_auto = tot_final_auto / n if n > 0 else 0
    avg_fs_oracle = tot_final_oracle / n if n > 0 else 0
    
    avg_kis_auto = kis_auto / n_kis if n_kis > 0 else 0
    avg_kis_oracle = kis_oracle / n_kis if n_kis > 0 else 0
    
    avg_trake_auto = trake_auto / n_trake if n_trake > 0 else 0
    avg_trake_oracle = trake_oracle / n_trake if n_trake > 0 else 0
    
    # Calculate average latency based on dp_method
    if flags.get('dp_method') == '1-best':
        avg_latency = np.mean(latency_stats['1-best']) if latency_stats['1-best'] else 0
    else:
        avg_latency = np.mean(latency_stats['k-best']) if latency_stats['k-best'] else 0
        
    # LLM latency: only applicable when use_llm_plan is True
    llm_latency = LLM_LATENCY_MS if flags.get("use_llm_plan", False) else 0.0
        
    return {
        "Config": config_name,
        "n_kis": n_kis,
        "n_trake": n_trake,
        "mAP (KIS)": round(avg_kis_auto, 4),
        "mAP (TRAKE)": round(avg_trake_auto, 4),
        "mAP (All)": round(avg_fs_auto, 4),
        "Oracle (All)": round(avg_fs_oracle, 4),
        "DP Lat (ms)": round(avg_latency, 2),
    }

def main():
    print("Loading system components...")
    
    # Kaggle environment fallback for artifact paths
    if os.path.exists("/kaggle/input"):
        if not os.environ.get("ARTIFACT_ROOT"):
            os.environ["ARTIFACT_ROOT"] = "/kaggle/input/artifacts-dataset/data/artifacts"
            print(f"Running on Kaggle: Set ARTIFACT_ROOT to {os.environ['ARTIFACT_ROOT']}")
            
        # We also need to set the DECOMPOSE_CACHE fallback if running on Kaggle
        if not os.environ.get("DECOMPOSE_CACHE") and not os.path.exists("data/decompose_cache.json"):
            if os.path.exists("/kaggle/working/data/decompose_cache.json"):
                os.environ["DECOMPOSE_CACHE"] = "/kaggle/working/data/decompose_cache.json"

    s = service.get()
    
    # 5 configurations
    configs = [
        {
            "name": "1. Baseline",
            "flags": {
                "use_llm_plan": False,
                "use_text_index": False,
                "fusion_method": "rrf",
                "dp_method": "1-best",
                "autofill_method": "old",
                "weights": {"siglip": 1.0, "meta": 0.0, "asr": 0.0, "ocr": 0.0}
            }
        },
        {
            "name": "2. + LLM",
            "flags": {
                "use_llm_plan": True,
                "use_text_index": False,
                "fusion_method": "rrf",
                "dp_method": "1-best",
                "autofill_method": "old",
                "weights": {"siglip": 1.0, "meta": 0.0, "asr": 0.0, "ocr": 0.0}
            }
        },
        {
            "name": "3. + Trigram OCR/ASR",
            "flags": {
                "use_llm_plan": True,
                "use_text_index": True,
                "fusion_method": "rrf",
                "dp_method": "1-best",
                "autofill_method": "old",
                "weights": {"siglip": 1.0, "meta": 0.2, "asr": 0.2, "ocr": 0.2}
            }
        },
        {
            "name": "4. + Z-Score Fusion",
            "flags": {
                "use_llm_plan": True,
                "use_text_index": True,
                "fusion_method": "z-score",
                "dp_method": "1-best",
                "autofill_method": "old",
                "weights": {"siglip": 1.0, "meta": 0.2, "asr": 0.2, "ocr": 0.2}
            }
        },
        {
            "name": "5. Full System (Ours)",
            "flags": {
                "use_llm_plan": True,
                "use_text_index": True,
                "fusion_method": "z-score",
                "dp_method": "k-best",
                "autofill_method": "deferred_mmr",
                "weights": {"siglip": 1.0, "meta": 0.2, "asr": 0.2, "ocr": 0.2}
            }
        }
    ]
    
    print("Loading dataset...")
    try:
        qset = load_synth_dev()
        print(f"Loaded {len(qset)} queries for evaluation.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    results = []
    
    for conf in configs:
        res = evaluate_config(conf["name"], conf["flags"], qset, s)
        results.append(res)
        
    df = pd.DataFrame(results)
    
    # Output Table
    print("\n# Ablation Study & Latency Evaluation Results\n")
    print(df.to_string(index=False))
    
    # Save to CSV
    csv_path = "ablation_results.csv"
    if os.path.exists("/kaggle/working/"):
        csv_path = "/kaggle/working/ablation_results.csv"
    
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
if __name__ == "__main__":
    main()
