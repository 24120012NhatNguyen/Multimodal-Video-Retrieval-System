import os
import sys
import pandas as pd

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import load_synth_dev, evaluate_config
from retrieval import service

def main():
    print("Loading system components...")
    
    # Kaggle environment fallback for artifact paths
    if os.path.exists("/kaggle/input"):
        if not os.environ.get("ARTIFACT_ROOT"):
            os.environ["ARTIFACT_ROOT"] = "/kaggle/input/artifacts-dataset/data/artifacts"
            
        if not os.environ.get("DECOMPOSE_CACHE") and not os.path.exists("data/decompose_cache.json"):
            if os.path.exists("/kaggle/working/data/decompose_cache.json"):
                os.environ["DECOMPOSE_CACHE"] = "/kaggle/working/data/decompose_cache.json"

    s = service.get()
    
    print("Loading dataset...")
    try:
        qset = load_synth_dev()
        print(f"Loaded {len(qset)} queries for evaluation.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Sweep configurations
    configs = []
    
    # 1. Sweep Tau with fixed Gamma = 0.5
    for tau in [1.0, 1.5, 2.0, 2.5, 3.0]:
        configs.append({
            "name": f"Tau={tau:.1f}, Gamma=0.5",
            "flags": {
                "use_llm_plan": True,
                "use_text_index": True,
                "fusion_method": "z-score",
                "dp_method": "k-best",
                "autofill_method": "deferred_mmr",
                "dp_tau": tau,
                "dp_gamma": 0.5,
                "weights": {"siglip": 1.0, "meta": 0.2, "asr": 0.2, "ocr": 0.2}
            }
        })
        
    # 2. Sweep Gamma with fixed Tau = 2.0
    for gamma in [0.1, 2.0]:
        configs.append({
            "name": f"Tau=2.0, Gamma={gamma:.1f}",
            "flags": {
                "use_llm_plan": True,
                "use_text_index": True,
                "fusion_method": "z-score",
                "dp_method": "k-best",
                "autofill_method": "deferred_mmr",
                "dp_tau": 2.0,
                "dp_gamma": gamma,
                "weights": {"siglip": 1.0, "meta": 0.2, "asr": 0.2, "ocr": 0.2}
            }
        })

    results = []
    
    for conf in configs:
        res = evaluate_config(conf["name"], conf["flags"], qset, s)
        results.append(res)
        
    df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("ABLATION STUDY RESULTS (Table 2)")
    print("="*50)
    print(df[["Config", "mAP (KIS)", "mAP (TRAKE)"]].to_string(index=False))
    
    # Save to CSV
    csv_path = "ablation_params_results.csv"
    if os.path.exists("/kaggle/working/"):
        csv_path = "/kaggle/working/ablation_params_results.csv"
    
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to {csv_path}")

if __name__ == "__main__":
    main()
