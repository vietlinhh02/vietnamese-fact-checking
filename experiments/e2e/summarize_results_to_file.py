import json
import os

file_path = 'experiments/e2e/e2e_run_results.json'
output_path = 'experiments/e2e/summary.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Error reading JSON: {e}")
    exit(1)

results = data.get("results", [])

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(f"Total results: {len(results)}\n\n")

    for i, res in enumerate(results):
        claim = res.get("claim", {})
        verdict = res.get("verdict", {})
        evidence = res.get("evidence", [])
        
        f.write(f"--- Claim {i+1} ---\n")
        f.write(f"Text: {claim.get('text', 'N/A')}\n")
        f.write(f"Verdict: {verdict.get('label', 'N/A')}\n")
        
        conf_scores = verdict.get("confidence_scores", {})
        f.write(f"Confidence: {conf_scores}\n")
        
        f.write(f"Evidence count: {len(evidence)}\n")
        
        explanation = verdict.get('explanation', '')
        f.write(f"Explanation length: {len(explanation)}\n")
        
        if verdict.get('label') == 'not_enough_info':
            f.write("ALERT: Verdict is not_enough_info\n")
        
        if not evidence and verdict.get('label') != 'not_enough_info':
            f.write("ALERT: No evidence but verdict is not not_enough_info\n")
            
        f.write("\n")
