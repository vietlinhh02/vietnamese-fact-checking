import json
import os

file_path = 'experiments/e2e/e2e_run_results.json'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error reading JSON: {e}")
    exit(1)

results = data.get("results", [])
print(f"Total results: {len(results)}\n")

for i, res in enumerate(results):
    claim = res.get("claim", {})
    verdict = res.get("verdict", {})
    evidence = res.get("evidence", [])
    
    print(f"--- Claim {i+1} ---")
    print(f"Text: {claim.get('text', 'N/A').encode('utf-8', 'ignore').decode('utf-8')}")
    print(f"Verdict: {verdict.get('label', 'N/A')}")
    
    conf_scores = verdict.get("confidence_scores", {})
    print(f"Confidence: {conf_scores}")
    
    print(f"Evidence count: {len(evidence)}")
    explanation = verdict.get('explanation', '')
    print(f"Explanation length: {len(explanation)}")
    print(f"Explanation snippet: {explanation[:100].replace(chr(10), ' ')}...")

    # Check for specific failure indicators
    if verdict.get('label') == 'not_enough_info':
        print("ALERT: Verdict is not_enough_info")
    
    if not evidence and verdict.get('label') != 'not_enough_info':
        print("ALERT: No evidence but verdict is not not_enough_info")
        
    print("")
