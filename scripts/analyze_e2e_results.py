import json
from pathlib import Path

# Load results
results_path = Path(r"experiments/end_to_end/e2e_run_results.json")
data = json.loads(results_path.read_text(encoding="utf-8"))

# Handle both old format (list) and new format (dict with 'results' key)
if isinstance(data, dict):
    results = data.get("results", [])
    metadata = data.get("metadata", {})
    print(f"Metadata: {json.dumps(metadata, ensure_ascii=False, indent=2)}")
    print("="*60)
else:
    results = data

print(f"Total verification results in file: {len(results)}")
print("="*60)

for i, result in enumerate(results):
    claim_text = result.get("claim", {}).get("text", "N/A")
    verdict_label = result.get("verdict", {}).get("label", "N/A")
    evidence_count = len(result.get("evidence", []))
    
    print(f"\n[Result {i+1}]")
    print(f"  Claim: {claim_text[:100]}...")
    print(f"  Verdict: {verdict_label}")
    print(f"  Evidence count: {evidence_count}")
    
    # Check evidence sources
    sources = set()
    for ev in result.get("evidence", []):
        url = ev.get("source_url", "")
        if url:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            sources.add(domain)
    print(f"  Source domains: {sources}")
