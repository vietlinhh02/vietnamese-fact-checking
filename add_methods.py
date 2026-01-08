
    def _apply_confidence_cap(
        self, 
        confidence_scores: Dict[str, float], 
        evidence_list: List[Evidence],
        temporal_warnings: List[str]
    ) -> Tuple[Dict[str, float], Optional[str], Optional[str]]:
        """Apply confidence capping based on evidence quality and temporal issues.
        
        Rules:
        1. If avg evidence credibility < 0.5: cap at 0.7
        2. If evidence has FUTURE/PENDING temporal marker: cap at 0.6
        3. If single source only: cap at 0.7
        4. Never allow 1.0 unless multiple high-credibility sources agree
        """
        capped_scores = confidence_scores.copy()
        cap_reason = None
        temporal_notes = "; ".join(temporal_warnings) if temporal_warnings else None
        
        # Calculate caps
        caps = []
        
        # 1. Credibility Cap
        if evidence_list:
            avg_credibility = sum(e.credibility_score for e in evidence_list) / len(evidence_list)
            if avg_credibility < 0.5:
                caps.append((0.7, f"Low average credibility ({avg_credibility:.2f})"))
            elif avg_credibility < 0.7:
                caps.append((0.85, f"Medium average credibility ({avg_credibility:.2f})"))
        else:
            caps.append((0.5, "No evidence collected"))
            
        # 2. Temporal Cap
        if temporal_warnings:
            caps.append((0.6, "Evidence contains future/pending policies"))
            
        # 3. Source Diversity Cap
        domains = set()
        from urllib.parse import urlparse
        for e in evidence_list:
            if e.source_url:
                try:
                    domain = urlparse(e.source_url).netloc.lower()
                    domains.add(domain)
                except:
                    pass
        
        if len(domains) < 2:
            caps.append((0.7, "Single source domain"))
            
        # Apply lowest cap
        if caps:
            min_cap, reason = min(caps, key=lambda x: x[0])
            
            # Apply to major scores
            for label, score in capped_scores.items():
                if score > min_cap:
                    capped_scores[label] = min_cap
                    cap_reason = reason
            
            # Re-normalize if needed (distribute remaining to 'not_enough_info' or 'mixed')
            total = sum(capped_scores.values())
            if total < 1.0:
                # Add remainder to uncertainty
                remainder = 1.0 - total
                if "not_enough_info" in capped_scores:
                    capped_scores["not_enough_info"] += remainder
                elif "mixed" in capped_scores:
                    capped_scores["mixed"] += remainder
                else:
                    # Distribute proportionally
                    for label in capped_scores:
                        capped_scores[label] /= total
                        
        return capped_scores, cap_reason, temporal_notes

    def _aggregate_sub_verdicts(self, sub_verdicts: List[Verdict]) -> Verdict:
        """Aggregate multiple sub-claim verdicts into final verdict."""
        if not sub_verdicts:
            return Verdict(
                claim_id="aggregated",
                label="not_enough_info", 
                confidence_scores={"not_enough_info": 1.0}
            )
            
        labels = [v.label for v in sub_verdicts]
        avg_scores = {}
        
        # Initialize scores
        all_keys = set()
        for v in sub_verdicts:
            all_keys.update(v.confidence_scores.keys())
            
        for key in all_keys:
            scores = [v.confidence_scores.get(key, 0.0) for v in sub_verdicts]
            avg_scores[key] = sum(scores) / len(scores)
            
        # Determine label logic
        if all(l == "supported" for l in labels):
            final_label = "supported"
        elif all(l == "refuted" for l in labels):
            final_label = "refuted"
        elif all(l == "not_enough_info" for l in labels):
            final_label = "not_enough_info"
        elif "refuted" in labels and "supported" in labels:
            final_label = "mixed"
        elif "supported" in labels:
            final_label = "partially_supported"
        elif "refuted" in labels:
            final_label = "mixed" # Partially refuted is often mixed
        else:
            final_label = "context_dependent"
            
        return Verdict(
            claim_id=sub_verdicts[0].claim_id, # Use first claim ID as primary
            label=final_label,
            confidence_scores=avg_scores,
            explanation=f"Aggregated from {len(sub_verdicts)} sub-claims: {', '.join(labels)}"
        )
