"""Run the complete end-to-end fact-checking pipeline from CLI."""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.data_models import Claim, ReasoningStep as ModelReasoningStep
from src.exa_search_client import ExaSearchClient
from src.credibility_analyzer import CredibilityAnalyzer
from src.llm_controller import create_llm_controller
from src.react_agent import create_react_agent
from src.rag_explanation_generator import RAGExplanationGenerator
from src.web_crawler import WebCrawler, WebContent
from src.cache_manager import CacheManager
from src.search_query_generator import SearchQueryGenerator
from src.claim_decomposer import decompose_claims

logger = logging.getLogger(__name__)


def _simple_sentence_split(text: str) -> List[str]:
    sentences = re.split(r"[.!?]+\\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _load_text(input_file: str) -> str:
    return Path(input_file).read_text(encoding="utf-8")


def _extract_claims(
    claim_text: Optional[str],
    input_file: Optional[str],
    use_claim_detector: bool,
    model_path: Optional[str],
    confidence_threshold: float,
    use_sliding_window: bool,
    max_claims: int
) -> List[Claim]:
    if claim_text:
        return [Claim(text=claim_text, language="vi")]

    if not input_file:
        raise ValueError("Either --claim or --input-file must be provided")

    text = _load_text(input_file)
    claims: List[Claim] = []

    if use_claim_detector:
        try:
            from src.claim_detector import detect_claims_in_text
            claims = detect_claims_in_text(
                text=text,
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                use_sliding_window=use_sliding_window
            )
        except Exception as exc:
            logger.warning("Claim detector failed, using fallback extraction: %s", exc)
            claims = []

    if not claims:
        sentences = _simple_sentence_split(text)
        for sentence in sentences[:max_claims]:
            claims.append(Claim(text=sentence, language="vi"))

    return claims[:max_claims]


def _convert_reasoning_steps(steps) -> List[ModelReasoningStep]:
    converted = []
    for step in steps:
        action = step.action or "none"
        observation = step.observation or "no_observation"
        action_input = step.action_params or {}
        timestamp = step.timestamp.isoformat() if hasattr(step.timestamp, "isoformat") else None
        converted.append(
            ModelReasoningStep(
                iteration=step.step_number,
                thought=step.thought,
                action=action,
                action_input=action_input,
                observation=observation,
                timestamp=timestamp
            )
        )
    return converted


def _score_evidence(evidence_list, credibility_analyzer: CredibilityAnalyzer) -> None:
    for evidence in evidence_list:
        if evidence.credibility_score > 0:
            continue
        content = WebContent(
            url=evidence.source_url,
            title=evidence.source_title or "",
            main_text=evidence.text,
            author=evidence.source_author,
            publish_date=evidence.publish_date,
            extraction_success=True,
            extraction_method="e2e_pipeline"
        )
        try:
            score = credibility_analyzer.analyze_credibility(content)
            evidence.credibility_score = score.overall_score
        except Exception as exc:
            logger.warning("Credibility scoring failed: %s", exc)


def _serialize_result(claim, verdict, evidence_list, reasoning_steps, explanation, verification_metadata):
    return {
        "claim": claim.to_dict(),
        "verdict": verdict.to_dict(),
        "evidence": [ev.to_dict() for ev in evidence_list],
        "reasoning_steps": [step.to_dict() for step in reasoning_steps],
        "explanation": explanation,
        "verification_metadata": verification_metadata or {}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end fact-checking pipeline")
    parser.add_argument("--claim", help="Claim text to verify")
    parser.add_argument("--input-file", help="Path to input text file for claim extraction")
    parser.add_argument("--config", default=None, help="Path to config file (yaml/json)")
    parser.add_argument("--output-dir", default="experiments/end_to_end", help="Output directory")
    parser.add_argument("--run-name", default="e2e_run", help="Run name prefix")
    parser.add_argument("--llm-provider", default=None, help="Override LLM provider (gemini, groq, local_llama, openai_compat)")
    parser.add_argument("--openai-compat-base-url", default=None, help="OpenAI-compatible base URL (e.g. http://localhost:8317)")
    parser.add_argument("--openai-compat-model", default=None, help="OpenAI-compatible model name")
    parser.add_argument("--openai-compat-api-key", default=None, help="OpenAI-compatible API key (optional)")
    parser.add_argument("--max-claims", type=int, default=3, help="Max claims to process")
    parser.add_argument("--use-selenium", action="store_true", help="Enable Selenium crawling")
    parser.add_argument("--use-claim-detector", action="store_true", help="Use claim detector on input file")
    parser.add_argument("--claim-detector-model", default=None, help="Path to claim detector model")
    parser.add_argument("--claim-threshold", type=float, default=0.5, help="Claim detector threshold")
    parser.add_argument("--sliding-window", action="store_true", help="Use sliding window for claim detection")
    parser.add_argument("--no-self-verification", action="store_true", help="Disable self-verification")
    parser.add_argument("--decompose-claims", action="store_true", default=True, help="Decompose compound claims into atomic sub-claims (default: True)")
    parser.add_argument("--no-decompose", action="store_true", help="Disable claim decomposition")
    parser.add_argument("--use-llm-decompose", action="store_true", help="Use LLM for claim decomposition")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    system_config = load_config(args.config)

    provider_name = args.llm_provider or system_config.model.llm_provider
    llm_controller = create_llm_controller(
        {
            "providers": [provider_name],
            "gemini_api_key": system_config.agent.gemini_api_key,
            "groq_api_key": system_config.agent.groq_api_key,
            "local_model": system_config.model.llm_model,
            "openai_compat_base_url": args.openai_compat_base_url,
            "openai_compat_model": args.openai_compat_model,
            "openai_compat_api_key": args.openai_compat_api_key
        }
    )
    
    from src.search_query_generator import SearchQueryGenerator
    
    # Initialize cache manager
    cache_manager = CacheManager(default_ttl_hours=system_config.search.cache_ttl_hours)
    
    search_client = ExaSearchClient(
        api_key=system_config.search.exa_api_key,
        cache_manager=cache_manager,
        max_results=system_config.search.max_search_results,
        search_type=system_config.search.search_type,
        use_context=system_config.search.use_context,
        rate_limit_rpm=system_config.search.rate_limit_rpm
    )
    web_crawler = WebCrawler(config=system_config.search, use_selenium=args.use_selenium)
    credibility_analyzer = CredibilityAnalyzer(web_crawler=web_crawler)
    
    # Initialize query generator
    query_generator = SearchQueryGenerator()
    
    agent = create_react_agent(
        llm_controller=llm_controller,
        search_client=search_client,
        web_crawler=web_crawler,
        credibility_analyzer=credibility_analyzer,
        max_iterations=system_config.agent.max_iterations
    )
    
    # Initialize RAG generator with search capabilities for self-verification
    rag_generator = RAGExplanationGenerator(
        llm_controller=llm_controller,
        enable_self_verification=not args.no_self_verification,
        search_client=search_client,
        web_crawler=web_crawler,
        query_generator=query_generator
    )

    # Auto-detect local model and enable detector if available
    detect_model_path = args.claim_detector_model
    use_detector = args.use_claim_detector
    
    # Check for local model if path not provided
    if detect_model_path is None:
        local_model_path = REPO_ROOT / "models" / "phobert_claim_detection"
        if local_model_path.exists():
            logger.info(f"Found local trained model at {local_model_path}, using as default.")
            detect_model_path = str(local_model_path)
            
            # Enable detector if not explicitly enabled, as we found a trained model
            if not use_detector:
                logger.info("Enabling claim detector because local trained model was found.")
                use_detector = True

    claims = _extract_claims(
        claim_text=args.claim,
        input_file=args.input_file,
        use_claim_detector=use_detector,
        model_path=detect_model_path,
        confidence_threshold=args.claim_threshold,
        use_sliding_window=args.sliding_window,
        max_claims=args.max_claims
    )

    if not claims:
        raise ValueError("No claims detected to process")

    # Decompose compound claims into atomic sub-claims
    should_decompose = args.decompose_claims and not args.no_decompose
    claim_mapping = {}
    
    if should_decompose:
        logger.info("Decomposing compound claims into atomic sub-claims...")
        claims, claim_mapping = decompose_claims(
            claims=claims,
            llm_controller=llm_controller if args.use_llm_decompose else None,
            use_llm=args.use_llm_decompose
        )
        logger.info(f"After decomposition: {len(claims)} atomic claims to verify")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create shared components for parallel processing
    def _process_single_claim(claim_idx_tuple):
        """Process a single claim and return result."""
        idx, claim = claim_idx_tuple
        try:
            logger.info("Processing claim %s/%s: %s", idx, len(claims), claim.text[:120])
            state = agent.verify_claim(claim)
            verdict = state.final_verdict
            if verdict is None:
                logger.warning("No verdict returned for claim %s", idx)
                return None

            evidence_list = state.collected_evidence
            _score_evidence(evidence_list, credibility_analyzer)
            reasoning_steps = _convert_reasoning_steps(state.reasoning_steps)

            verdict.reasoning_trace = reasoning_steps
            explanation, verification_metadata = rag_generator.generate_explanation(
                claim=claim,
                verdict=verdict,
                evidence_list=evidence_list,
                reasoning_steps=reasoning_steps,
                apply_self_verification=not args.no_self_verification
            )
            verdict.explanation = explanation

            # Verdict logic validation and auto-correction
            try:
                from src.verdict_logic_validator import VerdictLogicValidator
                validator = VerdictLogicValidator()
                verdict, was_corrected, correction_reason = validator.auto_correct_verdict(
                    verdict=verdict,
                    evidence_list=evidence_list,
                    claim_text=claim.text
                )
                if was_corrected:
                    logger.info(f"Verdict auto-corrected: {correction_reason}")
                    if verification_metadata:
                        verification_metadata["verdict_corrected"] = True
                        verification_metadata["correction_reason"] = correction_reason
            except ImportError as e:
                logger.warning(f"Verdict validator not available: {e}")

            result = _serialize_result(
                claim=claim,
                verdict=verdict,
                evidence_list=evidence_list,
                reasoning_steps=reasoning_steps,
                explanation=explanation,
                verification_metadata=verification_metadata
            )
            
            # Add parent claim info if this was decomposed
            if claim.parent_claim_id and claim.parent_claim_id in claim_mapping:
                result["parent_claim_info"] = claim_mapping[claim.parent_claim_id]
            
            return result
        except Exception as e:
            logger.error(f"Error processing claim {idx}: {e}")
            return None

    # Determine parallelism - use up to 3 workers for claim processing
    # (LLM and search are the bottlenecks, so more workers may not help much)
    max_claim_workers = min(3, len(claims))
    
    if len(claims) > 1 and max_claim_workers > 1:
        # Parallel processing for multiple claims
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        logger.info(f"Processing {len(claims)} claims in parallel with {max_claim_workers} workers")
        all_results = []
        
        with ThreadPoolExecutor(max_workers=max_claim_workers) as executor:
            # Submit all claim processing tasks with index
            future_to_claim = {
                executor.submit(_process_single_claim, (idx, claim)): (idx, claim)
                for idx, claim in enumerate(claims, 1)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_claim):
                idx, claim = future_to_claim[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                except Exception as e:
                    logger.error(f"Claim {idx} raised exception: {e}")
    else:
        # Sequential processing for single claim or when parallelism disabled
        all_results = []
        for idx, claim in enumerate(claims, 1):
            result = _process_single_claim((idx, claim))
            if result:
                all_results.append(result)

    # Add decomposition summary to output
    output_data = {
        "results": all_results,
        "metadata": {
            "total_claims_processed": len(all_results),
            "decomposition_enabled": should_decompose,
            "claim_decomposition_mapping": claim_mapping if should_decompose else None,
            "parallel_processing": len(claims) > 1 and max_claim_workers > 1
        }
    }

    output_path = output_dir / f"{args.run_name}_results.json"
    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()

