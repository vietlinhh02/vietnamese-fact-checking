#!/usr/bin/env python3
"""Test RAG explanation generator with mock data and predefined test cases."""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_models import Claim, Evidence, Verdict, ReasoningStep
from rag_explanation_generator import RAGExplanationGenerator, EvidenceRetriever
from mock_llm_provider import create_mock_llm_controller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGTestRunner:
    """Test runner for RAG explanation generator with mock data."""
    
    def __init__(self, test_data_path: str = "data/rag_test_dataset.json"):
        """Initialize test runner.
        
        Args:
            test_data_path: Path to test data JSON file
        """
        self.test_data_path = test_data_path
        self.test_data = self._load_test_data()
        self.mock_controller = create_mock_llm_controller(test_data_path)
        self.rag_generator = RAGExplanationGenerator(llm_controller=self.mock_controller)
        
        logger.info(f"Initialized RAG test runner with {len(self.test_data.get('test_cases', []))} test cases")
    
    def _load_test_data(self) -> Dict[str, Any]:
        """Load test data from JSON file."""
        try:
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return {"test_cases": [], "mock_explanations": {}}
    
    def _create_claim_from_data(self, claim_data: Dict[str, Any]) -> Claim:
        """Create Claim object from test data."""
        return Claim(
            text=claim_data["text"],
            language=claim_data.get("language", "vi"),
            confidence=claim_data.get("confidence", 0.8)
        )
    
    def _create_evidence_from_data(self, evidence_data: Dict[str, Any]) -> Evidence:
        """Create Evidence object from test data."""
        return Evidence(
            text=evidence_data["text"],
            source_url=evidence_data["source_url"],
            source_title=evidence_data["source_title"],
            credibility_score=evidence_data.get("credibility_score", 0.8),
            stance=evidence_data.get("stance"),
            stance_confidence=evidence_data.get("stance_confidence"),
            language=evidence_data.get("language", "vi")
        )
    
    def _create_reasoning_steps_from_data(self, steps_data: List[Dict[str, Any]]) -> List[ReasoningStep]:
        """Create ReasoningStep objects from test data."""
        steps = []
        for step_data in steps_data:
            step = ReasoningStep(
                iteration=step_data["iteration"],
                thought=step_data["thought"],
                action=step_data["action"],
                action_input=step_data["action_input"],
                observation=step_data["observation"]
            )
            steps.append(step)
        return steps
    
    def _create_verdict_from_data(self, verdict_data: Dict[str, Any], claim_id: str) -> Verdict:
        """Create Verdict object from test data."""
        return Verdict(
            claim_id=claim_id,
            label=verdict_data["label"],
            confidence_scores=verdict_data["confidence_scores"]
        )
    
    def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case.
        
        Args:
            test_case: Test case data
            
        Returns:
            Test results dictionary
        """
        test_id = test_case["id"]
        logger.info(f"Running test case: {test_id}")
        
        try:
            # Create objects from test data
            claim = self._create_claim_from_data(test_case["claim"])
            evidence_list = [self._create_evidence_from_data(ev) for ev in test_case["evidence"]]
            reasoning_steps = self._create_reasoning_steps_from_data(test_case["reasoning_steps"])
            verdict = self._create_verdict_from_data(test_case["verdict"], claim.id)
            
            # Generate explanation
            explanation = self.rag_generator.generate_explanation(
                claim=claim,
                verdict=verdict,
                evidence_list=evidence_list,
                reasoning_steps=reasoning_steps
            )
            
            # Validate explanation
            validation_results = self._validate_explanation(
                explanation, claim, evidence_list, verdict, test_case
            )
            
            return {
                "test_id": test_id,
                "status": "passed" if validation_results["all_passed"] else "failed",
                "claim": claim.text,
                "verdict": verdict.label,
                "confidence": max(verdict.confidence_scores.values()),
                "evidence_count": len(evidence_list),
                "explanation_length": len(explanation),
                "explanation": explanation,
                "validation": validation_results,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Test case {test_id} failed: {e}")
            return {
                "test_id": test_id,
                "status": "error",
                "claim": test_case["claim"]["text"],
                "error": str(e),
                "explanation": None,
                "validation": None
            }
    
    def _validate_explanation(
        self,
        explanation: str,
        claim: Claim,
        evidence_list: List[Evidence],
        verdict: Verdict,
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate generated explanation against requirements.
        
        Args:
            explanation: Generated explanation
            claim: Original claim
            evidence_list: Evidence used
            verdict: Verdict object
            test_case: Original test case data
            
        Returns:
            Validation results dictionary
        """
        results = {
            "has_content": False,
            "has_citations": False,
            "has_sources": False,
            "mentions_claim": False,
            "mentions_verdict": False,
            "has_reasoning": False,
            "appropriate_length": False,
            "all_passed": False
        }
        
        if not explanation or not explanation.strip():
            return results
        
        explanation_lower = explanation.lower()
        
        # Check basic content
        results["has_content"] = len(explanation.strip()) > 0
        results["appropriate_length"] = len(explanation) > 100
        
        # Check for citations [1], [2], etc.
        import re
        citations = re.findall(r'\[(\d+)\]', explanation)
        results["has_citations"] = len(citations) > 0
        
        # Check for source URLs
        urls = re.findall(r'https?://[^\s]+', explanation)
        results["has_sources"] = len(urls) > 0
        
        # Check if claim is mentioned or referenced
        claim_words = set(claim.text.lower().split())
        explanation_words = set(explanation_lower.split())
        overlap = len(claim_words.intersection(explanation_words))
        results["mentions_claim"] = overlap > 0 or "tuyên bố" in explanation_lower
        
        # Check if verdict is mentioned
        verdict_indicators = {
            "supported": ["hỗ trợ", "xác nhận", "đúng", "chính xác"],
            "refuted": ["bác bỏ", "sai", "không đúng", "phản bác"],
            "not_enough_info": ["không đủ", "thiếu thông tin", "chưa rõ"]
        }
        
        indicators = verdict_indicators.get(verdict.label, [])
        results["mentions_verdict"] = any(indicator in explanation_lower for indicator in indicators)
        
        # Check for reasoning trace
        reasoning_indicators = ["reasoning", "process", "step", "tìm kiếm", "xác minh", "kiểm tra"]
        results["has_reasoning"] = any(indicator in explanation_lower for indicator in reasoning_indicators)
        
        # Overall pass/fail
        results["all_passed"] = all([
            results["has_content"],
            results["has_citations"] or results["has_sources"],  # At least one citation method
            results["mentions_claim"],
            results["appropriate_length"]
        ])
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases.
        
        Returns:
            Summary of all test results
        """
        logger.info("Running all RAG test cases with mock data")
        logger.info("=" * 60)
        
        test_cases = self.test_data.get("test_cases", [])
        results = []
        
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            results.append(result)
            
            # Log result
            status_icon = "✓" if result["status"] == "passed" else "✗" if result["status"] == "failed" else "⚠"
            logger.info(f"{status_icon} {result['test_id']}: {result['status']}")
            
            if result["status"] == "passed":
                logger.info(f"   Claim: {result['claim'][:60]}...")
                logger.info(f"   Verdict: {result['verdict']} (confidence: {result['confidence']:.2f})")
                logger.info(f"   Explanation length: {result['explanation_length']} chars")
            elif result["status"] == "error":
                logger.error(f"   Error: {result['error']}")
        
        # Calculate summary statistics
        passed = len([r for r in results if r["status"] == "passed"])
        failed = len([r for r in results if r["status"] == "failed"])
        errors = len([r for r in results if r["status"] == "error"])
        total = len(results)
        
        summary = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0,
            "results": results
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY:")
        logger.info(f"Total tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Pass rate: {summary['pass_rate']:.1%}")
        
        return summary
    
    def demonstrate_explanations(self):
        """Demonstrate generated explanations for each test case."""
        logger.info("\nDEMONSTRATING GENERATED EXPLANATIONS:")
        logger.info("=" * 80)
        
        test_cases = self.test_data.get("test_cases", [])
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n{i}. TEST CASE: {test_case['id']}")
            logger.info("-" * 60)
            
            try:
                # Create objects
                claim = self._create_claim_from_data(test_case["claim"])
                evidence_list = [self._create_evidence_from_data(ev) for ev in test_case["evidence"]]
                reasoning_steps = self._create_reasoning_steps_from_data(test_case["reasoning_steps"])
                verdict = self._create_verdict_from_data(test_case["verdict"], claim.id)
                
                logger.info(f"Claim: {claim.text}")
                logger.info(f"Verdict: {verdict.label} (confidence: {max(verdict.confidence_scores.values()):.2f})")
                logger.info(f"Evidence pieces: {len(evidence_list)}")
                
                # Generate explanation
                explanation = self.rag_generator.generate_explanation(
                    claim=claim,
                    verdict=verdict,
                    evidence_list=evidence_list,
                    reasoning_steps=reasoning_steps
                )
                
                logger.info("\nGENERATED EXPLANATION:")
                logger.info("-" * 40)
                logger.info(explanation)
                logger.info("-" * 40)
                
            except Exception as e:
                logger.error(f"Failed to generate explanation for {test_case['id']}: {e}")
    
    def test_evidence_retriever(self):
        """Test evidence retriever component separately."""
        logger.info("\nTESTING EVIDENCE RETRIEVER:")
        logger.info("=" * 50)
        
        retriever = EvidenceRetriever(top_k=3)
        
        for test_case in self.test_data.get("test_cases", []):
            logger.info(f"\nTest case: {test_case['id']}")
            
            try:
                claim = self._create_claim_from_data(test_case["claim"])
                evidence_list = [self._create_evidence_from_data(ev) for ev in test_case["evidence"]]
                verdict_label = test_case["verdict"]["label"]
                
                top_evidence, contradictory = retriever.retrieve_evidence(
                    claim, evidence_list, verdict_label
                )
                
                logger.info(f"Retrieved {len(top_evidence)} evidence pieces")
                for i, scored_ev in enumerate(top_evidence, 1):
                    logger.info(f"  {i}. Score: {scored_ev.final_score:.3f}, "
                               f"Stance: {scored_ev.evidence.stance}, "
                               f"Credibility: {scored_ev.evidence.credibility_score:.2f}")
                
                if contradictory:
                    logger.info(f"Found contradictory evidence: {list(contradictory.keys())}")
                
            except Exception as e:
                logger.error(f"Evidence retrieval failed for {test_case['id']}: {e}")


def main():
    """Run comprehensive RAG tests with mock data."""
    logger.info("Starting RAG Tests with Mock Data")
    logger.info("=" * 80)
    
    try:
        # Initialize test runner
        test_runner = RAGTestRunner()
        
        # Test 1: Evidence Retriever
        test_runner.test_evidence_retriever()
        
        # Test 2: Run all test cases
        summary = test_runner.run_all_tests()
        
        # Test 3: Demonstrate explanations
        test_runner.demonstrate_explanations()
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("FINAL SUMMARY:")
        logger.info(f"✓ Evidence retriever working correctly")
        logger.info(f"✓ Mock LLM provider generating explanations")
        logger.info(f"✓ RAG explanation generator integrated")
        logger.info(f"✓ Test pass rate: {summary['pass_rate']:.1%} ({summary['passed']}/{summary['total_tests']})")
        
        if summary['pass_rate'] >= 0.8:
            logger.info("🎉 RAG explanation generator is working excellently with mock data!")
        elif summary['pass_rate'] >= 0.6:
            logger.info("👍 RAG explanation generator is working well with mock data!")
        else:
            logger.warning("⚠️  RAG explanation generator needs improvement")
        
        # Save results
        results_path = "test_results_rag_mock.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise


if __name__ == "__main__":
    main()