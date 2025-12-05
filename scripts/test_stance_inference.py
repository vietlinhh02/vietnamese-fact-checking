"""
Test stance detection inference pipeline.
"""

import logging
import argparse
from src.stance_detector import StanceDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to test stance detection."""
    parser = argparse.ArgumentParser(
        description="Test stance detection inference"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/xlmr_stance/best_model",
        help="Path to trained model"
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    detector = StanceDetector(model_path=args.model_path)
    
    # Test cases
    test_cases = [
        # Vietnamese-Vietnamese pairs
        {
            "claim": "Việt Nam có dân số khoảng 98 triệu người",
            "evidence": "Theo số liệu thống kê, dân số Việt Nam đạt 98 triệu người",
            "expected": "support",
            "claim_lang": "vi",
            "evidence_lang": "vi"
        },
        {
            "claim": "Thủ đô của Việt Nam là Hà Nội",
            "evidence": "Thủ đô của Việt Nam là Thành phố Hồ Chí Minh, không phải Hà Nội",
            "expected": "refute",
            "claim_lang": "vi",
            "evidence_lang": "vi"
        },
        {
            "claim": "GDP Việt Nam năm 2023 đạt 430 tỷ USD",
            "evidence": "Việt Nam là một quốc gia ở Đông Nam Á với nền kinh tế đang phát triển",
            "expected": "neutral",
            "claim_lang": "vi",
            "evidence_lang": "vi"
        },
        # Vietnamese-English pairs
        {
            "claim": "Việt Nam có dân số khoảng 98 triệu người",
            "evidence": "According to statistics, Vietnam's population reaches 98 million people",
            "expected": "support",
            "claim_lang": "vi",
            "evidence_lang": "en"
        },
        {
            "claim": "Thủ đô của Việt Nam là Hà Nội",
            "evidence": "The capital of Vietnam is Ho Chi Minh City, not Hanoi",
            "expected": "refute",
            "claim_lang": "vi",
            "evidence_lang": "en"
        }
    ]
    
    logger.info(f"\nTesting {len(test_cases)} examples:")
    logger.info("=" * 80)
    
    correct = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        claim = test_case["claim"]
        evidence = test_case["evidence"]
        expected = test_case["expected"]
        claim_lang = test_case["claim_lang"]
        evidence_lang = test_case["evidence_lang"]
        
        # Run inference
        result = detector.detect_stance(
            claim=claim,
            evidence=evidence,
            claim_lang=claim_lang,
            evidence_lang=evidence_lang
        )
        
        # Check result
        is_correct = result.stance == expected
        if is_correct:
            correct += 1
        
        # Print results
        logger.info(f"\nTest {i}:")
        logger.info(f"Claim ({claim_lang}): {claim}")
        logger.info(f"Evidence ({evidence_lang}): {evidence}")
        logger.info(f"Expected: {expected}")
        logger.info(f"Predicted: {result.stance} {'✓' if is_correct else '✗'}")
        logger.info(f"Confidence scores:")
        for stance, score in result.confidence_scores.items():
            logger.info(f"  {stance}: {score:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
    
    # Test batch inference
    logger.info("\n" + "=" * 80)
    logger.info("Testing batch inference:")
    
    pairs = [(tc["claim"], tc["evidence"]) for tc in test_cases]
    batch_results = detector.batch_detect_stance(pairs)
    
    logger.info(f"Processed {len(batch_results)} pairs in batch")
    for i, result in enumerate(batch_results, 1):
        logger.info(f"  Pair {i}: {result.stance} (confidence: {result.confidence_scores[result.stance]:.4f})")


if __name__ == "__main__":
    main()
