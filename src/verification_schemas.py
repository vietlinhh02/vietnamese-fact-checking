"""JSON Schemas for structured self-verification outputs using Gemini."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


class VerificationSchemas:
    """JSON Schemas for Gemini structured outputs in self-verification."""
    
    @staticmethod
    def get_claim_verification_schema() -> Dict[str, Any]:
        """Schema for individual claim verification result."""
        return {
            "type": "object",
            "properties": {
                "claim_text": {
                    "type": "string",
                    "description": "The text of the claim being verified"
                },
                "is_verified": {
                    "type": "boolean",
                    "description": "Whether the claim is verified by evidence"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score for the verification (0-1)"
                },
                "verification_method": {
                    "type": "string",
                    "enum": ["evidence_match", "search_verification", "relaxed_threshold", "no_verification"],
                    "description": "Method used for verification"
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of the verification result"
                },
                "supporting_evidence_count": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Number of supporting evidence pieces found"
                }
            },
            "required": ["claim_text", "is_verified", "confidence", "verification_method", "explanation", "supporting_evidence_count"]
        }
    
    @staticmethod
    def get_quality_score_schema() -> Dict[str, Any]:
        """Schema for quality score assessment."""
        return {
            "type": "object",
            "properties": {
                "overall_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Overall quality score (0-1)"
                },
                "verification_rate": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Rate of verified claims (0-1)"
                },
                "verified_claims": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Number of verified claims"
                },
                "total_claims": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Total number of claims"
                },
                "flagged_claims": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Number of flagged (unverified) claims"
                },
                "quality_level": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                    "description": "Quality level assessment"
                },
                "confidence_scores": {
                    "type": "object",
                    "description": "Confidence scores by verification method",
                    "additionalProperties": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "explanation": {
                    "type": "string",
                    "description": "Detailed explanation of quality assessment"
                }
            },
            "required": ["overall_score", "verification_rate", "verified_claims", "total_claims", "flagged_claims", "quality_level", "confidence_scores", "explanation"]
        }
    
    @staticmethod
    def get_verification_summary_schema() -> Dict[str, Any]:
        """Schema for complete verification summary."""
        return {
            "type": "object",
            "properties": {
                "quality_assessment": VerificationSchemas.get_quality_score_schema(),
                "verification_results": {
                    "type": "array",
                    "items": VerificationSchemas.get_claim_verification_schema(),
                    "description": "List of individual claim verification results"
                },
                "correction_applied": {
                    "type": "boolean",
                    "description": "Whether correction was applied to the explanation"
                },
                "correction_strategy": {
                    "type": "string",
                    "enum": ["flag", "revise", "remove", "adaptive", "none"],
                    "description": "Strategy used for correction"
                },
                "original_length": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Length of original explanation in characters"
                },
                "corrected_length": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Length of corrected explanation in characters"
                },
                "length_change": {
                    "type": "integer",
                    "description": "Change in length (corrected - original)"
                },
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of recommendations for improvement"
                }
            },
            "required": ["quality_assessment", "verification_results", "correction_applied", "correction_strategy", "original_length", "corrected_length", "length_change", "recommendations"]
        }
    
    @staticmethod
    def get_fact_check_explanation_schema() -> Dict[str, Any]:
        """Schema for structured fact-check explanation."""
        return {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The original claim being fact-checked"
                },
                "verdict": {
                    "type": "string",
                    "enum": ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"],
                    "description": "Final verdict for the claim"
                },
                "confidence_scores": {
                    "type": "object",
                    "properties": {
                        "supported": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence that claim is supported"
                        },
                        "refuted": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence that claim is refuted"
                        },
                        "not_enough_info": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence that there's not enough info"
                        }
                    },
                    "required": ["supported", "refuted", "not_enough_info"],
                    "description": "Confidence scores for each verdict option"
                },
                "evidence_summary": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Evidence text"
                            },
                            "source": {
                                "type": "string",
                                "description": "Source of the evidence"
                            },
                            "credibility": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Credibility score of the source"
                            },
                            "stance": {
                                "type": "string",
                                "enum": ["support", "refute", "neutral"],
                                "description": "Stance of evidence toward the claim"
                            }
                        },
                        "required": ["text", "source", "credibility", "stance"]
                    },
                    "description": "Summary of key evidence pieces"
                },
                "reasoning_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_number": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Step number in reasoning process"
                            },
                            "description": {
                                "type": "string",
                                "description": "Description of this reasoning step"
                            },
                            "evidence_used": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Evidence pieces used in this step"
                            }
                        },
                        "required": ["step_number", "description", "evidence_used"]
                    },
                    "description": "Step-by-step reasoning process"
                },
                "contradictory_evidence": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Evidence that contradicts the claim"
                },
                "verification_metadata": VerificationSchemas.get_verification_summary_schema(),
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Source title"
                            },
                            "url": {
                                "type": "string",
                                "format": "uri",
                                "description": "Source URL"
                            },
                            "credibility": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Source credibility score"
                            }
                        },
                        "required": ["title", "url", "credibility"]
                    },
                    "description": "List of sources used for verification"
                }
            },
            "required": ["claim", "verdict", "confidence_scores", "evidence_summary", "reasoning_steps", "contradictory_evidence", "verification_metadata", "sources"]
        }


@dataclass
class StructuredVerificationResult:
    """Structured verification result for Gemini output."""
    quality_assessment: Dict[str, Any]
    verification_results: List[Dict[str, Any]]
    correction_applied: bool
    correction_strategy: str
    original_length: int
    corrected_length: int
    length_change: int
    recommendations: List[str]


def create_verification_prompt_with_schema(
    explanation: str,
    evidence_summary: str,
    schema_type: str = "summary"
) -> tuple[str, Dict[str, Any]]:
    """Create prompt and schema for structured verification output.
    
    Args:
        explanation: Explanation text to verify
        evidence_summary: Summary of available evidence
        schema_type: Type of schema to use ("summary", "quality", "claim")
        
    Returns:
        Tuple of (prompt, schema)
    """
    schemas = VerificationSchemas()
    
    if schema_type == "summary":
        schema = schemas.get_verification_summary_schema()
    elif schema_type == "quality":
        schema = schemas.get_quality_score_schema()
    elif schema_type == "claim":
        schema = schemas.get_claim_verification_schema()
    elif schema_type == "fact_check":
        schema = schemas.get_fact_check_explanation_schema()
    else:
        raise ValueError(f"Unknown schema type: {schema_type}")
    
    prompt = f"""
Analyze the following explanation for factual accuracy and provide a structured verification report.

EXPLANATION TO VERIFY:
{explanation}

AVAILABLE EVIDENCE:
{evidence_summary}

Please provide a comprehensive verification analysis including:
1. Quality assessment with overall score and verification rate
2. Individual claim verification results with confidence scores
3. Correction recommendations and strategies
4. Detailed explanations for all assessments

Focus on identifying potential hallucinations, unsupported claims, and providing actionable recommendations for improvement.
"""
    
    return prompt, schema


if __name__ == "__main__":
    # Example usage
    schemas = VerificationSchemas()
    
    # Test schema generation
    summary_schema = schemas.get_verification_summary_schema()
    print("Verification Summary Schema:")
    print(f"Properties: {len(summary_schema['properties'])}")
    print(f"Required fields: {summary_schema['required']}")
    
    # Test prompt creation
    prompt, schema = create_verification_prompt_with_schema(
        "Việt Nam có 63 tỉnh thành và GDP 430 tỷ USD",
        "Evidence shows 63 provinces and GDP data",
        "summary"
    )
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"Schema type: {schema['type']}")