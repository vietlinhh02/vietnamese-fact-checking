"""Run evaluation and optional ablation study."""

import argparse
import logging

from src.config import load_config
from src.exa_search_client import ExaSearchClient
from src.credibility_analyzer import CredibilityAnalyzer
from src.web_crawler import WebCrawler
from src.llm_controller import create_llm_controller
from src.self_verification import SelfVerificationModule
from src.evaluation_framework import (
    AblationStudyRunner,
    ComponentConfig,
    EvaluationConfig,
    EvaluationPipeline,
    ReActPredictor
)
from src.experiment_tracking import ExperimentTracker


def build_predictor(system_config, use_selenium: bool) -> ReActPredictor:
    llm_controller = create_llm_controller(
        {
            "providers": [system_config.model.llm_provider],
            "gemini_api_key": system_config.agent.gemini_api_key,
            "groq_api_key": system_config.agent.groq_api_key,
            "local_model": system_config.model.llm_model
        }
    )
    search_client = ExaSearchClient(
        api_key=system_config.search.exa_api_key,
        max_results=system_config.search.max_search_results,
        search_type=system_config.search.search_type,
        use_context=system_config.search.use_context,
        rate_limit_rpm=system_config.search.rate_limit_rpm
    )
    web_crawler = WebCrawler(config=system_config.search, use_selenium=use_selenium)
    credibility_analyzer = CredibilityAnalyzer(
        mbfc_api_key=None,
        web_crawler=web_crawler
    )
    self_verification = SelfVerificationModule()
    return ReActPredictor(
        llm_controller=llm_controller,
        search_client=search_client,
        web_crawler=web_crawler,
        credibility_analyzer=credibility_analyzer,
        max_iterations=system_config.agent.max_iterations,
        self_verification_module=self_verification
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON/JSONL")
    parser.add_argument("--output-dir", default="experiments/evaluation", help="Output directory")
    parser.add_argument("--run-name", default="evaluation", help="Run name prefix")
    parser.add_argument("--config", default=None, help="Path to config file (yaml/json)")
    parser.add_argument("--max-claims", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--use-selenium", action="store_true", help="Enable Selenium crawling")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    system_config = load_config(args.config)

    predictor = build_predictor(system_config, use_selenium=args.use_selenium)
    tracker = ExperimentTracker(db_path="experiments/experiments.db")
    pipeline = EvaluationPipeline(predictor=predictor, experiment_tracker=tracker)

    eval_config = EvaluationConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        run_name=args.run_name,
        random_seed=system_config.random_seed,
        max_claims=args.max_claims,
        component_config=ComponentConfig()
    )

    if args.ablation:
        ablation = AblationStudyRunner(pipeline)
        ablation_configs = {
            "no_search": ComponentConfig(use_search=False, name="no_search"),
            "no_crawl": ComponentConfig(use_crawl=False, name="no_crawl"),
            "no_credibility": ComponentConfig(use_credibility=False, name="no_credibility"),
            "no_self_verification": ComponentConfig(use_self_verification=False, name="no_self_verification")
        }
        ablation.run(eval_config, ablation_configs)
    else:
        pipeline.run(eval_config)


if __name__ == "__main__":
    main()
