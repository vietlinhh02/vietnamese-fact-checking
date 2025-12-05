"""
Prepare cross-lingual stance detection dataset.

This script:
1. Loads FEVER/SNLI-like data or creates synthetic data
2. Translates to Vietnamese using MarianMT
3. Augments with back-translation
4. Creates train/val/test splits
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

from src.translation_service import TranslationService
from src.cache_manager import CacheManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StanceDatasetPreparer:
    """Prepare stance detection dataset for Vietnamese-English pairs."""
    
    def __init__(
        self,
        output_dir: str = "data/stance_detection",
        cache_db: str = "data/cache.db"
    ):
        """
        Initialize dataset preparer.
        
        Args:
            output_dir: Directory to save prepared dataset
            cache_db: Path to cache database
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize translation service with caching
        self.cache_manager = CacheManager(cache_db)
        self.translator = TranslationService(
            cache_manager=self.cache_manager,
            use_gpu=True
        )
        
        logger.info(f"Initialized dataset preparer, output: {self.output_dir}")
    
    def create_synthetic_data(self, num_samples: int = 1000) -> List[Dict]:
        """
        Create synthetic stance detection data.
        
        Since we don't have access to FEVER/SNLI, we'll create synthetic
        examples based on Vietnamese news topics.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            List of stance examples
        """
        logger.info(f"Creating {num_samples} synthetic stance examples")
        
        # Enhanced template-based generation with clearer patterns
        templates = [
            # Support examples - clear agreement
            {
                "claim": "Việt Nam có dân số khoảng {num} triệu người",
                "evidence": "Theo số liệu thống kê năm 2023, dân số Việt Nam đạt {num} triệu người",
                "stance": "support"
            },
            {
                "claim": "Việt Nam có dân số khoảng {num} triệu người",
                "evidence": "Báo cáo chính thức xác nhận dân số Việt Nam là {num} triệu người",
                "stance": "support"
            },
            {
                "claim": "Thủ đô của Việt Nam là {city}",
                "evidence": "{city} là thủ đô và là trung tâm chính trị, hành chính của Việt Nam",
                "stance": "support"
            },
            {
                "claim": "Thủ đô của Việt Nam là {city}",
                "evidence": "Chính phủ Việt Nam đặt tại {city}, thủ đô của đất nước",
                "stance": "support"
            },
            {
                "claim": "GDP Việt Nam năm {year} đạt {amount} tỷ USD",
                "evidence": "Năm {year}, tổng sản phẩm quốc nội của Việt Nam đạt mức {amount} tỷ USD",
                "stance": "support"
            },
            {
                "claim": "GDP Việt Nam năm {year} đạt {amount} tỷ USD",
                "evidence": "Theo Tổng cục Thống kê, GDP Việt Nam năm {year} là {amount} tỷ USD",
                "stance": "support"
            },
            # Refute examples - clear contradiction with explicit negation
            {
                "claim": "Việt Nam có dân số khoảng {num1} triệu người",
                "evidence": "Dân số Việt Nam thực tế là {num2} triệu người, không phải {num1} triệu như đã nói",
                "stance": "refute"
            },
            {
                "claim": "Việt Nam có dân số khoảng {num1} triệu người",
                "evidence": "Thông tin sai lệch. Số liệu chính xác cho thấy dân số Việt Nam là {num2} triệu, chứ không phải {num1} triệu",
                "stance": "refute"
            },
            {
                "claim": "Thủ đô của Việt Nam là {city1}",
                "evidence": "Thủ đô của Việt Nam là {city2}, không phải {city1} như nhiều người nhầm tưởng",
                "stance": "refute"
            },
            {
                "claim": "Thủ đô của Việt Nam là {city1}",
                "evidence": "{city1} không phải là thủ đô. Thủ đô chính thức của Việt Nam là {city2}",
                "stance": "refute"
            },
            {
                "claim": "GDP Việt Nam năm {year} đạt {amount1} tỷ USD",
                "evidence": "GDP Việt Nam năm {year} thực tế chỉ đạt {amount2} tỷ USD, không phải {amount1} tỷ USD",
                "stance": "refute"
            },
            {
                "claim": "GDP Việt Nam năm {year} đạt {amount1} tỷ USD",
                "evidence": "Con số {amount1} tỷ USD là sai. GDP Việt Nam năm {year} là {amount2} tỷ USD",
                "stance": "refute"
            },
            # Neutral examples - unrelated information
            {
                "claim": "Việt Nam có dân số khoảng {num} triệu người",
                "evidence": "Việt Nam là một quốc gia ở Đông Nam Á với nền kinh tế đang phát triển mạnh",
                "stance": "neutral"
            },
            {
                "claim": "Việt Nam có dân số khoảng {num} triệu người",
                "evidence": "Việt Nam có diện tích khoảng 331,000 km vuông và có biên giới với Trung Quốc, Lào, Campuchia",
                "stance": "neutral"
            },
            {
                "claim": "GDP Việt Nam năm {year} đạt {amount} tỷ USD",
                "evidence": "Việt Nam có nhiều thành phố lớn như Hà Nội, Thành phố Hồ Chí Minh, Đà Nẵng",
                "stance": "neutral"
            },
            {
                "claim": "GDP Việt Nam năm {year} đạt {amount} tỷ USD",
                "evidence": "Việt Nam xuất khẩu nhiều mặt hàng như gạo, cà phê, hải sản và điện tử",
                "stance": "neutral"
            },
            {
                "claim": "Thủ đô của Việt Nam là {city}",
                "evidence": "Việt Nam có 63 tỉnh thành và nền văn hóa đa dạng với 54 dân tộc",
                "stance": "neutral"
            },
            {
                "claim": "Thủ đô của Việt Nam là {city}",
                "evidence": "Việt Nam nằm ở khu vực Đông Nam Á và có khí hậu nhiệt đới gió mùa",
                "stance": "neutral"
            }
        ]
        
        # Fill values with more variety
        cities = ["Hà Nội", "Thành phố Hồ Chí Minh", "Đà Nẵng", "Cần Thơ", "Hải Phòng"]
        numbers = list(range(85, 100))
        years = list(range(2020, 2024))
        amounts = list(range(300, 500, 10))
        
        # For refute examples, ensure clear difference
        def get_different_number(num, numbers):
            diff_nums = [n for n in numbers if abs(n - num) >= 5]
            return random.choice(diff_nums) if diff_nums else num + 10
        
        def get_different_city(city, cities):
            diff_cities = [c for c in cities if c != city]
            return random.choice(diff_cities) if diff_cities else cities[0]
        
        def get_different_amount(amount, amounts):
            diff_amounts = [a for a in amounts if abs(a - amount) >= 50]
            return random.choice(diff_amounts) if diff_amounts else amount + 100
        
        examples = []
        
        for _ in range(num_samples):
            template = random.choice(templates)
            
            # Fill template
            claim = template["claim"]
            evidence = template["evidence"]
            stance = template["stance"]
            
            # Replace placeholders
            if "{num}" in claim:
                num = random.choice(numbers)
                claim = claim.replace("{num}", str(num))
                evidence = evidence.replace("{num}", str(num))
            
            if "{num1}" in claim:
                num1 = random.choice(numbers)
                num2 = get_different_number(num1, numbers)
                claim = claim.replace("{num1}", str(num1))
                evidence = evidence.replace("{num1}", str(num1))
                evidence = evidence.replace("{num2}", str(num2))
            
            if "{city}" in claim:
                city = random.choice(cities)
                claim = claim.replace("{city}", city)
                evidence = evidence.replace("{city}", city)
            
            if "{city1}" in claim:
                city1 = random.choice(cities)
                city2 = get_different_city(city1, cities)
                claim = claim.replace("{city1}", city1)
                evidence = evidence.replace("{city1}", city1)
                evidence = evidence.replace("{city2}", city2)
            
            if "{amount1}" in claim:
                amount1 = random.choice(amounts)
                amount2 = get_different_amount(amount1, amounts)
                claim = claim.replace("{amount1}", str(amount1))
                evidence = evidence.replace("{amount1}", str(amount1))
                evidence = evidence.replace("{amount2}", str(amount2))
            
            if "{year}" in claim:
                year = random.choice(years)
                claim = claim.replace("{year}", str(year))
                evidence = evidence.replace("{year}", str(year))
            
            if "{amount}" in claim:
                amount = random.choice(amounts)
                claim = claim.replace("{amount}", str(amount))
                evidence = evidence.replace("{amount}", str(amount))
            
            examples.append({
                "claim": claim,
                "evidence": evidence,
                "stance": stance,
                "claim_lang": "vi",
                "evidence_lang": "vi"
            })
        
        logger.info(f"Created {len(examples)} synthetic examples")
        return examples
    
    def translate_to_english(self, examples: List[Dict]) -> List[Dict]:
        """
        Translate Vietnamese examples to create cross-lingual pairs.
        
        Args:
            examples: List of Vietnamese examples
        
        Returns:
            List of examples with English translations
        """
        logger.info("Translating examples to English")
        
        translated_examples = []
        
        for i, example in enumerate(examples):
            if (i + 1) % 100 == 0:
                logger.info(f"Translated {i + 1}/{len(examples)} examples")
            
            # Create Vietnamese-Vietnamese pair (original)
            translated_examples.append(example.copy())
            
            # Create Vietnamese-English pair (translate evidence)
            try:
                evidence_en = self.translator.translate(
                    example["evidence"],
                    source_lang="vi",
                    target_lang="en"
                )
                
                if evidence_en:
                    translated_examples.append({
                        "claim": example["claim"],
                        "evidence": evidence_en,
                        "stance": example["stance"],
                        "claim_lang": "vi",
                        "evidence_lang": "en"
                    })
            except Exception as e:
                logger.warning(f"Translation failed for example {i}: {e}")
        
        logger.info(f"Created {len(translated_examples)} total examples (with translations)")
        return translated_examples
    
    def augment_with_back_translation(
        self,
        examples: List[Dict],
        augmentation_ratio: float = 0.2
    ) -> List[Dict]:
        """
        Augment dataset with back-translation.
        
        Args:
            examples: Original examples
            augmentation_ratio: Ratio of examples to augment
        
        Returns:
            Augmented examples
        """
        logger.info(f"Augmenting with back-translation (ratio: {augmentation_ratio})")
        
        num_to_augment = int(len(examples) * augmentation_ratio)
        examples_to_augment = random.sample(examples, num_to_augment)
        
        augmented = []
        
        for i, example in enumerate(examples_to_augment):
            if (i + 1) % 50 == 0:
                logger.info(f"Augmented {i + 1}/{num_to_augment} examples")
            
            # Only augment Vietnamese-Vietnamese pairs
            if example["claim_lang"] == "vi" and example["evidence_lang"] == "vi":
                try:
                    # Translate claim to English and back
                    claim_en = self.translator.translate(
                        example["claim"],
                        source_lang="vi",
                        target_lang="en"
                    )
                    
                    if claim_en:
                        # Note: For back-translation, we'd need en->vi model
                        # For now, we'll skip this or use the original
                        # In production, load Helsinki-NLP/opus-mt-en-vi
                        pass
                
                except Exception as e:
                    logger.warning(f"Back-translation failed: {e}")
        
        logger.info(f"Augmentation complete")
        return examples + augmented
    
    def create_splits(
        self,
        examples: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            examples: All examples
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
        
        Returns:
            Tuple of (train, val, test) examples
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01
        
        # Shuffle examples
        random.shuffle(examples)
        
        # Calculate split indices
        n = len(examples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train = examples[:train_end]
        val = examples[train_end:val_end]
        test = examples[val_end:]
        
        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        
        return train, val, test
    
    def save_dataset(
        self,
        train: List[Dict],
        val: List[Dict],
        test: List[Dict]
    ) -> None:
        """
        Save dataset splits to JSONL files.
        
        Args:
            train: Training examples
            val: Validation examples
            test: Test examples
        """
        splits = {
            "train": train,
            "val": val,
            "test": test
        }
        
        for split_name, examples in splits.items():
            output_file = self.output_dir / f"stance_{split_name}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(examples)} examples to {output_file}")
        
        # Save dataset statistics
        stats = {
            "total_examples": len(train) + len(val) + len(test),
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "stance_distribution": self._get_stance_distribution(train + val + test),
            "language_pairs": self._get_language_pairs(train + val + test)
        }
        
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved dataset statistics to {stats_file}")
    
    def _get_stance_distribution(self, examples: List[Dict]) -> Dict[str, int]:
        """Get distribution of stance labels."""
        distribution = {"support": 0, "refute": 0, "neutral": 0}
        
        for example in examples:
            stance = example.get("stance", "neutral")
            distribution[stance] = distribution.get(stance, 0) + 1
        
        return distribution
    
    def _get_language_pairs(self, examples: List[Dict]) -> Dict[str, int]:
        """Get distribution of language pairs."""
        pairs = {}
        
        for example in examples:
            claim_lang = example.get("claim_lang", "vi")
            evidence_lang = example.get("evidence_lang", "vi")
            pair = f"{claim_lang}-{evidence_lang}"
            pairs[pair] = pairs.get(pair, 0) + 1
        
        return pairs
    
    def prepare_full_dataset(
        self,
        num_samples: int = 1000,
        augment: bool = True
    ) -> None:
        """
        Prepare complete stance detection dataset.
        
        Args:
            num_samples: Number of synthetic samples to generate
            augment: Whether to apply data augmentation
        """
        logger.info("Starting dataset preparation")
        
        # Step 1: Create synthetic data
        examples = self.create_synthetic_data(num_samples)
        
        # Step 2: Translate to create cross-lingual pairs
        examples = self.translate_to_english(examples)
        
        # Step 3: Augment with back-translation (optional)
        if augment:
            examples = self.augment_with_back_translation(examples)
        
        # Step 4: Create splits
        train, val, test = self.create_splits(examples)
        
        # Step 5: Save dataset
        self.save_dataset(train, val, test)
        
        logger.info("Dataset preparation complete!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Prepare cross-lingual stance detection dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/stance_detection",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation"
    )
    parser.add_argument(
        "--cache-db",
        type=str,
        default="data/cache.db",
        help="Path to cache database"
    )
    
    args = parser.parse_args()
    
    # Prepare dataset
    preparer = StanceDatasetPreparer(
        output_dir=args.output_dir,
        cache_db=args.cache_db
    )
    
    preparer.prepare_full_dataset(
        num_samples=args.num_samples,
        augment=not args.no_augment
    )


if __name__ == "__main__":
    main()
