"""Data preparation module for claim detection training."""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LabeledSentence:
    """A sentence with its claim/non-claim label."""
    text: str
    is_claim: bool
    sentence_type: str  # "factual_claim", "opinion", "question", "command"
    source: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'text': self.text,
            'is_claim': self.is_claim,
            'sentence_type': self.sentence_type,
            'source': self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "LabeledSentence":
        """Create from dictionary."""
        return cls(
            text=data['text'],
            is_claim=data['is_claim'],
            sentence_type=data['sentence_type'],
            source=data.get('source', '')
        )


class ClaimDatasetBuilder:
    """Builder for claim detection training dataset."""
    
    def __init__(self, data_dir: str = "data/claim_detection"):
        """Initialize dataset builder.
        
        Args:
            data_dir: Directory to store dataset files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_file = self.data_dir / "train.jsonl"
        self.val_file = self.data_dir / "val.jsonl"
        self.test_file = self.data_dir / "test.jsonl"
    
    def create_weak_supervision_dataset(
        self,
        num_samples: int = 1000,
        seed: int = 42
    ) -> Tuple[List[LabeledSentence], List[LabeledSentence], List[LabeledSentence]]:
        """Create a weakly supervised dataset using heuristics.
        
        This generates synthetic examples for initial training.
        In production, this should be replaced with real Vietnamese news data.
        
        Args:
            num_samples: Total number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        random.seed(seed)
        
        # Template-based generation for weak supervision
        claim_templates = [
            "Việt Nam có dân số {number} triệu người.",
            "Thủ đô của {country} là {city}.",
            "{person} là Tổng thống của {country}.",
            "Công ty {company} được thành lập năm {year}.",
            "Nhiệt độ trung bình tại {city} là {number} độ C.",
            "{event} xảy ra vào ngày {date}.",
            "GDP của {country} đạt {number} tỷ USD.",
            "{person} sinh năm {year}.",
            "Diện tích của {country} là {number} km².",
            "{company} có {number} nhân viên.",
        ]
        
        opinion_templates = [
            "Tôi nghĩ rằng {topic} rất quan trọng.",
            "{topic} có vẻ thú vị.",
            "Theo quan điểm của tôi, {topic} cần được cải thiện.",
            "Có lẽ {topic} sẽ tốt hơn trong tương lai.",
            "Tôi cảm thấy {topic} khá phức tạp.",
            "{topic} thật tuyệt vời!",
            "Tôi không chắc về {topic}.",
            "Có thể {topic} là một ý tưởng hay.",
        ]
        
        question_templates = [
            "Dân số của {country} là bao nhiêu?",
            "{person} là ai?",
            "Khi nào {event} xảy ra?",
            "Tại sao {topic} lại quan trọng?",
            "Làm thế nào để {action}?",
            "{company} được thành lập khi nào?",
            "Ở đâu có {thing}?",
        ]
        
        command_templates = [
            "Hãy kiểm tra {topic}.",
            "Vui lòng xác nhận {information}.",
            "Đừng quên {action}.",
            "Hãy đọc thêm về {topic}.",
            "Xem xét {option}.",
        ]
        
        # Generate samples
        all_samples = []
        
        # Generate factual claims
        for _ in range(num_samples // 4):
            template = random.choice(claim_templates)
            text = self._fill_template(template)
            all_samples.append(LabeledSentence(
                text=text,
                is_claim=True,
                sentence_type="factual_claim",
                source="weak_supervision"
            ))
        
        # Generate opinions
        for _ in range(num_samples // 4):
            template = random.choice(opinion_templates)
            text = self._fill_template(template)
            all_samples.append(LabeledSentence(
                text=text,
                is_claim=False,
                sentence_type="opinion",
                source="weak_supervision"
            ))
        
        # Generate questions
        for _ in range(num_samples // 4):
            template = random.choice(question_templates)
            text = self._fill_template(template)
            all_samples.append(LabeledSentence(
                text=text,
                is_claim=False,
                sentence_type="question",
                source="weak_supervision"
            ))
        
        # Generate commands
        for _ in range(num_samples // 4):
            template = random.choice(command_templates)
            text = self._fill_template(template)
            all_samples.append(LabeledSentence(
                text=text,
                is_claim=False,
                sentence_type="command",
                source="weak_supervision"
            ))
        
        # Shuffle and split
        random.shuffle(all_samples)
        
        train_size = int(0.7 * len(all_samples))
        val_size = int(0.15 * len(all_samples))
        
        train_data = all_samples[:train_size]
        val_data = all_samples[train_size:train_size + val_size]
        test_data = all_samples[train_size + val_size:]
        
        logger.info(f"Generated {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
        
        return train_data, val_data, test_data
    
    def _fill_template(self, template: str) -> str:
        """Fill template with random values."""
        replacements = {
            '{number}': str(random.randint(1, 100)),
            '{country}': random.choice(['Việt Nam', 'Mỹ', 'Trung Quốc', 'Nhật Bản', 'Hàn Quốc']),
            '{city}': random.choice(['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Cần Thơ', 'Hải Phòng']),
            '{person}': random.choice(['Nguyễn Văn A', 'Trần Thị B', 'Lê Văn C', 'Phạm Thị D']),
            '{company}': random.choice(['VinGroup', 'FPT', 'Viettel', 'VNPT', 'Masan']),
            '{year}': str(random.randint(1990, 2023)),
            '{date}': f"{random.randint(1, 28)}/{random.randint(1, 12)}/{random.randint(2000, 2023)}",
            '{event}': random.choice(['Hội nghị', 'Sự kiện', 'Lễ hội', 'Cuộc họp']),
            '{topic}': random.choice(['giáo dục', 'y tế', 'kinh tế', 'môi trường', 'công nghệ']),
            '{action}': random.choice(['cải thiện', 'phát triển', 'thực hiện', 'hoàn thành']),
            '{information}': random.choice(['thông tin', 'dữ liệu', 'số liệu', 'báo cáo']),
            '{option}': random.choice(['phương án', 'lựa chọn', 'giải pháp', 'kế hoạch']),
            '{thing}': random.choice(['tài liệu', 'thông tin', 'dữ liệu', 'nguồn lực']),
        }
        
        text = template
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        
        return text
    
    def load_from_news_articles(
        self,
        articles_file: str,
        annotate_fn: Optional[callable] = None
    ) -> List[LabeledSentence]:
        """Load and annotate sentences from Vietnamese news articles.
        
        Args:
            articles_file: Path to file containing news articles
            annotate_fn: Optional function to automatically annotate sentences
            
        Returns:
            List of labeled sentences
        """
        # Placeholder for real implementation
        # In production, this would:
        # 1. Load Vietnamese news articles
        # 2. Split into sentences using Vietnamese sentence tokenizer
        # 3. Either manually annotate or use weak supervision
        # 4. Return labeled sentences
        
        logger.warning("load_from_news_articles not fully implemented - using weak supervision")
        return []
    
    def save_dataset(
        self,
        train_data: List[LabeledSentence],
        val_data: List[LabeledSentence],
        test_data: List[LabeledSentence]
    ) -> None:
        """Save dataset splits to JSONL files.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
        """
        self._save_jsonl(train_data, self.train_file)
        self._save_jsonl(val_data, self.val_file)
        self._save_jsonl(test_data, self.test_file)
        
        logger.info(f"Saved dataset to {self.data_dir}")
        logger.info(f"  Train: {len(train_data)} samples")
        logger.info(f"  Val: {len(val_data)} samples")
        logger.info(f"  Test: {len(test_data)} samples")
    
    def _save_jsonl(self, data: List[LabeledSentence], filepath: Path) -> None:
        """Save data to JSONL file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + '\n')
    
    def load_dataset(self) -> Tuple[List[LabeledSentence], List[LabeledSentence], List[LabeledSentence]]:
        """Load dataset from saved files.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        train_data = self._load_jsonl(self.train_file)
        val_data = self._load_jsonl(self.val_file)
        test_data = self._load_jsonl(self.test_file)
        
        logger.info(f"Loaded dataset from {self.data_dir}")
        logger.info(f"  Train: {len(train_data)} samples")
        logger.info(f"  Val: {len(val_data)} samples")
        logger.info(f"  Test: {len(test_data)} samples")
        
        return train_data, val_data, test_data
    
    def _load_jsonl(self, filepath: Path) -> List[LabeledSentence]:
        """Load data from JSONL file."""
        if not filepath.exists():
            return []
        
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(LabeledSentence.from_dict(item))
        
        return data
    
    def get_statistics(self) -> Dict[str, any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        train_data, val_data, test_data = self.load_dataset()
        
        def count_by_type(data: List[LabeledSentence]) -> Dict[str, int]:
            counts = {}
            for item in data:
                counts[item.sentence_type] = counts.get(item.sentence_type, 0) + 1
            return counts
        
        return {
            'train': {
                'total': len(train_data),
                'by_type': count_by_type(train_data)
            },
            'val': {
                'total': len(val_data),
                'by_type': count_by_type(val_data)
            },
            'test': {
                'total': len(test_data),
                'by_type': count_by_type(test_data)
            }
        }


def prepare_claim_detection_data(
    data_dir: str = "data/claim_detection",
    num_samples: int = 1000,
    seed: int = 42
) -> None:
    """Prepare claim detection training data.
    
    This is the main entry point for data preparation.
    
    Args:
        data_dir: Directory to store dataset
        num_samples: Number of samples to generate
        seed: Random seed
    """
    builder = ClaimDatasetBuilder(data_dir)
    
    # Create weak supervision dataset
    train_data, val_data, test_data = builder.create_weak_supervision_dataset(
        num_samples=num_samples,
        seed=seed
    )
    
    # Save dataset
    builder.save_dataset(train_data, val_data, test_data)
    
    # Print statistics
    stats = builder.get_statistics()
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Example usage
    prepare_claim_detection_data()
