"""
Test Phase 6: Claim Detection Module
Kiểm tra model PhoBERT đã train có hoạt động đúng không

Requirements test:
- 1.1: Identify sentences containing verifiable factual claims
- 1.2: Distinguish between factual claims and opinions/questions/commands
- 1.5: Classify with confidence scores
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Cấu hình
MODEL_PATH = "models/phobert_claim_detection"
TEST_DATA_PATH = "data/claim_detection/claim_detection_test.jsonl"
MAX_LENGTH = 128

class ClaimDetector:
    """
    Claim Detection Module
    Phát hiện câu nào là claim cần kiểm chứng
    """
    
    def __init__(self, model_path):
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
    
    def predict(self, text, return_confidence=True):
        """
        Dự đoán một câu có phải claim không
        
        Args:
            text: Câu cần phân loại
            return_confidence: Trả về confidence score
        
        Returns:
            label: 'claim' hoặc 'non-claim'
            confidence: Độ tin cậy (nếu return_confidence=True)
        """
        encoding = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        
        label = 'claim' if pred == 1 else 'non-claim'
        confidence = probs[0][pred].item()
        
        if return_confidence:
            return label, confidence
        return label
    
    def predict_batch(self, texts, batch_size=32):
        """
        Dự đoán nhiều câu cùng lúc
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                label, conf = self.predict(text)
                results.append({'text': text, 'label': label, 'confidence': conf})
        return results

def load_test_data(file_path):
    """Load test data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def test_requirement_1_1(detector):
    """
    Test Requirement 1.1: Identify sentences containing verifiable factual claims
    """
    print("\n" + "="*70)
    print("TEST REQUIREMENT 1.1: Identify Verifiable Factual Claims")
    print("="*70)
    
    test_cases = [
        # Claims với thông tin cụ thể
        ("Việt Nam có 54 dân tộc anh em", "claim"),
        ("Dân số Hà Nội năm 2023 đạt 8 triệu người", "claim"),
        ("TP HCM là thành phố lớn nhất Việt Nam", "claim"),
        ("Tổng thống Mỹ Joe Biden sinh năm 1942", "claim"),
        ("Chiến tranh thế giới thứ 2 kết thúc năm 1945", "claim"),
    ]
    
    correct = 0
    for text, expected in test_cases:
        label, conf = detector.predict(text)
        is_correct = (label == expected)
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"\n{status} Text: {text}")
        print(f"  Expected: {expected}, Got: {label} (conf: {conf:.3f})")
    
    accuracy = correct / len(test_cases)
    print(f"\n{'='*70}")
    print(f"Requirement 1.1 Result: {correct}/{len(test_cases)} = {accuracy*100:.1f}%")
    print(f"Status: {'PASS ✓' if accuracy >= 0.8 else 'FAIL ✗'}")
    
    return accuracy >= 0.8

def test_requirement_1_2(detector):
    """
    Test Requirement 1.2: Distinguish between claims and opinions/questions/commands
    """
    print("\n" + "="*70)
    print("TEST REQUIREMENT 1.2: Distinguish Claims from Non-Claims")
    print("="*70)
    
    test_cases = [
        # Non-claims: Questions
        ("Bạn có biết điều này không?", "non-claim"),
        ("Tại sao lại như vậy?", "non-claim"),
        ("Khi nào sự kiện này diễn ra?", "non-claim"),
        
        # Non-claims: Opinions
        ("Tôi nghĩ rằng đây là quyết định đúng đắn", "non-claim"),
        ("Có lẽ chúng ta nên chờ đợi thêm thông tin", "non-claim"),
        ("Theo tôi, vấn đề này cần xem xét kỹ hơn", "non-claim"),
        
        # Non-claims: Commands
        ("Hãy xem xét kỹ lưỡng vấn đề này", "non-claim"),
        ("Đừng quên kiểm tra thông tin trước khi chia sẻ", "non-claim"),
        
        # Claims
        ("Việt Nam có 54 dân tộc", "claim"),
        ("Dân số thế giới đạt 8 tỷ người năm 2022", "claim"),
    ]
    
    correct = 0
    for text, expected in test_cases:
        label, conf = detector.predict(text)
        is_correct = (label == expected)
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"\n{status} Text: {text}")
        print(f"  Expected: {expected}, Got: {label} (conf: {conf:.3f})")
    
    accuracy = correct / len(test_cases)
    print(f"\n{'='*70}")
    print(f"Requirement 1.2 Result: {correct}/{len(test_cases)} = {accuracy*100:.1f}%")
    print(f"Status: {'PASS ✓' if accuracy >= 0.8 else 'FAIL ✗'}")
    
    return accuracy >= 0.8

def test_requirement_1_5(detector):
    """
    Test Requirement 1.5: Output confidence scores
    """
    print("\n" + "="*70)
    print("TEST REQUIREMENT 1.5: Confidence Scores")
    print("="*70)
    
    test_cases = [
        "Việt Nam có 54 dân tộc",
        "Bạn có biết điều này không?",
        "Tôi nghĩ đây là quyết định đúng",
    ]
    
    all_have_confidence = True
    for text in test_cases:
        label, conf = detector.predict(text)
        has_conf = (0 <= conf <= 1)
        all_have_confidence = all_have_confidence and has_conf
        
        status = "✓" if has_conf else "✗"
        print(f"\n{status} Text: {text}")
        print(f"  Label: {label}, Confidence: {conf:.3f}")
        print(f"  Valid confidence: {has_conf}")
    
    print(f"\n{'='*70}")
    print(f"Requirement 1.5 Result: All outputs have valid confidence scores")
    print(f"Status: {'PASS ✓' if all_have_confidence else 'FAIL ✗'}")
    
    return all_have_confidence

def test_on_test_set(detector, test_data_path):
    """
    Test trên toàn bộ test set
    """
    print("\n" + "="*70)
    print("TEST ON FULL TEST SET")
    print("="*70)
    
    # Load test data
    test_df = load_test_data(test_data_path)
    print(f"\nTest set size: {len(test_df)}")
    print(f"Label distribution:")
    print(test_df['label'].value_counts())
    
    # Predict
    print("\nPredicting...")
    predictions = []
    confidences = []
    
    for idx, row in test_df.iterrows():
        label, conf = detector.predict(row['text'])
        predictions.append(label)
        confidences.append(conf)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(test_df)}...")
    
    # Convert labels
    true_labels = [1 if l == 'claim' else 0 for l in test_df['label']]
    pred_labels = [1 if l == 'claim' else 0 for l in predictions]
    
    # Calculate metrics
    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    print(f"\n{'='*70}")
    print("METRICS:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\n{'='*70}")
    print("CLASSIFICATION REPORT:")
    print(classification_report(true_labels, pred_labels, target_names=['non-claim', 'claim']))
    
    # Confidence distribution
    avg_conf = sum(confidences) / len(confidences)
    print(f"\nAverage confidence: {avg_conf:.3f}")
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_confidence': avg_conf
    }

def main():
    print("="*70)
    print("PHASE 6 TEST: CLAIM DETECTION MODULE")
    print("="*70)
    
    # Load model
    detector = ClaimDetector(MODEL_PATH)
    
    # Test requirements
    results = {}
    
    # Test 1.1
    results['req_1.1'] = test_requirement_1_1(detector)
    
    # Test 1.2
    results['req_1.2'] = test_requirement_1_2(detector)
    
    # Test 1.5
    results['req_1.5'] = test_requirement_1_5(detector)
    
    # Test on full test set
    test_metrics = test_on_test_set(detector, TEST_DATA_PATH)
    results['test_metrics'] = test_metrics
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\nRequirements Test:")
    print(f"  1.1 Identify Claims: {'PASS ✓' if results['req_1.1'] else 'FAIL ✗'}")
    print(f"  1.2 Distinguish Types: {'PASS ✓' if results['req_1.2'] else 'FAIL ✗'}")
    print(f"  1.5 Confidence Scores: {'PASS ✓' if results['req_1.5'] else 'FAIL ✗'}")
    
    print("\nTest Set Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    
    # Overall pass/fail
    all_pass = all([results['req_1.1'], results['req_1.2'], results['req_1.5']])
    performance_pass = test_metrics['f1'] >= 0.70  # F1 >= 70%
    
    print("\n" + "="*70)
    if all_pass and performance_pass:
        print("✓ PHASE 6 TEST: PASSED")
        print(f"  All requirements met")
        print(f"  F1-Score: {test_metrics['f1']:.4f} (>= 0.70)")
    else:
        print("✗ PHASE 6 TEST: FAILED")
        if not all_pass:
            print("  Some requirements not met")
        if not performance_pass:
            print(f"  F1-Score: {test_metrics['f1']:.4f} (< 0.70)")
    print("="*70)
    
    return results

if __name__ == "__main__":
    results = main()
