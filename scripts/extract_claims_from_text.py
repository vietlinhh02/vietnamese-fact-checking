"""
Claim Extraction Pipeline
Trích xuất tất cả claims từ một đoạn văn bản dài

Task 6.3: Implement claim extraction pipeline
- Tách văn bản thành câu
- Phân loại từng câu (claim/non-claim)
- Trả về danh sách claims với confidence scores
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

class ClaimExtractor:
    """
    Pipeline trích xuất claims từ văn bản
    """
    
    def __init__(self, model_path="models/phobert_claim_detection", confidence_threshold=0.7):
        """
        Args:
            model_path: Đường dẫn đến model
            confidence_threshold: Ngưỡng confidence để coi là claim
        """
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        print(f"✓ Model loaded on {self.device}")
    
    def split_sentences(self, text):
        """
        Tách văn bản thành các câu
        Xử lý tiếng Việt
        """
        # Thêm khoảng trắng sau dấu câu nếu thiếu
        text = re.sub(r'([.!?])([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ])', r'\1 \2', text)
        
        # Tách câu bằng dấu chấm, chấm hỏi, chấm than
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ"\'])', text)
        
        # Nếu không tách được, thử cách khác
        if len(sentences) <= 1:
            sentences = re.split(r'[.!?]\s+', text)
        
        # Lọc câu rỗng và quá ngắn
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    def predict_sentence(self, text):
        """
        Dự đoán một câu
        Returns: (label, confidence)
        """
        encoding = self.tokenizer(
            text,
            max_length=128,
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
        
        return label, confidence
    
    def extract_claims(self, text, return_all=False):
        """
        Trích xuất claims từ văn bản
        
        Args:
            text: Văn bản cần trích xuất
            return_all: Nếu True, trả về cả non-claims
        
        Returns:
            List of dict với keys: text, label, confidence
        """
        # Tách câu
        sentences = self.split_sentences(text)
        
        # Phân loại từng câu
        results = []
        for sent in sentences:
            label, confidence = self.predict_sentence(sent)
            
            # Chỉ lấy claims có confidence >= threshold
            if label == 'claim' and confidence >= self.confidence_threshold:
                results.append({
                    'text': sent,
                    'label': label,
                    'confidence': confidence
                })
            elif return_all:
                results.append({
                    'text': sent,
                    'label': label,
                    'confidence': confidence
                })
        
        return results
    
    def extract_claims_with_context(self, text, context_window=1):
        """
        Trích xuất claims kèm context xung quanh
        
        Args:
            text: Văn bản
            context_window: Số câu trước/sau để lấy làm context
        
        Returns:
            List of dict với keys: claim, context_before, context_after, confidence
        """
        sentences = self.split_sentences(text)
        
        results = []
        for i, sent in enumerate(sentences):
            label, confidence = self.predict_sentence(sent)
            
            if label == 'claim' and confidence >= self.confidence_threshold:
                # Lấy context trước
                context_before = []
                for j in range(max(0, i - context_window), i):
                    context_before.append(sentences[j])
                
                # Lấy context sau
                context_after = []
                for j in range(i + 1, min(len(sentences), i + 1 + context_window)):
                    context_after.append(sentences[j])
                
                results.append({
                    'claim': sent,
                    'context_before': ' '.join(context_before) if context_before else '',
                    'context_after': ' '.join(context_after) if context_after else '',
                    'confidence': confidence
                })
        
        return results

def demo():
    """Demo với đoạn văn mẫu"""
    
    # Đoạn văn mẫu
    text = """
    Dự án có tổng mức đầu tư sơ bộ khoảng 104.410 tỷ đồng, gồm cả lãi vay. Theo đề xuất, nhà đầu tư BT sẽ bố trí toàn bộ vốn, gồm chi phí giải phóng mặt bằng, không sử dụng ngân sách; Nhà nước thanh toán bằng quỹ đất tương đương giá trị công trình.
    
    Vingroup kiến nghị được giao làm nhà đầu tư, dự kiến khởi công trong năm 2026, hoàn thành sau 3 năm. Doanh nghiệp đề xuất thành phố sớm phê duyệt báo cáo nghiên cứu tiền khả thi và cho phép áp dụng cơ chế lựa chọn nhà đầu tư trong trường hợp đặc biệt nhằm đẩy nhanh tiến độ.
    
    Theo Vingroup, Cần Giờ và Long Sơn - Vũng Tàu đều có tiềm năng lớn về du lịch sinh thái, công nghiệp - cảng biển và logistics, song chưa có tuyến giao thông trực tiếp. Hiện việc di chuyển phải đi vòng qua quốc lộ 51 hoặc dùng phà Cần Giờ - Vũng Tàu, mất 90-120 phút.
    """
    
    print("="*70)
    print("DEMO: CLAIM EXTRACTION FROM TEXT")
    print("="*70)
    
    # Load extractor
    extractor = ClaimExtractor(confidence_threshold=0.7)
    
    print("\n" + "="*70)
    print("ĐOẠN VĂN GỐC:")
    print("="*70)
    print(text.strip())
    
    # Trích xuất claims
    print("\n" + "="*70)
    print("CLAIMS ĐƯỢC TRÍCH XUẤT:")
    print("="*70)
    
    claims = extractor.extract_claims(text)
    
    if claims:
        for i, claim in enumerate(claims, 1):
            print(f"\n[{i}] {claim['text']}")
            print(f"    Confidence: {claim['confidence']:.3f}")
    else:
        print("Không tìm thấy claims nào")
    
    # Trích xuất với context
    print("\n" + "="*70)
    print("CLAIMS VỚI CONTEXT:")
    print("="*70)
    
    claims_with_context = extractor.extract_claims_with_context(text, context_window=1)
    
    for i, item in enumerate(claims_with_context, 1):
        print(f"\n[{i}] CLAIM: {item['claim']}")
        print(f"    Confidence: {item['confidence']:.3f}")
        if item['context_before']:
            print(f"    Context before: {item['context_before']}")
        if item['context_after']:
            print(f"    Context after: {item['context_after']}")
    
    # Xem tất cả câu (kể cả non-claims)
    print("\n" + "="*70)
    print("TẤT CẢ CÂU (CLAIMS + NON-CLAIMS):")
    print("="*70)
    
    all_sentences = extractor.extract_claims(text, return_all=True)
    
    for i, sent in enumerate(all_sentences, 1):
        label_icon = "✓" if sent['label'] == 'claim' else "✗"
        print(f"\n[{i}] {label_icon} {sent['label'].upper()}")
        print(f"    {sent['text']}")
        print(f"    Confidence: {sent['confidence']:.3f}")
    
    print("\n" + "="*70)
    print(f"TỔNG KẾT: Tìm thấy {len(claims)} claims trong {len(all_sentences)} câu")
    print("="*70)

def test_with_custom_text():
    """Test với văn bản tùy chỉnh"""
    
    print("\n" + "="*70)
    print("TEST VỚI VĂN BẢN TỰ NHẬP")
    print("="*70)
    
    # Load extractor
    extractor = ClaimExtractor(confidence_threshold=0.7)
    
    # Nhập văn bản
    print("\nNhập đoạn văn bản (Enter 2 lần để kết thúc):")
    lines = []
    while True:
        line = input()
        if line == "":
            if lines and lines[-1] == "":
                break
            lines.append(line)
        else:
            lines.append(line)
    
    text = "\n".join(lines[:-1])  # Bỏ dòng trống cuối
    
    if not text.strip():
        print("Không có văn bản để xử lý")
        return
    
    # Trích xuất
    claims = extractor.extract_claims(text)
    
    print("\n" + "="*70)
    print(f"KẾT QUẢ: Tìm thấy {len(claims)} claims")
    print("="*70)
    
    for i, claim in enumerate(claims, 1):
        print(f"\n[{i}] {claim['text']}")
        print(f"    Confidence: {claim['confidence']:.3f}")

if __name__ == "__main__":
    # Chạy demo
    demo()
    
    # Uncomment để test với văn bản tự nhập
    # test_with_custom_text()
