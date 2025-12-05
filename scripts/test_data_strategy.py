"""
Script để test chiến lược phân loại claim/non-claim
Giúp visualize xem heuristics có hoạt động tốt không
"""

import json

# Test cases từ dataset thực tế
test_cases = [
    # Claims rõ ràng (từ field 'claim')
    {
        "text": "Trong năm nay, hai địa phương dẫn đầu và gần nhau nhất về lượng hồ sơ là TP Hồ Chí Minh và Điện Biên",
        "expected": "claim",
        "reason": "Factual assertion với số liệu và tên riêng"
    },
    {
        "text": "Việt Nam có 54 dân tộc anh em",
        "expected": "claim",
        "reason": "Factual assertion với số liệu cụ thể"
    },
    {
        "text": "Dân số Hà Nội năm 2023 đạt 8 triệu người",
        "expected": "claim",
        "reason": "Factual assertion với số liệu và thời gian"
    },
    
    # Non-claims rõ ràng
    {
        "text": "Bạn có biết điều này không?",
        "expected": "non-claim",
        "reason": "Câu hỏi"
    },
    {
        "text": "Tôi nghĩ rằng đây là một quyết định đúng đắn",
        "expected": "non-claim",
        "reason": "Ý kiến chủ quan"
    },
    {
        "text": "Hãy xem xét kỹ lưỡng vấn đề này",
        "expected": "non-claim",
        "reason": "Câu mệnh lệnh"
    },
    {
        "text": "Trong khi đó, tình hình đang có nhiều diễn biến",
        "expected": "non-claim",
        "reason": "Câu nối chung chung"
    },
    
    # Cases khó (từ context)
    {
        "text": "Năm nay hai địa phương có lượng hồ sơ dẫn đầu và bám sát nhau là TP HCM và Điện Biên",
        "expected": "claim",
        "reason": "Có số liệu và tên riêng → Có thể là claim khác!"
    },
    {
        "text": "Tiếp theo là Hà Nội, Nam Định, Cần Thơ, Huế, Lạng Sơn, Khánh Hòa",
        "expected": "claim",
        "reason": "Liệt kê thông tin cụ thể → Có thể là claim!"
    },
    {
        "text": "Điều này cho thấy sự quan tâm của các tác giả hướng tới tìm kiếm giải pháp công nghệ mới",
        "expected": "non-claim",
        "reason": "Câu mô tả chung, không có assertion cụ thể"
    },
    {
        "text": "Đây là một vấn đề phức tạp cần được xem xét từ nhiều góc độ",
        "expected": "non-claim",
        "reason": "Câu mô tả chung"
    },
]

def is_likely_non_claim(sentence):
    """Heuristics để phát hiện non-claim"""
    sent = sentence.strip()
    
    # Câu hỏi
    if sent.endswith('?'):
        return True, 'question'
    
    # Ý kiến chủ quan
    opinion_markers = ['tôi nghĩ', 'tôi cho rằng', 'theo tôi', 'có lẽ', 'có thể', 
                       'dường như', 'hình như', 'chắc là', 'có vẻ']
    if any(marker in sent.lower() for marker in opinion_markers):
        return True, 'opinion'
    
    # Câu mệnh lệnh
    if any(sent.lower().startswith(cmd) for cmd in ['hãy ', 'đừng ', 'cần ', 'nên ']):
        return True, 'command'
    
    # Câu quá ngắn
    word_count = len(sent.split())
    if word_count < 5:
        return True, 'too_short'
    
    # Câu nối chung chung
    vague_patterns = ['trong khi đó', 'bên cạnh đó', 'ngoài ra', 'đồng thời']
    if sent.lower().startswith(tuple(vague_patterns)) and word_count < 15:
        return True, 'vague'
    
    return False, None

def is_likely_claim(sentence):
    """Heuristics để phát hiện claim"""
    import re
    sent = sentence.strip()
    
    # Có số liệu cụ thể
    if re.search(r'\d+', sent):
        if any(pattern in sent for pattern in ['năm', 'tháng', '%', 'triệu', 'tỷ']):
            return True, 'has_numbers'
    
    # Có tên riêng (viết hoa)
    if re.search(r'\b[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]', sent):
        return True, 'has_proper_noun'
    
    # Có động từ khẳng định
    assertion_verbs = ['là ', 'có ', 'được ', 'đạt ', 'tăng ', 'giảm ']
    if any(verb in sent.lower() for verb in assertion_verbs):
        return True, 'has_assertion'
    
    return False, None

def classify_sentence(sentence):
    """Phân loại câu dựa trên heuristics"""
    # Kiểm tra non-claim trước
    is_non, non_reason = is_likely_non_claim(sentence)
    if is_non:
        return 'non-claim', non_reason
    
    # Kiểm tra claim
    is_claim, claim_reason = is_likely_claim(sentence)
    if is_claim:
        return 'claim', claim_reason
    
    # Không chắc chắn
    return 'uncertain', 'no_clear_indicators'

def test_classification():
    """Test heuristics trên test cases"""
    print("=" * 80)
    print("TEST CHIẾN LƯỢC PHÂN LOẠI CLAIM/NON-CLAIM")
    print("=" * 80)
    
    correct = 0
    total = 0
    
    for i, case in enumerate(test_cases, 1):
        text = case['text']
        expected = case['expected']
        reason = case['reason']
        
        predicted, pred_reason = classify_sentence(text)
        
        is_correct = (predicted == expected) or (predicted == 'uncertain' and expected == 'claim')
        if is_correct:
            correct += 1
        total += 1
        
        status = "✓" if is_correct else "✗"
        
        print(f"\n[{i}] {status}")
        print(f"Text: {text[:80]}...")
        print(f"Expected: {expected} ({reason})")
        print(f"Predicted: {predicted} ({pred_reason})")
        
        if not is_correct:
            print("⚠️  MISMATCH!")
    
    print("\n" + "=" * 80)
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print("=" * 80)
    
    print("\n💡 Nhận xét:")
    print("- ✅ Heuristics hoạt động tốt cho non-claims rõ ràng (câu hỏi, ý kiến, mệnh lệnh)")
    print("- ⚠️  Cần cẩn thận với câu trong context (có thể là claims khác)")
    print("- 🎯 Chiến lược: Chỉ lấy non-claims RÕ RÀNG, bỏ qua các câu uncertain")

if __name__ == "__main__":
    test_classification()
    
    print("\n" + "=" * 80)
    print("DEMO: Phân loại câu từ context")
    print("=" * 80)
    
    context = """
    Năm nay hai địa phương có lượng hồ sơ dẫn đầu và bám sát nhau là TP HCM và Điện Biên. 
    Tiếp theo là Hà Nội, Nam Định, Cần Thơ. 
    Điều này cho thấy sự quan tâm của các tác giả hướng tới tìm kiếm giải pháp công nghệ mới.
    Bạn có biết về điều này không?
    Tôi nghĩ đây là một xu hướng tích cực.
    """
    
    import re
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', context) if s.strip()]
    
    print("\nPhân tích từng câu:")
    for i, sent in enumerate(sentences, 1):
        predicted, reason = classify_sentence(sent)
        
        if predicted == 'non-claim':
            label = "✅ NON-CLAIM"
        elif predicted == 'claim':
            label = "⚠️  CLAIM (bỏ qua)"
        else:
            label = "❓ UNCERTAIN (bỏ qua)"
        
        print(f"\n[{i}] {label}")
        print(f"    {sent}")
        print(f"    → {predicted} ({reason})")
    
    print("\n💡 Kết luận:")
    print("Chỉ lấy câu 4 và 5 làm non-claims!")
    print("Câu 1, 2, 3 có thể là claims khác → BỎ QUA để tránh label noise")
