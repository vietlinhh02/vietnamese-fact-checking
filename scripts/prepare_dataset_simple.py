"""
Script chuẩn bị dataset Claim Detection
Trích xuất claims và non-claims từ dataset ise-dsc01

Chiến lược:
1. Claims: Lấy từ field 'claim' trong dataset
2. Non-claims: Trích xuất từ context (các câu KHÔNG phải claim/evidence)
3. Lọc thông minh để đảm bảo chất lượng
"""

import json
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

def load_ise_dataset(file_path):
    """Load dataset ise-dsc01"""
    print(f"Loading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for key, value in data.items():
        records.append({
            'id': key,
            'context': value['context'],
            'claim': value['claim'],
            'evidence': value.get('evidence', ''),
            'verdict': value['verdict'],
            'domain': value.get('domain', '')
        })
    
    return pd.DataFrame(records)

def split_sentences(text):
    """
    Tách văn bản thành các câu
    Xử lý tiếng Việt tốt hơn
    """
    # Thêm khoảng trắng sau dấu câu nếu thiếu
    text = re.sub(r'([.!?])([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ])', r'\1 \2', text)
    
    # Tách câu bằng dấu chấm, chấm hỏi, chấm than
    # Nhưng không tách nếu là số thập phân hoặc viết tắt
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ"\'])', text)
    
    # Nếu không tách được, thử cách khác
    if len(sentences) <= 1:
        sentences = re.split(r'[.!?]\s+', text)
    
    # Lọc câu rỗng và quá ngắn
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return sentences

def is_question(text):
    """Kiểm tra câu hỏi"""
    text = text.strip()
    if text.endswith('?'):
        return True
    question_words = ['ai ', 'gì ', 'nào ', 'đâu ', 'sao ', 'bao giờ', 'bao nhiêu', 'như thế nào', 'tại sao', 'vì sao', 'có phải', 'liệu ']
    return any(text.lower().startswith(w) or f' {w}' in text.lower() for w in question_words)

def is_opinion(text):
    """Kiểm tra ý kiến chủ quan"""
    opinion_markers = [
        'tôi nghĩ', 'tôi cho rằng', 'theo tôi', 'tôi tin', 'tôi cảm thấy',
        'có lẽ', 'có thể', 'dường như', 'hình như', 'chắc là', 'có vẻ',
        'theo quan điểm', 'theo ý kiến', 'cá nhân tôi', 'riêng tôi'
    ]
    return any(marker in text.lower() for marker in opinion_markers)

def is_command(text):
    """Kiểm tra câu mệnh lệnh"""
    command_starters = ['hãy ', 'đừng ', 'cần ', 'nên ', 'phải ', 'xin ', 'mời ', 'vui lòng']
    return any(text.lower().startswith(cmd) for cmd in command_starters)

def is_connector(text):
    """Kiểm tra câu nối/chuyển tiếp"""
    connectors = [
        'trong khi đó', 'bên cạnh đó', 'ngoài ra', 'đồng thời', 'tuy nhiên',
        'mặc dù vậy', 'do đó', 'vì vậy', 'theo đó', 'như vậy', 'tóm lại',
        'nói cách khác', 'mặt khác', 'hơn nữa', 'thêm vào đó'
    ]
    return any(text.lower().startswith(conn) for conn in connectors)

def has_specific_info(text):
    """
    Kiểm tra câu có thông tin cụ thể (có thể là claim)
    - Có số liệu
    - Có tên riêng
    - Có ngày tháng
    """
    # Có số
    if re.search(r'\d+', text):
        return True
    
    # Có tên riêng (chữ hoa ở giữa câu)
    words = text.split()
    for i, word in enumerate(words):
        if i > 0 and word and word[0].isupper():
            # Không phải đầu câu và viết hoa
            return True
    
    return False

def is_likely_non_claim(text):
    """
    Xác định câu có khả năng là non-claim
    Returns: (is_non_claim, reason)
    """
    text = text.strip()
    
    # Câu quá ngắn
    word_count = len(text.split())
    if word_count < 5:
        return True, 'too_short'
    
    # Câu quá dài
    if word_count > 60:
        return True, 'too_long'
    
    # Câu hỏi
    if is_question(text):
        return True, 'question'
    
    # Ý kiến
    if is_opinion(text):
        return True, 'opinion'
    
    # Mệnh lệnh
    if is_command(text):
        return True, 'command'
    
    # Câu nối ngắn
    if is_connector(text) and word_count < 15:
        return True, 'connector'
    
    return False, None

def extract_non_claims_from_context(row, max_per_context=3):
    """
    Trích xuất non-claims từ context
    Chỉ lấy các câu RÕ RÀNG không phải claim
    """
    context = row['context']
    claim = row['claim']
    evidence = row.get('evidence', '') or ''
    
    non_claims = []
    
    # Tách câu
    sentences = split_sentences(context)
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        
        # Bỏ qua nếu trùng hoặc chứa claim
        if claim.lower() in sent.lower() or sent.lower() in claim.lower():
            continue
        
        # Bỏ qua nếu trùng hoặc chứa evidence
        if evidence and (evidence.lower() in sent.lower() or sent.lower() in evidence.lower()):
            continue
        
        # Kiểm tra có phải non-claim không
        is_non, reason = is_likely_non_claim(sent)
        
        if is_non:
            non_claims.append({
                'text': sent,
                'label': 'non-claim',
                'source': f'context_{reason}',
                'domain': row['domain']
            })
        elif not has_specific_info(sent):
            # Câu không có thông tin cụ thể → có thể là non-claim
            non_claims.append({
                'text': sent,
                'label': 'non-claim',
                'source': 'context_general',
                'domain': row['domain']
            })
    
    # Giới hạn số lượng
    if len(non_claims) > max_per_context:
        non_claims = non_claims[:max_per_context]
    
    return non_claims

def create_claims(df):
    """Tạo claims từ field 'claim'"""
    claims = []
    for _, row in df.iterrows():
        claims.append({
            'text': row['claim'],
            'label': 'claim',
            'source': 'claim_field',
            'domain': row['domain']
        })
    return claims

def create_template_non_claims():
    """
    Tạo non-claims từ templates
    Dùng để bổ sung nếu không đủ non-claims từ context
    """
    non_claims = []
    
    # Câu hỏi đa dạng
    questions = [
        "Bạn có biết về điều này không?",
        "Tại sao lại như vậy?",
        "Khi nào sự kiện này diễn ra?",
        "Ai là người chịu trách nhiệm?",
        "Làm thế nào để giải quyết vấn đề?",
        "Điều gì sẽ xảy ra tiếp theo?",
        "Có phải đây là sự thật không?",
        "Chúng ta nên làm gì bây giờ?",
        "Liệu điều này có đúng không?",
        "Bạn nghĩ sao về vấn đề này?",
        "Có ai biết thông tin về việc này?",
        "Khi nào chúng ta sẽ có câu trả lời?",
        "Tại sao không có ai nói về điều này?",
        "Làm sao để xác minh thông tin?",
        "Có bằng chứng nào chứng minh không?",
        "Ai đã đưa ra tuyên bố này?",
        "Nguồn thông tin từ đâu?",
        "Có thể tin tưởng được không?",
        "Điều này có ảnh hưởng gì?",
        "Chúng ta cần làm gì tiếp theo?",
        "Vấn đề này bắt đầu từ khi nào?",
        "Ai là người đầu tiên phát hiện ra?",
        "Có giải pháp nào khác không?",
        "Tình hình hiện tại như thế nào?",
        "Có ai phản đối điều này không?",
    ]
    
    # Ý kiến đa dạng
    opinions = [
        "Tôi nghĩ rằng đây là một quyết định đúng đắn",
        "Theo tôi, vấn đề này cần được xem xét kỹ lưỡng hơn",
        "Có lẽ chúng ta nên chờ đợi thêm thông tin",
        "Tôi cảm thấy điều này không hoàn toàn chính xác",
        "Dường như tình hình đang có những chuyển biến tích cực",
        "Tôi tin rằng mọi thứ sẽ tốt đẹp hơn",
        "Có vẻ như đây là một xu hướng đáng chú ý",
        "Theo quan điểm của tôi, đây là vấn đề quan trọng",
        "Tôi cho rằng cần có thêm nghiên cứu",
        "Có thể nói rằng đây là một bước tiến lớn",
        "Tôi không chắc chắn về điều này",
        "Theo ý kiến cá nhân, đây là vấn đề phức tạp",
        "Tôi nghĩ chúng ta nên thận trọng",
        "Có lẽ đây không phải là giải pháp tốt nhất",
        "Tôi cảm thấy cần thêm thời gian để đánh giá",
        "Dường như có nhiều khía cạnh cần xem xét",
        "Tôi tin rằng sẽ có cách giải quyết tốt hơn",
        "Có vẻ như tình hình đang được cải thiện",
        "Theo tôi thấy, đây là hướng đi đúng đắn",
        "Tôi cho rằng cần có sự thay đổi",
        "Cá nhân tôi không đồng ý với quan điểm này",
        "Tôi nghĩ đây chỉ là một phần của vấn đề",
        "Theo tôi hiểu, tình hình phức tạp hơn nhiều",
        "Tôi cảm thấy lo ngại về điều này",
        "Có lẽ chúng ta đang bỏ qua điều gì đó",
    ]
    
    # Mệnh lệnh đa dạng
    commands = [
        "Hãy xem xét kỹ lưỡng vấn đề này",
        "Đừng quên kiểm tra thông tin trước khi chia sẻ",
        "Cần phải có thêm nghiên cứu về chủ đề này",
        "Nên tham khảo ý kiến chuyên gia trước khi quyết định",
        "Hãy đọc kỹ tài liệu trước khi đưa ra kết luận",
        "Đừng tin vào thông tin chưa được xác minh",
        "Cần kiểm tra nguồn gốc của thông tin",
        "Hãy suy nghĩ thật kỹ trước khi hành động",
        "Đừng vội vàng đưa ra nhận định",
        "Hãy tìm hiểu thêm về vấn đề này",
        "Cần phải xác minh từ nhiều nguồn",
        "Nên chờ đợi thêm thông tin chính thức",
        "Hãy giữ thái độ khách quan",
        "Đừng lan truyền thông tin sai lệch",
        "Cần có bằng chứng cụ thể",
        "Hãy cân nhắc tất cả các khía cạnh",
        "Đừng bỏ qua những chi tiết quan trọng",
        "Cần lắng nghe nhiều ý kiến khác nhau",
        "Hãy đặt câu hỏi trước khi tin",
        "Nên so sánh với các nguồn khác",
    ]
    
    # Câu mô tả chung
    descriptions = [
        "Đây là một vấn đề phức tạp cần được xem xét từ nhiều góc độ",
        "Tình hình hiện tại đang có nhiều diễn biến khác nhau",
        "Vấn đề này đã thu hút sự quan tâm của dư luận",
        "Nhiều người đang thảo luận về chủ đề này",
        "Trong khi đó, các chuyên gia vẫn đang tranh luận",
        "Bên cạnh đó, còn có nhiều yếu tố cần xem xét",
        "Ngoài ra, cần chú ý đến các khía cạnh khác",
        "Đồng thời, vấn đề này cũng liên quan đến nhiều lĩnh vực",
        "Tuy nhiên, vẫn còn nhiều điều chưa rõ ràng",
        "Mặc dù vậy, cần thêm thời gian để đánh giá",
        "Do đó, chúng ta cần thận trọng trong việc đưa ra kết luận",
        "Vì vậy, cần có thêm nghiên cứu sâu hơn",
        "Theo đó, tình hình đang được theo dõi chặt chẽ",
        "Như vậy, vấn đề vẫn đang được xem xét",
        "Tóm lại, đây là một chủ đề đáng quan tâm",
        "Nhìn chung, tình hình vẫn đang diễn biến phức tạp",
        "Trên thực tế, có nhiều yếu tố cần cân nhắc",
        "Về cơ bản, đây là vấn đề cần được giải quyết",
        "Nói chung, mọi người đều quan tâm đến điều này",
        "Cuối cùng, chúng ta cần chờ đợi thêm thông tin",
    ]
    
    all_templates = questions + opinions + commands + descriptions
    
    for text in all_templates:
        non_claims.append({
            'text': text,
            'label': 'non-claim',
            'source': 'template',
            'domain': 'general'
        })
    
    return non_claims

def create_dataset(input_files, output_prefix):
    """
    Tạo dataset claim detection
    """
    print("=" * 70)
    print("TẠO DATASET CLAIM DETECTION")
    print("=" * 70)
    
    # Load data
    print("\n[1] Đang load dữ liệu...")
    all_data = []
    for file_path in input_files:
        try:
            df = load_ise_dataset(file_path)
            all_data.append(df)
            print(f"✓ Loaded {len(df)} samples")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    df_all = pd.concat(all_data, ignore_index=True)
    print(f"✓ Tổng: {len(df_all)} samples")
    
    # Tạo claims
    print("\n[2] Tạo claims từ field 'claim'...")
    claims = create_claims(df_all)
    print(f"✓ {len(claims)} claims")
    
    # Trích xuất non-claims từ context
    print("\n[3] Trích xuất non-claims từ context...")
    context_non_claims = []
    for idx, row in df_all.iterrows():
        extracted = extract_non_claims_from_context(row, max_per_context=2)
        context_non_claims.extend(extracted)
        
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(df_all)} samples...")
    
    print(f"✓ {len(context_non_claims)} non-claims từ context")
    
    # Thống kê nguồn non-claims
    source_counts = {}
    for nc in context_non_claims:
        source = nc['source']
        source_counts[source] = source_counts.get(source, 0) + 1
    print("  Phân bố nguồn:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    - {source}: {count}")
    
    # Tạo template non-claims để bổ sung
    print("\n[4] Tạo template non-claims...")
    template_non_claims = create_template_non_claims()
    print(f"✓ {len(template_non_claims)} templates")
    
    # Kết hợp non-claims
    all_non_claims = context_non_claims + template_non_claims
    print(f"✓ Tổng non-claims: {len(all_non_claims)}")
    
    # Cân bằng dataset
    print("\n[5] Cân bằng dataset...")
    num_claims = len(claims)
    
    if len(all_non_claims) < num_claims:
        # Duplicate non-claims nếu cần
        multiplier = (num_claims // len(all_non_claims)) + 1
        all_non_claims = all_non_claims * multiplier
    
    # Shuffle và lấy đủ số lượng
    np.random.seed(42)
    np.random.shuffle(all_non_claims)
    all_non_claims = all_non_claims[:num_claims]
    
    print(f"✓ Claims: {len(claims)}")
    print(f"✓ Non-claims: {len(all_non_claims)}")
    
    # Kết hợp
    all_samples = claims + all_non_claims
    df_final = pd.DataFrame(all_samples)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n[6] Tổng: {len(df_final)} samples")
    print("Phân bố labels:")
    print(df_final['label'].value_counts())
    print("\nPhân bố nguồn:")
    print(df_final['source'].value_counts())
    
    # Chia train/val/test
    print("\n[7] Chia train/val/test (70/15/15)...")
    train_val, test = train_test_split(
        df_final, test_size=0.15, random_state=42, stratify=df_final['label']
    )
    train, val = train_test_split(
        train_val, test_size=0.176, random_state=42, stratify=train_val['label']
    )
    
    print(f"✓ Train: {len(train)}")
    print(f"✓ Val: {len(val)}")
    print(f"✓ Test: {len(test)}")
    
    # Lưu files
    print("\n[8] Lưu files...")
    for df, name in [(train, 'train'), (val, 'val'), (test, 'test')]:
        file_path = f"{output_prefix}_{name}.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                json_obj = {'text': row['text'], 'label': row['label']}
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        print(f"✓ {file_path}")
    
    print("\n" + "=" * 70)
    print("✓ HOÀN THÀNH!")
    print("=" * 70)
    
    return train, val, test

if __name__ == "__main__":
    # Cấu hình
    input_files = [
        "data/dataset/Dataset/ise-dsc01-train.json",
        "data/dataset/Dataset/ise-dsc01-train_ver2.json"
    ]
    
    output_prefix = "data/claim_detection/claim_detection"
    
    # Tạo dataset
    train, val, test = create_dataset(input_files, output_prefix)
    
    print("\n📝 Files đã tạo:")
    print(f"  - {output_prefix}_train.jsonl")
    print(f"  - {output_prefix}_val.jsonl")
    print(f"  - {output_prefix}_test.jsonl")
    
    print("\n💡 Bước tiếp theo:")
    print("  1. Upload 3 files JSONL lên Kaggle")
    print("  2. Chạy finetune_phobert_kaggle.py")
