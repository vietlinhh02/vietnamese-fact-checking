"""
Script kiểm tra dataset đã tạo
"""

import json
import pandas as pd
from collections import Counter

def check_dataset(file_path):
    """Kiểm tra một file dataset"""
    print(f"\n{'='*70}")
    print(f"Kiểm tra: {file_path}")
    print('='*70)
    
    # Load data
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Thống kê cơ bản
    print(f"\n📊 Thống kê:")
    print(f"  Tổng số samples: {len(df)}")
    print(f"\n  Phân bố labels:")
    print(df['label'].value_counts().to_string().replace('\n', '\n  '))
    
    # Độ dài text
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print(f"\n  Độ dài text:")
    print(f"    - Trung bình: {df['text_length'].mean():.1f} chars")
    print(f"    - Min: {df['text_length'].min()} chars")
    print(f"    - Max: {df['text_length'].max()} chars")
    
    print(f"\n  Số từ:")
    print(f"    - Trung bình: {df['word_count'].mean():.1f} words")
    print(f"    - Min: {df['word_count'].min()} words")
    print(f"    - Max: {df['word_count'].max()} words")
    
    # Xem mẫu claims
    claims = df[df['label'] == 'claim']
    print(f"\n✅ Mẫu CLAIMS (5 samples):")
    for i, row in claims.head(5).iterrows():
        text = row['text']
        if len(text) > 100:
            text = text[:100] + "..."
        print(f"  {i+1}. {text}")
    
    # Xem mẫu non-claims
    non_claims = df[df['label'] == 'non-claim']
    print(f"\n❌ Mẫu NON-CLAIMS (10 samples):")
    for i, row in non_claims.head(10).iterrows():
        print(f"  {i+1}. {row['text']}")
    
    # Kiểm tra duplicate
    duplicates = df[df.duplicated(subset=['text'], keep=False)]
    if len(duplicates) > 0:
        print(f"\n⚠️  Cảnh báo: Có {len(duplicates)} duplicates!")
    else:
        print(f"\n✓ Không có duplicates")
    
    # Kiểm tra empty
    empty = df[df['text'].str.strip() == '']
    if len(empty) > 0:
        print(f"⚠️  Cảnh báo: Có {len(empty)} empty texts!")
    else:
        print(f"✓ Không có empty texts")
    
    return df

def main():
    print("="*70)
    print("KIỂM TRA DATASET CLAIM DETECTION")
    print("="*70)
    
    files = [
        'data/claim_detection/claim_detection_train.jsonl',
        'data/claim_detection/claim_detection_val.jsonl',
        'data/claim_detection/claim_detection_test.jsonl'
    ]
    
    all_dfs = []
    for file_path in files:
        try:
            df = check_dataset(file_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"\n✗ Error: {e}")
    
    # Tổng kết
    print(f"\n{'='*70}")
    print("TỔNG KẾT")
    print('='*70)
    
    total_samples = sum(len(df) for df in all_dfs)
    total_claims = sum(len(df[df['label'] == 'claim']) for df in all_dfs)
    total_non_claims = sum(len(df[df['label'] == 'non-claim']) for df in all_dfs)
    
    print(f"\n📊 Tổng cộng:")
    print(f"  - Total: {total_samples:,} samples")
    print(f"  - Claims: {total_claims:,} ({total_claims/total_samples*100:.1f}%)")
    print(f"  - Non-claims: {total_non_claims:,} ({total_non_claims/total_samples*100:.1f}%)")
    
    print(f"\n✅ Đánh giá:")
    
    # Kiểm tra cân bằng
    balance_ratio = total_claims / total_non_claims
    if 0.9 <= balance_ratio <= 1.1:
        print(f"  ✓ Dataset cân bằng tốt (ratio: {balance_ratio:.2f})")
    else:
        print(f"  ⚠️  Dataset không cân bằng (ratio: {balance_ratio:.2f})")
    
    # Kiểm tra kích thước
    if total_samples >= 10000:
        print(f"  ✓ Dataset đủ lớn ({total_samples:,} samples)")
    else:
        print(f"  ⚠️  Dataset nhỏ ({total_samples:,} samples)")
    
    # Kiểm tra split
    train_size = len(all_dfs[0])
    val_size = len(all_dfs[1])
    test_size = len(all_dfs[2])
    
    train_ratio = train_size / total_samples
    val_ratio = val_size / total_samples
    test_ratio = test_size / total_samples
    
    print(f"\n  Split ratio:")
    print(f"    - Train: {train_ratio*100:.1f}% (expected: 70%)")
    print(f"    - Val: {val_ratio*100:.1f}% (expected: 15%)")
    print(f"    - Test: {test_ratio*100:.1f}% (expected: 15%)")
    
    if 0.68 <= train_ratio <= 0.72 and 0.13 <= val_ratio <= 0.17 and 0.13 <= test_ratio <= 0.17:
        print(f"  ✓ Split ratio đúng")
    else:
        print(f"  ⚠️  Split ratio không chuẩn")
    
    print(f"\n{'='*70}")
    print("✅ KIỂM TRA HOÀN TẤT!")
    print('='*70)

if __name__ == "__main__":
    main()
