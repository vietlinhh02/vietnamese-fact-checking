import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.self_verification import SelfVerificationModule, print_verification_results
from src.data_models import Evidence

def test_2026_scenario():
    print("="*60)
    print("KIỂM THỬ KỊCH BẢN TƯƠNG LAI: VIỆT NAM NĂM 2026")
    print("="*60)

    # Giả lập một câu trả lời từ AI cũ (Học dữ liệu 2023)
    # Simulated answer from an outdated AI model (trained on 2023 data)
    explanation = """
    Việt Nam hiện nay có 63 tỉnh thành phố trực thuộc trung ương trải dài từ Bắc vào Nam.
    Quy mô dân số khoảng hơn 98 triệu người.
    """
    
    # Dữ liệu "Sự thật" mới do người dùng cung cấp (Bối cảnh 2026)
    # New "Ground Truth" provided by user (2026 Context)
    evidence_list = [
        Evidence(
            text="Theo Nghị quyết mới của Quốc hội, sau khi thực hiện sáp nhập các đơn vị hành chính, tính đến đầu năm 2026, Việt Nam chính thức chỉ còn 33 thành phố trực thuộc trung ương.",
            source_url="https://chinhphu.vn/nghi-quyet-sap-nhap-2025",
            source_title="Nghị quyết Sáp nhập Đơn vị Hành chính Quốc gia 2026",
            credibility_score=1.0,
            language="vi"
        ),
        Evidence(
            text="Tổng cục Thống kê công bố dân số Việt Nam đầu năm 2026 đã vượt mốc 101 triệu người.",
            source_url="https://gso.gov.vn/dan-so-2026",
            source_title="Báo cáo Dân số 2026",
            credibility_score=0.95,
            language="vi"
        )
    ]

    print("\n1. NỘI DUNG CẦN KIỂM CHỨNG (Kiến thức cũ):")
    print("-" * 50)
    print(explanation.strip())

    print("\n2. DỮ LIỆU THỰC TẾ (Bối cảnh 2026 - 33 Thành phố):")
    print("-" * 50)
    for i, ev in enumerate(evidence_list, 1):
        print(f"{i}. {ev.text}")
        print(f"   (Nguồn: {ev.source_title})")

    print("\n3. ĐANG CHẠY KIỂM CHỨNG (SELF-VERIFICATION)...")
    print("-" * 50)
    
    verifier = SelfVerificationModule()
    # Hệ thống sẽ so sánh "63" (trong tuyên bố) với "33" (trong bằng chứng)
    # The system will compare "63" (in claim) with "33" (in evidence)
    quality_score, results = verifier.verify_explanation(explanation, evidence_list)

    print("\n4. KẾT QUẢ PHÂN TÍCH:")
    # In kết quả chi tiết
    print_verification_results(quality_score, results, "detailed")
    
    # Đánh giá kết quả mong đợi
    print("\n" + "="*60)
    print("ĐÁNH GIÁ KẾT QUẢ:")
    
    found_hallucination = False
    for res in results:
        if "63" in res.claim.text and not res.is_verified:
            print("✅ ĐÃ PHÁT HIỆN LỖI SAI: Hệ thống đã bác bỏ thông tin '63 tỉnh thành' do xung đột với dữ liệu 33 thành phố.")
            found_hallucination = True
            break
            
    if not found_hallucination:
        print("❌ CẢNH BÁO: Hệ thống chưa phát hiện ra lỗi sai về số lượng tỉnh thành.")
    else:
        print("-> Hệ thống hoạt động chính xác theo dữ liệu mới năm 2026.")

if __name__ == "__main__":
    test_2026_scenario()
