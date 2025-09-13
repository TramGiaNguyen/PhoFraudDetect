import pandas as pd
import os
import time
import random
from groq import Groq

# Thiết lập API key (thay bằng key thực của bạn, tốt nhất dùng os.environ để bảo mật)
GROQ_API_KEY = "your-api-key-here"
client = Groq(api_key=GROQ_API_KEY)

# Đường dẫn file input (sẽ ghi đè trực tiếp)
input_file = 'expanded_scam_all_types.csv'

# Max retry cho error handling
MAX_RETRIES = 5

def generate_expanded_dialogue(loai, mo_ta, variant_idx):
    """
    Hàm generate hội thoại mở rộng với prompt động dựa trên loại
    """
    # Xác định loại prompt dựa trên loại lừa đảo
    if 'tình cảm' in loai.lower() or 'lòng tin' in loai.lower() or 'romance' in loai.lower():
        # Đối với loại lợi dụng lòng tin (romance scam): Generate dài, với build-up
        prompt = f"""
        Dựa trên loại lừa đảo: {loai}
        Mô tả ngắn gọn: {mo_ta}
        
        Hãy generate một cuộc hội thoại CHI TIẾT và DÀI giữa kẻ lừa đảo (Kẻ lừa) và nạn nhân (Nạn nhân). 
        Hội thoại phải:
        - Bắt đầu từ lời chào thân thiện và xây dựng lòng tin dần dần (chia sẻ chuyện cá nhân, khen ngợi, tạo sự đồng cảm trong 10-15 lượt đầu).
        - Tập trung vào ngữ nghĩa trọng tâm: Thao túng cảm xúc, dẫn đến yêu cầu tiền hoặc thông tin cá nhân.
        - Có ít nhất 20-30 lượt nói qua lại để simulate realtime kéo dài (như chat qua nhiều ngày).
        - Sử dụng ngôn ngữ tiếng Việt tự nhiên, thêm lỗi chính tả, emoji, hoặc từ lóng để giống thật (biến thể {variant_idx} để đa dạng).
        - Kết thúc bằng hành động lừa đảo thành công hoặc nạn nhân nghi ngờ.
        - Độ dài tổng: 300-600 từ.
        
        Định dạng output: 
        Kẻ lừa: [lời nói]
        Nạn nhân: [lời nói]
        ...
        """
    else:
        # Đối với các loại khác (như mạo danh thuế, hỗ trợ kỹ thuật, đầu tư...): Generate ngắn hơn, khẩn cấp
        prompt = f"""
        Dựa trên loại lừa đảo: {loai}
        Mô tả ngắn gọn: {mo_ta}
        
        Hãy generate một cuộc hội thoại CHI TIẾT giữa kẻ lừa đảo (Kẻ lừa) và nạn nhân (Nạn nhân). 
        Hội thoại phải:
        - Bắt đầu từ lời chào và đi thẳng vào vấn đề (giả mạo danh tính, tạo tình huống khẩn cấp).
        - Tập trung vào ngữ nghĩa trọng tâm: Yêu cầu OTP, thông tin ngân hàng, hoặc hành động nhanh chóng.
        - Có ít nhất 10-15 lượt nói qua lại để simulate cuộc gọi hoặc chat realtime.
        - Sử dụng ngôn ngữ tiếng Việt tự nhiên, thêm giọng điệu thuyết phục hoặc đe dọa nhẹ (biến thể {variant_idx} để đa dạng).
        - Kết thúc bằng lời cảnh báo hoặc hành động lừa đảo.
        - Độ dài tổng: 200-400 từ.
        
        Định dạng output: 
        Kẻ lừa: [lời nói]
        Nạn nhân: [lời nói]
        ...
        """
    
    retries = 0
    while retries < MAX_RETRIES:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Bạn là một AI chuyên generate script hội thoại lừa đảo đa dạng để mục đích nghiên cứu và phát hiện gian lận. Giữ an toàn, không khuyến khích hành vi xấu."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.7 + random.uniform(-0.1, 0.1),
                max_tokens=1024,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            retries += 1
            print(f"Lỗi khi generate cho {loai} (variant {variant_idx}), thử lại {retries}/{MAX_RETRIES}: {e}")
            time.sleep(2 ** retries)
    
    return "Generate thất bại sau nhiều lần thử."

def main():
    print("Bắt đầu quá trình dò lại và generate lại dữ liệu bị lỗi...")
    
    # Đọc file CSV gốc
    try:
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        print(f"Đã đọc file: {input_file} với {len(df)} dòng dữ liệu")
    except Exception as e:
        print(f"Lỗi khi đọc file {input_file}: {e}")
        return
    
    # Kiểm tra cột cần thiết
    required_columns = ['Loại lừa đảo', 'Hội thoại', 'Hoi_thoai_mo_rong', 'Variant', 'Original_Index']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Thiếu các cột: {missing_columns}")
        return
    
    # Tìm các dòng có "Generate thất bại sau nhiều lần thử."
    failed_mask = df['Hoi_thoai_mo_rong'] == "Generate thất bại sau nhiều lần thử."
    failed_count = failed_mask.sum()
    
    if failed_count == 0:
        print("Không có dữ liệu nào bị lỗi cần generate lại!")
        return
    
    print(f"Tìm thấy {failed_count} dòng cần generate lại...")
    
    # Tạo bản sao để xử lý
    df_fixed = df.copy()
    
    # Xử lý từng dòng bị lỗi
    processed_count = 0
    for idx, row in df[failed_mask].iterrows():
        loai = row['Loại lừa đảo']
        mo_ta = row['Hội thoại']
        variant = row['Variant']
        
        print(f"Đang xử lý dòng {idx + 1}: {loai} (Variant {variant})")
        
        # Generate lại hội thoại
        new_dialogue = generate_expanded_dialogue(loai, mo_ta, variant)
        
        # Cập nhật dữ liệu
        df_fixed.loc[idx, 'Hoi_thoai_mo_rong'] = new_dialogue
        
        processed_count += 1
        print(f"Đã xử lý: {processed_count}/{failed_count}")
        
        # Lưu file NGAY SAU MỖI DÒNG để tránh mất dữ liệu
        df_fixed.to_csv(input_file, index=False, encoding='utf-8-sig')
        print(f"Đã lưu dòng {processed_count} vào file")
        
        # Sleep để tránh rate limit
        time.sleep(1)
    
    # Lưu file cuối cùng (ghi đè file gốc) - thực ra không cần vì đã lưu từng dòng rồi
    print(f"\nHoàn thành! Đã xử lý {processed_count} dòng bị lỗi.")
    print(f"Đã ghi đè trực tiếp vào file: {input_file}")
    print("Lưu ý: Mỗi dòng đã được lưu ngay sau khi xử lý thành công!")
    
    # Thống kê kết quả
    final_failed = (df_fixed['Hoi_thoai_mo_rong'] == "Generate thất bại sau nhiều lần thử.").sum()
    print(f"Số dòng vẫn bị lỗi sau khi xử lý: {final_failed}")
    
    if final_failed > 0:
        print("Một số dòng vẫn bị lỗi. Bạn có thể chạy lại script này để thử generate lại.")

if __name__ == "__main__":
    main()
