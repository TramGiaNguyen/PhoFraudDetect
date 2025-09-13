import pandas as pd
import os
import time
import asyncio
from groq import Groq
import random

# Thiết lập API key (thay bằng key thực của bạn, tốt nhất dùng os.environ để bảo mật)
GROQ_API_KEY = "your-api-key-here"
client = Groq(api_key=GROQ_API_KEY)

# Đường dẫn file input và output
input_file = 'scam_dataset.csv'
output_file = 'expanded_scam_all_types.csv'

# Đọc file CSV với encoding UTF-8 để hỗ trợ tiếng Việt
df = pd.read_csv(input_file, encoding='utf-8')

# Kiểm tra cột chính: 'Loai_lua_dao' và 'Mo_ta'. Nếu khác, raise error để bạn chỉnh sửa
if 'Loại lừa đảo' not in df.columns or 'Hội thoại' not in df.columns:
    raise ValueError("File CSV phải có cột 'Loại lừa đảo' và 'Hội thoại'. Hãy kiểm tra và rename cột nếu cần.")

# Thêm cột mới cho hội thoại mở rộng (sẽ append nhiều variants nếu cần)
expanded_data = []  # Sử dụng list để lưu nhiều variants

# Số lượng variants generate cho mỗi loại (tăng dataset, ví dụ: 5 để có 60 mẫu từ 12 loại)
num_variants = 5

# Max retry cho error handling
MAX_RETRIES = 3

# Hàm generate hội thoại mở rộng với prompt động dựa trên loại (async)
def generate_expanded_dialogue(loai, mo_ta, variant_idx):
    # Xác định loại prompt dựa trên loại lừa đảo (để hoàn thiện cho tất cả hình thức)
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
            chat_completion = client.chat.completions.create(  # Bỏ await
                messages=[
                    {"role": "system", "content": "Bạn là một AI chuyên generate script hội thoại lừa đảo đa dạng để mục đích nghiên cứu và phát hiện gian lận. Giữ an toàn, không khuyến khích hành vi xấu."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.7 + random.uniform(-0.1, 0.1),
                max_tokens=1024,  # Giảm xuống 1024 để tránh lỗi
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            retries += 1
            print(f"Lỗi khi generate cho {loai} (variant {variant_idx}), thử lại {retries}/{MAX_RETRIES}: {e}")
            time.sleep(2 ** retries)  # Dùng time.sleep thay vì asyncio.sleep
    raise Exception(f"Không thể generate hội thoại cho {loai} (variant {variant_idx}) sau {MAX_RETRIES} lần thử. Dừng chương trình để tránh tạo dữ liệu lỗi.")

# Main function async
def main():
    # Kiểm tra file output đã tồn tại chưa
    if os.path.exists(output_file):
        # Đọc file cũ để tiếp tục
        existing_df = pd.read_csv(output_file, encoding='utf-8-sig')
        print(f"Tiếp tục từ file cũ: {len(existing_df)} mẫu đã có")
        
        # Lấy index cuối cùng đã xử lý
        if 'Original_Index' in existing_df.columns:
            last_processed_index = existing_df['Original_Index'].max()
            print(f"Đã xử lý đến index: {last_processed_index}")
            
            # Tiếp tục từ index tiếp theo
            start_index = last_processed_index + 1
            remaining_df = df.iloc[start_index:]
            print(f"Tiếp tục từ index {start_index}, còn lại: {len(remaining_df)} records")
        else:
            # File cũ không có cột index, xử lý từ đầu
            remaining_df = df
            start_index = 0
            print("File cũ không có index, bắt đầu từ đầu")
    else:
        # File mới, khởi tạo empty
        expanded_data = []
        remaining_df = df
        start_index = 0
        print(f"Bắt đầu mới: {len(remaining_df)} records cần xử lý")
    
    # Khởi tạo expanded_data từ file cũ nếu có
    if os.path.exists(output_file):
        expanded_data = existing_df.to_dict('records')
    else:
        expanded_data = []
    
    # Loop qua từng row còn lại và generate nhiều variants
    for idx, (index, row) in enumerate(remaining_df.iterrows()):
        current_index = start_index + idx
        loai = row['Loại lừa đảo']
        mo_ta = row['Hội thoại']
        print(f"Đang xử lý index {current_index}: {loai} (với {num_variants} variants)")
        
        for variant_idx in range(1, num_variants + 1):
            try:
                expanded = generate_expanded_dialogue(loai, mo_ta, variant_idx)
                
                # Lưu vào list: Sao chép row gốc và thêm hội thoại
                new_row = row.copy()
                new_row['Hoi_thoai_mo_rong'] = expanded
                new_row['Variant'] = variant_idx
                new_row['Original_Index'] = current_index  # Lưu index gốc
                expanded_data.append(new_row)
            except Exception as e:
                print(f"Lỗi nghiêm trọng khi xử lý {loai} (variant {variant_idx}): {e}")
                print("Dừng chương trình để tránh tạo dữ liệu lỗi.")
                # Lưu dữ liệu đã xử lý trước khi dừng
                if expanded_data:
                    temp_df = pd.DataFrame(expanded_data)
                    temp_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                    print(f"Đã lưu {len(expanded_data)} mẫu đã xử lý thành công vào {output_file}")
                raise e  # Dừng chương trình
        
        # Lưu file ngay sau khi xử lý xong 1 record (để tránh mất dữ liệu)
        temp_df = pd.DataFrame(expanded_data)
        temp_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Đã lưu tạm thời: {len(temp_df)} mẫu (đến index {current_index})")
        
        # Sleep để tránh rate limit của Groq
        time.sleep(1)
    
    # Lưu file output cuối cùng
    final_df = pd.DataFrame(expanded_data)
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Hoàn thành! File output: {output_file}. Tổng số mẫu: {len(final_df)} (từ {len(df)} loại gốc).")

# Chạy main function
if __name__ == "__main__":
    main()