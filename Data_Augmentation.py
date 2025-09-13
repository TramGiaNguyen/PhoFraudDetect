# Import các thư viện cần thiết
import pandas as pd
import random
from googletrans import Translator  # Cài đặt: pip install googletrans==4.0.0-rc1 (nếu chưa có)
import re  # Để làm sạch text nếu cần
import asyncio
import json
import os
from datetime import datetime

# Khởi tạo Translator cho back-translation
translator = Translator()

# Định nghĩa dict synonym cho replacement (rule-based, dựa trên các từ phổ biến trong dataset lừa đảo)
# Bạn có thể mở rộng dict này dựa trên dataset
synonyms = {
    "chuyển": ["gửi", "chuyển khoản", "nộp"],
    "ngay": ["lập tức", "gấp", "nhanh chóng"],
    "OTP": ["mã xác thực", "mã OTP", "code xác nhận"],
    "tài khoản": ["STK", "số tài khoản", "account"],
    "quét": ["scan", "quét mã"],
    "QR": ["mã QR", "QR code"],
    "trúng thưởng": ["trúng giải", "nhận thưởng", "giải thưởng"],
    "phí": ["lệ phí", "chi phí", "tiền phí"],
    "lừa": ["gian lận", "lừa đảo"],  # Thêm nếu cần, nhưng tránh thay đổi nghĩa
    # Thêm nhiều hơn dựa trên dataset của bạn
}

# Danh sách noise sentences để thêm vào hội thoại (mô phỏng hội thoại dài hơn)
noises = [
    "Chào anh/chị, ",
    "Bạn khỏe không? ",
    "Tôi gọi từ công ty... ",
    "Xin lỗi làm phiền, nhưng... ",
    "Bạn có thời gian không? ",
    "Đây là cuộc gọi quan trọng... ",
    "Tôi muốn xác nhận thông tin... ",
]

# Hàm back_translate: Dịch sang tiếng Anh rồi ngược về tiếng Việt để tạo biến thể
async def back_translate(text: str) -> str:
    try:
        # Dịch sang tiếng Anh
        en_text = await translator.translate(text, dest='en')
        # Dịch ngược về tiếng Việt
        vi_text = await translator.translate(en_text.text, dest='vi')
        return vi_text.text
    except Exception as e:
        print(f"Lỗi back-translation: {e}")
        return text  # Giữ nguyên nếu lỗi

# Hàm back_translate với retry và fallback
async def back_translate_with_fallback(text: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            # Dịch sang tiếng Anh
            en_text = await translator.translate(text, dest='en')
            if not en_text or not en_text.text:
                continue
                
            # Dịch ngược về tiếng Việt
            vi_text = await translator.translate(en_text.text, dest='vi')
            if vi_text and vi_text.text:
                return vi_text.text
                
        except Exception as e:
            if attempt == max_retries - 1:  # Lần thử cuối
                print(f"Back-translation thất bại sau {max_retries} lần thử: {e}")
                raise e  # Ném lỗi để xử lý ở cấp cao hơn
            else:
                print(f"Lần thử {attempt + 1} thất bại, thử lại sau 2 giây...")
                await asyncio.sleep(2)  # Đợi 2 giây trước khi thử lại
    
    return text  # Giữ nguyên nếu tất cả đều thất bại

# Hàm add_noise: Thêm câu noise ngẫu nhiên vào đầu hoặc cuối hội thoại để mô phỏng dài hơn
def add_noise(text: str) -> str:
    if random.random() < 0.5:  # Xác suất 50% thêm noise
        noise = random.choice(noises)
        if random.random() < 0.5:
            text = noise + text  # Thêm đầu
        else:
            text = text + " " + noise  # Thêm cuối
    return text

# Hàm synonym_replacement: Thay thế từ đồng nghĩa với xác suất
def synonym_replacement(text: str) -> str:
    words = text.split()  # Tách từ (đơn giản, không dùng tokenizer phức tạp)
    for i, word in enumerate(words):
        if word in synonyms and random.random() < 0.3:  # Xác suất 30% thay thế
            replacement = random.choice(synonyms[word])
            words[i] = replacement
    return ' '.join(words)

# Hàm clean_text: Làm sạch text cơ bản (lowercase, xóa khoảng trắng thừa, giữ dấu tiếng Việt)
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Xóa khoảng trắng thừa
    text = re.sub(r'[^\w\s\.,!?]', '', text)  # Loại ký tự đặc biệt không cần (giữ dấu câu)
    return text.strip()

# Hàm augment_example: Áp dụng augmentation cho một hàng dữ liệu
async def augment_example(row, variant_counter, use_extended=True):
    # Chọn nguồn text để augment
    if use_extended and pd.notna(row['Hoi_thoai_mo_rong']) and row['Hoi_thoai_mo_rong'].strip():
        # Sử dụng Hoi_thoai_mo_rong nếu có và không rỗng
        original_conversation = row['Hoi_thoai_mo_rong']
        source_column = 'Hoi_thoai_mo_rong'
    else:
        # Fallback về Hội thoại nếu Hoi_thoai_mo_rong rỗng
        original_conversation = row['Hội thoại']
        source_column = 'Hội thoại'
    
    # Clean text
    conversation = clean_text(original_conversation)
    
    # Synonym replacement
    conversation = synonym_replacement(conversation)
    
    # Back-translation với xác suất 40%
    if random.random() < 0.4:
        conversation = await back_translate_with_fallback(conversation)
    
    # Add noise
    conversation = add_noise(conversation)
    
    # Tạo row mới
    augmented_row = row.copy()
    augmented_row['Hội thoại'] = conversation  # Luôn lưu vào cột Hội thoại
    augmented_row['Hoi_thoai_mo_rong'] = conversation  # Cũng lưu vào cột mở rộng
    augmented_row['Variant'] = f"aug_{variant_counter}_{source_column[:3]}"  # Đánh dấu nguồn gốc
    augmented_row['Original_Index'] = row.name  # Giữ index gốc để trace back
    return augmented_row

# Hàm lưu checkpoint
def save_checkpoint(checkpoint_file: str, current_index: int, total_processed: int, augmented_rows: list):
    checkpoint_data = {
        'current_index': current_index,
        'total_processed': total_processed,
        'timestamp': datetime.now().isoformat(),
        'status': 'in_progress'
    }
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Checkpoint đã lưu: index {current_index}, đã xử lý {total_processed} mẫu")

# Hàm load checkpoint
def load_checkpoint(checkpoint_file: str):
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            print(f"📂 Tìm thấy checkpoint: index {checkpoint_data['current_index']}, đã xử lý {checkpoint_data['total_processed']} mẫu")
            return checkpoint_data
        except Exception as e:
            print(f"⚠️  Không thể đọc checkpoint: {e}")
    
    return None

# Hàm lưu file tạm thời
def save_temp_file(temp_file: str, augmented_rows: list):
    temp_df = pd.DataFrame(augmented_rows)
    temp_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
    print(f"💾 File tạm thời đã lưu: {temp_file}")

# Main function: Đọc file, augment, lưu file mới với checkpoint
async def augment_dataset_with_checkpoint(input_file: str, output_file: str, augmentation_factor: float = 1.0, 
                                       batch_size: int = 100, checkpoint_file: str = 'augmentation_checkpoint.json'):
    """
    augmentation_factor: Số lần augment mỗi mẫu
    batch_size: Số mẫu xử lý trước khi lưu checkpoint
    checkpoint_file: File lưu trạng thái để resume
    """
    # Đọc CSV
    df = pd.read_csv(input_file, encoding='utf-8')
    
    # Kiểm tra cấu trúc file
    expected_columns = ['Loại lừa đảo', 'Hội thoại', 'Label', 'Hoi_thoai_mo_rong', 'Variant', 'Original_Index']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Thiếu các cột sau: {missing_columns}")
        print(f"Các cột hiện có: {list(df.columns)}")
    
    print(f"📊 Số lượng mẫu gốc: {len(df)}")
    
    # Thống kê về việc sử dụng các cột
    extended_count = df['Hoi_thoai_mo_rong'].notna().sum()
    basic_count = len(df) - extended_count
    print(f"📊 Số mẫu có Hoi_thoai_mo_rong: {extended_count}")
    print(f"📊 Số mẫu chỉ có Hội thoại: {basic_count}")
    
    # Kiểm tra checkpoint
    checkpoint = load_checkpoint(checkpoint_file)
    start_index = 0
    augmented_rows = []
    
    if checkpoint:
        start_index = checkpoint['current_index']
        print(f"🔄 Tiếp tục từ index {start_index}")
        
        # Load file tạm thời nếu có
        temp_file = f"temp_augmented_{start_index}.csv"
        if os.path.exists(temp_file):
            temp_df = pd.read_csv(temp_file, encoding='utf-8')
            augmented_rows = temp_df.to_dict('records')
            print(f"📂 Đã load {len(augmented_rows)} mẫu từ file tạm thời")
    
    # Tạo list các row augmented
    variant_counter = len(augmented_rows) + 1
    
    print(f"\n🚀 Bắt đầu augmentation từ index {start_index}...")
    
    try:
        for idx in range(start_index, len(df)):
            row = df.iloc[idx]
            
            # Hiển thị tiến trình mỗi 100 mẫu
            if idx % 100 == 0:
                print(f"📊 Đang xử lý mẫu {idx}/{len(df)}... (đã tạo {len(augmented_rows)} mẫu)")
            
            # Luôn giữ mẫu gốc
            augmented_rows.append(row)
            
            # Tạo thêm phiên bản augmented dựa trên factor
            num_augs = int(augmentation_factor) + (1 if random.random() < (augmentation_factor % 1) else 0)
            
            for aug_type in range(num_augs):
                try:
                    # Tạo variant từ Hoi_thoai_mo_rong (nếu có)
                    if pd.notna(row['Hoi_thoai_mo_rong']) and row['Hoi_thoai_mo_rong'].strip():
                        aug_row_extended = await augment_example(row, variant_counter, use_extended=True)
                        augmented_rows.append(aug_row_extended)
                        variant_counter += 1
                    
                    # Tạo variant từ Hội thoại (luôn có)
                    aug_row_basic = await augment_example(row, variant_counter, use_extended=False)
                    augmented_rows.append(aug_row_basic)
                    variant_counter += 1
                    
                except Exception as e:
                    print(f"❌ Lỗi khi augment mẫu {idx}: {e}")
                    raise e  # Ném lỗi để dừng chương trình
            
            # Lưu checkpoint và file tạm thời mỗi batch_size mẫu
            if (idx + 1) % batch_size == 0:
                save_checkpoint(checkpoint_file, idx + 1, len(augmented_rows), augmented_rows)
                temp_file = f"temp_augmented_{idx + 1}.csv"
                save_temp_file(temp_file, augmented_rows)
        
        print(f"✅ Hoàn thành augmentation!")
        
        # Tạo dataframe mới từ list rows
        augmented_df = pd.DataFrame(augmented_rows)
        
        # Shuffle để tránh bias
        print("🔄 Đang shuffle dữ liệu...")
        augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)
        
        # Lưu file output cuối cùng
        print("💾 Đang lưu file cuối cùng...")
        augmented_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # Xóa checkpoint và file tạm thời
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        # Xóa các file tạm thời
        for i in range(0, len(df), batch_size):
            temp_file = f"temp_augmented_{i}.csv"
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"✅ Số lượng mẫu sau augment: {len(augmented_df)}")
        print(f"✅ File augmented lưu tại: {output_file}")
        
        # Thống kê sau augment
        print(f"\n📊 Thống kê sau augmentation:")
        print(f"- Tổng số mẫu: {len(augmented_df)}")
        print(f"- Số mẫu gốc: {len(df)}")
        print(f"- Số mẫu được tạo thêm: {len(augmented_df) - len(df)}")
        
    except Exception as e:
        print(f"\n❌ Lỗi xảy ra tại index {idx}: {e}")
        print(f"💾 Đang lưu checkpoint và file tạm thời...")
        
        # Lưu checkpoint khi gặp lỗi
        save_checkpoint(checkpoint_file, idx, len(augmented_rows), augmented_rows)
        
        # Lưu file tạm thời
        temp_file = f"temp_augmented_{idx}.csv"
        save_temp_file(temp_file, augmented_rows)
        
        print(f"📋 Để tiếp tục, chạy lại script với cùng tham số")
        print(f"📁 Checkpoint: {checkpoint_file}")
        print(f"📁 File tạm thời: {temp_file}")
        
        raise e

# Chạy main function
if __name__ == "__main__":
    input_file = 'expanded_scam_all_types.csv'
    output_file = 'augmented_expanded_scam_dataset.csv'
    augmentation_factor = 1.0
    batch_size = 100  # Lưu checkpoint mỗi 100 mẫu
    checkpoint_file = 'augmentation_checkpoint.json'
    
    print("🚀 Bắt đầu chạy Data Augmentation với Checkpoint...")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")
    print(f"🔢 Augmentation factor: {augmentation_factor}")
    print(f"📦 Batch size: {batch_size}")
    print(f"📋 Checkpoint file: {checkpoint_file}")
    print("-" * 60)
    
    try:
        asyncio.run(augment_dataset_with_checkpoint(input_file, output_file, augmentation_factor, batch_size, checkpoint_file))
    except KeyboardInterrupt:
        print("\n⏹️  Chương trình bị dừng bởi người dùng")
        print("💾 Checkpoint đã được lưu, bạn có thể chạy lại để tiếp tục")
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {e}")
        print("💾 Checkpoint đã được lưu, bạn có thể chạy lại để tiếp tục")