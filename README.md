# Hệ thống Phát hiện Lừa đảo sử dụng PhoBERT

## Tổng quan

Dự án này xây dựng một mô hình phát hiện lừa đảo tự động sử dụng mô hình PhoBERT (Vietnamese BERT) để phân loại các cuộc hội thoại có phải là lừa đảo hay không. Hệ thống được thiết kế để xử lý các loại lừa đảo phổ biến tại Việt Nam.

## Cấu trúc dự án

```
eureka/
├── scam_dataset.csv                    # Dataset gốc (2,347 mẫu)
├── expanded_scam_all_types.csv         # Dataset đã mở rộng bằng AI
├── augmented_expanded_scam_dataset.csv # Dataset đã augment
├── preprocessed_scam_dataset/          # Dataset đã tiền xử lý
│   ├── train/                         # Tập huấn luyện
│   ├── validation/                    # Tập validation
│   └── test/                          # Tập kiểm thử
├── model/                             # Mô hình đã huấn luyện
│   ├── config.json
│   ├── model.safetensors               # Mô hình đã huấn luyện
│   ├── training_args.bin
│   └── ...
├── Generate_data.py                   # Tạo dữ liệu mở rộng
├── Data_Augmentation.py              # Augment dữ liệu
├── preprocessing.py                   # Tiền xử lý dữ liệu
├── training-phobert.ipynb            # Notebook huấn luyện
├── inference_phobert.py              # Script suy luận
├── regenerate_failed_data.py         # Tạo lại dữ liệu lỗi
├── test-token.py                     # Kiểm tra tokenization
├── sample1.txt                       # Mẫu hội thoại bình thường
├── sample2.txt                       # Mẫu hội thoại lừa đảo
└── scam_analysis_summary.txt         # Thống kê dataset
```

## Yêu cầu hệ thống

### Phần cứng
- **RAM**: Tối thiểu 8GB, khuyến nghị 16GB+
- **GPU**: NVIDIA GPU với VRAM ≥ 6GB (khuyến nghị RTX 3070+)
- **Dung lượng**: ~5GB cho dataset và mô hình

### Phần mềm
- Python 3.8+
- CUDA 11.8+ (nếu sử dụng GPU)
- Các thư viện Python (xem requirements.txt)

## Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd eureka
```

### 2. Cài đặt dependencies
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.2 accelerate>=0.34.0 datasets>=2.20.0 sentencepiece evaluate
pip install pandas numpy scikit-learn googletrans==4.0.0-rc1 psutil
```

### 3. Cấu hình API Key (tùy chọn)
Nếu muốn sử dụng Groq API để tạo dữ liệu:
```python
# Trong Generate_data.py và regenerate_failed_data.py
GROQ_API_KEY = "your-api-key-here"
```

## Quy trình chạy chương trình

### Bước 1: Tạo dữ liệu mở rộng (Tùy chọn)

Nếu muốn tạo thêm dữ liệu từ dataset gốc:

```bash
python Generate_data.py
```

**Mô tả**: Script này sử dụng Groq API để tạo thêm các biến thể hội thoại lừa đảo từ dataset gốc `scam_dataset.csv`, tạo ra file `expanded_scam_all_types.csv`.

**Tham số có thể điều chỉnh**:
- `num_variants`: Số biến thể tạo cho mỗi mẫu (mặc định: 5)
- `MAX_RETRIES`: Số lần thử lại khi gặp lỗi API (mặc định: 3)

### Bước 2: Augment dữ liệu

```bash
python Data_Augmentation.py
```

**Mô tả**: Script này áp dụng các kỹ thuật data augmentation:
- **Back-translation**: Dịch sang tiếng Anh rồi dịch ngược về tiếng Việt
- **Synonym replacement**: Thay thế từ đồng nghĩa
- **Noise injection**: Thêm câu noise để mô phỏng hội thoại dài hơn
- **Text cleaning**: Làm sạch và chuẩn hóa text

**Tính năng**:
- Hỗ trợ checkpoint để tiếp tục khi bị gián đoạn
- Xử lý batch để tiết kiệm RAM
- Tự động lưu file tạm thời

### Bước 3: Tiền xử lý dữ liệu

```bash
python preprocessing.py
```

**Mô tả**: Script này thực hiện:
- **Tokenization**: Sử dụng PhoBERT tokenizer
- **Sliding window**: Tạo các cửa sổ trượt cho hội thoại dài
- **Keyword extraction**: Đếm từ khóa lừa đảo
- **Train/Val/Test split**: Chia dataset theo tỷ lệ 80:10:10
- **Memory optimization**: Xử lý theo chunk để tiết kiệm RAM

**Cấu hình**:
- `MAX_LENGTH = 256`: Độ dài tối đa của sequence
- `SLIDING_STRIDE = 64`: Bước nhảy của sliding window
- `BATCH_SIZE = 16`: Kích thước batch xử lý
- `CHUNK_SIZE = 1000`: Kích thước chunk để tiết kiệm RAM

### Bước 4: Huấn luyện mô hình

Mở file `training-phobert.ipynb` và chạy các cell theo thứ tự:

```python
# Cell 1: Cài đặt và import
# Cell 2: Load dataset
# Cell 3: Cấu hình model và tokenizer
# Cell 4: Thiết lập training arguments
# Cell 5: Khởi tạo trainer
# Cell 6: Bắt đầu huấn luyện
# Cell 7: Đánh giá mô hình
```

**Cấu hình huấn luyện**:
- **Model**: `vinai/phobert-large`
- **Learning rate**: 2e-5
- **Batch size**: Tự động điều chỉnh theo VRAM
- **Epochs**: 3-5
- **Warmup steps**: 500
- **Weight decay**: 0.01

### Bước 5: Kiểm tra mô hình

```bash
python test-token.py
```

**Mô tả**: Kiểm tra tokenization và độ dài sequence.

### Bước 6: Suy luận

```bash
# Suy luận từ text
python inference_phobert.py --text "Nội dung hội thoại cần kiểm tra"

# Suy luận từ file
python inference_phobert.py --file sample1.txt

# Suy luận với auto-markers
python inference_phobert.py --file sample2.txt --add_markers
```

**Tham số**:
- `--model_dir`: Đường dẫn đến thư mục model (mặc định: "model")
- `--text`: Text cần phân loại
- `--file`: File chứa text cần phân loại
- `--max_length`: Độ dài tối đa sequence (mặc định: 256)
- `--add_markers`: Tự động thêm [USER]/[AGENT] markers

## Xử lý lỗi

### Tạo lại dữ liệu bị lỗi

```bash
python regenerate_failed_data.py
```

**Mô tả**: Script này tìm và tạo lại các mẫu dữ liệu bị lỗi trong quá trình generate.

### Kiểm tra checkpoint

Nếu quá trình augment bị gián đoạn, script sẽ tự động tạo checkpoint. Để tiếp tục:
```bash
python Data_Augmentation.py
```

## Kết quả mong đợi

### Dataset
- **Dataset gốc**: 2,347 mẫu
- **Dataset mở rộng**: ~11,000+ mẫu (sau generate và augment)
- **55 loại lừa đảo** khác nhau
- **Tỷ lệ**: 80% train, 10% validation, 10% test

### Hiệu suất mô hình
- **Accuracy**: >90%
- **F1-score**: >0.9
- **Precision**: >0.9
- **Recall**: >0.9

## Các loại lừa đảo được hỗ trợ

1. **Mạo danh cơ quan nhà nước** (Công an, Thuế, Tòa án)
2. **Lừa đảo trúng thưởng/quà tặng**
3. **Lừa đảo đầu tư** (tiền ảo, forex, chứng khoán)
4. **Lừa đảo tình cảm** (romance scam)
5. **Lừa đảo du lịch/vé máy bay**
6. **Lừa đảo việc làm**
7. **Lừa đảo qua mạng xã hội**
8. **Và 47 loại khác...**

---

**Lưu ý**: Dự án này chỉ phục vụ mục đích nghiên cứu và giáo dục. Không sử dụng để thực hiện các hành vi lừa đảo.
