# Import các thư viện cần thiết
import pandas as pd
import re
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, ClassLabel
from sklearn.model_selection import train_test_split
from functools import lru_cache  # Thêm cache decorator
import torch  # Để kiểm tra device nếu cần
import warnings  # Để warn nếu sequence dài
import gc  # Garbage collector để giải phóng bộ nhớ
import psutil  # Để theo dõi sử dụng RAM
import os

# Load PhoBERT tokenizer (large để max_length=256 theo giới hạn position embeddings)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")

# Định nghĩa max_length phù hợp cho PhoBERT-large
MAX_LENGTH = 256
SLIDING_STRIDE = 64

# Cấu hình tiết kiệm RAM
BATCH_SIZE = 16  # Tăng batch size để tận dụng vectorization
CHUNK_SIZE = 1000  # Tăng chunk size để giảm số lần xử lý I/O
MAX_MEMORY_MB = 4000  # Tăng giới hạn RAM để tận dụng hiệu quả hơn
PREFETCH_SIZE = 2  # Số lượng batch để prefetch

# Keywords mở rộng chi tiết dựa trên phân loại lừa đảo thực tế
keywords = [
    # === OTP VÀ XÁC THỰC ===
    "otp", "mã otp", "mã xác thực", "mã bảo mật", "mã xác nhận", "gửi otp", "nhập otp", "xác thực otp",
    "mã bảo mật", "mã xác minh", "mã bảo vệ", "mã khôi phục", "mã đăng nhập", "mã xác minh danh tính",
    "mã bảo mật 2fa", "mã xác thực 2 lớp", "mã bảo vệ tài khoản", "mã xác minh giao dịch", "mã bảo mật giao dịch",
    "mã xác nhận chuyển tiền", "mã bảo vệ chuyển tiền", "mã xác minh rút tiền", "mã bảo mật rút tiền",
    "mã xác thực đăng nhập", "mã bảo vệ đăng nhập", "mã xác minh đăng ký", "mã bảo mật đăng ký",
    "mã xác nhận thay đổi thông tin", "mã bảo vệ thay đổi thông tin", "mã xác minh khôi phục", "mã bảo mật khôi phục"
    
    # === CHUYỂN TIỀN VÀ TÀI CHÍNH ===
    "chuyển tiền", "chuyển ngay", "chuyển gấp", "chuyển khẩn cấp", "nộp phí", "đóng phí", "trả phí",
    "chuyển nhầm tiền", "chuyển sai tài khoản", "chuyển nhầm số tiền", "chuyển nhầm ngân hàng",
    "stk", "số tài khoản", "tài khoản ngân hàng", "visa", "mastercard", "thẻ tín dụng", "thẻ ghi nợ",
    "thẻ atm", "thẻ debit", "thẻ prepaid", "thẻ quốc tế", "thẻ nội địa", "thẻ chip", "thẻ từ",
    "wallet", "ví điện tử", "momo", "zalo pay", "vnpay", "airpay", "grabpay", "paypal", "wechat pay",
    "shinhan pay", "kakao pay", "line pay", "apple pay", "google pay", "samsung pay", "huawei pay",
    "hoàn tiền", "hoàn vốn", "hoàn thuế", "hoàn phí", "hoàn phí dịch vụ", "hoàn phí giao dịch",
    "tiền mặt", "vàng", "forex", "chứng khoán", "cổ phiếu", "trái phiếu", "quỹ đầu tư", "bảo hiểm nhân thọ"
    
    # === QR VÀ LINK ===
    "quét qr", "scan qr", "mã qr", "click link", "nhấp link", "truy cập link", "đường link",
    "nhấp vào link", "bấm vào link", "mở link", "vào link", "truy cập website", "vào website",
    "url", "website", "trang web", "trang mạng", "site", "web", "app", "ứng dụng", "tải app", "cài đặt app",
    "ứng dụng giả", "app giả", "phần mềm giả", "tải phần mềm", "cài đặt phần mềm", "tải game",
    "cài đặt game", "tải ứng dụng", "cài đặt ứng dụng", "tải tool", "cài đặt tool", "tải driver",
    "cài đặt driver", "tải plugin", "cài đặt plugin", "tải extension", "cài đặt extension"
    
    # === TÌNH HUỐNG KHẨN CẤP ===
    "gấp", "khẩn cấp", "nguy hiểm", "cấp cứu", "tình huống khẩn", "vấn đề nghiêm trọng",
    "vấn đề cấp bách", "tình huống nguy hiểm", "sự cố khẩn cấp", "trường hợp khẩn cấp",
    "lỗi hệ thống", "sự cố", "trục trặc", "bảo trì", "nâng cấp hệ thống", "cận tết",
    "cuối năm", "cuối tháng", "cuối tuần", "cuối ngày", "hết hạn", "sắp hết hạn",
    "hạn chót", "deadline", "thời hạn cuối", "kỳ hạn cuối", "ngày cuối", "giờ cuối"
    
    # === LỪA ĐẢO TÀI CHÍNH ===
    "trúng thưởng", "nhận thưởng", "giải thưởng", "tiền thưởng", "quà tặng", "tặng quà",
    "trúng giải", "nhận giải", "giải đặc biệt", "giải nhất", "giải nhì", "giải ba", "giải khuyến khích",
    "đầu tư", "đầu tư sinh lời", "lãi suất cao", "nhân đôi tiền", "gấp đôi", "lợi nhuận",
    "lãi suất ưu đãi", "lãi suất đặc biệt", "lãi suất khủng", "lãi suất cao nhất", "lãi suất tốt nhất",
    "cơ hội kiếm tiền", "kiếm tiền online", "công việc online", "part time", "làm giàu nhanh",
    "làm giàu trong 1 tháng", "làm giàu trong 1 tuần", "làm giàu trong 1 ngày", "kiếm tiền dễ dàng",
    "kiếm tiền không cần vốn", "kiếm tiền không cần kinh nghiệm", "kiếm tiền không cần bằng cấp",
    "đầu tư bất động sản", "đầu tư tiền điện tử", "đầu tư vàng", "đầu tư bitcoin", "đầu tư ethereum",
    "đầu tư crypto", "đầu tư coin", "đầu tư token", "đầu tư nft", "đầu tư metaverse",
    "đa cấp", "bán hàng đa cấp", "kinh doanh đa cấp", "marketing đa cấp", "bán hàng trực tuyến đa cấp"
    
    # === MẠO DANH VÀ GIẢ MẠO ===
    "mạo danh", "giả mạo", "giả vờ", "giả bộ", "giả danh", "mạo nhận", "giả nhận",
    "ngân hàng", "ngân hàng nhà nước", "ngân hàng thương mại", "ngân hàng quốc tế", "ngân hàng nước ngoài",
    "công an", "cảnh sát", "cảnh sát giao thông", "cảnh sát hình sự", "cảnh sát kinh tế", "cảnh sát môi trường",
    "cơ quan nhà nước", "chính phủ", "bộ tài chính", "bộ công an", "bộ quốc phòng", "bộ ngoại giao",
    "công ty", "tập đoàn", "công ty đa quốc gia", "công ty quốc tế", "công ty nước ngoài",
    "nhân viên", "cán bộ", "quản lý", "giám đốc", "phó giám đốc", "trưởng phòng", "phó phòng",
    "nhân viên bảo hiểm", "nhân viên ngân hàng", "nhân viên thuế", "nhân viên hải quan",
    "tổ chức", "tổ chức phi chính phủ", "tổ chức quốc tế", "tổ chức từ thiện", "tổ chức nhân đạo",
    "trường học", "đại học", "cao đẳng", "trung cấp", "trung học", "tiểu học", "mầm non",
    "người quen", "bạn bè", "đồng nghiệp", "hàng xóm", "người thân", "gia đình", "bạn học",
    "tiktok", "facebook", "zalo", "telegram", "viber", "wechat", "weibo", "line", "kakao talk",
    "điện lực", "bảo hiểm", "bưu điện", "cơ quan thuế", "cơ quan hải quan", "cơ quan bảo hiểm xã hội",
    "cơ quan bảo hiểm y tế", "cơ quan bảo hiểm thất nghiệp", "cơ quan bảo hiểm tai nạn lao động"
    
    # === CÔNG NGHỆ VÀ HACK ===
    "hack", "bị hack", "tài khoản bị hack", "bảo mật", "bảo vệ tài khoản", "khóa tài khoản",
    "tài khoản bị khóa", "tài khoản bị đóng", "tài khoản bị đình chỉ", "tài khoản bị tạm khóa",
    "deepfake", "video giả", "ảnh giả", "tin nhắn giả", "cuộc gọi giả", "công nghệ cao",
    "công nghệ tiên tiến", "công nghệ mới nhất", "công nghệ độc quyền", "công nghệ độc đáo",
    "gọi điện tự động", "gọi điện robot", "gọi điện máy", "gọi điện tự động", "gọi điện hàng loạt",
    "email giả mạo", "email giả", "email lừa đảo", "email spam", "email rác", "email độc hại",
    "sim rác", "sim số đẹp", "sim giả", "sim lừa đảo", "sim spam", "sim độc hại",
    "ngân hàng giả", "ngân hàng lừa đảo", "ngân hàng spam", "ngân hàng độc hại",
    "website giả", "website lừa đảo", "website spam", "website độc hại", "app giả", "app lừa đảo"
    
    # === THÔNG TIN CÁ NHÂN ===
    "cmnd", "cccd", "căn cước", "hộ chiếu", "giấy tờ tùy thân", "thông tin cá nhân",
    "thông tin riêng tư", "thông tin bí mật", "thông tin nhạy cảm", "thông tin quan trọng",
    "số điện thoại", "số di động", "số mobile", "số liên lạc", "số liên hệ", "số hotline",
    "email", "địa chỉ email", "email liên hệ", "email liên lạc", "email công việc", "email cá nhân",
    "địa chỉ", "địa chỉ nhà", "địa chỉ cư trú", "địa chỉ thường trú", "địa chỉ tạm trú",
    "ngày sinh", "nơi sinh", "quê quán", "quốc tịch", "dân tộc", "tôn giáo", "nghề nghiệp",
    "thừa kế", "di chúc", "tài sản thừa kế", "quyền thừa kế", "người thừa kế", "người được thừa kế"
    
    # === DU LỊCH VÀ GIẢI TRÍ ===
    "du lịch", "mùa du lịch", "vé concert", "vé máy bay", "tour du lịch", "du lịch giá rẻ",
    "du lịch khuyến mãi", "du lịch giảm giá", "du lịch sale", "du lịch ưu đãi", "du lịch đặc biệt",
    "vé concert", "vé nhạc hội", "vé show", "vé biểu diễn", "vé ca nhạc", "vé hòa nhạc",
    "vé máy bay", "vé bay", "vé hàng không", "vé khứ hồi", "vé một chiều", "vé nội địa", "vé quốc tế",
    "tour du lịch", "chuyến du lịch", "hành trình du lịch", "lịch trình du lịch", "kế hoạch du lịch",
    "đặt phòng", "đặt khách sạn", "đặt resort", "đặt homestay", "đặt villa", "đặt căn hộ",
    "vé xem phim", "vé phim", "vé rạp chiếu phim", "vé cinema", "vé movie", "vé show phim",
    "đặt cọc thuê nhà", "đặt cọc thuê phòng", "đặt cọc thuê căn hộ", "đặt cọc thuê villa",
    "khách sạn", "resort", "homestay", "villa", "căn hộ", "nhà nghỉ", "motel", "hostel"
    
    # === MUA SẮM VÀ BÁN HÀNG ===
    "bán hàng online", "mua sắm online", "bán hàng trực tuyến", "bán hàng đa cấp",
    "bán hàng qua mạng", "bán hàng internet", "bán hàng website", "bán hàng app",
    "mua hàng không giao", "mua hàng bị lừa", "mua hàng giả", "mua hàng kém chất lượng",
    "đặt cọc", "đặt cọc mua hàng", "đặt cọc đặt hàng", "đặt cọc đặt sản phẩm",
    "thanh toán trước", "thanh toán trước khi giao hàng", "thanh toán trước khi nhận hàng",
    "hoàn tiền", "hoàn vốn", "hoàn phí", "hoàn phí vận chuyển", "hoàn phí giao hàng",
    "xe máy", "xe tay ga", "xe số", "xe côn tay", "xe mô tô", "xe gắn máy",
    "nạp thẻ điện thoại", "nạp tiền điện thoại", "nạp thẻ sim", "nạp tiền sim",
    "số đẹp", "sim số đẹp", "số điện thoại đẹp", "số di động đẹp", "số mobile đẹp",
    "sản phẩm giả", "hàng giả", "hàng nhái", "hàng kém chất lượng", "hàng không chính hãng"
    
    # === VIỆC LÀM VÀ TUYỂN DỤNG ===
    "tuyển dụng", "việc nhẹ lương cao", "lương cao", "tuyển người mẫu", "công việc online",
    "tuyển dụng online", "tuyển dụng qua mạng", "tuyển dụng internet", "tuyển dụng website",
    "việc nhẹ lương cao", "việc dễ lương cao", "việc đơn giản lương cao", "việc không cần kinh nghiệm",
    "lương cao", "lương khủng", "lương tốt", "lương hấp dẫn", "lương cạnh tranh", "lương thị trường",
    "tuyển người mẫu", "tuyển diễn viên", "tuyển ca sĩ", "tuyển người đẹp", "tuyển người nổi tiếng",
    "công việc online", "việc làm online", "công việc qua mạng", "việc làm qua mạng",
    "part time", "full time", "làm việc tại nhà", "làm việc từ xa", "làm việc online",
    "kiếm tiền online", "kiếm tiền qua mạng", "kiếm tiền internet", "kiếm tiền website",
    "cơ hội việc làm", "cơ hội nghề nghiệp", "cơ hội thăng tiến", "cơ hội phát triển"
    
    # === TỪ THIỆN VÀ VAY MƯỢN ===
    "kêu gọi từ thiện", "quyên góp", "ủng hộ", "vay tiền online", "vay tiền nhanh",
    "kêu gọi quyên góp", "kêu gọi ủng hộ", "kêu gọi từ thiện", "kêu gọi nhân đạo",
    "quyên góp", "ủng hộ", "hỗ trợ", "giúp đỡ", "chia sẻ", "đóng góp", "tài trợ",
    "vay tiền online", "vay tiền qua mạng", "vay tiền internet", "vay tiền website",
    "vay tiền nhanh", "vay tiền gấp", "vay tiền khẩn cấp", "vay tiền cấp bách",
    "vay tiền không cần thế chấp", "vay tiền không cần bảo lãnh", "vay tiền không cần giấy tờ",
    "vay tiền lãi suất thấp", "vay tiền lãi suất ưu đãi", "vay tiền lãi suất tốt",
    "cầm đồ", "thế chấp", "bảo lãnh", "đảm bảo", "cam kết", "hứa hẹn", "đảm bảo trả nợ"
    
    # === GAME VÀ GIẢI TRÍ ONLINE ===
    "game online", "chơi game kiếm tiền", "game thủ", "esports", "streaming", "youtube",
    "game online", "game mobile", "game pc", "game console", "game điện tử", "game video",
    "chơi game kiếm tiền", "game kiếm tiền", "game thắng tiền", "game đánh bài", "game casino",
    "game thủ", "game player", "người chơi game", "cộng đồng game", "hội game thủ",
    "esports", "thể thao điện tử", "thi đấu game", "giải đấu game", "champion game",
    "streaming", "phát trực tiếp", "live stream", "phát sóng trực tiếp", "phát sóng live",
    "youtube", "tiktok", "instagram", "facebook", "zalo", "telegram", "viber", "wechat", "weibo",
    "mạng xã hội", "social media", "platform", "nền tảng", "ứng dụng mạng xã hội"
    
    # === QUẢNG CÁO VÀ MARKETING ===
    "chạy quảng cáo", "quảng cáo online", "marketing online", "seo", "google ads", "facebook ads",
    "chạy quảng cáo", "đặt quảng cáo", "mua quảng cáo", "đăng quảng cáo", "phát quảng cáo",
    "quảng cáo online", "quảng cáo internet", "quảng cáo mạng", "quảng cáo website", "quảng cáo app",
    "marketing online", "marketing internet", "marketing mạng", "marketing số", "marketing điện tử",
    "seo", "tối ưu hóa công cụ tìm kiếm", "tối ưu seo", "seo website", "seo từ khóa",
    "google ads", "facebook ads", "tiktok ads", "youtube ads", "instagram ads", "twitter ads",
    "quảng cáo google", "quảng cáo facebook", "quảng cáo tiktok", "quảng cáo youtube",
    "influencer", "kols", "người nổi tiếng", "người có ảnh hưởng", "người có tầm ảnh hưởng",
    "người có sức ảnh hưởng", "người có uy tín", "người có danh tiếng", "người có tiếng nói"
    
    # === TỪ NGỮ CẢNH BÁO ===
    "cảnh báo", "chú ý", "quan trọng", "nghiêm trọng", "nguy hiểm", "cẩn thận",
    "cảnh báo", "chú ý", "quan trọng", "nghiêm trọng", "nguy hiểm", "cẩn thận",
    "cảnh báo quan trọng", "cảnh báo khẩn cấp", "cảnh báo nguy hiểm", "cảnh báo nghiêm trọng",
    "chú ý quan trọng", "chú ý khẩn cấp", "chú ý nguy hiểm", "chú ý nghiêm trọng",
    "quan trọng", "rất quan trọng", "cực kỳ quan trọng", "vô cùng quan trọng",
    "nghiêm trọng", "rất nghiêm trọng", "cực kỳ nghiêm trọng", "vô cùng nghiêm trọng",
    "nguy hiểm", "rất nguy hiểm", "cực kỳ nguy hiểm", "vô cùng nguy hiểm",
    "cẩn thận", "rất cẩn thận", "cực kỳ cẩn thận", "vô cùng cẩn thận",
    "không được chia sẻ", "bí mật", "riêng tư", "không được tiết lộ", "tuyệt mật",
    "không được chia sẻ", "không được tiết lộ", "không được nói", "không được kể",
    "bí mật", "riêng tư", "tuyệt mật", "cực kỳ bí mật", "vô cùng bí mật"
    
    # === TỪ NGỮ THUYẾT PHỤC ===
    "tin tôi đi", "đảm bảo", "chắc chắn", "100%", "không có gì phải lo",
    "tin tôi đi", "tin tôi", "hãy tin tôi", "tôi đảm bảo", "tôi chắc chắn", "tôi cam kết",
    "đảm bảo", "chắc chắn", "100%", "không có gì phải lo", "không có gì phải sợ",
    "đảm bảo 100%", "chắc chắn 100%", "không có gì phải lo", "không có gì phải sợ",
    "cơ hội duy nhất", "chỉ hôm nay", "giới hạn thời gian", "số lượng có hạn",
    "cơ hội duy nhất", "cơ hội cuối cùng", "cơ hội hiếm có", "cơ hội đặc biệt",
    "chỉ hôm nay", "chỉ ngày hôm nay", "chỉ trong ngày", "chỉ trong hôm nay",
    "giới hạn thời gian", "thời gian có hạn", "thời gian giới hạn", "thời gian cuối",
    "số lượng có hạn", "số lượng giới hạn", "số lượng cuối", "số lượng cuối cùng",
    "ưu đãi đặc biệt", "giảm giá sốc", "khuyến mãi lớn", "sale off",
    "ưu đãi đặc biệt", "ưu đãi cuối cùng", "ưu đãi hiếm có", "ưu đãi độc quyền",
    "giảm giá sốc", "giảm giá khủng", "giảm giá cực sốc", "giảm giá không tưởng",
    "khuyến mãi lớn", "khuyến mãi khủng", "khuyến mãi cực lớn", "khuyến mãi không tưởng",
    "sale off", "sale khủng", "sale cực khủng", "sale không tưởng"
    
    # === TỪ NGỮ TẠO ÁP LỰC ===
    "phải làm ngay", "không được chậm trễ", "hậu quả nghiêm trọng", "sẽ bị phạt",
    "phải làm ngay", "phải làm gấp", "phải làm khẩn cấp", "phải làm ngay lập tức",
    "không được chậm trễ", "không được trì hoãn", "không được để lâu", "không được để chậm",
    "hậu quả nghiêm trọng", "hậu quả khủng khiếp", "hậu quả không lường trước", "hậu quả đáng sợ",
    "sẽ bị phạt", "sẽ bị khóa", "sẽ bị mất", "sẽ bị đóng", "sẽ bị hủy", "sẽ bị tịch thu",
    "sẽ bị phạt", "sẽ bị phạt tiền", "sẽ bị phạt nặng", "sẽ bị phạt rất nặng",
    "sẽ bị khóa", "sẽ bị đóng", "sẽ bị đình chỉ", "sẽ bị tạm khóa", "sẽ bị tạm đóng",
    "sẽ bị mất", "sẽ bị mất vĩnh viễn", "sẽ bị mất hoàn toàn", "sẽ bị mất tất cả",
    "sẽ bị đóng", "sẽ bị đóng vĩnh viễn", "sẽ bị đóng hoàn toàn", "sẽ bị đóng tất cả",
    "sẽ bị hủy", "sẽ bị hủy vĩnh viễn", "sẽ bị hủy hoàn toàn", "sẽ bị hủy tất cả",
    "sẽ bị tịch thu", "sẽ bị kiểm tra", "sẽ bị điều tra", "sẽ bị bắt", "sẽ bị phạt tiền",
    "sẽ bị tịch thu", "sẽ bị tịch thu hoàn toàn", "sẽ bị tịch thu tất cả", "sẽ bị tịch thu vĩnh viễn",
    "sẽ bị kiểm tra", "sẽ bị điều tra", "sẽ bị bắt", "sẽ bị phạt tiền",
    "sẽ bị kiểm tra", "sẽ bị kiểm tra gắt gao", "sẽ bị kiểm tra nghiêm ngặt", "sẽ bị kiểm tra kỹ lưỡng",
    "sẽ bị điều tra", "sẽ bị điều tra gắt gao", "sẽ bị điều tra nghiêm ngặt", "sẽ bị điều tra kỹ lưỡng",
    "sẽ bị bắt", "sẽ bị bắt ngay", "sẽ bị bắt gấp", "sẽ bị bắt khẩn cấp",
    "sẽ bị phạt tiền", "sẽ bị phạt tiền nặng", "sẽ bị phạt tiền rất nặng", "sẽ bị phạt tiền khủng khiếp"
    
    # === TỪ NGỮ THỜI GIAN ===
    "hôm nay", "ngay bây giờ", "trong vòng 24h", "trước 12h đêm", "cuối tuần",
    "hôm nay", "ngày hôm nay", "hôm nay", "ngày hôm nay", "hôm nay", "ngày hôm nay",
    "ngay bây giờ", "ngay lập tức", "ngay tức khắc", "ngay tức thì", "ngay tức khắc",
    "trong vòng 24h", "trong vòng 1 ngày", "trong vòng 24 giờ", "trong vòng 1 ngày",
    "trước 12h đêm", "trước 12 giờ đêm", "trước 12 giờ tối", "trước 12 giờ tối",
    "cuối tuần", "cuối tháng", "cuối năm", "tết", "lễ", "kỳ nghỉ", "mùa cao điểm",
    "cuối tuần", "cuối tuần này", "cuối tuần tới", "cuối tuần sau", "cuối tuần trước",
    "cuối tháng", "cuối tháng này", "cuối tháng tới", "cuối tháng sau", "cuối tháng trước",
    "cuối năm", "cuối năm này", "cuối năm tới", "cuối năm sau", "cuối năm trước",
    "tết", "lễ", "kỳ nghỉ", "mùa cao điểm", "mùa du lịch", "mùa lễ hội", "mùa đặc biệt"
    
    # === TỪ NGỮ SỐ LƯỢNG ===
    "chỉ còn", "số lượng có hạn", "giới hạn", "đã hết", "sắp hết", "cuối cùng",
    "chỉ còn", "chỉ còn lại", "chỉ còn sót lại", "chỉ còn thừa lại", "chỉ còn dư lại",
    "số lượng có hạn", "số lượng giới hạn", "số lượng cuối", "số lượng cuối cùng",
    "giới hạn", "giới hạn cuối", "giới hạn cuối cùng", "giới hạn cuối cùng",
    "đã hết", "sắp hết", "cuối cùng", "lần cuối", "cơ hội cuối", "ưu đãi cuối", "giảm giá cuối",
    "đã hết", "đã hết sạch", "đã hết hoàn toàn", "đã hết tất cả", "đã hết vĩnh viễn",
    "sắp hết", "sắp hết sạch", "sắp hết hoàn toàn", "sắp hết tất cả", "sắp hết vĩnh viễn",
    "cuối cùng", "lần cuối", "cơ hội cuối", "ưu đãi cuối", "giảm giá cuối",
    "cuối cùng", "cuối cùng", "cuối cùng", "cuối cùng", "cuối cùng",
    "lần cuối", "lần cuối cùng", "lần cuối cùng", "lần cuối cùng", "lần cuối cùng",
    "cơ hội cuối", "ưu đãi cuối", "giảm giá cuối", "cơ hội cuối", "ưu đãi cuối", "giảm giá cuối"
    
    # === TỪ NGỮ ĐỊA LÝ ===
    "từ nước ngoài", "quốc tế", "toàn cầu", "châu âu", "châu mỹ", "châu á",
    "từ nước ngoài", "từ quốc tế", "từ toàn cầu", "từ châu âu", "từ châu mỹ", "từ châu á",
    "quốc tế", "toàn cầu", "châu âu", "châu mỹ", "châu á", "châu phi", "châu úc",
    "singapore", "hong kong", "đài loan", "hàn quốc", "nhật bản", "trung quốc",
    "singapore", "singapore", "hong kong", "đài loan", "hàn quốc", "nhật bản", "trung quốc",
    "mỹ", "anh", "đức", "pháp", "canada", "australia", "new zealand", "switzerland", "netherlands",
    "mỹ", "anh", "đức", "pháp", "canada", "australia", "new zealand", "switzerland", "netherlands",
    "united states", "united kingdom", "germany", "france", "canada", "australia", "new zealand",
    "switzerland", "netherlands", "belgium", "austria", "sweden", "norway", "denmark", "finland"
]

# Hàm clean_text: Làm sạch hội thoại (lowercase, xóa thừa, giữ dấu tiếng Việt)
@lru_cache(maxsize=10000)
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Lowercase để đồng nhất  
    text = re.sub(r'\s+', ' ', text)  # Xóa khoảng trắng thừa
    text = re.sub(r'[^\w\s\.,!?]', '', text)  # Loại ký tự đặc biệt không cần (giữ dấu câu)
    return text.strip()

@lru_cache(maxsize=10000)
def cached_tokenize(text):
    return tokenizer(
        text,
        truncation=True,
        padding='max_length', 
        max_length=MAX_LENGTH,
        return_overflowing_tokens=True,
        stride=SLIDING_STRIDE,
    )

# Hàm tối ưu hóa keywords để tiết kiệm RAM
def optimize_keywords():
    """Tối ưu hóa keywords để tiết kiệm RAM"""
    global keywords
    # Loại bỏ duplicates và sắp xếp
    keywords = list(set(keywords))
    keywords.sort()
    print(f"🔍 Đã tối ưu hóa keywords: {len(keywords)} từ khóa duy nhất")
    return keywords

# Hàm preprocess theo batch: tạo các cửa sổ trượt cho mỗi hội thoại trong batch
# Trả về danh sách các cửa sổ (phẳng) với orig_id cho phép gộp lại khi đánh giá
def preprocess_batch(examples, indices):
    out_input_ids = []
    out_attention_mask = []
    out_labels = []
    out_keyword_count = []
    out_core_text = []
    out_extended_text = []
    out_orig_id = []
    out_text = []
    
    texts_ext = examples.get('Hoi_thoai_mo_rong')
    texts_std = examples.get('Hội thoại')
    labels = examples.get('Label')
    
    batch_size = len(indices)
    for i in range(batch_size):
        # Lấy cả 2 loại text
        core_text = texts_std[i] if texts_std is not None else ''
        extended_text = texts_ext[i] if texts_ext is not None else ''
        
        # Clean cả 2 loại text
        core_conv = clean_text(core_text)
        extended_conv = clean_text(extended_text)
        
        # Ghép 2 text với [SEP] token để model phân biệt
        full_text = core_conv + " [SEP] " + extended_conv
        
        # Tokenize text đã ghép
        enc = cached_tokenize(full_text)
        num_windows = len(enc['input_ids'])
        
        # Tính số cửa sổ cần thiết
        num_required = max(3, num_windows // 128)  # Ít nhất 3 cửa sổ, thêm 1 cửa sổ cho mỗi 128 token
        selected_windows = [int(i * (num_windows - 1) / (num_required - 1)) for i in range(num_required)]
        
        # Lấy các cửa sổ đã chọn
        for idx in selected_windows:
            # Lấy tokens từ cửa sổ được chọn và chuyển về list
            window_input_ids = enc['input_ids'][idx]
            window_attention_mask = enc['attention_mask'][idx]
            
            # Chuyển tensor về list nếu cần
            if hasattr(window_input_ids, 'tolist'):
                window_input_ids = window_input_ids.tolist()
            if hasattr(window_attention_mask, 'tolist'):
                window_attention_mask = window_attention_mask.tolist()
            
            # Đảm bảo đây là list
            if not isinstance(window_input_ids, list):
                window_input_ids = list(window_input_ids)
            if not isinstance(window_attention_mask, list):
                window_attention_mask = list(window_attention_mask)
            
            # Validation: đảm bảo độ dài đúng
            if len(window_input_ids) != MAX_LENGTH:
                print(f"⚠️ Warning: Window {idx} có độ dài {len(window_input_ids)} != {MAX_LENGTH}")
            
            # Validation: đảm bảo attention_mask có cùng độ dài
            if len(window_attention_mask) != len(window_input_ids):
                print(f"⚠️ Warning: Attention mask và input_ids có độ dài khác nhau: {len(window_attention_mask)} vs {len(window_input_ids)}")
            
            out_input_ids.append(window_input_ids)
            out_attention_mask.append(window_attention_mask)
            out_labels.append(int(labels[i]))
            kw_count = sum(1 for kw in keywords if kw in (core_conv + extended_conv))
            out_keyword_count.append(kw_count)
            # Lưu text đã ghép
            out_text.append(full_text)
            # Lưu core_text và extended_text cho mỗi cửa sổ
            out_core_text.append(core_conv)
            out_extended_text.append(extended_conv)
            out_orig_id.append(int(indices[i]))
     
    # Debug: kiểm tra format trước khi return
    if out_input_ids and out_attention_mask:
        sample_input_ids = out_input_ids[0]
        sample_attention_mask = out_attention_mask[0]
        print(f"🔍 Debug - Sample input_ids type: {type(sample_input_ids)}, length: {len(sample_input_ids) if isinstance(sample_input_ids, list) else 'N/A'}")
        print(f"🔍 Debug - Sample attention_mask type: {type(sample_attention_mask)}, length: {len(sample_attention_mask) if isinstance(sample_attention_mask, list) else 'N/A'}")
        print(f"🔍 Debug - First 5 input_ids: {sample_input_ids[:5] if isinstance(sample_input_ids, list) else sample_input_ids}")
        print(f"🔍 Debug - First 5 attention_mask: {sample_attention_mask[:5] if isinstance(sample_attention_mask, list) else sample_attention_mask}")
    
    return {
        'input_ids': out_input_ids,
        'attention_mask': out_attention_mask,
        'labels': out_labels,
        'keyword_count': out_keyword_count,
        'core_text': out_core_text,
        'extended_text': out_extended_text,
        'orig_id': out_orig_id,
        'text': out_text,
    }

# Hàm tiết kiệm RAM: xử lý từng chunk nhỏ
def preprocess_chunk(chunk_data, chunk_id):
    """Xử lý từng chunk nhỏ để tiết kiệm RAM"""
    print(f"🔄 Đang xử lý chunk {chunk_id + 1}...")
    
    # Chuyển chunk thành Dataset
    chunk_dataset = Dataset.from_pandas(chunk_data)
    
    # Preprocess chunk với batch size nhỏ
    processed_chunk = chunk_dataset.map(
        preprocess_batch,
        with_indices=True,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=1,
        remove_columns=chunk_dataset.column_names,
    )
    
    # Giải phóng bộ nhớ chunk gốc
    del chunk_dataset
    gc.collect()
    
    return processed_chunk

# Hàm theo dõi RAM
def monitor_memory():
    """Theo dõi sử dụng RAM"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"💾 RAM hiện tại: {memory_mb:.1f} MB")
        return memory_mb
    except:
        print("⚠️ Không thể theo dõi RAM (psutil không có sẵn)")
        return 0

# Main function: Đọc file, preprocess, stratified split (handle imbalance), lưu dataset
def preprocess_dataset(input_file: str, output_dir: str = './preprocessed_scam_dataset'):
    """
    input_file: Đường dẫn đến augmented_expanded_scam_dataset.csv
    output_dir: Thư mục lưu DatasetDict tokenized (dễ load cho train)
    """
    global CHUNK_SIZE  # Khai báo global ở đầu function
    
    print("🚀 Bắt đầu preprocessing với chiến lược tiết kiệm RAM...")
    monitor_memory()
    
    # Đọc CSV theo chunk để tiết kiệm RAM
    print(f"📖 Đang đọc file: {input_file}")
    chunk_list = []
    total_rows = 0
    
    # Đọc file theo chunk
    for chunk_id, chunk_df in enumerate(pd.read_csv(input_file, encoding='utf-8', chunksize=CHUNK_SIZE)):
        chunk_list.append(chunk_df)
        total_rows += len(chunk_df)
        print(f"📊 Đã đọc chunk {chunk_id + 1}: {len(chunk_df)} mẫu")
        monitor_memory()
    
    print(f"✅ Tổng số mẫu: {total_rows}")
    
    # Xử lý từng chunk để tiết kiệm RAM
    processed_chunks = []
    for chunk_id, chunk_df in enumerate(chunk_list):
        print(f"\n🔄 Xử lý chunk {chunk_id + 1}/{len(chunk_list)}")
        
        # Xử lý chunk
        processed_chunk = preprocess_chunk(chunk_df, chunk_id)
        processed_chunks.append(processed_chunk)
        
        # Giải phóng bộ nhớ chunk gốc
        del chunk_df
        gc.collect()
        monitor_memory()
        
        # Nếu RAM quá cao, force garbage collection
        current_ram = monitor_memory()
        if current_ram > MAX_MEMORY_MB:  # Nếu RAM vượt quá giới hạn
            print(f"⚠️ RAM cao ({current_ram:.1f} MB), đang force garbage collection...")
            gc.collect()
            import time
            time.sleep(3)  # Đợi lâu hơn để hệ thống giải phóng bộ nhớ
            
            # Nếu vẫn cao, giảm chunk size
            if monitor_memory() > MAX_MEMORY_MB:
                print("🚨 RAM vẫn cao, đang giảm chunk size...")
                CHUNK_SIZE = max(50, CHUNK_SIZE // 2)  # Giảm xuống tối thiểu 50
                print(f"📉 Chunk size mới: {CHUNK_SIZE}")
                
                # Force garbage collection thêm
                gc.collect()
                time.sleep(2)
    
    # Giải phóng bộ nhớ chunk list
    del chunk_list
    gc.collect()
    
    print("\n🔗 Đang gộp các chunk đã xử lý...")
    
    # Gộp các chunk đã xử lý theo batch nhỏ để tiết kiệm RAM
    print("🔗 Đang gộp các chunk theo batch nhỏ...")
    
    # Import concatenate_datasets
    from datasets import concatenate_datasets
    
    # Gộp từng batch nhỏ để tránh OOM
    batch_size = 3  # Gộp 3 chunk một lần
    preprocessed_dataset = None
    
    for i in range(0, len(processed_chunks), batch_size):
        batch_chunks = processed_chunks[i:i+batch_size]
        print(f"🔄 Gộp batch {i//batch_size + 1}: chunks {i+1}-{min(i+batch_size, len(processed_chunks))}")
        
        if preprocessed_dataset is None:
            preprocessed_dataset = batch_chunks[0]
            for chunk in batch_chunks[1:]:
                preprocessed_dataset = concatenate_datasets([preprocessed_dataset, chunk])
        else:
            for chunk in batch_chunks:
                preprocessed_dataset = concatenate_datasets([preprocessed_dataset, chunk])
        
        # Giải phóng bộ nhớ batch đã xử lý
        del batch_chunks
        gc.collect()
        monitor_memory()
    
    # Giải phóng bộ nhớ các chunk đã xử lý
    del processed_chunks
    gc.collect()
    monitor_memory()
    
    print("✅ Đã gộp xong các chunk!")

    # Giữ các cột cần thiết
    print("🧹 Đang dọn dẹp các cột không cần thiết...")
    keep_cols = ['input_ids', 'attention_mask', 'labels', 'keyword_count', 'text', 'orig_id']
    cols_to_drop = [c for c in preprocessed_dataset.column_names if c not in keep_cols]
    if cols_to_drop:
        try:
            preprocessed_dataset = preprocessed_dataset.remove_columns(cols_to_drop)
            print(f"✅ Đã loại bỏ {len(cols_to_drop)} cột không cần thiết")
        except Exception as e:
            print(f"⚠️ Không thể loại bỏ cột: {e}")
            print("🔄 Tiếp tục với cột hiện có...")
    
    # Chuyển cột labels sang ClassLabel theo batch nhỏ để tiết kiệm RAM
    print("🏷️ Đang chuyển đổi labels sang ClassLabel...")
    
    # Chia dataset thành các batch nhỏ để xử lý
    total_size = len(preprocessed_dataset)
    batch_size_labels = 100  # Giảm xuống 100 mẫu một lần để tiết kiệm RAM
    
    processed_batches = []
    for i in range(0, total_size, batch_size_labels):
        end_idx = min(i + batch_size_labels, total_size)
        print(f"🔄 Xử lý labels batch {i//batch_size_labels + 1}: {i+1}-{end_idx}")
        
        try:
            # Lấy batch nhỏ
            batch_dataset = preprocessed_dataset.select(range(i, end_idx))
            
            # Chuyển đổi labels cho batch này
            batch_dataset = batch_dataset.cast_column('labels', ClassLabel(names=['0', '1']))
            
            processed_batches.append(batch_dataset)
            
            # Giải phóng bộ nhớ batch
            del batch_dataset
            gc.collect()
            monitor_memory()
            
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý batch {i//batch_size_labels + 1}: {e}")
            print("🔄 Thử giảm batch size...")
            
            # Thử với batch size nhỏ hơn
            try:
                smaller_batch_size = batch_size_labels // 2
                for j in range(i, end_idx, smaller_batch_size):
                    sub_end = min(j + smaller_batch_size, end_idx)
                    sub_batch = preprocessed_dataset.select(range(j, sub_end))
                    sub_batch = sub_batch.cast_column('labels', ClassLabel(names=['0', '1']))
                    processed_batches.append(sub_batch)
                    del sub_batch
                    gc.collect()
            except Exception as e2:
                print(f"❌ Không thể xử lý batch này: {e2}")
                continue
    
    # Gộp lại các batch đã xử lý
    if processed_batches:
        preprocessed_dataset = concatenate_datasets(processed_batches)
        del processed_batches
        gc.collect()
        monitor_memory()
    else:
        print("❌ Không thể xử lý labels, sử dụng labels gốc")
        # Giữ nguyên labels gốc nếu không thể chuyển đổi
    
    print("✂️ Đang chia dataset thành train/val/test...")
    
    # Split trực tiếp trên HF Dataset theo batch nhỏ để tiết kiệm RAM
    print("✂️ Đang chia dataset thành train/val/test theo batch...")
    
    # Tính toán kích thước split
    total_size = len(preprocessed_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"📊 Kích thước: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Split theo index để tiết kiệm RAM
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))
    
    # Tạo các dataset con
    train_ds = preprocessed_dataset.select(train_indices)
    val_ds = preprocessed_dataset.select(val_indices)
    test_ds = preprocessed_dataset.select(test_indices)
    
    # Giải phóng bộ nhớ dataset gốc
    del preprocessed_dataset
    gc.collect()
    monitor_memory()

    # Tạo DatasetDict
    try:
        final_dataset = DatasetDict({
            'train': train_ds,
            'validation': val_ds,
            'test': test_ds
        })
        
        # Lưu dataset tokenized theo từng phần để tiết kiệm RAM
        print(f"💾 Đang lưu dataset vào: {output_dir}")
    except Exception as e:
        print(f"❌ Lỗi khi tạo DatasetDict: {e}")
        return None  # Trả về None nếu có lỗi
    
    try:
        # Lưu từng phần riêng biệt
        os.makedirs(output_dir, exist_ok=True)
        
        print("💾 Lưu train dataset...")
        train_ds.save_to_disk(os.path.join(output_dir, 'train'))
        
        print("💾 Lưu validation dataset...")
        val_ds.save_to_disk(os.path.join(output_dir, 'validation'))
        
        print("💾 Lưu test dataset...")
        test_ds.save_to_disk(os.path.join(output_dir, 'test'))
        
        print(f"✅ Dataset preprocessed đã lưu thành công tại: {output_dir}")
        print(f"📊 Kích thước (cửa sổ): Train={len(final_dataset['train'])}, Val={len(final_dataset['validation'])}, Test={len(final_dataset['test'])}")
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu dataset: {e}")
        # Thử lưu vào thư mục local nếu không lưu được
        local_output = './preprocessed_scam_dataset'
        print(f"Thử lưu vào thư mục local: {local_output}")
        
        try:
            os.makedirs(local_output, exist_ok=True)
            train_ds.save_to_disk(os.path.join(local_output, 'train'))
            val_ds.save_to_disk(os.path.join(local_output, 'validation'))
            test_ds.save_to_disk(os.path.join(local_output, 'test'))
            print(f"✅ Đã lưu vào thư mục local: {local_output}")
        except Exception as e2:
            print(f"❌ Không thể lưu vào local: {e2}")
            print("💡 Gợi ý: Giảm CHUNK_SIZE hoặc BATCH_SIZE để tiết kiệm RAM")
    
    try:
        # Giải phóng bộ nhớ cuối cùng
        del train_ds, val_ds, test_ds
        gc.collect()
        monitor_memory()
        
        return final_dataset
    except Exception as e:
        print(f"❌ Lỗi khi giải phóng bộ nhớ: {e}")
        return None  # Trả về None nếu có lỗi

# Hàm load dataset đã lưu
def load_preprocessed_dataset(dataset_dir):
    """Load dataset đã được preprocess và lưu"""
    try:
        from datasets import load_from_disk
        dataset = DatasetDict({
            'train': load_from_disk(os.path.join(dataset_dir, 'train')),
            'validation': load_from_disk(os.path.join(dataset_dir, 'validation')),
            'test': load_from_disk(os.path.join(dataset_dir, 'test'))
        })
        print(f"✅ Đã load dataset từ: {dataset_dir}")
        return dataset
    except Exception as e:
        print(f"❌ Lỗi khi load dataset: {e}")
        return None

# Chạy main function chỉ khi file được chạy trực tiếp
if __name__ == '__main__':
    # Tối ưu hóa keywords trước khi chạy
    optimize_keywords()
    
    input_file = 'augmented_expanded_scam_dataset.csv'
    output_dir = './preprocessed_scam_dataset'
    
    try:
        preprocessed_dataset = preprocess_dataset(input_file, output_dir)
        
        if preprocessed_dataset is not None:
            # Hiển thị sample
            sample = preprocessed_dataset['train'][0]
            print("\n📋 Sample sau preprocess:")
            print("Input IDs:", sample['input_ids'])
            print("Attention Mask:", sample['attention_mask'])
            print("Label:", sample['labels'])
            print("Keyword Count:", sample['keyword_count'])
            print("Orig ID:", sample['orig_id'])
            
            # In thêm thông tin về kích thước
            print("\n📊 Thông tin dataset:")
            for split in ['train', 'validation', 'test']:
                print(f"Số lượng mẫu {split}: {len(preprocessed_dataset[split])}")
                
            # In thông tin về cấu trúc dữ liệu
            print("\n🔍 Cấu trúc dữ liệu:")
            for key, value in sample.items():
                print(f"{key}: {type(value)}")
            
            # In thêm thông tin về kích thước
            print("\n📊 Thông tin dataset:")
            print(f"Số lượng mẫu train: {len(preprocessed_dataset['train'])}")
            print(f"Số lượng mẫu validation: {len(preprocessed_dataset['validation'])}")
            print(f"Số lượng mẫu test: {len(preprocessed_dataset['test'])}")
            
            # In thêm thông tin về kích thước
            print("\n📊 Thông tin dataset:")
            print(f"Số lượng mẫu train: {len(preprocessed_dataset['train'])}")
            print(f"Số lượng mẫu validation: {len(preprocessed_dataset['validation'])}")
            print(f"Số lượng mẫu test: {len(preprocessed_dataset['test'])}")
            
            # In thông tin về cấu trúc dữ liệu
            print("\n🔍 Cấu trúc dữ liệu và nội dung:")
            for key, value in sample.items():
                print(f"\n{key}:")
                print(f"Kiểu dữ liệu: {type(value)}")
                if key == 'text':
                    print("Nội dung text đầy đủ:", value)
                    print("Độ dài text:", len(value))
                    # Hiển thị một số từ khóa tìm thấy trong text
                    found_keywords = [kw for kw in keywords if kw in value]
                    if found_keywords:
                        print("Các từ khóa tìm thấy:", found_keywords)
        else:
            print("❌ Không thể tạo dataset")
        
        # Giải phóng bộ nhớ
        del preprocessed_dataset
        gc.collect()
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình preprocessing: {e}")
        import traceback
        traceback.print_exc()