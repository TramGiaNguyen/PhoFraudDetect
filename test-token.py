from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("vinai/phobert-large")
text = """Dưới đây là một cuộc hội thoại chi tiết giữa kẻ lừa đảo và nạn nhân:

Kẻ lừa: Xin chào, anh chào nhé? Tôi là nhân viên bán vé của công ty tổ chức concert TWICE tại Sân vận động Quốc gia.

Nạn nhân: Chào, anh cần giúp tôi gì?

Kẻ lừa (loại 1 - Thuyết phục): Anh ạ, anh đang tìm vé concert TWICE không? Hiện tại, chúng tôi có một số vé VIP còn lại. Anh có muốn mua không?

Nạn nhân: À, tôi muốn mua vé VIP.

Kẻ lừa (loại 1 - Thuyết phục): Tuyệt vời, anh ạ! Vé VIP của chúng tôi rất hiếm. Anh cần chuyển 10 triệu để đặt vé. Sau khi chuyển tiền, anh sẽ nhận được mã xác thực qua SMS.

Nạn nhân: Vé VIP thì phải là 10 triệu?

Kẻ lừa (loại 2 - Đe dọa): À, anh không tin được. Con số đó là đúng. Nếu anh không chuyển tiền trong thời gian sớm nhất, vé VIP sẽ bị bán hết cho người khác. Anh không muốn bỏ lỡ cơ hội này, đúng không?

Nạn nhân: À, tôi không biết... Vễ VIP có phải có code không?

Kẻ lừa (loại 2 - Đe dọa): Code sẽ được gửi tới số điện thoại của anh sau khi anh chuyển tiền. Không chuyển tiền, anh sẽ không nhận được code.

Nạn nhân: À, tôi sẽ chuyển tiền. Nhưng anh cho tôi số tài khoản để chuyển nhé!

Kẻ lừa (loại 3 - Khống chế): À, anh ạ, anh cần chuyển tiền qua ngân hàng. Số tài khoản của chúng tôi là: 0123456789. Anh cần chuyển 10 triệu vào tài khoản này.

Nạn nhân: À, tôi sẽ chuyển tiền. Nhưng anh cho tôi mã xác thực để chuyển tiền nhé!

Kẻ lừa (loại 3 - Khống chế): Mã xác thực sẽ được gửi tới số điện thoại của anh sau khi anh chuyển tiền. Anh cần chuyển tiền trước khi nhận được mã xác thực.

Nạn nhân: À, tôi sẽ chuyển tiền. Nhưng anh cho tôi thông tin về vé VIP nữa nhé!

Kẻ lừa (loại 4 - Mạo danh): À, anh ạ, vé VIP của chúng tôi có giá 10 triệu. Vé này bao gồm quyền vào trước sân, chỗ ngồi VIP và được gặp gỡ các member TWICE. Anh có muốn mua vé này không?

Nạn nhân: À, tôi muốn mua vé VIP. Nhưng anh cho tôi số tài khoản để chuyển tiền nhé!

Kẻ lừa (loại 4 - Mạo danh): À, anh ạ, số tài khoản của chúng tôi là: 0123456789. Anh cần chuyển 10 triệu vào tài khoản này. Sau khi chuyển tiền, anh sẽ nhận được mã xác thực qua SMS.

Nạn nhân: À, tôi sẽ chuyển tiền. Cảm ơn anh!

Kẻ lừa (loại 4 - Mạo danh): À, anh ạ, anh cần chuyển tiền càng nhanh càng tốt. Nếu anh không chuyển tiền trong thời gian sớm nhất, vé VIP sẽ bị bán hết cho người khác.

Lời cảnh báo: Lừa đảo vé concert TWICE thường bị lặp lại với nhiều biến thể. Người dùng cần cẩn thận khi mua vé trực tuyến và không cung cấp thông tin cá nhân hoặc tiền bạc cho người lạ. Hãy kiểm tra thông tin chính thức của công ty tổ chức concert để tránh bị lừa đảo."""
n_tokens = len(tok.encode(text, add_special_tokens=True))
print("Số token:", n_tokens)