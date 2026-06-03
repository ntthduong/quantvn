### Đánh giá Bot Duongtaptrade

1. Mô tả chiến lược

Chiến lược giao dịch chính là chỉ mua khi cổ phiếu đang trong xu hướng tăng. Tức là mua bằng breakout hoặc pullback, không mua vào khi volatility quá cao hoặc giá đã tăng quá xa EMA. Đặt stop-loss theo ATR/hard stop.
Nếu giá tăng, trailing stop được kéo lên để bảo vệ lợi nhuận. Bán khi mất trend, RSI yếu, thủng hỗ trợ, chạm stop hoặc giữ quá lâu không có lãi.

2. Các chỉ số đạt được
- Trade Win Rate đạt khoảng 27.77%, tức là số lệnh thắng không quá cao, nhưng chiến lược vẫn có lợi thế nhờ các lệnh thắng có biên lợi nhuận lớn.
- Profit factor là 1.18 cũng cho thấy là bot vẫn cho tỷ suất sinh lời ở một mức nhất định
- Max Drawdown còn rất cao khoảng 34.91% cho thấy là có giai đoạn sụt giảm sâu.

3. Điểm mạnh

Chiến lược đánh theo trend tăng nên có thể sinh ra lợi nhuận lớn nếu có và follow được trend

4. Điểm yếu & rủi ro

Nếu không có uptrend thì bot hoạt động trong sideway và downtrend không hiệu quả lắm. Chỉ số Max Drawdown còn khá tệ nên nếu chạy thực tế thì khó giữ được kỉ luật

5. Đề xuất cải thiện

Nếu có thêm thời gian, hướng cải thiện chính sẽ là giảm Max Drawdown. Có thể bổ sung thêm market filter ví dụ chỉ mua khi thị trường chung hoặc nhóm ngành cổ phiếu công nghệ cùng xác nhận trạng thái uptrend. Hoặc siết lại trailing stop và giảm time stop để thoát nhanh hơn giai đoạn trend fail.
