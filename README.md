# Sức mạnh của Data Preparation qua Data Storytelling  
### Case study: Rossmann Store Sales

## Giới thiệu

Hãy tưởng tượng bạn là Data Scientist của chuỗi cửa hàng **Rossmann**. Trước mặt bạn là bộ dữ liệu hơn **1 triệu bản ghi** dạng **time series**, ghi lại doanh thu theo từng ngày và thông tin chi tiết của **1.115 cửa hàng** (loại cửa hàng, khuyến mãi, ngày nghỉ lễ, cạnh tranh, v.v.). Nhiệm vụ: 

> Dự đoán **Doanh thu (Sales)** của một cửa hàng vào một ngày bất kỳ, dựa trên thông tin thời gian và đặc trưng cửa hàng.

**Vấn đề chính**: phần lớn đặc trưng là **biến rời rạc/phân loại**, vốn không “thân thiện” với mô hình như các biến liên tục. Trên nền đó, dữ liệu còn có **missing values** và cấu trúc phức tạp: nhiều mã hoá khó hiểu, nhiều trường “Since/Duration”, cùng sự đa dạng của 1.115 cửa hàng cần được “giải mã” trước khi đưa vào mô hình.

Dự án này được thiết kế để cho thấy **Data Preparation có thể thay đổi kết quả bài toán như thế nào** thông qua bốn bước chính:

- **EDA / Data Understanding**: Khám phá bức tranh tổng thể, hiểu rõ cấu trúc và bối cảnh dữ liệu Rossmann.  
- **Deep Analysis**: Đi sâu vào các nhóm biến quan trọng (thời gian, mùa vụ, cạnh tranh, promotion, loại cửa hàng) để hiểu rõ hơn hành vi doanh thu.  
- **Chuẩn bị dữ liệu (Data Preparation)**: Làm sạch, xử lý missing values, biến đổi và thiết kế lại đặc trưng – đặc biệt cho các biến phân loại và chuỗi thời gian.  
- **Đánh giá mô hình (RAW vs CLEAN)**: So sánh mô hình trên dữ liệu thô và dữ liệu đã chuẩn bị kỹ lưỡng.

Phần mở đầu này đặt nền cho các phần tiếp theo trong project: từ một bộ dữ liệu phức tạp, nhiều biến phân loại khó xử lý, chúng ta sẽ từng bước **hiểu – chuẩn hóa – tối ưu dữ liệu**, tạo cơ sở vững chắc cho Data Storytelling và phân tích kỹ thuật ở các phần sau.
