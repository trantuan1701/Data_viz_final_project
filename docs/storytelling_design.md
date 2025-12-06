# Storytelling Design – Rossmann Data Story

Tài liệu này mô tả cách nhóm thiết kế câu chuyện dữ liệu cho Rossmann Store Sales,
dựa trên các nguyên tắc trong *Storytelling with Data* và được hiện thực hóa
trong slide deck `[GROUP 5] Rossmann.pdf`.

---

## 1. Ngữ cảnh (Context) & Big Idea

### 1.1. Đối tượng (Who)

- **Ban lãnh đạo chuỗi cửa hàng** Rossmann.
- **Đội ngũ vận hành & điều hành hệ thống** (operations, marketing, cửa hàng trưởng).

Họ bận rộn và quan tâm chủ yếu đến:

- Doanh thu, biên lợi nhuận, rủi ro mất doanh thu.
- Những quyết định thực tế: nên chạy promo ở đâu, khi nào, cho cửa hàng nào, mặt hàng nào.

### 1.2. Họ cần biết / làm gì? (What)

Chúng mình muốn khán giả:

1. Hiểu **nhịp điệu doanh thu** và hành vi khách hàng theo thời gian, ngày lễ, promo, cạnh tranh.
2. Nhìn thấy **cơ hội & rủi ro** rõ ràng (ví dụ: tháng 12, ngày lễ lớn, khu vực cạnh tranh cao).
3. Tin tưởng rằng một **mô hình dự báo doanh thu** được chuẩn bị dữ liệu tốt
   có thể trở thành công cụ hỗ trợ ra quyết định.
4. Nhận ra rằng **đầu tư vào Data Preparation** là cần thiết để khai thác hết giá trị dữ liệu hiện có.

### 1.3. Cách thức (How)

- Hình thức: **thuyết trình trực tiếp** với slide.
- Slide được thiết kế:
  - Nhiều biểu đồ, ít chữ.
  - Tiêu đề dạng **câu khẳng định** (action titles).
  - Màu sắc bám theo brand Rossmann: đỏ chủ đạo + xám trung tính.

### 1.4. Big Idea

> **“Chất lượng mô hình dự báo là phản chiếu trực tiếp của chất lượng dữ liệu và cách ta chuẩn bị nó.”**

Toàn bộ câu chuyện hướng tới việc chứng minh Big Idea này:

- Từ EDA, hiểu business → thiết kế feature có logic.
- Từ việc xử lý thiếu hợp lý → giảm sai số.
- Từ Entity Embedding → mô hình “hiểu” từng cửa hàng tốt hơn.

### 1.5. Câu chuyện 3 phút

Nếu chỉ có 3 phút để kể:

1. Rossmann đang ngồi trên một **kho dữ liệu khổng lồ** về doanh thu, khách hàng, promo, ngày lễ.
2. Phân tích cho thấy:
   - Tháng 12 là “mỏ vàng” doanh thu.
   - Promo có hiệu quả rất khác nhau tùy **loại cửa hàng**, **loại hàng hóa**, **thời điểm**, **vị trí**.
   - Ngày lễ tạo ra các kiểu **tâm lý mua sắm** khác nhau.
3. Khi đưa dữ liệu đó vào mô hình:
   - Pipeline **thô** chỉ cho kết quả trung bình.
   - Pipeline **có business logic** giảm đáng kể sai số.
   - Khi dùng **Entity Embeddings**, mô hình học được cấu trúc thật giữa các cửa hàng,
     RMSPE tiếp tục giảm và dự báo bám sát thực tế.
4. Kết luận: Đầu tư vào **chuẩn bị dữ liệu & pipeline dự báo** là khoản đầu tư trực tiếp
   vào doanh thu và lợi thế cạnh tranh của chuỗi cửa hàng.

---

## 2. Cấu trúc câu chuyện – 3 hồi (Beginning / Middle / End)

### 2.1. Beginning – Khai thác & hiểu sâu dữ liệu (Chương 1)

Mục đích:

- Thiết lập bối cảnh và “tình trạng mất cân bằng”: Rossmann có rất nhiều dữ liệu, nhưng chưa khai thác hết.
- Trả lời câu hỏi: **Tại sao ban lãnh đạo nên quan tâm?**

Nội dung chính:

- Giới thiệu dataset (số dòng, số cột, số cửa hàng).
- Phân phối Sales & Customers, seasonality theo ngày / tháng / năm.
- Các insight về:
  - **StoreType & Assortment** – nghịch lý về giá trị bán hàng.
  - **Competition** – khoảng cách đối thủ, thời điểm đối thủ mở cửa.
  - **Promo** – tác động lên Sales, Customers, Basket Size.
  - **Holiday** – 3 kiểu tâm lý mua sắm: “Nước đến chân mới nhảy”, “Người có kế hoạch”, “Lo xa”.

Xung đột được giới thiệu:

> Doanh thu của chuỗi chịu tác động mạnh của thời gian, promo, cạnh tranh, ngày lễ,
> nhưng nếu không có một công cụ dự báo tốt, ban lãnh đạo đang “đi trong sương mù”.

### 2.2. Middle – Khai thác dữ liệu hiệu quả hơn cho mô hình dự báo (Chương 2)

Mục đích:

- Phát triển “điều có thể xảy ra”: nếu chuẩn bị dữ liệu tốt hơn, chúng ta có thể có dự báo tốt hơn.
- Thuyết phục ban lãnh đạo rằng **Data Preparation** không phải chi tiết kỹ thuật, mà là đòn bẩy chiến lược.

Nội dung:

1. **Cố định mô hình**: XGBoost + RMSPE + dự báo theo ngày từng cửa hàng.
2. **Baseline RAW**:
   - Missing = 0, Date tách tối thiểu, categorical = Label Encoding.
   - Đây là “mốc 0” để so sánh.
3. **Business Logic Features**:
   - Xử lý thiếu bằng median, tạo CompetitionMonthsOpen, Promo2Weeks, IsPromo2Month,
     DayOfWeek, WeekOfYear, WeekOfMonth, IsWeekend, IsStateHoliday…
   - Dự đoán log(Sales) thay vì Sales gốc.
   - RMSPE giảm đáng kể so với RAW.
4. **Entity Embeddings**:
   - Động lực: One-hot & Label Encoding đều không giúp mô hình “hiểu” độ giống nhau giữa các cửa hàng.
   - Thiết kế embedding cho các biến phân loại → tổng cộng 90 chiều.
   - Kết quả:
     - RMSPE tiếp tục giảm.
     - PC1 của store embedding có quan hệ gần tuyến tính với doanh thu trung bình.
     - Khoảng cách embedding tương quan với chênh lệch doanh thu giữa các cửa hàng.

### 2.3. End – Kết luận & Call to Action

Kết lại:

- Nhắc lại xung đột: dữ liệu lớn nhưng mô hình thô thì sai số cao.
- Nhắc lại những gì đã đạt được:
  - Chuẩn hóa dữ liệu theo business logic → mô hình ổn định hơn, sai số giảm.
  - Entity Embeddings → mô hình hiểu “chân dung hành vi” của từng cửa hàng.
- **Call to Action**:
  - Đầu tư nghiêm túc vào Data Preparation & hệ thống dự báo doanh thu.
  - Sử dụng mô hình như một **công cụ hỗ trợ ra quyết định**: plan promo, tồn kho, staffing, mở rộng cửa hàng.

---

## 3. Narrative Flow & Chiến lược truyền tải

### 3.1. Thứ tự tường thuật

Nhóm kết hợp hai kiểu flow:

1. **Theo thời gian & quá trình phân tích**:
   - Từ EDA → hiểu business → thiết kế feature → xây pipeline dự báo.
2. **Bing – Bang – Bongo**:
   - **Bing**: Mở đầu với slide “Câu chuyện này dành cho ai?” và “Cấu trúc câu chuyện”.
   - **Bang**: Đi sâu vào Chương 1 & Chương 2 với dữ liệu, biểu đồ, mô hình.
   - **Bongo**: Kết thúc bằng slide “Mô hình tốt chỉ khi dữ liệu tốt” và tóm tắt lại thông điệp.

### 3.2. Logic ngang (Horizontal Logic)

- Tiêu đề slide được viết dưới dạng **câu khẳng định** (action title), ví dụ:
  - “Tháng 12 là ‘mỏ vàng’ doanh thu”
  - “Sức mạnh của Promo”
  - “Promo là vũ khí phòng thủ và tấn công”
  - “Cần một cách biểu diễn ‘thông minh’ hơn cho biến phân loại”
- Nếu đọc liên tiếp các tiêu đề, khán giả vẫn nắm được **xương sống câu chuyện** mà không cần xem chi tiết.

### 3.3. Logic dọc (Vertical Logic)

- Mỗi slide chỉ tập trung vào **một ý chính**.
- Toàn bộ biểu đồ, bảng, text trên slide cùng nhau hỗ trợ cho tiêu đề.
  - Ví dụ:
    - Slide “Tháng 12 là ‘mỏ vàng’ doanh thu” đi kèm biểu đồ seasonality và bảng so sánh chỉ số tăng trưởng.
    - Slide “Sức mạnh của Promo” dùng các boxplot, violinplot để minh họa tăng trưởng doanh số, khách, basket size.
    - Slide về Entity Embeddings có bảng số chiều embedding cho từng feature và biểu đồ PC1 vs doanh thu.

---

## 4. Thiết kế trực quan (Visual Design)

### 4.1. Chọn hình ảnh trực quan phù hợp

- **Biểu đồ đường** cho dữ liệu liên tục theo thời gian (seasonality, trend, trước & sau lễ).
- **Biểu đồ cột** cho so sánh giữa các nhóm (StoreType, Assortment, mức độ hiệu quả Promo, loại cửa hàng).
- **Boxplot / Violin plot** để thể hiện phân phối doanh thu/khoảng giá trị.
- **Bảng highlight / heatmap** để thể hiện liên hợp 2 chiều (loại store × loại hàng, mức độ “nghiện” promo).

### 4.2. Giảm clutter

- Nền sáng, loại bỏ gridline không cần thiết.
- Giữ lại đủ trục & tick để đọc số, nhưng không làm rối mắt.
- Tập trung vào **1–2 series chính** trên mỗi chart.

### 4.3. Tập trung sự chú ý bằng thuộc tính tiền chú ý

- Màu sắc:
  - **Đỏ Rossmann** cho series quan trọng / kết quả muốn nhấn mạnh.
  - **Xám** cho baseline hoặc nhóm đối chứng.
- Kích thước & độ đậm:
  - Đường/ cột chính đậm hơn, dày hơn.
- Vị trí:
  - Ký hiệu, annotation, label được đặt sát điểm cần chú ý (ví dụ: điểm cực trị, mức tăng trưởng đặc biệt).

### 4.4. Tư duy như một nhà thiết kế

- Dùng một **hệ thống cấp bậc rõ ràng**:
  - Tiêu đề lớn, phụ đề nhỏ, text hỗ trợ ngắn gọn.
- Tận dụng **white space** để slide không bị ngộp.
- Giữ **tính nhất quán**:
  - Cùng loại thông tin → cùng kiểu chart, cùng palette màu.

---

## 5. Kết luận

Storytelling design của project Rossmann được xây dựng xoay quanh:

1. **Big Idea rõ ràng**: *Data Preparation quyết định sức mạnh của mô hình dự báo.*
2. **Cấu trúc 3 hồi**:  
   - Beginning – hiểu dữ liệu & business.  
   - Middle – xây & so sánh các pipeline dữ liệu.  
   - End – kết luận & lời kêu gọi hành động.
3. **Flow mạch lạc** (Bing–Bang–Bongo) và logic ngang/dọc rõ ràng.
4. **Thiết kế trực quan có chủ đích**, dùng màu sắc, biểu đồ, và bố cục để dẫn dắt sự chú ý của khán giả.

Tài liệu này là cầu nối giữa **lý thuyết Storytelling with Data** và **câu chuyện cụ thể cho Rossmann**, giúp đảm bảo phần slide và phần code thống nhất về thông điệp:  

> *Muốn mô hình thông minh, trước hết dữ liệu phải được chuẩn bị một cách thông minh.*
