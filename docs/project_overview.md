# Project Overview – Rossmann Store Sales

## 1. Bối cảnh & Bài toán kinh doanh

Rossmann là chuỗi cửa hàng dược phẩm với hơn 1.000 cửa hàng trên khắp châu Âu.  
Ban lãnh đạo muốn tận dụng dữ liệu lịch sử để:

- Hiểu sâu hơn về **nhịp điệu doanh thu** theo thời gian, loại cửa hàng, loại hàng hóa, khuyến mãi, ngày lễ...
- Chủ động trong **kế hoạch tồn kho, nhân sự và vận hành chuỗi cửa hàng**.
- Sở hữu một **mô hình dự báo doanh thu theo ngày** cho từng cửa hàng để hỗ trợ ra quyết định.

Bài toán cốt lõi:

> Dự báo **Sales** theo ngày cho từng **Store** dựa trên lịch sử và các đặc điểm cửa hàng,
> đồng thời chứng minh rằng **Data Preparation** là đòn bẩy quan trọng quyết định chất lượng mô hình.

---

## 2. Mục tiêu phân tích & mô hình

### 2.1. Mục tiêu phân tích (EDA & Insight)

1. Khai phá và hiểu sâu bộ dữ liệu Rossmann:
   - Phân phối doanh thu & khách hàng, xu hướng theo ngày/tuần/tháng/năm.
   - Ảnh hưởng của loại cửa hàng, loại hàng hóa, khoảng cách & thời điểm xuất hiện đối thủ.
   - Vai trò của Promo, Promo2 và các loại ngày lễ (Public Holiday, Easter, Christmas).

2. Rút ra các **insight mang tính hành động**:
   - Tháng 12 là “mỏ vàng” doanh thu; Giáng Sinh là tháng “sống còn”.
   - Promo có tác động khác nhau theo **loại cửa hàng**, **loại hàng hóa**, **thời điểm** và **vị trí cạnh tranh**.
   - Mỗi loại ngày lễ gắn với một kiểu **tâm lý mua sắm** khác nhau.

### 2.2. Mục tiêu mô hình & Data Preparation

1. Xây dựng các pipeline chuẩn bị dữ liệu cho bài toán dự báo doanh thu:

   - **RAW (Naive)**: xử lý tối thiểu, dùng như mốc 0.
   - **Business Logic Features (CLEAN)**: chuẩn hóa lại dữ liệu theo logic business (time features, holiday, competition, promo).
   - **Entity Embeddings**: biểu diễn các biến phân loại (Store, DayOfWeek, StoreType, Assortment, ...) bằng vector học được.

2. Cố định **mô hình (XGBoost)** và **metric (RMSPE)** để so sánh công bằng giữa các pipeline.

3. Định lượng tác động của Data Preparation:
   - So sánh RMSPE của RAW vs CLEAN vs Entity Embeddings.
   - Quan sát trực tiếp đường **Actual vs Predicted** theo từng cửa hàng.
   - “Soi” vào không gian embedding để xem mô hình đã thực sự hiểu gì về hành vi cửa hàng.

---

## 3. Bộ dữ liệu

Nguồn dữ liệu: **Rossmann Store Sales (Kaggle)**.

- Số dòng: ~1.017.209
- Số cột: 18
- Số cửa hàng: 1.115

Các file chính:

- `train.csv` – Lịch sử doanh thu từng ngày cho từng cửa hàng (có Sales & Customers).
- `test.csv` – Bộ test của Kaggle (không có Sales).
- `store.csv` – Thông tin tĩnh của từng cửa hàng (StoreType, Assortment, CompetitionDistance, Promo2, ...).
- `sample_submission.csv` – File submission mẫu.

Chi tiết ý nghĩa từng trường được mô tả trong `docs/data_dictionary.md`.

---

## 4. Phạm vi kỹ thuật

### 4.1. Mô hình

- Thuật toán: **XGBoost Regressor** (dữ liệu dạng bảng).
- Mục tiêu dự báo: `log(Sales)` để ổn định phân phối; kết quả được chuyển ngược về Sales bằng `expm1`.
- Metric: **RMSPE (Root Mean Squared Percentage Error)** – đo độ lệch dự báo theo tỷ lệ phần trăm.

### 4.2. Các pipeline dữ liệu

1. **RAW (Naive)**
   - Fill toàn bộ missing bằng 0.
   - `Date` → tách tối thiểu `Year`, `Month`, `Day`.
   - Biến phân loại → Label Encoding.
   - Bỏ các cột không dùng (ví dụ: `Date` sau khi đã tách).

2. **Business Logic Features (CLEAN)**
   - Xử lý thiếu có logic:
     - Median cho CompetitionDistance & các mốc CompetitionOpenSince*.
     - Tạo `Promo2Weeks`, `IsPromo2Month` cho Promo2.
   - Tạo thêm feature:
     - Time: `DayOfWeek`, `WeekOfYear`, `WeekOfMonth`, `IsWeekend`.
     - Holiday: `IsStateHoliday`.
     - Competition: `CompetitionMonthsOpen`.
   - Vẫn giữ Label Encoding cho biến phân loại để so sánh công bằng với RAW.

3. **Entity Embeddings**
   - Thêm lớp embedding cho các biến phân loại:
     - `Store`, `DayOfWeek`, `Year`, `Month`, `Day`, `StateHoliday`, `StoreType`, `Assortment`, `PromoInterval`, `WeekOfMonth`.
   - Tổng cộng 90 chiều embedding.
   - Kết hợp embedding với các biến liên tục (Competition, Promo, Holiday, ...) làm đầu vào cho XGBoost.

---

## 5. Deliverables

1. **Slide / PDF câu chuyện dữ liệu**  
   - Chương 1: Khai thác & hiểu sâu dữ liệu (EDA, insight business).  
   - Chương 2: Khai thác dữ liệu hiệu quả hơn cho mô hình dự báo (Data Preparation & Entity Embeddings).

2. **Báo cáo phân tích kỹ thuật (PDF)**  
   - Giải thích chi tiết logic xử lý dữ liệu, thiết kế biểu đồ, chiến lược kể chuyện.

3. **Repository GitHub (repo này)**
   - Code Python trong `data_preparation/`.
   - Notebook phân tích trong `notebooks/`.
   - Tài liệu mô tả trong `docs/`.
   - `README.md` hướng dẫn cài đặt & chạy.

---

## 6. Thông điệp kết luận

Thông điệp trung tâm của toàn bộ project:

> **“Mô hình tốt chỉ khi dữ liệu tốt.”**  
> Chất lượng dự báo không đến từ thuật toán “thần thánh”,  
> mà đến từ việc **chuẩn hóa & kể lại câu chuyện dữ liệu** theo ngôn ngữ của business.
