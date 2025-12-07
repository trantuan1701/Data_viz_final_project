# 📁 Dataset Overview — Rossmann Store Sales

Dữ liệu Rossmann được lấy từ cuộc thi Kaggle **“Rossmann Store Sales”**.  
Đây là **dữ liệu bảng theo ngày (panel data)** của chuỗi hơn **1.115 cửa hàng Rossmann** tại 7 nước châu Âu. Nhiệm vụ gốc của Kaggle là dự đoán cột **`Sales`** cho các ngày tương lai. :contentReference[oaicite:0]{index=0}  

Trong phạm vi dự án môn học, **nhóm chỉ sử dụng 2 file**:

- `train.csv` – dữ liệu lịch sử bán hàng theo ngày.
- `store.csv` – thông tin bổ sung (metadata) của từng cửa hàng.

> Gốc Kaggle còn có `test.csv` và `sample_submission.csv`, nhưng ta **không dùng** vì không tham gia leaderboard.

---

## 📄 1. Mô tả chi tiết các file sử dụng

### 1.1. `train.csv`

- **Nội dung**: Doanh thu hằng ngày của từng cửa hàng, kèm một số thông tin bối cảnh.
- **Độ chi tiết (granularity)**:  
  Mỗi dòng tương ứng với **1 cửa hàng – 1 ngày** (`Store`, `Date`).
- **Kích thước điển hình**:  
  Khoảng **1.017.209 dòng, 9 cột**, thu thập từ **1.115 cửa hàng** trong **942 ngày** (từ 2013-01-01 đến 2015-07-31). :contentReference[oaicite:1]{index=1}  
- **Mục đích trong dự án**:
  - Là **data chính cho EDA** và **data preparation** (làm sạch, biến đổi, feature engineering).
  - Chia tiếp thành `train` / `validation` / `holdout` để xây dựng & đánh giá mô hình dự đoán `Sales`.

---

### 1.2. `store.csv`

- **Nội dung**: Thông tin tĩnh của mỗi cửa hàng (loại cửa hàng, mức độ đa dạng sản phẩm, cạnh tranh, chương trình khuyến mãi dài hạn…).  
- **Độ chi tiết**: Mỗi dòng tương ứng **1 cửa hàng** (`Store`).
- **Kích thước điển hình**: **1.115 dòng, 10 cột**. :contentReference[oaicite:2]{index=2}  
- **Mục đích trong dự án**:
  - Dùng để **bổ sung đặc trưng** cho `train.csv` thông qua phép `merge` trên khóa `Store`.
  - Tạo thêm các feature về **thời gian có đối thủ cạnh tranh**, **thời gian tham gia chương trình Promo2**, v.v.

---

# 📑 2. Data Dictionary Chi Tiết

## 2.1. `train.csv`

| Biến | Kiểu | Mô tả chi tiết |
|------|------|----------------|
| **Store** | `int` | ID duy nhất cho mỗi cửa hàng. Đây là **khóa để join** với `store.csv`. Có 1.115 giá trị khác nhau. :contentReference[oaicite:3]{index=3} |
| **DayOfWeek** | `int` (1–7) | Thứ trong tuần của ngày đó: thường **1 = Monday, …, 7 = Sunday**. Dùng để phân tích pattern theo ngày trong tuần (weekday vs weekend). :contentReference[oaicite:4]{index=4} |
| **Date** | `date/string` (`YYYY-MM-DD`) | Ngày ghi nhận doanh thu. Khoảng thời gian phủ từ **2013-01-01 đến 2015-07-31**. Có thể tách thêm các trường **Year, Month, Day, WeekOfYear** để làm feature. :contentReference[oaicite:5]{index=5} |
| **Sales** | `int/float` | **Doanh thu (turnover)** của cửa hàng trong ngày đó. Đây là **biến mục tiêu** trong bài toán dự đoán. Giá trị luôn bằng 0 khi `Open = 0` (cửa hàng đóng cửa). :contentReference[oaicite:6]{index=6} |
| **Customers** | `int` | **Số lượng khách hàng** ghé cửa hàng trong ngày. Dùng làm feature để hiểu mối quan hệ `Sales ~ Customers` (thường tương quan dương mạnh). Không xuất hiện trong tập test Kaggle gốc. :contentReference[oaicite:7]{index=7} |
| **Open** | `int` (0/1) | Trạng thái mở cửa của cửa hàng trong ngày đó: **0 = đóng, 1 = mở**. Một số dòng trong train có `Open = 0` và `Sales = 0` do cửa hàng đóng cửa (ví dụ chủ nhật hoặc sửa chữa). Thường ta **loại các dòng `Open = 0` khi huấn luyện** vì không mang thông tin về pattern doanh thu. :contentReference[oaicite:8]{index=8} |
| **Promo** | `int` (0/1) | Cho biết **cửa hàng có đang chạy chương trình khuyến mãi (Promo)** trong ngày đó hay không: **1 = có khuyến mãi, 0 = không**. Khác với `Promo2` (khuyến mãi dài hạn) trong `store.csv`. :contentReference[oaicite:9]{index=9} |
| **StateHoliday** | `object` (`'0'`, `'a'`, `'b'`, `'c'`) | Biến chỉ **ngày nghỉ lễ cấp bang/quốc gia**. Ý nghĩa giá trị: **`'0'` = không phải ngày lễ; `'a'` = public holiday; `'b'` = Easter holiday; `'c'` = Christmas**. Thường trong ngày `StateHoliday ≠ '0'`, hầu hết cửa hàng sẽ đóng cửa; đồng thời **tất cả trường học đều nghỉ** vào public holidays & cuối tuần. :contentReference[oaicite:10]{index=10} |
| **SchoolHoliday** | `int` (0/1) | Cho biết ngày đó có bị ảnh hưởng bởi **kỳ nghỉ của trường học** hay không: **1 = trùng nghỉ học, 0 = ngày thường**. Biến này có thể giao thoa với `StateHoliday` (public holiday thường cũng là school holiday). :contentReference[oaicite:11]{index=11} |

**Gợi ý sử dụng cho Data Storytelling**

- So sánh phân phối `Sales` theo **`DayOfWeek`**, **`Promo`**, **`StateHoliday`**, **`SchoolHoliday`** để cho thấy **business pattern rõ ràng hơn sau khi làm sạch & mã hoá biến**.
- Đưa ví dụ các lỗi/dirty data như:
  - `StateHoliday` vừa có dạng `'0'` (string) vừa `0` (int) → cần chuẩn hoá. :contentReference[oaicite:12]{index=12}  
  - Ngày cửa hàng đóng (`Open = 0`) nhưng vẫn còn trong dữ liệu → phải quyết định **loại bỏ hay giữ lại** tuỳ bài toán.

---

## 2.2. `store.csv`

| Biến | Kiểu | Mô tả chi tiết |
|------|------|----------------|
| **Store** | `int` | ID cửa hàng – **khóa chính** để nối (join) với `train.csv`. Có 1.115 cửa hàng khác nhau. :contentReference[oaicite:13]{index=13} |
| **StoreType** | `object` (`'a'` / `'b'` / `'c'` / `'d'`) | **Loại hình cửa hàng** (4 mô hình khác nhau của Rossmann). Ví dụ có thể là: cửa hàng tiêu chuẩn, cửa hàng trung tâm, cửa hàng nhỏ trong khu dân cư,… – Kaggle không ghi cụ thể, nhưng dùng như một biến phân loại để bắt khác biệt về cấu trúc doanh số. :contentReference[oaicite:14]{index=14} |
| **Assortment** | `object` (`'a'` / `'b'` / `'c'`) | **Mức độ đa dạng danh mục sản phẩm**: **`'a'` = basic**, **`'b'` = extra**, **`'c'` = extended**. Cửa hàng `Assortment` cao thường có doanh thu cao hơn nhưng cũng phụ thuộc vị trí & cạnh tranh. :contentReference[oaicite:15]{index=15} |
| **CompetitionDistance** | `int/float` | **Khoảng cách (mét)** đến **cửa hàng đối thủ gần nhất**. Giá trị nhỏ = đối thủ gần; **NA** nghĩa là “không rõ hoặc không có đối thủ trong vùng”. Trong EDA thường thấy phân phối lệch phải (nhiều đối thủ gần, một số rất xa). :contentReference[oaicite:16]{index=16} |
| **CompetitionOpenSinceMonth** | `int` (1–12, có thể NA) | Tháng **đối thủ gần nhất bắt đầu hoạt động**. Là giá trị xấp xỉ do công ty cung cấp, không phải timestamp tuyệt đối. :contentReference[oaicite:17]{index=17} |
| **CompetitionOpenSinceYear** | `int` (ví dụ 1990–2010, có thể NA) | Năm **đối thủ gần nhất bắt đầu hoạt động**. Kết hợp với `CompetitionOpenSinceMonth` để suy ra số tháng/ năm đã có cạnh tranh. :contentReference[oaicite:18]{index=18} |
| **Promo2** | `int` (0/1) | Cho biết cửa hàng có tham gia **chương trình khuyến mãi kéo dài nhiều kỳ (Promo2)** hay không: **0 = không tham gia**, **1 = có tham gia**. Đây là loại promo “liên tục, lặp lại” khác với `Promo` (khuyến mãi ngắn hạn theo ngày). :contentReference[oaicite:19]{index=19} |
| **Promo2SinceWeek** | `int` (1–52, có thể NA) | **Tuần (ISO calendar week)** trong năm mà cửa hàng **bắt đầu tham gia Promo2**. Chỉ có ý nghĩa khi `Promo2 = 1`. :contentReference[oaicite:20]{index=20} |
| **Promo2SinceYear** | `int` (có thể NA) | **Năm** cửa hàng bắt đầu tham gia Promo2. Kết hợp với `Promo2SinceWeek` để tính “số tuần đã tham gia Promo2” tại một ngày bất kỳ. :contentReference[oaicite:21]{index=21} |
| **PromoInterval** | `string` (ví dụ `"Feb,May,Aug,Nov"`, có thể NA) | Mô tả **các đợt kích hoạt lặp lại của Promo2 trong năm** – là tên tháng ngăn cách bằng dấu phẩy. Ví dụ `"Feb,May,Aug,Nov"` nghĩa là mỗi năm cửa hàng chạy Promo2 bắt đầu các tháng **2, 5, 8, 11**. Để dùng trong mô hình, thường tách chuỗi này thành các biến nhị phân theo tháng. :contentReference[oaicite:22]{index=22} |
---
|