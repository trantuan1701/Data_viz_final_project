# Rossmann Store Sales — Sức mạnh của Data Preparation qua Data Storytelling

Dự án này là bài **Project cuối kỳ** cho môn Machine Learning.  
Mục tiêu chính:

- Khai thác bộ dữ liệu **Rossmann Store Sales** (train.csv + store.csv).
- Thể hiện **vai trò quan trọng của Data Preparation** (làm sạch, biến đổi, feature engineering…) thông qua:
  - **Câu chuyện dữ liệu (Data Storytelling)**: trực quan hóa, so sánh, diễn giải.
  - **Mô hình ML đơn giản**: so sánh kết quả khi dùng dữ liệu *thô* vs *đã chuẩn bị*.

> Giai đoạn đầu, **nhóm chỉ tập trung làm việc trên các file notebook (thử nghiệm, EDA, vẽ biểu đồ)**.  
> Khi đã ổn, **mới refactor notebook thành các module Python trong `src/`**.

---

## 1. Bộ dữ liệu sử dụng

Trong repo này, chúng ta chỉ dùng:

- `data/raw/train.csv`  
- `data/raw/store.csv`

Hai file này là dữ liệu gốc từ Kaggle (không chỉnh sửa thủ công).

---

## 2. Cấu trúc thư mục (dự kiến)

```text
rossmann-data-prep-storytelling/
├── README.md
├── .gitignore
├── requirements.txt           # (sẽ bổ sung sau)
│
├── data/
│   ├── raw/                   # Dữ liệu gốc (train.csv, store.csv)
│   ├── interim/               # Dữ liệu trung gian (sau một số bước xử lý)
│   └── processed/             # Dữ liệu cuối cùng để train/evaluate
│
├── notebooks/
│   ├── 01_eda_raw_data.ipynb              # EDA dữ liệu thô
│   ├── 02_data_cleaning_preparation.ipynb # Làm sạch & chuẩn bị dữ liệu
│   ├── 03_feature_engineering.ipynb       # Tạo đặc trưng mới
│   ├── 04_model_raw_vs_clean.ipynb        # So sánh model raw vs clean
│   └── sandbox_<tên_thành_viên>.ipynb     # Notebook thử nghiệm cá nhân
│
├── src/                        # (Giai đoạn 1 hầu như để trống / rất ít code)
│   ├── __init__.py
│   ├── data/                   # load_data, preprocess (sẽ tách từ notebook)
│   ├── features/               # build_features (sẽ tách từ notebook)
│   ├── models/                 # train, evaluate (sẽ tách từ notebook)
│   └── visualization/          # hàm vẽ biểu đồ dùng lại
│
├── scripts/                    # (dùng ở giai đoạn 2, sau khi đã có module)
│   ├── run_prepare_data.py
│   ├── run_train_baseline.py
│   └── run_evaluate.py
│
├── reports/
│   ├── figures/                # Lưu hình vẽ dùng cho slide/báo cáo
│   ├── slides/                 # File slide thuyết trình
│   └── final_report/           # PDF gộp Storytelling + Phân tích kỹ thuật
│
└── docs/
    ├── data_dictionary.md      # Mô tả chi tiết các biến (train.csv, store.csv)
    ├── project_overview.md     # Ghi chú flow chung của project
    └── storytelling_design.md  # Ý tưởng lựa chọn biểu đồ & chiến lược kể chuyện
