import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# =========================
#  CONSTANTS (MÀU SẮC)
# =========================
ROSSMANN_RED = "#C3002D"
GRAY_NEUTRAL = "#888888"
GRAY_LIGHT   = "#C3C3C3"


# =========================
#  METRIC HỖ TRỢ XGBOOST
# =========================
def rmspe_xg_raw(y_true, y_pred):
    """
    RMSPE cho target dạng raw (Sales).
    Dùng được làm eval_metric cho XGBRegressor (Scikit-Learn API).
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mask = y_true > 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    percent_error = (y_true - y_pred) / y_true
    score = np.sqrt(np.mean(percent_error ** 2))
    return score


def rmspe_xg_log(y_true, y_pred):
    """
    RMSPE cho target dạng log1p(Sales).
    Khi đánh giá, phải expm1 để quay lại thang Sales rồi mới tính RMSPE.
    Dùng được làm eval_metric cho XGBRegressor (Scikit-Learn API).
    """
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mask = y_true > 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    percent_error = (y_true - y_pred) / y_true
    score = np.sqrt(np.mean(percent_error ** 2))
    return score


# =========================
#  ROSSAMNN COMPARER
# =========================
class RossmannComparer:
    """
    So sánh nhiều pipeline Data Preparation trên cùng bài toán Rossmann:
    - Áp dụng preprocessor (Raw / Business / Embedding...)
    - Train XGBoost với cùng hyperparam
    - Tính RMSPE trên tập test time-based
    - Lưu learning curves & prediction trên test để vẽ biểu đồ
    """

    def __init__(self, model_params, train_df, store_df):
        """
        Parameters
        ----------
        model_params : dict
            Tham số dùng cho XGBRegressor (chung cho mọi pipeline).
        train_df : pd.DataFrame
            Dữ liệu train gốc (có cột Date, Store, Sales, Open, ...).
        store_df : pd.DataFrame
            Thông tin store, merge với train_df theo Store.
        """
        self.model_params = model_params
        self.full_data = pd.merge(train_df, store_df, on="Store", how="left")

        # Kết quả tổng hợp từng pipeline
        self.results_ = {}     # pipeline_name -> dict(RMSPE, Features, Log Target)
        self.history_ = {}     # pipeline_name -> evals_result_ từ XGBoost
        self.predictions_ = {} # pipeline_name -> DataFrame(Date, Store, y_true, y_pred)
        self.split_date_ = None  # mốc split train/test (datetime)

    # -------------------------
    #  RMSPE nội bộ
    # -------------------------
    def rmspe(self, y_true, y_pred):
        """RMSPE dùng nội bộ (không log-transform)."""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        mask = y_true > 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        percent_error = (y_true - y_pred) / y_true
        return np.sqrt(np.mean(percent_error ** 2))

    # -------------------------
    #  HÀM ĐÁNH GIÁ 1 PIPELINE
    # -------------------------
    def evaluate(self, pipeline_name, preprocessor, use_log_target=False):
        """
        Chạy full pipeline:
        - Time-based split (6 tuần cuối làm test).
        - Fit preprocessor trên train.
        - Train XGBoost.
        - Tuning correction factor (nếu dùng log target) trên TRAIN (tránh leakage).
        - Tính RMSPE trên test.
        - Lưu:
            + self.results_[pipeline_name]
            + self.history_[pipeline_name]
            + self.predictions_[pipeline_name] (để vẽ actual vs predicted theo Store)
        """
        print(f"\n--- Evaluating: {pipeline_name} ---")

        # 1. Time-based split
        data = self.full_data.sort_values("Date")
        split_date = data["Date"].max() - pd.Timedelta(days=6 * 7)
        self.split_date_ = split_date

        train_set = data[data["Date"] < split_date].copy()
        test_set = data[data["Date"] >= split_date].copy()

        test_open_mask = test_set["Open"] == 0
        y_test_real = test_set["Sales"].values

        print(
            f"Train range: {train_set['Date'].min().date()} -> {train_set['Date'].max().date()} "
            f"({len(train_set):,} rows)"
        )
        print(
            f"Test  range: {test_set['Date'].min().date()} -> {test_set['Date'].max().date()} "
            f"({len(test_set):,} rows)"
        )

        # 2. Preprocess
        print("Preprocessing...")
        preprocessor.fit(train_set, train_set["Sales"])

        X_train = preprocessor.transform(train_set)
        X_test = preprocessor.transform(test_set)

        # Đảm bảo không có cột Sales trong X (tránh leak)
        if isinstance(X_train, pd.DataFrame) and "Sales" in X_train.columns:
            X_train = X_train.drop("Sales", axis=1)
        if isinstance(X_test, pd.DataFrame) and "Sales" in X_test.columns:
            X_test = X_test.drop("Sales", axis=1)

        # 3. Target & eval_metric
        if use_log_target:
            y_train = np.log1p(train_set["Sales"].values)
            y_test_eval = np.log1p(test_set["Sales"].values)
            custom_feval = rmspe_xg_log
        else:
            y_train = train_set["Sales"].values
            y_test_eval = test_set["Sales"].values
            custom_feval = rmspe_xg_raw

        # 4. Train XGBoost
        print(f"Training XGBoost (log target = {use_log_target})...")
        params = self.model_params.copy()
        params["disable_default_eval_metric"] = 1  # tắt metric mặc định

        model = xgb.XGBRegressor(
            **params,
            eval_metric=custom_feval,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test_eval)],
            verbose=False,
        )

        self.history_[pipeline_name] = model.evals_result()

        # 5. Predict + (nếu log) tune correction factor TRÊN TRAIN, rồi đánh giá TEST
        if use_log_target:
            print("Optimizing correction factor (on TRAIN, not TEST)...")

            # 5.1. Predict trên TRAIN để tune factor
            train_open_mask = train_set["Open"] == 0
            y_train_real = train_set["Sales"].values

            preds_log_train = model.predict(X_train)
            preds_train_basic = np.expm1(preds_log_train)
            preds_train_basic[train_open_mask] = 0

            best_score = float("inf")
            best_factor = 1.0

            for factor in np.arange(0.98, 1.02, 0.001):
                score = self.rmspe(y_train_real, preds_train_basic * factor)
                if score < best_score:
                    best_score = score
                    best_factor = factor

            print(f"Best factor (from TRAIN): {best_factor:.4f}")

            # 5.2. Predict trên TEST, APPLY factor đã tune
            preds_log_test = model.predict(X_test)
            preds_test_basic = np.expm1(preds_log_test)
            preds_test_basic[test_open_mask] = 0

            preds_final = preds_test_basic * best_factor
            final_score = self.rmspe(y_test_real, preds_final)
        else:
            preds_raw = model.predict(X_test)
            preds_raw[test_open_mask] = 0
            preds_final = preds_raw
            final_score = self.rmspe(y_test_real, preds_final)

        # 6. Lưu kết quả & prediction theo Store/Date (để vẽ sau này)
        self.results_[pipeline_name] = {
            "RMSPE": final_score,
            "Features": X_train.shape[1] if hasattr(X_train, "shape") else None,
            "Log Target": use_log_target,
        }
        print(f"Done. RMSPE: {final_score:.5f}")

        test_pred_df = test_set[["Date", "Store"]].copy()
        test_pred_df["y_true"] = y_test_real
        test_pred_df["y_pred"] = preds_final

        self.predictions_[pipeline_name] = test_pred_df

        return final_score

    # -------------------------
    #  TÓM TẮT KẾT QUẢ
    # -------------------------
    def get_summary(self):
        """
        Trả về bảng tổng kết pipeline:
        - RMSPE
        - Số lượng features
        - Có dùng log target hay không
        """
        if not self.results_:
            return pd.DataFrame()
        return pd.DataFrame(self.results_).T.sort_values("RMSPE")

    # -------------------------
    #  VẼ LEARNING CURVES
    # -------------------------
    def plot_learning_curves(self):
        """
        Vẽ learning curve Train/Test RMSPE của từng pipeline.
        Dùng màu brand:
        - Train: xám
        - Test : đỏ Rossmann
        """
        if not self.history_:
            print("No history to plot. Run evaluate() first.")
            return

        plt.style.use("default")

        n_plots = len(self.history_)
        fig, axes = plt.subplots(
            1,
            n_plots,
            figsize=(5 * n_plots, 4),
            sharey=True,
        )

        if n_plots == 1:
            axes = [axes]

        fig.patch.set_facecolor("white")

        for ax, (name, history) in zip(axes, self.history_.items()):
            ax.set_facecolor("white")

            # metric_name ví dụ: 'rmspe_xg_raw' hoặc 'rmspe_xg_log'
            metric_name = list(history["validation_0"].keys())[0]
            train_loss = history["validation_0"][metric_name]
            test_loss = history["validation_1"][metric_name]

            ax.plot(
                train_loss,
                label="Train RMSPE",
                color=GRAY_NEUTRAL,
                linewidth=1.8,
            )
            ax.plot(
                test_loss,
                label="Test RMSPE",
                color=ROSSMANN_RED,
                linewidth=2.0,
            )

            ax.set_title(
                f"{name}\nFinal Test: {test_loss[-1]:.4f}",
                fontsize=11,
                color="#333333",
                pad=10,
            )
            ax.set_xlabel("Iteration", fontsize=9)
            ax.set_ylabel("RMSPE", fontsize=9)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)
            ax.tick_params(axis="both", labelsize=8)

            ax.legend(frameon=False, fontsize=8)

        plt.tight_layout()
        plt.show()
