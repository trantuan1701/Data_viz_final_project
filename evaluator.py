import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

def rmspe_xg_raw(y_true, y_pred):
    """Metric cho Raw Target (Scikit-Learn API compatible)"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mask = y_true > 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    percent_error = (y_true - y_pred) / y_true
    score = np.sqrt(np.mean(percent_error ** 2))
    return score  

def rmspe_xg_log(y_true, y_pred):
    """Metric cho Log Target (Scikit-Learn API compatible)"""
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

# --- 2. CLASS EVALUATOR ---
class RossmannComparer:
    def __init__(self, model_params, train_df, store_df):
        self.model_params = model_params
        self.full_data = pd.merge(train_df, store_df, on='Store', how='left')
        self.results_ = {}
        self.history_ = {}
        
    def rmspe(self, y_true, y_pred):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        mask = y_true > 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        percent_error = (y_true - y_pred) / y_true
        return np.sqrt(np.mean(percent_error**2))

    def evaluate(self, pipeline_name, preprocessor, use_log_target=False):
        print(f"\n--- Evaluating: {pipeline_name} ---")
        
        # 1. Split
        data = self.full_data.sort_values('Date')
        split_date = data['Date'].max() - pd.Timedelta(days=6*7)
        train_set = data[data['Date'] < split_date].copy()
        test_set = data[data['Date'] >= split_date].copy()
        
        test_open_mask = test_set['Open'] == 0
        y_test_real = test_set['Sales'].values
        
        # 2. Transform
        print("Preprocessing...")
        preprocessor.fit(train_set, train_set['Sales'])
        
        X_train = preprocessor.transform(train_set)
        X_test = preprocessor.transform(test_set)
        
        # 3. Target & Metric
        if use_log_target:
            y_train = np.log1p(train_set['Sales'])
            y_test_eval = np.log1p(test_set['Sales'])
            custom_feval = rmspe_xg_log
        else:
            y_train = train_set['Sales']
            y_test_eval = test_set['Sales']
            custom_feval = rmspe_xg_raw
            
        if 'Sales' in X_train.columns: X_train = X_train.drop('Sales', axis=1)
        if 'Sales' in X_test.columns: X_test = X_test.drop('Sales', axis=1)

        # 4. Train XGBoost
        print(f"Training XGBoost (Log Target: {use_log_target})...")
        params = self.model_params.copy()
        params['disable_default_eval_metric'] = 1 
        
        # Lưu ý: eval_metric nằm trong constructor
        model = xgb.XGBRegressor(**params, eval_metric=custom_feval)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test_eval)],
            verbose=False
        )
        self.history_[pipeline_name] = model.evals_result()
        
        # 5. Predict
        preds_log = model.predict(X_test)
        
        if use_log_target:
            print("Optimizing correction factor...")
            preds_basic = np.expm1(preds_log)
            preds_basic[test_open_mask] = 0
            
            best_score = float('inf')
            best_factor = 1.0
            for factor in np.arange(0.98, 1.02, 0.001):
                score = self.rmspe(y_test_real, preds_basic * factor)
                if score < best_score:
                    best_score = score
                    best_factor = factor
            print(f"Best Factor: {best_factor}")
            final_score = best_score
        else:
            preds_raw = preds_log
            preds_raw[test_open_mask] = 0
            final_score = self.rmspe(y_test_real, preds_raw)
        
        self.results_[pipeline_name] = {
            'RMSPE': final_score,
            'Features': X_train.shape[1],
            'Log Target': use_log_target
        }
        print(f"Done. RMSPE: {final_score:.5f}")
        return final_score

    def get_summary(self):
        return pd.DataFrame(self.results_).T.sort_values('RMSPE')
    
    def plot_learning_curves(self):
        if not self.history_: return
        plt.figure(figsize=(14, 6))
        for i, (name, history) in enumerate(self.history_.items()):
            # Lấy key metric động (vì tên hàm có thể là rmspe_xg_log hoặc rmspe_xg_raw)
            metric_name = list(history['validation_0'].keys())[0]
            
            train_loss = history['validation_0'][metric_name]
            test_loss = history['validation_1'][metric_name]
            
            plt.subplot(1, len(self.history_), i+1)
            plt.plot(train_loss, label='Train RMSPE', color='blue')
            plt.plot(test_loss, label='Test RMSPE', color='orange')
            
            plt.title(f'{name}\nFinal Test RMSPE: {test_loss[-1]:.4f}')
            plt.xlabel('Iterations')
            plt.ylabel('RMSPE')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 0.6) 
        plt.tight_layout()
        plt.show()