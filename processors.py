import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import os

from models import EntityEmbeddingModel

class BaseRossmannTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def _clean_state_holiday(self, df):
        """
        Chuẩn hóa StateHoliday siêu mạnh tay:
        1. Fill NaN bằng '0'.
        2. Chuyển về string.
        3. Gom nhóm '0.0', '0.00', 'nan' về '0'.
        """
        if 'StateHoliday' in df.columns:
            # Fill NaN trước để tránh biến thành chuỗi 'nan' sau này
            df['StateHoliday'] = df['StateHoliday'].fillna(0)
            
            # Chuyển về string
            df['StateHoliday'] = df['StateHoliday'].astype(str)
            
            # Quy hoạch lại các biến thể lạ về '0'
            # Thêm 'nan' vào list replace để chắc chắn sạch sẽ
            df['StateHoliday'] = df['StateHoliday'].replace(['0.0', '0.00', 'nan'], '0')
        return df

class RawFeatureGenerator(BaseRossmannTransformer):
    def __init__(self):
        self.label_encoders = {}
        self.cat_cols = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']

    def fit(self, X, y=None):
        temp_df = X.copy()
        
        # 1. Clean StateHoliday TRƯỚC
        temp_df = self._clean_state_holiday(temp_df)
        
        # 2. Fillna 0 cho các cột còn lại
        temp_df.fillna(0, inplace=True)
        
        for col in self.cat_cols:
            le = LabelEncoder()
            # Fit trên dữ liệu đã sạch
            vals = temp_df[col].astype(str)
            le.fit(vals)
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        df = X.copy()
        
        # 1. Clean StateHoliday
        df = self._clean_state_holiday(df)
        
        # 2. Fillna 0 (Naive)
        df.fillna(0, inplace=True)
        
        # 3. Date
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        
        # 4. Encoding
        for col in self.cat_cols:
            vals = df[col].astype(str)
            if col in self.label_encoders:
                le = self.label_encoders[col]
                classes = set(le.classes_)
                # Map unseen labels về class đầu tiên (thường là '0')
                vals = vals.map(lambda x: x if x in classes else list(classes)[0])
                df[col] = le.transform(vals)
            
        drop_cols = ['Date', 'Customers', 'Open'] 
        df = df.drop([c for c in drop_cols if c in df.columns], axis=1)
        return df

class OptimizedFeatureGenerator(BaseRossmannTransformer):
    def __init__(self):
        self.medians = {}
        self.label_encoders = {}
        self.cat_cols = ['StoreType', 'Assortment']

    def fit(self, X, y=None):
        self.medians = {
            "CompetitionDistance": X["CompetitionDistance"].median(),
            "CompetitionOpenSinceYear": X["CompetitionOpenSinceYear"].median(),
            "CompetitionOpenSinceMonth": X["CompetitionOpenSinceMonth"].median()
        }
        for col in self.cat_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        df = X.copy()
        
        # --- QUAN TRỌNG: Clean StateHoliday ngay đầu ---
        df = self._clean_state_holiday(df)
        
        # Feature Engineering
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek + 1
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
        df["WeekOfMonth"] = df["Day"].apply(lambda d: (d-1) // 7 + 1)
        df["IsWeekend"] = (df["DayOfWeek"] >= 6).astype("int8")
        
        # Logic StateHoliday: Vì đã clean về '0', logic này giờ chạy đúng cho cả NaN cũ
        df["IsStateHoliday"] = df["StateHoliday"].apply(lambda x: 0 if x == '0' else 1).astype("int8")

        # Competition
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(self.medians["CompetitionDistance"])
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(self.medians["CompetitionOpenSinceYear"]).astype(int)
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(self.medians["CompetitionOpenSinceMonth"]).astype(int)
        df["CompetitionMonthsOpen"] = 12 * (df["Year"] - df["CompetitionOpenSinceYear"]) + (df["Month"] - df["CompetitionOpenSinceMonth"])
        df["CompetitionMonthsOpen"] = df["CompetitionMonthsOpen"].apply(lambda x: x if x > 0 else 0)

        # Promo2
        df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(df["Year"]).astype(int)
        df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(df["WeekOfYear"]).astype(int)
        df["PromoInterval"] = df["PromoInterval"].fillna("")
        df["Promo2Weeks"] = 52 * (df["Year"] - df["Promo2SinceYear"]) + (df["WeekOfYear"] - df["Promo2SinceWeek"])
        df["Promo2Weeks"] = df["Promo2Weeks"].apply(lambda x: x if x > 0 else 0)
        
        month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        df['MonthStr'] = df['Month'].map(month_map)
        
        def check_promo(row):
            if row['PromoInterval'] == "" or row['Promo2'] == 0: return 0
            if row['MonthStr'] in row['PromoInterval']: return 1
            return 0
        df["IsPromo2Month"] = df.apply(check_promo, axis=1).astype("int8")
        df.loc[df["Promo2Weeks"] <= 0, "IsPromo2Month"] = 0
        
        # Encoding
        for col in self.cat_cols:
            df[col] = self.label_encoders[col].transform(df[col].astype(str))
            
        drop_cols = ['Date', 'Customers', 'Open', 'PromoInterval', 'StateHoliday', 'MonthStr']
        df = df.drop([c for c in drop_cols if c in df.columns], axis=1)
        return df
    



class EntityEmbeddingFeatureGenerator(BaseRossmannTransformer):
    def __init__(self, model_save_path='checkpoints\emb_model_compare.pth', epochs=5, batch_size=512):
        self.model_save_path = model_save_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.cat_cols = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'StoreType', 'Assortment', 'PromoInterval', 'WeekOfMonth']
        self.cont_cols = ['CompetitionDistance', 'CompetitionMonthsOpen', 'Promo2Weeks', 'IsPromo2Month', 'Promo', 'SchoolHoliday', 'IsWeekend']
        
        self.medians = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.emb_dims = [] 

    def _feature_engineering(self, df):
        df = df.copy()
        df = self._clean_state_holiday(df)
        
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
        df["WeekOfMonth"] = df["Day"].apply(lambda d: (d-1) // 7 + 1)
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype("int8")

        df["CompetitionMonthsOpen"] = 12 * (df["Year"] - df["CompetitionOpenSinceYear"]) + (df["Month"] - df["CompetitionOpenSinceMonth"])
        df["CompetitionMonthsOpen"] = df["CompetitionMonthsOpen"].fillna(0).clip(lower=0, upper=60).astype("int16")

        df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(df["Year"]).astype(int)
        df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(df["WeekOfYear"]).astype(int)
        df["PromoInterval"] = df["PromoInterval"].fillna("")
        
        df["Promo2Weeks"] = 52 * (df["Year"] - df["Promo2SinceYear"]) + (df["WeekOfYear"] - df["Promo2SinceWeek"])
        df["Promo2Weeks"] = df["Promo2Weeks"].clip(lower=0, upper=25).astype("int16")
        
        month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        df['MonthStr'] = df['Month'].map(month_map)
        def check_promo(row):
            if row['PromoInterval'] == "" or row['Promo2'] == 0: return 0
            if row['MonthStr'] in row['PromoInterval']: return 1
            return 0
        df["IsPromo2Month"] = df.apply(check_promo, axis=1).astype("int8")
        df.loc[df["Promo2Weeks"] <= 0, "IsPromo2Month"] = 0
        
        return df.drop('MonthStr', axis=1)

    def _train_embedding_model(self, X_processed, y_train):
        # --- LOGGING START ---
        print(f"\n[Embed Training] Device: {self.device} | Batch Size: {self.batch_size}")
        
        # Prepare Tensors
        X_cat = torch.tensor(X_processed[self.cat_cols].values, dtype=torch.long)
        X_cont = torch.tensor(X_processed[self.cont_cols].values, dtype=torch.float)
        y = torch.tensor(np.log1p(y_train.values), dtype=torch.float).view(-1, 1)
        
        dataset = TensorDataset(X_cat, X_cont, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Init Model (EntityEmbeddingModel được import từ models.py)
        model = EntityEmbeddingModel(self.emb_dims, n_cont=len(self.cont_cols)).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print(f"[Embed Training] Start training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            for cats, conts, targets in loader:
                cats, conts, targets = cats.to(self.device), conts.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(cats, conts)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * cats.size(0)
            
            avg_loss = running_loss / len(dataset)
            # --- LOGGING EPOCH ---
            print(f"  >> Epoch {epoch+1}/{self.epochs} | Loss (MSE): {avg_loss:.5f}")
            
        if not os.path.exists(os.path.dirname(self.model_save_path)):
            os.makedirs(os.path.dirname(self.model_save_path))
        torch.save(model.state_dict(), self.model_save_path)
        print(f"[Embed Training] Model saved to: {self.model_save_path}")

    def fit(self, X, y=None):
        print(f"\n[EntityEmbedding] 1. Feature Engineering (Train)...")
        df = self._feature_engineering(X)
        
        print(f"[EntityEmbedding] 2. Computing Medians...")
        self.medians = {
            "CompetitionDistance": df["CompetitionDistance"].median(),
            "CompetitionOpenSinceYear": df["CompetitionOpenSinceYear"].median(),
            "CompetitionOpenSinceMonth": df["CompetitionOpenSinceMonth"].median()
        }
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(self.medians["CompetitionDistance"])
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(self.medians["CompetitionOpenSinceYear"]).astype(int)
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(self.medians["CompetitionOpenSinceMonth"]).astype(int)

        print(f"[EntityEmbedding] 3. Fitting Label Encoders...")
        self.emb_dims = []
        for col in self.cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
            df[col] = le.transform(df[col].astype(str))
            
            num_unique = len(le.classes_)
            dim = min(50, (num_unique + 1) // 2)
            self.emb_dims.append((num_unique, dim))
            
        print(f"[EntityEmbedding] 4. Scaling Continuous Variables...")
        for col in self.cont_cols:
            df[col] = df[col].fillna(0)
        df[self.cont_cols] = self.scaler.fit_transform(df[self.cont_cols])
        
        if y is not None:
            self._train_embedding_model(df, y)
        else:
            print("[EntityEmbedding] Warning: 'y' is None. Skipping Neural Network training.")

        return self

    def transform(self, X):
        # 1. FE
        df = self._feature_engineering(X)
        
        # 2. Fill Missing
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(self.medians["CompetitionDistance"])
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(self.medians["CompetitionOpenSinceYear"]).astype(int)
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(self.medians["CompetitionOpenSinceMonth"]).astype(int)
        
        # 3. Transform Encoders
        for col in self.cat_cols:
            le = self.label_encoders[col]
            vals = df[col].astype(str).map(lambda x: x if x in set(le.classes_) else list(le.classes_)[0])
            df[col] = le.transform(vals)
            
        # 4. Transform Scaler
        df_scaled_cont = df[self.cont_cols].copy().fillna(0)
        df_scaled_cont = self.scaler.transform(df_scaled_cont)
        
        # --- LOAD EMBEDDINGS ---
        # Khởi tạo model architecture để load weights
        # Class EntityEmbeddingModel được import từ models.py nên không lỗi NameError nữa
        model = EntityEmbeddingModel(self.emb_dims, n_cont=len(self.cont_cols))
        try:
            model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing model file: {self.model_save_path}. Please run .fit() first!")
            
        model.eval()
        
        embeddings_dict = {}
        for i, col in enumerate(self.cat_cols):
            embeddings_dict[col] = model.embeddings[i].weight.detach().cpu().numpy()
            
        embedding_dfs = []
        df_final = pd.DataFrame(df_scaled_cont, columns=self.cont_cols, index=df.index)
        
        for i, col in enumerate(self.cat_cols):
            indices = df[col].values
            vectors = embeddings_dict[col][indices]
            dim = self.emb_dims[i][1]
            col_names = [f"{col}_emb_{j}" for j in range(dim)]
            emb_df = pd.DataFrame(vectors, columns=col_names, index=df.index)
            embedding_dfs.append(emb_df)
            
        if embedding_dfs:
            df_embeddings = pd.concat(embedding_dfs, axis=1)
            df_final = pd.concat([df_final, df_embeddings], axis=1)
            
        return df_final