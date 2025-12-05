import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

from models import EntityEmbeddingModel


class BaseRossmannTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def _clean_state_holiday(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "StateHoliday" in df.columns:
            df["StateHoliday"] = df["StateHoliday"].fillna(0)
            df["StateHoliday"] = df["StateHoliday"].astype(str)
            df["StateHoliday"] = df["StateHoliday"].replace(["0.0", "0.00", "nan"], "0")
        return df


class RawFeatureGenerator(BaseRossmannTransformer):
    def __init__(self):
        self.label_encoders = {}
        self.cat_cols = ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]

    def fit(self, X, y=None):
        temp_df = X.copy()
        temp_df = self._clean_state_holiday(temp_df)
        temp_df.fillna(0, inplace=True)

        for col in self.cat_cols:
            le = LabelEncoder()
            vals = temp_df[col].astype(str)
            le.fit(vals)
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        df = X.copy()
        df = self._clean_state_holiday(df)
        df.fillna(0, inplace=True)

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day

        for col in self.cat_cols:
            vals = df[col].astype(str)
            if col in self.label_encoders:
                le = self.label_encoders[col]
                classes = set(le.classes_)
                vals = vals.map(lambda x: x if x in classes else list(classes)[0])
                df[col] = le.transform(vals)

        drop_cols = ["Date", "Customers", "Open"]
        df = df.drop([c for c in drop_cols if c in df.columns], axis=1)
        return df


class OptimizedFeatureGenerator(BaseRossmannTransformer):
    def __init__(self, use_scaler: bool = False):
        self.medians = {}
        self.label_encoders = {}
        self.cat_cols = ["StoreType", "Assortment"]
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None
        self.cont_cols = [
            "Year",
            "Month",
            "DayOfWeek",
            "WeekOfYear",
            "WeekOfMonth",
            "IsWeekend",
            "IsStateHoliday",
            "CompetitionDistance",
            "CompetitionMonthsOpen",
            "Promo2Weeks",
            "Promo",
            "SchoolHoliday",
            "IsPromo2Month",
        ]

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek + 1
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
        df["WeekOfMonth"] = df["Day"].apply(lambda d: (d - 1) // 7 + 1)
        df["IsWeekend"] = (df["DayOfWeek"] >= 6).astype("int8")
        return df

    def fit(self, X, y=None):
        df = X.copy()
        df = self._clean_state_holiday(df)
        df = self._add_time_features(df)

        self.medians = {
            "CompetitionDistance": df["CompetitionDistance"].median(),
            "CompetitionOpenSinceYear": df["CompetitionOpenSinceYear"].median(),
            "CompetitionOpenSinceMonth": df["CompetitionOpenSinceMonth"].median(),
        }

        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
            self.medians["CompetitionDistance"]
        )
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(
            self.medians["CompetitionOpenSinceYear"]
        ).astype(int)
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(
            self.medians["CompetitionOpenSinceMonth"]
        ).astype(int)

        df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(df["Year"]).astype(int)
        df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(df["WeekOfYear"]).astype(int)
        df["PromoInterval"] = df["PromoInterval"].fillna("")

        df["IsStateHoliday"] = df["StateHoliday"].apply(
            lambda x: 0 if x == "0" else 1
        ).astype("int8")

        df["CompetitionMonthsOpen"] = (
            12 * (df["Year"] - df["CompetitionOpenSinceYear"])
            + (df["Month"] - df["CompetitionOpenSinceMonth"])
        )
        df["CompetitionMonthsOpen"] = df["CompetitionMonthsOpen"].apply(
            lambda x: x if x > 0 else 0
        )

        df["Promo2Weeks"] = (
            52 * (df["Year"] - df["Promo2SinceYear"])
            + (df["WeekOfYear"] - df["Promo2SinceWeek"])
        )
        df["Promo2Weeks"] = df["Promo2Weeks"].apply(lambda x: x if x > 0 else 0)

        month_map = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        df["MonthStr"] = df["Month"].map(month_map)

        def check_promo(row):
            if row["PromoInterval"] == "" or row["Promo2"] == 0:
                return 0
            if row["MonthStr"] in row["PromoInterval"]:
                return 1
            return 0

        df["IsPromo2Month"] = df.apply(check_promo, axis=1).astype("int8")
        df.loc[df["Promo2Weeks"] <= 0, "IsPromo2Month"] = 0

        for col in self.cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le

        if self.use_scaler:
            for c in self.cont_cols:
                if c in df.columns:
                    df[c] = df[c].fillna(0)
            self.scaler.fit(df[self.cont_cols])

        return self

    def transform(self, X):
        df = X.copy()
        df = self._clean_state_holiday(df)
        df = self._add_time_features(df)

        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
            self.medians["CompetitionDistance"]
        )
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(
            self.medians["CompetitionOpenSinceYear"]
        ).astype(int)
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(
            self.medians["CompetitionOpenSinceMonth"]
        ).astype(int)

        df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(df["Year"]).astype(int)
        df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(df["WeekOfYear"]).astype(int)
        df["PromoInterval"] = df["PromoInterval"].fillna("")

        df["IsStateHoliday"] = df["StateHoliday"].apply(
            lambda x: 0 if x == "0" else 1
        ).astype("int8")

        df["CompetitionMonthsOpen"] = (
            12 * (df["Year"] - df["CompetitionOpenSinceYear"])
            + (df["Month"] - df["CompetitionOpenSinceMonth"])
        )
        df["CompetitionMonthsOpen"] = df["CompetitionMonthsOpen"].apply(
            lambda x: x if x > 0 else 0
        )

        df["Promo2Weeks"] = (
            52 * (df["Year"] - df["Promo2SinceYear"])
            + (df["WeekOfYear"] - df["Promo2SinceWeek"])
        )
        df["Promo2Weeks"] = df["Promo2Weeks"].apply(lambda x: x if x > 0 else 0)

        month_map = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        df["MonthStr"] = df["Month"].map(month_map)

        def check_promo(row):
            if row["PromoInterval"] == "" or row["Promo2"] == 0:
                return 0
            if row["MonthStr"] in row["PromoInterval"]:
                return 1
            return 0

        df["IsPromo2Month"] = df.apply(check_promo, axis=1).astype("int8")
        df.loc[df["Promo2Weeks"] <= 0, "IsPromo2Month"] = 0

        for col in self.cat_cols:
            le = self.label_encoders[col]
            vals = df[col].astype(str)
            classes = set(le.classes_)
            vals = vals.map(lambda x: x if x in classes else list(classes)[0])
            df[col] = le.transform(vals)

        drop_cols = [
            "Date",
            "Customers",
            "Open",
            "PromoInterval",
            "StateHoliday",
            "MonthStr",
            "Promo2SinceYear",
            "Promo2SinceWeek",
        ]
        df = df.drop([c for c in drop_cols if c in df.columns], axis=1)

        if self.use_scaler:
            for c in self.cont_cols:
                if c in df.columns:
                    df[c] = df[c].fillna(0)
            df[self.cont_cols] = self.scaler.transform(df[self.cont_cols])

        return df


class EntityEmbeddingFeatureGenerator(BaseRossmannTransformer):
    def __init__(
        self,
        model_save_path: str = "checkpoints/emb_model_compare.pth",
        epochs: int = 5,
        batch_size: int = 512,
    ):
        self.model_save_path = model_save_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cat_cols = [
            "Store",
            "DayOfWeek",
            "Year",
            "Month",
            "Day",
            "StateHoliday",
            "StoreType",
            "Assortment",
            "PromoInterval",
            "WeekOfMonth",
        ]
        self.cont_cols = [
            "CompetitionDistance",
            "CompetitionMonthsOpen",
            "Promo2Weeks",
            "IsPromo2Month",
            "Promo",
            "SchoolHoliday",
            "IsWeekend",
        ]

        self.medians = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.emb_dims = []

    def _base_fe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._clean_state_holiday(df)

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
        df["WeekOfMonth"] = df["Day"].apply(lambda d: (d - 1) // 7 + 1)
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype("int8")
        df["PromoInterval"] = df["PromoInterval"].fillna("")
        return df

    def _train_embedding_model(self, X_processed: pd.DataFrame, y_train: pd.Series):
        X_cat = torch.tensor(X_processed[self.cat_cols].values, dtype=torch.long)
        X_cont = torch.tensor(X_processed[self.cont_cols].values, dtype=torch.float)
        y = torch.tensor(np.log1p(y_train.values), dtype=torch.float).view(-1, 1)

        dataset = TensorDataset(X_cat, X_cont, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = EntityEmbeddingModel(self.emb_dims, n_cont=len(self.cont_cols)).to(
            self.device
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            for cats, conts, targets in loader:
                cats = cats.to(self.device)
                conts = conts.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(cats, conts)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * cats.size(0)

        ckpt_dir = os.path.dirname(self.model_save_path)
        if ckpt_dir and not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        torch.save(model.state_dict(), self.model_save_path)

    def fit(self, X, y=None):
        df = self._base_fe(X)

        self.medians = {
            "CompetitionDistance": df["CompetitionDistance"].median(),
            "CompetitionOpenSinceYear": df["CompetitionOpenSinceYear"].median(),
            "CompetitionOpenSinceMonth": df["CompetitionOpenSinceMonth"].median(),
        }

        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
            self.medians["CompetitionDistance"]
        )
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(
            self.medians["CompetitionOpenSinceYear"]
        ).astype(int)
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(
            self.medians["CompetitionOpenSinceMonth"]
        ).astype(int)

        mask_promo2 = df["Promo2"] == 1

        df.loc[mask_promo2, "Promo2SinceYear"] = df.loc[
            mask_promo2, "Promo2SinceYear"
        ].fillna(df.loc[mask_promo2, "Year"])
        df.loc[mask_promo2, "Promo2SinceWeek"] = df.loc[
            mask_promo2, "Promo2SinceWeek"
        ].fillna(df.loc[mask_promo2, "WeekOfYear"])

        df.loc[~mask_promo2, "Promo2SinceYear"] = df.loc[~mask_promo2, "Year"]
        df.loc[~mask_promo2, "Promo2SinceWeek"] = df.loc[~mask_promo2, "WeekOfYear"]

        df["Promo2SinceYear"] = df["Promo2SinceYear"].astype(int)
        df["Promo2SinceWeek"] = df["Promo2SinceWeek"].astype(int)

        df["CompetitionMonthsOpen"] = (
            12 * (df["Year"] - df["CompetitionOpenSinceYear"])
            + (df["Month"] - df["CompetitionOpenSinceMonth"])
        )
        df["CompetitionMonthsOpen"] = df["CompetitionMonthsOpen"].clip(
            lower=0, upper=60
        ).astype("int16")

        df["Promo2Weeks"] = (
            52 * (df["Year"] - df["Promo2SinceYear"])
            + (df["WeekOfYear"] - df["Promo2SinceWeek"])
        )
        df["Promo2Weeks"] = df["Promo2Weeks"].clip(lower=0, upper=25).astype("int16")

        month_map = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        df["MonthStr"] = df["Month"].map(month_map)

        def check_promo(row):
            if row["PromoInterval"] == "" or row["Promo2"] == 0:
                return 0
            if row["MonthStr"] in row["PromoInterval"]:
                return 1
            return 0

        df["IsPromo2Month"] = df.apply(check_promo, axis=1).astype("int8")
        df.loc[df["Promo2Weeks"] <= 0, "IsPromo2Month"] = 0
        df = df.drop(columns=["MonthStr"])

        self.emb_dims = []
        for col in self.cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
            df[col] = le.transform(df[col].astype(str))

            num_unique = len(le.classes_)
            dim = min(50, (num_unique + 1) // 2)
            self.emb_dims.append((num_unique, dim))

        for col in self.cont_cols:
            df[col] = df[col].fillna(0)
        df[self.cont_cols] = self.scaler.fit_transform(df[self.cont_cols])

        if y is not None:
            self._train_embedding_model(df, y)

        return self

    def transform(self, X):
        df = self._base_fe(X)

        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
            self.medians["CompetitionDistance"]
        )
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(
            self.medians["CompetitionOpenSinceYear"]
        ).astype(int)
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(
            self.medians["CompetitionOpenSinceMonth"]
        ).astype(int)

        mask_promo2 = df["Promo2"] == 1

        df.loc[mask_promo2, "Promo2SinceYear"] = df.loc[
            mask_promo2, "Promo2SinceYear"
        ].fillna(df.loc[mask_promo2, "Year"])
        df.loc[mask_promo2, "Promo2SinceWeek"] = df.loc[
            mask_promo2, "Promo2SinceWeek"
        ].fillna(df.loc[mask_promo2, "WeekOfYear"])

        df.loc[~mask_promo2, "Promo2SinceYear"] = df.loc[~mask_promo2, "Year"]
        df.loc[~mask_promo2, "Promo2SinceWeek"] = df.loc[~mask_promo2, "WeekOfYear"]

        df["Promo2SinceYear"] = df["Promo2SinceYear"].astype(int)
        df["Promo2SinceWeek"] = df["Promo2SinceWeek"].astype(int)

        df["CompetitionMonthsOpen"] = (
            12 * (df["Year"] - df["CompetitionOpenSinceYear"])
            + (df["Month"] - df["CompetitionOpenSinceMonth"])
        )
        df["CompetitionMonthsOpen"] = df["CompetitionMonthsOpen"].clip(
            lower=0, upper=60
        ).astype("int16")

        df["Promo2Weeks"] = (
            52 * (df["Year"] - df["Promo2SinceYear"])
            + (df["WeekOfYear"] - df["Promo2SinceWeek"])
        )
        df["Promo2Weeks"] = df["Promo2Weeks"].clip(lower=0, upper=25).astype("int16")

        month_map = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        df["MonthStr"] = df["Month"].map(month_map)

        def check_promo(row):
            if row["PromoInterval"] == "" or row["Promo2"] == 0:
                return 0
            if row["MonthStr"] in row["PromoInterval"]:
                return 1
            return 0

        df["IsPromo2Month"] = df.apply(check_promo, axis=1).astype("int8")
        df.loc[df["Promo2Weeks"] <= 0, "IsPromo2Month"] = 0
        df = df.drop(columns=["MonthStr"])

        for col in self.cat_cols:
            le = self.label_encoders[col]
            vals = df[col].astype(str)
            classes = set(le.classes_)
            vals = vals.map(lambda x: x if x in classes else list(classes)[0])
            df[col] = le.transform(vals)

        df_scaled_cont = df[self.cont_cols].copy().fillna(0)
        df_scaled_cont = self.scaler.transform(df_scaled_cont)

        model = EntityEmbeddingModel(self.emb_dims, n_cont=len(self.cont_cols))
        try:
            model.load_state_dict(
                torch.load(self.model_save_path, map_location=self.device)
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Missing model file: {self.model_save_path}. Please run .fit() first!"
            )

        model.eval()

        embeddings_dict = {}
        for i, col in enumerate(self.cat_cols):
            embeddings_dict[col] = model.embeddings[i].weight.detach().cpu().numpy()

        df_final = pd.DataFrame(df_scaled_cont, columns=self.cont_cols, index=df.index)
        embedding_dfs = []

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
