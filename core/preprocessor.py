import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self):
        self.required_columns = ['date', 'description', 'amount']
        self.optional_columns = ['category']

    def load_csv(self, path):
        
        try:
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV file not found: {path}")

           
            if os.path.getsize(path) == 0:
                raise ValueError(f"CSV file is empty: {path}")

            
            encodings = ['utf-8', 'latin1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, encoding=encoding, on_bad_lines='skip', engine='python')
                    logger.info(f"Successfully loaded CSV with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
                except pd.errors.ParserError:
                    try:
                        df = pd.read_csv(path, encoding=encoding, sep=',', quotechar='"',
                                         on_bad_lines='skip', engine='python')
                        logger.warning("Loaded CSV with flexible parsing due to parsing errors")
                        break
                    except:
                        continue

            if df is None:
                raise ValueError(f"Could not read CSV file with any supported encoding: {path}")

            
            if df.empty or len(df.columns) < len(self.required_columns):
                logger.warning("Standard CSV parsing failed, attempting manual parsing")
                df = self._manual_csv_parse(path)
                if df is None or df.empty:
                    raise ValueError("Could not parse CSV file even with manual parsing")

            
            df = self._validate_columns(df)
            df = self._clean_data(df)
            df = self._standardize_data(df)

            logger.info(f"Successfully processed {len(df)} transactions")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise

    def _manual_csv_parse(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                return None

            header = lines[0].strip().split(',')
            if len(header) < 3:
                return None

            header = [h.strip().lower() for h in header]

            data = []
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue

                if '"' in line:
                    parts = []
                    cur = ""
                    quotes = False
                    for c in line:
                        if c == '"' and not quotes:
                            quotes = True
                        elif c == '"' and quotes:
                            quotes = False
                        elif c == ',' and not quotes:
                            parts.append(cur)
                            cur = ""
                        else:
                            cur += c
                    parts.append(cur)
                    row = [p.strip() for p in parts]
                else:
                    row = [p.strip() for p in line.split(',')]

                if len(row) >= len(self.required_columns):
                    data.append(row)

            if not data:
                return None

            return pd.DataFrame(data, columns=header[:len(data[0])])

        except Exception as e:
            logger.error(f"Manual CSV parsing failed: {str(e)}")
            return None

    def _validate_columns(self, df):
        missing_required = [c for c in self.required_columns if c not in df.columns]
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")

        if len(df) == 0:
            raise ValueError("CSV contains no data rows")

        logger.info(f"Validated columns: {list(df.columns)}")
        return df

    def _clean_data(self, df):
        df = df.dropna(how='all')
        df['description'] = df['description'].fillna('Unknown transaction')
        df['amount'] = df['amount'].fillna(0)

        if 'category' in df.columns:
            df['category'] = df['category'].fillna('uncategorized')

        duplicates = df.duplicated(subset=['date', 'description', 'amount'], keep='first')
        if duplicates.any():
            logger.info(f"Removed {duplicates.sum()} duplicate transactions")
            df = df[~duplicates]

        return df

    def _standardize_data(self, df):
        df = self._parse_dates(df)
        df = self._parse_amounts(df)
        df = self._clean_text(df)
        df = self._validate_data_ranges(df)
        return df

    def _parse_dates(self, df):
        formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y',
            '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y',
            '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S'
        ]

        parsed = []
        for d in df['date']:
            if pd.isna(d):
                parsed.append(pd.NaT)
                continue

            d = str(d).strip()
            ok = False
            for f in formats:
                try:
                    parsed.append(pd.to_datetime(d, format=f))
                    ok = True
                    break
                except:
                    continue

            if not ok:
                try:
                    parsed.append(pd.to_datetime(d))
                except:
                    logger.warning(f"Could not parse date: {d}, using NaT")
                    parsed.append(pd.NaT)

        df['date'] = parsed
        df = df.dropna(subset=['date'])
        return df

    def _parse_amounts(self, df):
        def clean(a):
            if pd.isna(a):
                return 0.0

            s = str(a).strip()
            s = re.sub(r'[$,€£¥₹₽₩₦₨₪₫₡₵₺₴₸₼₲₱₭₯₰₳₶₷₻₽₾₿]', '', s)
            s = s.replace(',', '')

            if s.startswith('(') and s.endswith(')'):
                s = '-' + s[1:-1]

            try:
                return float(s)
            except:
                logger.warning(f"Could not parse amount: {a}, setting 0")
                return 0.0

        df['amount'] = df['amount'].apply(clean)
        return df

    def _clean_text(self, df):
        def clean(desc):
            if pd.isna(desc):
                return 'unknown transaction'

            desc = str(desc).strip().lower()
            desc = re.sub(r'\s+', ' ', desc)
            desc = re.sub(r'[^\w\s]', '', desc)
            return desc if desc else 'unknown transaction'

        df['description'] = df['description'].apply(clean)

        if 'category' in df.columns:
            df['category'] = df['category'].str.strip().str.lower()

        return df

    def _validate_data_ranges(self, df):
        return df

    def get_anomaly_features(self, df):
        df["amount_z"] = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-9)
        df["hour"] = df["date"].dt.hour.fillna(12)
        df["day"] = df["date"].dt.day.fillna(15)
        df["month"] = df["date"].dt.month.fillna(6)
        df["weekday"] = df["date"].dt.weekday.fillna(2)
        return df

    def get_monthly_summary(self, df):
        df_copy = df.copy()
        df_copy["month"] = df_copy["date"].dt.to_period("M")
        result = df_copy.groupby("month")["amount"].sum().astype(float).to_dict()
        return {str(k): float(v) for k, v in result.items()}

    def get_category_summary(self, df):
        return df.groupby("category")["amount"].sum().astype(float).to_dict()