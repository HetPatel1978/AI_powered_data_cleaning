# """
# Simple AI-Powered Data Cleaner
# Clean CSV/Excel files automatically
# """
# import pandas as pd
# import numpy as np
# from sklearn.impute import KNNImputer
# from typing import Dict, List, Tuple


# class SimpleDataCleaner:
#     """Simple but powerful data cleaner"""
    
#     def __init__(self):
#         self.cleaning_report = []
    
#     def load_data(self, file) -> pd.DataFrame:
#         """Load CSV or Excel file"""
#         try:
#             if file.name.endswith('.csv'):
#                 df = pd.read_csv(file)
#             elif file.name.endswith(('.xlsx', '.xls')):
#                 df = pd.read_excel(file)
#             else:
#                 raise ValueError("Only CSV and Excel files supported")
#             return df
#         except Exception as e:
#             raise Exception(f"Error loading file: {str(e)}")
    
#     def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
#         """Analyze data quality and identify issues"""
#         issues = {
#             'total_rows': len(df),
#             'total_columns': len(df.columns),
#             'missing_values': {},
#             'duplicates': int(df.duplicated().sum()),
#             'data_types': {},
#             'outliers': {},
#             'quality_score': 0
#         }
        
#         # Check missing values
#         for col in df.columns:
#             missing = df[col].isna().sum()
#             if missing > 0:
#                 issues['missing_values'][col] = {
#                     'count': int(missing),
#                     'percentage': float(missing / len(df) * 100)
#                 }
        
#         # Check data types
#         for col in df.columns:
#             issues['data_types'][col] = str(df[col].dtype)
        
#         # Check outliers in numeric columns
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         for col in numeric_cols:
#             outliers = self._detect_outliers(df[col])
#             if outliers > 0:
#                 issues['outliers'][col] = int(outliers)
        
#         # Calculate quality score
#         issues['quality_score'] = self._calculate_quality_score(df)
        
#         return issues
    
#     def clean_data(self, df: pd.DataFrame, options: Dict) -> Tuple[pd.DataFrame, List]:
#         """Clean data based on selected options"""
#         self.cleaning_report = []
#         df_clean = df.copy()
        
#         # 1. Remove duplicates
#         if options.get('remove_duplicates', False):
#             before = len(df_clean)
#             df_clean = df_clean.drop_duplicates()
#             removed = before - len(df_clean)
#             if removed > 0:
#                 self.cleaning_report.append(f"✓ Removed {removed} duplicate rows")
        
#         # 2. Handle missing values
#         if options.get('handle_missing', False):
#             strategy = options.get('missing_strategy', 'auto')
#             df_clean = self._handle_missing(df_clean, strategy)
        
#         # 3. Fix data types
#         if options.get('fix_types', False):
#             df_clean = self._fix_types(df_clean)
        
#         # 4. Handle outliers
#         if options.get('handle_outliers', False):
#             df_clean = self._handle_outliers(df_clean, options.get('outlier_action', 'flag'))
        
#         # 5. Clean text
#         if options.get('clean_text', False):
#             df_clean = self._clean_text(df_clean)
        
#         if not self.cleaning_report:
#             self.cleaning_report.append("No cleaning needed - data looks good!")
        
#         return df_clean, self.cleaning_report
    
#     def _handle_missing(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
#         """Handle missing values intelligently"""
#         missing_cols = [col for col in df.columns if df[col].isna().any()]
        
#         if not missing_cols:
#             return df
        
#         for col in missing_cols:
#             missing_count = df[col].isna().sum()
#             missing_pct = missing_count / len(df) * 100
            
#             # If > 70% missing, drop column
#             if missing_pct > 70:
#                 df = df.drop(columns=[col])
#                 self.cleaning_report.append(f"✗ Dropped column '{col}' ({missing_pct:.1f}% missing)")
#                 continue
            
#             # Numeric columns
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 if strategy == 'auto' or strategy == 'median':
#                     df[col].fillna(df[col].median(), inplace=True)
#                     self.cleaning_report.append(f"✓ Filled {missing_count} missing values in '{col}' with median")
#                 elif strategy == 'mean':
#                     df[col].fillna(df[col].mean(), inplace=True)
#                     self.cleaning_report.append(f"✓ Filled {missing_count} missing values in '{col}' with mean")
#                 elif strategy == 'knn':
#                     # KNN imputation
#                     numeric_cols = df.select_dtypes(include=[np.number]).columns
#                     if len(numeric_cols) > 1:
#                         imputer = KNNImputer(n_neighbors=5)
#                         df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
#                         self.cleaning_report.append(f"✓ Applied KNN imputation to numeric columns")
            
#             # Categorical columns
#             else:
#                 mode_val = df[col].mode()
#                 if len(mode_val) > 0:
#                     df[col].fillna(mode_val[0], inplace=True)
#                     self.cleaning_report.append(f"✓ Filled {missing_count} missing values in '{col}' with mode")
        
#         return df
    
#     def _fix_types(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Auto-detect and fix data types"""
#         for col in df.columns:
#             if df[col].dtype == 'object':
#                 # Try numeric
#                 try:
#                     df[col] = pd.to_numeric(df[col])
#                     self.cleaning_report.append(f"✓ Converted '{col}' to numeric")
#                     continue
#                 except:
#                     pass
                
#                 # Try datetime
#                 try:
#                     df[col] = pd.to_datetime(df[col])
#                     self.cleaning_report.append(f"✓ Converted '{col}' to datetime")
#                     continue
#                 except:
#                     pass
        
#         return df
    
#     def _handle_outliers(self, df: pd.DataFrame, action: str) -> pd.DataFrame:
#         """Detect and handle outliers"""
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
        
#         for col in numeric_cols:
#             Q1 = df[col].quantile(0.25)
#             Q3 = df[col].quantile(0.75)
#             IQR = Q3 - Q1
            
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR
            
#             outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
#             if outliers > 0:
#                 if action == 'remove':
#                     before = len(df)
#                     df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
#                     self.cleaning_report.append(f"✓ Removed {before - len(df)} outliers from '{col}'")
#                 elif action == 'cap':
#                     df[col] = df[col].clip(lower_bound, upper_bound)
#                     self.cleaning_report.append(f"✓ Capped {outliers} outliers in '{col}'")
#                 elif action == 'flag':
#                     df[f'{col}_is_outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound))
#                     self.cleaning_report.append(f"✓ Flagged {outliers} outliers in '{col}'")
        
#         return df
    
#     def _clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Clean text columns"""
#         text_cols = df.select_dtypes(include=['object']).columns
        
#         for col in text_cols:
#             if df[col].dtype == 'object':
#                 # Strip whitespace
#                 df[col] = df[col].str.strip()
#                 # Remove extra spaces
#                 df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
#                 self.cleaning_report.append(f"✓ Cleaned text in '{col}'")
        
#         return df
    
#     def _detect_outliers(self, series: pd.Series) -> int:
#         """Count outliers using IQR method"""
#         if not pd.api.types.is_numeric_dtype(series) or series.isna().all():
#             return 0
        
#         Q1 = series.quantile(0.25)
#         Q3 = series.quantile(0.75)
#         IQR = Q3 - Q1
        
#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR
        
#         return int(((series < lower) | (series > upper)).sum())
    
#     def _calculate_quality_score(self, df: pd.DataFrame) -> float:
#         """Calculate data quality score (0-100)"""
#         score = 100.0
        
#         # Penalize missing values
#         missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
#         score -= min(missing_pct * 0.5, 30)
        
#         # Penalize duplicates
#         dup_pct = (df.duplicated().sum() / len(df)) * 100
#         score -= min(dup_pct * 0.3, 20)
        
#         # Penalize outliers
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         if len(numeric_cols) > 0:
#             outlier_cols = sum(1 for col in numeric_cols if self._detect_outliers(df[col]) > 0)
#             score -= min(outlier_cols * 3, 15)
        
#         return max(0.0, round(score, 2))
    
#     def generate_insights(self, df: pd.DataFrame) -> List[str]:
#         """Generate simple AI insights"""
#         insights = []
        
#         # Numeric column insights
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         for col in numeric_cols:
#             mean_val = df[col].mean()
#             median_val = df[col].median()
            
#             # Check if skewed
#             if abs(mean_val - median_val) / (df[col].std() + 0.001) > 0.5:
#                 insights.append(f"📊 '{col}' is skewed (mean={mean_val:.2f}, median={median_val:.2f})")
            
#             # Check variance
#             if df[col].std() / (mean_val + 0.001) > 1:
#                 insights.append(f"📈 '{col}' has high variance - values are spread out")
        
#         # Categorical insights
#         cat_cols = df.select_dtypes(include=['object']).columns
#         for col in cat_cols:
#             unique_pct = (df[col].nunique() / len(df)) * 100
#             if unique_pct < 5:
#                 insights.append(f"🏷️  '{col}' has only {df[col].nunique()} unique values")
#             elif unique_pct > 95:
#                 insights.append(f"🔍 '{col}' has very high cardinality ({df[col].nunique()} unique values)")
        
#         # Correlation insights (if multiple numeric columns)
#         if len(numeric_cols) >= 2:
#             corr_matrix = df[numeric_cols].corr()
#             high_corr = []
#             for i in range(len(corr_matrix.columns)):
#                 for j in range(i+1, len(corr_matrix.columns)):
#                     if abs(corr_matrix.iloc[i, j]) > 0.7:
#                         high_corr.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}")
            
#             if high_corr:
#                 insights.append(f"🔗 Strong correlations found: {', '.join(high_corr[:3])}")
        
#         return insights if insights else ["✨ Data looks good! No major patterns detected."]












































"""
Simple AI-Powered Data Cleaner
Clean CSV/Excel files automatically
"""
import pandas as pd
import numpy as np
import warnings
import re
from sklearn.impute import KNNImputer
from typing import Dict, List, Tuple

# Suppress specific pandas datetime warnings
warnings.filterwarnings('ignore', message='Could not infer format, so each element will be parsed individually')


class SimpleDataCleaner:
    """Simple but powerful data cleaner"""
    
    def __init__(self):
        self.cleaning_report = []
    
    def load_data(self, file) -> pd.DataFrame:
        """Load CSV or Excel file"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                raise ValueError("Only CSV and Excel files supported")
            return df
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """Analyze data quality and identify issues"""
        issues = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicates': int(df.duplicated().sum()),
            'data_types': {},
            'outliers': {},
            'quality_score': 0
        }
        
        # Check missing values
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                issues['missing_values'][col] = {
                    'count': int(missing),
                    'percentage': float(missing / len(df) * 100)
                }
        
        # Check data types
        for col in df.columns:
            issues['data_types'][col] = str(df[col].dtype)
        
        # Check outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers = self._detect_outliers(df[col])
            if outliers > 0:
                issues['outliers'][col] = int(outliers)
        
        # Calculate quality score
        issues['quality_score'] = self._calculate_quality_score(df)
        
        return issues
    
    def detect_datetime_format(self, series: pd.Series) -> str:
        """Detect most likely datetime format for a series"""
        sample = series.dropna().head(10).astype(str)
        
        if sample.empty:
            return None
        
        # Common patterns to test
        patterns = [
            (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),
            (r'^\d{2}/\d{2}/\d{4}$', '%m/%d/%Y'),
            (r'^\d{1,2}/\d{1,2}/\d{4}$', '%m/%d/%Y'),
            (r'^\d{4}/\d{2}/\d{2}$', '%Y/%m/%d'),
            (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', '%Y-%m-%d %H:%M:%S'),
            (r'^\d{2}-\d{2}-\d{4}$', '%m-%d-%Y'),
            (r'^\d{4}–\d{4}$', None),  # Range format like "2023–2024"
            (r'^\d{4}-\d{4}$', None),  # Range format like "2023-2024"
        ]
        
        for pattern, fmt in patterns:
            try:
                if sample.str.match(pattern).any():
                    return fmt
            except:
                continue
        
        return None  # Let pandas guess
    
    def clean_data(self, df: pd.DataFrame, options: Dict) -> Tuple[pd.DataFrame, List]:
        """Clean data based on selected options"""
        self.cleaning_report = []
        df_clean = df.copy()
        
        # 1. Remove duplicates
        if options.get('remove_duplicates', False):
            before = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed = before - len(df_clean)
            if removed > 0:
                self.cleaning_report.append(f"✓ Removed {removed} duplicate rows")
        
        # 2. Handle missing values
        if options.get('handle_missing', False):
            strategy = options.get('missing_strategy', 'auto')
            df_clean = self._handle_missing(df_clean, strategy)
        
        # 3. Fix data types
        if options.get('fix_types', False):
            df_clean = self._fix_types(df_clean)
        
        # 4. Handle outliers
        if options.get('handle_outliers', False):
            df_clean = self._handle_outliers(df_clean, options.get('outlier_action', 'flag'))
        
        # 5. Clean text
        if options.get('clean_text', False):
            df_clean = self._clean_text(df_clean)
        
        # 6. Parse currency fields (NEW FEATURE)
        if options.get('parse_currency', False):
            df_clean = self._parse_currency(df_clean)
        
        if not self.cleaning_report:
            self.cleaning_report.append("No cleaning needed - data looks good!")
        
        return df_clean, self.cleaning_report
    
    def _parse_currency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse currency strings to numeric values"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains currency-like strings
                sample = df[col].dropna().head(5).astype(str)
                if sample.str.contains(r'[$£€¥]').any() or sample.str.contains(r'\d+,\d+').any():
                    try:
                        # Remove currency symbols and commas, convert to float
                        cleaned = df[col].astype(str).str.replace(r'[$£€¥,"]', '', regex=True)
                        cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
                        df[col] = pd.to_numeric(cleaned, errors='coerce')
                        self.cleaning_report.append(f"✓ Converted currency strings in '{col}' to numeric")
                    except:
                        pass
        return df
    
    def _handle_missing(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values intelligently"""
        missing_cols = [col for col in df.columns if df[col].isna().any()]
        
        if not missing_cols:
            return df
        
        for col in missing_cols:
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df) * 100
            
            # If > 70% missing, drop column
            if missing_pct > 70:
                df = df.drop(columns=[col])
                self.cleaning_report.append(f"✗ Dropped column '{col}' ({missing_pct:.1f}% missing)")
                continue
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                if strategy == 'auto' or strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                    self.cleaning_report.append(f"✓ Filled {missing_count} missing values in '{col}' with median")
                elif strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                    self.cleaning_report.append(f"✓ Filled {missing_count} missing values in '{col}' with mean")
                elif strategy == 'knn':
                    # KNN imputation
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        imputer = KNNImputer(n_neighbors=5)
                        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                        self.cleaning_report.append(f"✓ Applied KNN imputation to numeric columns")
            
            # Categorical columns
            else:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
                    self.cleaning_report.append(f"✓ Filled {missing_count} missing values in '{col}' with mode")
        
        return df
    
    def _fix_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-detect and fix data types with improved datetime handling"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try numeric first (including currency parsing)
                try:
                    # Check if it looks like currency
                    sample = df[col].dropna().head(3).astype(str)
                    if sample.str.contains(r'[$£€¥]').any() or sample.str.contains(r'\d+,\d+').any():
                        # Remove currency symbols and commas
                        cleaned = df[col].astype(str).str.replace(r'[$£€¥,"]', '', regex=True)
                        cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
                        df[col] = pd.to_numeric(cleaned, errors='coerce')
                        self.cleaning_report.append(f"✓ Converted '{col}' from currency to numeric")
                        continue
                    else:
                        # Regular numeric conversion
                        df[col] = pd.to_numeric(df[col])
                        self.cleaning_report.append(f"✓ Converted '{col}' to numeric")
                        continue
                except:
                    pass
                
                # Try datetime with format detection
                try:
                    detected_format = self.detect_datetime_format(df[col])
                    
                    if detected_format is None:
                        # Skip datetime conversion for range formats or annotations
                        sample_vals = df[col].dropna().head(3).astype(str).tolist()
                        if any(('–' in str(val) or '-' in str(val) or '[' in str(val)) for val in sample_vals):
                            # This looks like a range format or annotation, skip datetime conversion
                            continue
                        else:
                            # Let pandas infer but suppress warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                df[col] = pd.to_datetime(df[col], errors='ignore')
                                if pd.api.types.is_datetime64_any_dtype(df[col]):
                                    self.cleaning_report.append(f"✓ Converted '{col}' to datetime (format inferred)")
                    else:
                        # Use detected format
                        df[col] = pd.to_datetime(df[col], format=detected_format, errors='ignore')
                        if pd.api.types.is_datetime64_any_dtype(df[col]):
                            self.cleaning_report.append(f"✓ Converted '{col}' to datetime using format {detected_format}")
                except:
                    pass
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, action: str) -> pd.DataFrame:
        """Detect and handle outliers with improved logic"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Skip outlier detection for ID-like columns
            if 'id' in col.lower() or 'rank' in col.lower():
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Use more conservative outlier detection for small datasets
            multiplier = 2.0 if len(df) < 100 else 1.5
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            # Only flag as outliers if more than 1% but less than 30% of data
            outlier_pct = outliers / len(df) * 100
            if outliers > 0 and 1 <= outlier_pct <= 30:
                if action == 'remove':
                    before = len(df)
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    self.cleaning_report.append(f"✓ Removed {before - len(df)} outliers from '{col}'")
                elif action == 'cap':
                    df[col] = df[col].clip(lower_bound, upper_bound)
                    self.cleaning_report.append(f"✓ Capped {outliers} outliers in '{col}'")
                elif action == 'flag':
                    df[f'{col}_is_outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound))
                    self.cleaning_report.append(f"✓ Flagged {outliers} outliers in '{col}' ({outlier_pct:.1f}%)")
        
        return df
    
    def _clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns with improved handling"""
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            if df[col].dtype == 'object':
                try:
                    # Clean whitespace and normalize text
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                    df[col] = df[col].str.replace(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', regex=True)
                    
                    # Handle special characters
                    df[col] = df[col].str.replace('†', '', regex=False)
                    df[col] = df[col].str.replace('‡', '', regex=False)
                    
                    self.cleaning_report.append(f"✓ Cleaned text in '{col}'")
                except:
                    pass
        
        return df
    
    def _detect_outliers(self, series: pd.Series) -> int:
        """Count outliers using improved IQR method"""
        if not pd.api.types.is_numeric_dtype(series) or series.isna().all():
            return 0
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:  # No variance
            return 0
        
        # Use more conservative multiplier for small datasets
        multiplier = 2.0 if len(series) < 100 else 1.5
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        
        outliers = ((series < lower) | (series > upper)).sum()
        
        # Only count as outliers if reasonable percentage (1-30%)
        outlier_pct = outliers / len(series) * 100
        return int(outliers) if 1 <= outlier_pct <= 30 else 0
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score (0-100) with improved logic"""
        score = 100.0
        
        # Penalize missing values
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        score -= min(missing_pct * 0.5, 30)
        
        # Penalize duplicates
        dup_pct = (df.duplicated().sum() / len(df)) * 100
        score -= min(dup_pct * 0.3, 20)
        
        # Penalize outliers (but less aggressively)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            outlier_cols = sum(1 for col in numeric_cols if self._detect_outliers(df[col]) > 0)
            score -= min(outlier_cols * 2, 10)
        
        # Bonus for good data type consistency
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            # Check if object columns contain mostly consistent data
            consistent_cols = 0
            for col in object_cols:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.8:  # Not too many unique values
                    consistent_cols += 1
            
            consistency_bonus = (consistent_cols / len(object_cols)) * 5
            score += consistency_bonus
        
        return max(0.0, round(score, 2))
    
    def generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate enhanced AI insights"""
        insights = []
        
        # Data size insights
        if len(df) > 10000:
            insights.append(f"📊 Large dataset with {len(df):,} rows - good for robust analysis")
        elif len(df) < 100:
            insights.append(f"⚠️ Small dataset ({len(df)} rows) - results may be less reliable")
        
        # Numeric column insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().all():
                continue
                
            mean_val = df[col].mean()
            median_val = df[col].median()
            
            # Check if skewed
            if abs(mean_val - median_val) / (df[col].std() + 0.001) > 0.5:
                skew_direction = "right" if mean_val > median_val else "left"
                insights.append(f"📈 '{col}' is {skew_direction}-skewed (mean={mean_val:.2f}, median={median_val:.2f})")
            
            # Check variance
            cv = df[col].std() / (abs(mean_val) + 0.001)  # Coefficient of variation
            if cv > 1:
                insights.append(f"📊 '{col}' has high variance (CV={cv:.2f}) - values vary widely")
            
            # Check for potential currency columns
            if mean_val > 1000 and col.lower() in ['price', 'cost', 'revenue', 'gross', 'salary']:
                insights.append(f"💰 '{col}' appears to be financial data (avg: ${mean_val:,.0f})")
        
        # Categorical insights
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            unique_pct = (df[col].nunique() / len(df)) * 100
            if unique_pct < 5:
                insights.append(f"🏷️ '{col}' has only {df[col].nunique()} categories - good for grouping")
            elif unique_pct > 90:
                insights.append(f"🔍 '{col}' has very high uniqueness ({df[col].nunique()} values) - might be an identifier")
            
            # Check for potential issues
            if df[col].str.contains(r'[\$£€¥]', na=False).any():
                insights.append(f"💱 '{col}' contains currency symbols - consider parsing to numeric")
        
        # Correlation insights (enhanced)
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            high_corr = []
            moderate_corr = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]} (r={corr_val:.2f})")
                    elif abs(corr_val) > 0.5:
                        moderate_corr.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}")
            
            if high_corr:
                insights.append(f"🔗 Strong correlations: {'; '.join(high_corr[:2])}")
            elif moderate_corr:
                insights.append(f"🔗 Moderate correlations found: {'; '.join(moderate_corr[:3])}")
        
        # Data quality insights
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            insights.append(f"🔍 Missing data in: {', '.join(missing_cols[:3])}")
        
        return insights if insights else ["✨ Data looks excellent! No major issues detected."]
