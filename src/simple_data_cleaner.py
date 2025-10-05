"""
Simple AI-Powered Data Cleaner
Clean CSV/Excel files automatically
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from typing import Dict, List, Tuple


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
        
        if not self.cleaning_report:
            self.cleaning_report.append("No cleaning needed - data looks good!")
        
        return df_clean, self.cleaning_report
    
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
        """Auto-detect and fix data types"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try numeric
                try:
                    df[col] = pd.to_numeric(df[col])
                    self.cleaning_report.append(f"✓ Converted '{col}' to numeric")
                    continue
                except:
                    pass
                
                # Try datetime
                try:
                    df[col] = pd.to_datetime(df[col])
                    self.cleaning_report.append(f"✓ Converted '{col}' to datetime")
                    continue
                except:
                    pass
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, action: str) -> pd.DataFrame:
        """Detect and handle outliers"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers > 0:
                if action == 'remove':
                    before = len(df)
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    self.cleaning_report.append(f"✓ Removed {before - len(df)} outliers from '{col}'")
                elif action == 'cap':
                    df[col] = df[col].clip(lower_bound, upper_bound)
                    self.cleaning_report.append(f"✓ Capped {outliers} outliers in '{col}'")
                elif action == 'flag':
                    df[f'{col}_is_outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound))
                    self.cleaning_report.append(f"✓ Flagged {outliers} outliers in '{col}'")
        
        return df
    
    def _clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns"""
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            if df[col].dtype == 'object':
                # Strip whitespace
                df[col] = df[col].str.strip()
                # Remove extra spaces
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                self.cleaning_report.append(f"✓ Cleaned text in '{col}'")
        
        return df
    
    def _detect_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        if not pd.api.types.is_numeric_dtype(series) or series.isna().all():
            return 0
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        return int(((series < lower) | (series > upper)).sum())
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score (0-100)"""
        score = 100.0
        
        # Penalize missing values
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        score -= min(missing_pct * 0.5, 30)
        
        # Penalize duplicates
        dup_pct = (df.duplicated().sum() / len(df)) * 100
        score -= min(dup_pct * 0.3, 20)
        
        # Penalize outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            outlier_cols = sum(1 for col in numeric_cols if self._detect_outliers(df[col]) > 0)
            score -= min(outlier_cols * 3, 15)
        
        return max(0.0, round(score, 2))
    
    def generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate simple AI insights"""
        insights = []
        
        # Numeric column insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean_val = df[col].mean()
            median_val = df[col].median()
            
            # Check if skewed
            if abs(mean_val - median_val) / (df[col].std() + 0.001) > 0.5:
                insights.append(f"📊 '{col}' is skewed (mean={mean_val:.2f}, median={median_val:.2f})")
            
            # Check variance
            if df[col].std() / (mean_val + 0.001) > 1:
                insights.append(f"📈 '{col}' has high variance - values are spread out")
        
        # Categorical insights
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            unique_pct = (df[col].nunique() / len(df)) * 100
            if unique_pct < 5:
                insights.append(f"🏷️  '{col}' has only {df[col].nunique()} unique values")
            elif unique_pct > 95:
                insights.append(f"🔍 '{col}' has very high cardinality ({df[col].nunique()} unique values)")
        
        # Correlation insights (if multiple numeric columns)
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}")
            
            if high_corr:
                insights.append(f"🔗 Strong correlations found: {', '.join(high_corr[:3])}")
        
        return insights if insights else ["✨ Data looks good! No major patterns detected."]