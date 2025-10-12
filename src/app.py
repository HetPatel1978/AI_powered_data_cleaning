# """
# Simple AI Data Cleaning App
# Upload CSV/Excel → Clean → Download
# """
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from simple_data_cleaner import SimpleDataCleaner

# # Page config
# st.set_page_config(
#     page_title="AI Data Cleaner",
#     page_icon="🤖",
#     layout="wide"
# )

# # Initialize
# if 'cleaner' not in st.session_state:
#     st.session_state.cleaner = SimpleDataCleaner()
# if 'df_original' not in st.session_state:
#     st.session_state.df_original = None
# if 'df_cleaned' not in st.session_state:
#     st.session_state.df_cleaned = None
# if 'quality_report' not in st.session_state:
#     st.session_state.quality_report = None

# # Title
# st.title("🤖 AI-Powered Data Cleaner")
# st.markdown("Upload your messy data → AI cleans it automatically → Download clean data")

# # Sidebar
# with st.sidebar:
#     st.header("📁 Upload Data")
#     uploaded_file = st.file_uploader(
#         "Choose CSV or Excel file",
#         type=['csv', 'xlsx', 'xls'],
#         help="Upload your data file to analyze and clean"
#     )
    
#     if uploaded_file:
#         if st.button("🔍 Analyze Data", type="primary"):
#             with st.spinner("Analyzing your data..."):
#                 try:
#                     # Load data
#                     st.session_state.df_original = st.session_state.cleaner.load_data(uploaded_file)
                    
#                     # Analyze quality
#                     st.session_state.quality_report = st.session_state.cleaner.analyze_data_quality(
#                         st.session_state.df_original
#                     )
                    
#                     st.success("✓ Analysis complete!")
#                 except Exception as e:
#                     st.error(f"Error: {str(e)}")

# # Main content
# if st.session_state.df_original is not None:
    
#     # Quality Dashboard
#     st.header("📊 Data Quality Report")
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric(
#             "Rows",
#             f"{st.session_state.quality_report['total_rows']:,}"
#         )
    
#     with col2:
#         st.metric(
#             "Columns",
#             st.session_state.quality_report['total_columns']
#         )
    
#     with col3:
#         score = st.session_state.quality_report['quality_score']
#         st.metric(
#             "Quality Score",
#             f"{score:.1f}/100",
#             delta=f"{'Good' if score > 70 else 'Needs Work'}"
#         )
    
#     with col4:
#         st.metric(
#             "Issues Found",
#             len(st.session_state.quality_report['missing_values']) + 
#             (1 if st.session_state.quality_report['duplicates'] > 0 else 0) +
#             len(st.session_state.quality_report['outliers'])
#         )
    
#     # Issues breakdown
#     st.subheader("⚠️ Issues Detected")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         # Missing values
#         if st.session_state.quality_report['missing_values']:
#             st.markdown("**Missing Values:**")
#             for col, info in st.session_state.quality_report['missing_values'].items():
#                 st.write(f"- `{col}`: {info['count']} ({info['percentage']:.1f}%)")
#         else:
#             st.success("✓ No missing values")
        
#         # Duplicates
#         if st.session_state.quality_report['duplicates'] > 0:
#             st.warning(f"⚠️ {st.session_state.quality_report['duplicates']} duplicate rows found")
#         else:
#             st.success("✓ No duplicates")
    
#     with col2:
#         # Outliers
#         if st.session_state.quality_report['outliers']:
#             st.markdown("**Outliers Detected:**")
#             for col, count in st.session_state.quality_report['outliers'].items():
#                 st.write(f"- `{col}`: {count} outliers")
#         else:
#             st.success("✓ No outliers detected")
    
#     # Preview original data
#     st.subheader("👀 Original Data Preview")
#     st.dataframe(st.session_state.df_original.head(10), use_container_width=True)
    
#     # Cleaning options
#     st.header("🧹 Cleaning Options")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         remove_duplicates = st.checkbox(
#             "Remove duplicate rows",
#             value=st.session_state.quality_report['duplicates'] > 0,
#             help="Remove exact duplicate rows"
#         )
        
#         handle_missing = st.checkbox(
#             "Handle missing values",
#             value=len(st.session_state.quality_report['missing_values']) > 0,
#             help="Fill or remove missing values intelligently"
#         )
        
#         if handle_missing:
#             missing_strategy = st.selectbox(
#                 "Strategy",
#                 ['auto', 'mean', 'median', 'knn'],
#                 help="auto: intelligent selection, knn: advanced AI imputation"
#             )
#         else:
#             missing_strategy = 'auto'
        
#         fix_types = st.checkbox(
#             "Fix data types",
#             value=True,
#             help="Auto-detect and convert data types"
#         )
    
#     with col2:
#         handle_outliers = st.checkbox(
#             "Handle outliers",
#             value=len(st.session_state.quality_report['outliers']) > 0,
#             help="Detect and handle outliers in numeric columns"
#         )
        
#         if handle_outliers:
#             outlier_action = st.selectbox(
#                 "Action",
#                 ['flag', 'cap', 'remove'],
#                 help="flag: add outlier column, cap: limit values, remove: delete rows"
#             )
#         else:
#             outlier_action = 'flag'
        
#         clean_text = st.checkbox(
#             "Clean text",
#             value=True,
#             help="Trim whitespace and normalize text"
#         )
    
#     # Clean button
#     if st.button("✨ Clean Data Now", type="primary", use_container_width=True):
#         with st.spinner("Cleaning your data with AI..."):
#             try:
#                 options = {
#                     'remove_duplicates': remove_duplicates,
#                     'handle_missing': handle_missing,
#                     'missing_strategy': missing_strategy,
#                     'fix_types': fix_types,
#                     'handle_outliers': handle_outliers,
#                     'outlier_action': outlier_action,
#                     'clean_text': clean_text
#                 }
                
#                 st.session_state.df_cleaned, report = st.session_state.cleaner.clean_data(
#                     st.session_state.df_original,
#                     options
#                 )
                
#                 st.session_state.cleaning_report = report
#                 st.success("✓ Data cleaned successfully!")
                
#             except Exception as e:
#                 st.error(f"Error cleaning data: {str(e)}")
    
#     # Show cleaned data
#     if st.session_state.df_cleaned is not None:
#         st.header("✨ Cleaned Data")
        
#         # Cleaning report
#         st.subheader("📋 What Was Done:")
#         for item in st.session_state.cleaning_report:
#             st.write(item)
        
#         # Comparison metrics
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.metric(
#                 "Rows",
#                 f"{len(st.session_state.df_cleaned):,}",
#                 delta=f"{len(st.session_state.df_cleaned) - len(st.session_state.df_original)}"
#             )
        
#         with col2:
#             original_missing = st.session_state.df_original.isna().sum().sum()
#             cleaned_missing = st.session_state.df_cleaned.isna().sum().sum()
#             st.metric(
#                 "Missing Values",
#                 cleaned_missing,
#                 delta=f"{cleaned_missing - original_missing}"
#             )
        
#         with col3:
#             new_quality = st.session_state.cleaner._calculate_quality_score(st.session_state.df_cleaned)
#             st.metric(
#                 "Quality Score",
#                 f"{new_quality:.1f}",
#                 delta=f"+{new_quality - st.session_state.quality_report['quality_score']:.1f}"
#             )
        
#         # Preview cleaned data
#         st.subheader("👀 Cleaned Data Preview")
#         st.dataframe(st.session_state.df_cleaned.head(10), use_container_width=True)
        
#         # AI Insights
#         st.subheader("🤖 AI Insights")
#         insights = st.session_state.cleaner.generate_insights(st.session_state.df_cleaned)
#         for insight in insights:
#             st.info(insight)
        
#         # Visualizations
#         if len(st.session_state.df_cleaned.select_dtypes(include=['number']).columns) > 0:
#             st.subheader("📈 Data Visualizations")
            
#             numeric_cols = st.session_state.df_cleaned.select_dtypes(include=['number']).columns
            
#             # Correlation heatmap
#             if len(numeric_cols) >= 2:
#                 fig = px.imshow(
#                     st.session_state.df_cleaned[numeric_cols].corr(),
#                     title="Correlation Heatmap",
#                     color_continuous_scale='RdBu',
#                     aspect='auto'
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
        
#         # Download section
#         st.header("💾 Download Cleaned Data")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # CSV download
#             csv = st.session_state.df_cleaned.to_csv(index=False)
#             st.download_button(
#                 "📥 Download as CSV",
#                 csv,
#                 file_name="cleaned_data.csv",
#                 mime="text/csv",
#                 use_container_width=True
#             )
        
#         with col2:
#             # Excel download
#             import io
#             buffer = io.BytesIO()
#             with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
#                 st.session_state.df_cleaned.to_excel(writer, index=False, sheet_name='Cleaned Data')
            
#             st.download_button(
#                 "📥 Download as Excel",
#                 buffer.getvalue(),
#                 file_name="cleaned_data.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#                 use_container_width=True
#             )

# else:
#     # Welcome screen
#     st.info("👈 Upload a CSV or Excel file from the sidebar to get started!")
    
#     st.markdown("""
#     ### 🚀 Features:
#     - **Automatic Data Quality Analysis** - Get instant insights about your data
#     - **AI-Powered Cleaning** - Intelligent handling of missing values, duplicates, and outliers
#     - **Smart Type Detection** - Automatically fixes data types
#     - **Business Insights** - AI generates actionable insights from your data
#     - **One-Click Download** - Export cleaned data in CSV or Excel format
    
#     ### 📊 Supported Operations:
#     - ✓ Remove duplicates
#     - ✓ Fill missing values (mean, median, KNN imputation)
#     - ✓ Detect and handle outliers
#     - ✓ Fix data types automatically
#     - ✓ Clean and normalize text
#     - ✓ Generate quality reports
    
#     ### 🎯 Perfect for:
#     - Business analysts cleaning sales data
#     - Data scientists preparing datasets
#     - Anyone working with messy spreadsheets
#     """)

# # Footer
# st.markdown("---")
# st.markdown("Made by het.. | AI-Powered Data Cleaning")
































"""
Simple AI Data Cleaning App
Upload CSV/Excel → Clean → Download
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from simple_data_cleaner import SimpleDataCleaner


# Page config
st.set_page_config(
    page_title="AI Data Cleaner",
    page_icon="🤖",
    layout="wide"
)


# Initialize
if 'cleaner' not in st.session_state:
    st.session_state.cleaner = SimpleDataCleaner()
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'quality_report' not in st.session_state:
    st.session_state.quality_report = None


# Title
st.title("🤖 AI-Powered Data Cleaner")
st.markdown("Upload your messy data → AI cleans it automatically → Download clean data")


# Sidebar
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your data file to analyze and clean"
    )
    
    if uploaded_file:
        if st.button("🔍 Analyze Data", type="primary"):
            with st.spinner("Analyzing your data..."):
                try:
                    # Load data
                    st.session_state.df_original = st.session_state.cleaner.load_data(uploaded_file)
                    
                    # Analyze quality
                    st.session_state.quality_report = st.session_state.cleaner.analyze_data_quality(
                        st.session_state.df_original
                    )
                    
                    st.success("✓ Analysis complete!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# Main content
if st.session_state.df_original is not None:
    
    # Quality Dashboard
    st.header("📊 Data Quality Report")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Rows",
            f"{st.session_state.quality_report['total_rows']:,}"
        )
    
    with col2:
        st.metric(
            "Columns",
            st.session_state.quality_report['total_columns']
        )
    
    with col3:
        score = st.session_state.quality_report['quality_score']
        st.metric(
            "Quality Score",
            f"{score:.1f}/100",
            delta=f"{'Good' if score > 70 else 'Needs Work'}"
        )
    
    with col4:
        st.metric(
            "Issues Found",
            len(st.session_state.quality_report['missing_values']) + 
            (1 if st.session_state.quality_report['duplicates'] > 0 else 0) +
            len(st.session_state.quality_report['outliers'])
        )
    
    # Issues breakdown
    st.subheader("⚠️ Issues Detected")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing values
        if st.session_state.quality_report['missing_values']:
            st.markdown("**Missing Values:**")
            for col, info in st.session_state.quality_report['missing_values'].items():
                st.write(f"- `{col}`: {info['count']} ({info['percentage']:.1f}%)")
        else:
            st.success("✓ No missing values")
        
        # Duplicates
        if st.session_state.quality_report['duplicates'] > 0:
            st.warning(f"⚠️ {st.session_state.quality_report['duplicates']} duplicate rows found")
        else:
            st.success("✓ No duplicates")
    
    with col2:
        # Outliers
        if st.session_state.quality_report['outliers']:
            st.markdown("**Outliers Detected:**")
            for col, count in st.session_state.quality_report['outliers'].items():
                st.write(f"- `{col}`: {count} outliers")
        else:
            st.success("✓ No outliers detected")
    
    # Preview original data
    st.subheader("👀 Original Data Preview")
    st.dataframe(st.session_state.df_original.head(10), use_container_width=True)
    
    # Cleaning options
    st.header("🧹 Cleaning Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        remove_duplicates = st.checkbox(
            "Remove duplicate rows",
            value=st.session_state.quality_report['duplicates'] > 0,
            help="Remove exact duplicate rows"
        )
        
        handle_missing = st.checkbox(
            "Handle missing values",
            value=len(st.session_state.quality_report['missing_values']) > 0,
            help="Fill or remove missing values intelligently"
        )
        
        if handle_missing:
            missing_strategy = st.selectbox(
                "Strategy",
                ['auto', 'mean', 'median', 'knn'],
                help="auto: intelligent selection, knn: advanced AI imputation"
            )
        else:
            missing_strategy = 'auto'
        
        fix_types = st.checkbox(
            "Fix data types",
            value=True,
            help="Auto-detect and convert data types"
        )
    
    with col2:
        handle_outliers = st.checkbox(
            "Handle outliers",
            value=len(st.session_state.quality_report['outliers']) > 0,
            help="Detect and handle outliers in numeric columns"
        )
        
        if handle_outliers:
            outlier_action = st.selectbox(
                "Action",
                ['flag', 'cap', 'remove'],
                help="flag: add outlier column, cap: limit values, remove: delete rows"
            )
        else:
            outlier_action = 'flag'
        
        clean_text = st.checkbox(
            "Clean text",
            value=True,
            help="Trim whitespace and normalize text"
        )
        
        # NEW: Currency parsing option
        parse_currency = st.checkbox(
            "Parse currency strings",
            value=True,
            help="Convert currency strings like '$780,000,000' to numeric values"
        )
    
    # Clean button
    if st.button("✨ Clean Data Now", type="primary", use_container_width=True):
        with st.spinner("Cleaning your data with AI..."):
            try:
                options = {
                    'remove_duplicates': remove_duplicates,
                    'handle_missing': handle_missing,
                    'missing_strategy': missing_strategy,
                    'fix_types': fix_types,
                    'handle_outliers': handle_outliers,
                    'outlier_action': outlier_action,
                    'clean_text': clean_text,
                    'parse_currency': parse_currency  # NEW FEATURE
                }
                
                st.session_state.df_cleaned, report = st.session_state.cleaner.clean_data(
                    st.session_state.df_original,
                    options
                )
                
                st.session_state.cleaning_report = report
                st.success("✓ Data cleaned successfully!")
                
            except Exception as e:
                st.error(f"Error cleaning data: {str(e)}")
    
    # Show cleaned data
    if st.session_state.df_cleaned is not None:
        st.header("✨ Cleaned Data")
        
        # Cleaning report
        st.subheader("📋 What Was Done:")
        for item in st.session_state.cleaning_report:
            st.write(item)
        
        # Comparison metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Rows",
                f"{len(st.session_state.df_cleaned):,}",
                delta=f"{len(st.session_state.df_cleaned) - len(st.session_state.df_original)}"
            )
        
        with col2:
            original_missing = st.session_state.df_original.isna().sum().sum()
            cleaned_missing = st.session_state.df_cleaned.isna().sum().sum()
            st.metric(
                "Missing Values",
                cleaned_missing,
                delta=f"{cleaned_missing - original_missing}"
            )
        
        with col3:
            new_quality = st.session_state.cleaner._calculate_quality_score(st.session_state.df_cleaned)
            st.metric(
                "Quality Score",
                f"{new_quality:.1f}",
                delta=f"+{new_quality - st.session_state.quality_report['quality_score']:.1f}"
            )
        
        # Preview cleaned data
        st.subheader("👀 Cleaned Data Preview")
        st.dataframe(st.session_state.df_cleaned.head(10), use_container_width=True)
        
        # AI Insights
        st.subheader("🤖 AI Insights")
        insights = st.session_state.cleaner.generate_insights(st.session_state.df_cleaned)
        for insight in insights:
            st.info(insight)
        
        # Visualizations
        if len(st.session_state.df_cleaned.select_dtypes(include=['number']).columns) > 0:
            st.subheader("📈 Data Visualizations")
            
            numeric_cols = st.session_state.df_cleaned.select_dtypes(include=['number']).columns
            
            # Correlation heatmap
            if len(numeric_cols) >= 2:
                fig = px.imshow(
                    st.session_state.df_cleaned[numeric_cols].corr(),
                    title="Correlation Heatmap",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Download section
        st.header("💾 Download Cleaned Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv = st.session_state.df_cleaned.to_csv(index=False)
            st.download_button(
                "📥 Download as CSV",
                csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel download
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.df_cleaned.to_excel(writer, index=False, sheet_name='Cleaned Data')
            
            st.download_button(
                "📥 Download as Excel",
                buffer.getvalue(),
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


else:
    # Welcome screen
    st.info("👈 Upload a CSV or Excel file from the sidebar to get started!")
    
    st.markdown("""
    ### 🚀 Features:
    - **Automatic Data Quality Analysis** - Get instant insights about your data
    - **AI-Powered Cleaning** - Intelligent handling of missing values, duplicates, and outliers
    - **Smart Type Detection** - Automatically fixes data types
    - **Currency Parsing** - Converts financial strings to numeric values
    - **Business Insights** - AI generates actionable insights from your data
    - **One-Click Download** - Export cleaned data in CSV or Excel format
    
    ### 📊 Supported Operations:
    - ✓ Remove duplicates
    - ✓ Fill missing values (mean, median, KNN imputation)
    - ✓ Detect and handle outliers
    - ✓ Fix data types automatically
    - ✓ Parse currency strings to numeric values
    - ✓ Clean and normalize text
    - ✓ Generate quality reports
    
    ### 🎯 Perfect for:
    - Business analysts cleaning sales data
    - Data scientists preparing datasets
    - Financial data processing
    - Concert/entertainment industry data
    - Anyone working with messy spreadsheets
    """)


# Footer
st.markdown("---")
st.markdown("Made with by Het | AI-Powered Data Cleaning | Built with Streamlit")
