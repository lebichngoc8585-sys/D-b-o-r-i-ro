import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# Thư viện Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# ==============================================================================
# STEP 1: CONFIGURATION (Lệnh Streamlit đầu tiên bắt buộc)
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống học máy phát hiện giao dịch gian lận tại Agribank",
    page_icon="🤖"
)

# ==============================================================================
# STEP 2: CACHED FUNCTIONS (Hàm nạp dữ liệu dùng chung)
# ==============================================================================
@st.cache_data(show_spinner="Đang xử lý dữ liệu...")
def load_data(file_bytes, file_name):
    """
    Đọc dữ liệu từ bytes để tối ưu cache và tránh lỗi hashable.
    Đồng thời thực hiện các bước tạo biến phái sinh giống trong Notebook (nếu có).
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {str(e)}")
        return None

# ==============================================================================
# STEP 3: SIDEBAR - VÙNG CẤU HÌNH (THÀNH PHẦN 1)
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải file dữ liệu nguồn
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn file dữ liệu mẫu đã dùng trong Notebook để huấn luyện mô hình."
    )
    
    st.divider()
    
    st.subheader("🤖 Cấu hình Mô hình")
    task_type = st.selectbox(
        "Loại bài toán",
        options=["Phân loại (Classification)", "Hồi quy (Regression)"],
        help="Chọn loại bài toán tương ứng với mục tiêu trong Notebook."
    )
    
    model_option = st.selectbox(
        "Thuật toán sử dụng",
        options=["Random Forest", "Linear/Logistic Regression"],
        help="Chọn thuật toán mô hình toán học."
    )
    
    st.subheader("🎛️ Siêu tham số (Hyperparameters)")
    
    if model_option == "Random Forest":
        n_estimators = st.slider("Số cây (n_estimators)", min_value=10, max_value=500, value=100, step=10)
        max_depth = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=50, value=10)
        criterion = st.selectbox("Tiêu chí đánh giá", options=["gini", "entropy"] if "Phân loại" in task_type else ["squared_error", "absolute_error"])
    else:
        c_value = st.number_input("Hệ số điều hòa (C / Alpha)", min_value=0.001, max_value=100.0, value=1.0, step=0.1)
        
    with st.expander("🛠️ Tham số hệ thống nâng cao"):
        random_state = st.number_input("Cố định dữ liệu (random_state)", min_value=0, max_value=9999, value=42, step=1)
        test_size = st.slider("Tỷ lệ tập kiểm tra (test_size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    st.divider()
    
    trigger_train = st.button("🚀 Huấn luyện Mô hình", type="primary", use_container_width=True)

# ==============================================================================
# STEP 4: HEADER & ĐIỀU PHỐI TRẠNG THÁI (THÀNH PHẦN 2)
# ==============================================================================
st.title("📊 Hệ thống học máy phát hiện giao dịch gian lận tại Agribank")
st.caption("Ứng dụng hỗ trợ nạp dữ liệu, phân tích khám phá, huấn luyện tự động và dự báo trực tiếp dựa trên kiến trúc pipeline chuẩn hóa.")

if uploaded_file is None:
    st.info("👋 Chào mừng! Vui lòng tải file dữ liệu ở **Thanh bên (Sidebar)** để bắt đầu ứng dụng.")
    st.stop()
else:
    file_bytes = uploaded_file.getvalue()
    df = load_data(file_bytes, uploaded_file.name)
    
    if df is None:
        st.error("Không thể xử lý tệp dữ liệu. Vui lòng kiểm tra định dạng.")
        st.stop()
        
    st.caption(f"📁 **Đang sử dụng tệp:** {uploaded_file.name} | **Kích thước:** {df.shape[0]} dòng, {df.shape[1]} cột")
    st.divider()

all_columns = df.columns.tolist()
y_column = st.sidebar.selectbox("Biến mục tiêu (y)", options=all_columns, index=len(all_columns)-1)
x_columns = [col for col in all_columns if col != y_column]

numeric_features = df[x_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df[x_columns].select_dtypes(include=['object', 'category']).columns.tolist()

# ==============================================================================
# STEP 5: KHỐI HUẤN LUYỆN MÔ HÌNH
# ==============================================================================
if trigger_train:
    with st.spinner("🔄 Hệ thống đang xử lý đường ống dữ liệu và huấn luyện..."):
        try:
            X = df[x_columns]
            y = df[y_column]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )
            
            if "Phân loại" in task_type:
                if model_option == "Random Forest":
                    model_core = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=int(random_state))
                else:
                    from sklearn.linear_model import LogisticRegression
                    model_core = LogisticRegression(C=c_value, random_state=int(random_state), max_iter=1000)
            else:
                if model_option == "Random Forest":
                    model_core = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=int(random_state))
                else:
                    from sklearn.linear_model import LinearRegression
                    model_core = LinearRegression()
            
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_core)])
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            metrics_results = {}
            
            if "Phân loại" in task_type:
                metrics_results['type'] = 'classification'
                metrics_results['accuracy'] = accuracy_score(y_test, y_pred)
                metrics_results['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics_results['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics_results['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics_results['cm'] = confusion_matrix(y_test, y_pred)
                metrics_results['report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            else:
                metrics_results['type'] = 'regression'
                metrics_results['r2'] = r2_score(y_test, y_pred)
                metrics_results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                metrics_results['mae'] = mean_absolute_error(y_test, y_pred)
                metrics_results['y_test'] = y_test.values
                metrics_results['y_pred'] = y_pred
                
            st.session_state['trained_pipeline'] = pipeline
            st.session_state['metrics'] = metrics_results
            st.session_state['x_columns'] = x_columns
            st.session_state['numeric_features'] = numeric_features
