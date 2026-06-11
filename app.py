import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# Thư viện Machine Learning (Thay đổi tùy theo Notebook của bạn)
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
    page_title="Hệ thống Huấn luyện & Dự báo ML Tự động",
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
        
        # [MỞ RỘNG]: Thêm các bước Tiền xử lý / Feature Engineering cố định từ Notebook vào đây
        # Ví dụ: df['new_col'] = df['col1'] / df['col2']
        
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
    
    # Kịch bản giả định dựa trên Notebook: Chọn bài toán & Thuật toán
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
    
    # Siêu tham số động theo thuật toán (Đọc từ Notebook và đặt mặc định)
    st.subheader("🎛️ Siêu tham số (Hyperparameters)")
    
    if model_option == "Random Forest":
        n_estimators = st.slider("Số cây (n_estimators)", min_value=10, max_value=500, value=100, step=10, help="Số lượng cây quyết định trong rừng.")
        max_depth = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=50, value=10, help="Độ sâu tối đa của từng cây cơ sở.")
        criterion = st.selectbox("Tiêu chí đánh giá", options=["gini", "entropy"] if "Phân loại" in task_type else ["squared_error", "absolute_error"])
    else:
        # Ví dụ cho mô hình Tuyến tính / Logistic
        c_value = st.number_input("Hệ số điều hòa (C / Alpha)", min_value=0.001, max_value=100.0, value=1.0, step=0.1, help="Tham số kiểm soát quá khớp.")
        
    # Gom tham số nâng cao vào Expander
    with st.expander("🛠️ Tham số hệ thống nâng cao"):
        random_state = st.number_input("Cố định dữ liệu (random_state)", min_value=0, max_value=9999, value=42, step=1)
        test_size = st.slider("Tỷ lệ tập kiểm tra (test_size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    st.divider()
    
    # NÚT HÀNH ĐỘNG DUY NHẤT ĐỂ KÍCH HOẠT HUẤN LUYỆN
    trigger_train = st.button("🚀 Huấn luyện Mô hình", type="primary", use_container_width=True, help="Bấm để kích hoạt luồng Pipeline huấn luyện.")

# ==============================================================================
# STEP 4: HEADER & ĐIỀU PHỐI TRẠNG THÁI (THÀNH PHẦN 2)
# ==============================================================================
st.title("📊 Hệ thống học máy phát hiện giao dịch gian lận Agribank")
st.caption("Ứng dụng hỗ trợ nạp dữ liệu, phân tích khám phá, huấn luyện tự động và dự báo trực tiếp dựa trên kiến trúc pipeline chuẩn hóa.")

if uploaded_file is None:
    st.info("👋 Chào mừng! Vui lòng tải file dữ liệu ở **Thanh bên (Sidebar)** để bắt đầu ứng dụng.")
    st.stop() # Dừng ứng dụng tại đây nếu chưa có dữ liệu
else:
    # Đọc dữ liệu qua hàm cache chung
    file_bytes = uploaded_file.getvalue()
    df = load_data(file_bytes, uploaded_file.name)
    
    if df is None:
        st.error("Không thể xử lý tệp dữ liệu. Vui lòng kiểm tra định dạng.")
        st.stop()
        
    st.caption(f"📁 **Đang sử dụng tệp:** {uploaded_file.name} | **Kích thước:** {df.shape[0]} dòng, {df.shape[1]} cột")
    st.divider()

# ==============================================================================
# CẤU HÌNH MỤC TIÊU VÀ BIẾN ĐẦU VÀO (Ánh xạ chính xác từ Notebook của bạn)
# ==============================================================================
# LƯU Ý: Thay đổi tên cột 'target_column' bằng tên cột thực tế trong dữ liệu của bạn
all_columns = df.columns.tolist()
y_column = st.sidebar.selectbox("Biến mục tiêu (y)", options=all_columns, index=len(all_columns)-1, help="Chọn cột làm nhãn/biến phụ thuộc mục tiêu.")
x_columns = [col for col in all_columns if col != y_column]

# Tách biệt biến số và biến phân loại cho khâu Tiền xử lý tự động
numeric_features = df[x_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df[x_columns].select_dtypes(include=['object', 'category']).columns.tolist()

# ==============================================================================
# STEP 5: KHỐI HUẤN LUYỆN MÔ HÌNH (Chỉ chạy một lần khi bấm nút, lưu vào Session State)
# ==============================================================================
if trigger_train:
    with st.spinner("🔄 Hệ thống đang xử lý đường ống dữ liệu và huấn luyện..."):
        try:
            X = df[x_columns]
            y = df[y_column]
            
            # 1. Chia tách dữ liệu tập Train/Test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
            
            # 2. Xây dựng bộ tiền xử lý tự động (Pipeline mã hóa chuẩn Scikit-Learn giống Notebook)
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )
            
            # 3. Khởi tạo mô hình tương ứng lựa chọn từ UI
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
            
            # 4. Đóng gói toàn bộ vào Pipeline thống nhất
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_core)])
            
            # Fit mô hình
            pipeline.fit(X_train, y_train)
            
            # 5. Đánh giá kiểm định mô hình
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
                
            # Lưu trữ 3 thành phần cốt lõi vào session_state bền vững
            st.session_state['trained_pipeline'] = pipeline
            st.session_state['metrics'] = metrics_results
            st.session_state['x_columns'] = x_columns
            st.session_state['numeric_features'] = numeric_features
            st.session_state['categorical_features'] = categorical_features
            
            st.success("🎉 Huấn luyện mô hình thành công! Hãy chuyển sang các Tab kết quả bên dưới để kiểm tra.")
            
        except Exception as e:
            st.error(f"❌ Quá trình huấn luyện gặp lỗi cấu hình dữ liệu: {str(e)}")

# ==============================================================================
# STEP 6: CHIA CHỨC NĂNG THEO TABS RA VÙNG CHÍNH
# ==============================================================================
tabs = st.tabs([
    "📋 Tổng quan dữ liệu", 
    "📊 Trực quan hóa", 
    "🎯 Kết quả huấn luyện & Kiểm định", 
    "🔮 Sử dụng mô hình dự báo"
])

# ------------------------------------------------------------------------------
# TAB 1: TỔNG QUAN DỮ LIỆU (THÀNH PHẦN 3)
# ------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("📌 Chỉ số cấu trúc file")
    m1, m2, m3 = st.columns(3)
    m1.metric("Số lượng dòng (Records)", f"{df.shape[0]:,}")
    m2.metric("Số lượng cột (Features)", f"{df.shape[1]}")
    m3.metric("Dung lượng tệp bộ nhớ", f"{uploaded_file.size / (1024*1024):.2f} MB")
    
    st.write("### 🔍 Xem trước 5 hàng dữ liệu đầu tiên (Head)")
    st.dataframe(df.head(5), use_container_width=True)
    
    st.write("### 📈 Thống kê mô tả các biến mô hình")
    # Chỉ mô tả tập các biến đưa vào mô hình (X và y)
    st.dataframe(df[x_columns + [y_column]].describe(include='all').T, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: TRỰC QUAN HÓA DỮ LIỆU (THÀNH PHẦN 4)
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("🎨 Phân tích trực quan các biến đặc trưng")
    
    # Giới hạn phân tích tối đa 4 biến ưu tiên hiển thị đồng thời để tránh crash giao diện
    viz_features = st.multiselect("Chọn các biến muốn trực quan hóa (Hệ thống tự động nhận diện kiểu đồ thị)", 
                                  options=[y_column] + x_columns, 
                                  default=([y_column] + x_columns[:3])[:4])
    
    if not viz_features:
        st.warning("Vui lòng chọn ít nhất một biến để hiển thị đồ thị.")
    else:
        # Bố trí dạng lưới 2x2 linh hoạt
        cols = st.columns(2)
        for idx, col_name in enumerate(viz_features):
            current_col = cols[idx % 2]
            with current_col:
                st.write(f"**Biến: {col_name}** {'*(Mục tiêu)*' if col_name == y_column else ''}")
                
                # Biểu đồ tự chọn thông minh dựa trên kiểu dữ liệu (Data Type)
                if df[col_name].dtype in ['int64', 'float64']:
                    # Biến số liên tục -> Vẽ biểu đồ phân phối Histogram
                    fig = px.histogram(df, x=col_name, marginal="box", nbins=30, color_discrete_sequence=['#4A90E2'])
                else:
                    # Biến phân loại / chuỗi rời rạc -> Vẽ biểu đồ cột Bar tần suất
                    counts = df[col_name].value_counts().reset_index()
                    fig = px.bar(counts, x=counts.columns[0], y=counts.columns[1], color_discrete_sequence=['#50E3C2'])
                
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Ngắt hàng sau mỗi 2 biểu đồ
            if idx % 2 == 1 and idx < len(viz_features) - 1:
                st.divider()

# ------------------------------------------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN (THÀNH PHẦN 5)
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("🎯 Đánh giá hiệu năng chi tiết của Mô hình")
    
    # Kiểm tra trạng thái đồng bộ kiểm định
    if 'trained_pipeline' not in st.session_state:
        st.info("💡 Vui lòng cấu hình tham số ở Sidebar bên trái và bấm nút **[🚀 Huấn luyện Mô hình]** để xem kết quả phân tích tại đây.")
    else:
        res = st.session_state['metrics']
        
        if res['type'] == 'classification':
            # 1. Hiển thị Metrics vô hướng
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Độ chính xác (Accuracy)", f"{res['accuracy']:.4f}")
            c2.metric("Precision (Weighted)", f"{res['precision']:.4f}")
            c3.metric("Recall (Weighted)", f"{res['recall']:.4f}")
            c4.metric("F1-Score (Weighted)", f"{res['f1']:.4f}")
            
            st.divider()
            
            # 2. Hiển thị Đồ thị Ma trận nhầm lẫn
            cc1, cc2 = st.columns([1, 1])
            with cc1:
                st.write("#### 🧩 Ma trận nhầm lẫn (Confusion Matrix)")
                fig_cm = px.imshow(res['cm'], text_auto=True, color_continuous_scale='Blues',
                                   labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế"))
                st.plotly_chart(fig_cm, use_container_width=True)
            with cc2:
                st.write("#### 📝 Báo cáo phân loại chi tiết (Classification Report)")
                report_df = pd.DataFrame(res['report']).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='Greens', subset=['precision','recall','f1-score']), use_container_width=True)
                
        elif res['type'] == 'regression':
            # 1. Hiển thị Metrics bài toán hồi quy
            r1, r2, r3 = st.columns(3)
            r1.metric("Hệ số xác định R² Score", f"{res['r2']:.4f}")
            r2.metric("Lỗi bình phương trung bình (RMSE)", f"{res['rmse']:.4f}")
            r3.metric("Sai số tuyệt đối trung bình (MAE)", f"{res['mae']:.4f}")
            
            st.divider()
            
            # 2. Biểu đồ Dự đoán vs Thực tế
            st.write("#### 📉 Đồ thị so sánh Giá trị Thực tế vs Dự báo")
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=res['y_test'], y=res['y_pred'], mode='markers', name='Dữ liệu test', marker=dict(color='#9013FE')))
            # Đường chéo lý tưởng y = x
            min_val = min(min(res['y_test']), min(res['y_pred']))
            max_val = max(max(res['y_test']), max(res['y_pred']))
            fig_reg.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Đường lý tưởng 1:1', line=dict(dash='dash', color='red')))
            fig_reg.update_layout(xaxis_title="Giá trị Thực tế", yaxis_title="Giá trị Mô hình dự báo", height=400)
            st.plotly_chart(fig_reg, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: SỬ DỤNG MÔ HÌNH DỰ BÁO (THÀNH PHẦN 6)
# ------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("🔮 Triển khai chấm điểm dữ liệu trực tuyến")
    
    if 'trained_pipeline' not in st.session_state:
        st.info("💡 Vui lòng huấn luyện mô hình thành công trước khi sử dụng tính năng dự báo thực tế.")
    else:
        pipeline = st.session_state['trained_pipeline']
        saved_x_cols = st.session_state['x_columns']
        saved_num_feats = st.session_state['numeric_features']
        saved_cat_feats = st.session_state['categorical_features']
        
        mode = st.radio("Chọn phương thức nhập đầu vào:", options=["Nhập thông số trực tiếp qua Form", "Tập dữ liệu kiểm tra hàng loạt (.csv/.xlsx)"])
        
        # ---------------------------------------------------------
        # CHẾ ĐỘ 1: NHẬP TRỰC TIẾP FORM
        # ---------------------------------------------------------
        if mode == "Nhập thông số trực tiếp qua Form":
            st.write("### 📝 Điền các thông số đầu vào")
            
            with st.form("single_prediction_form"):
                form_cols = st.columns(2)
                input_data = {}
                
                # Tạo tự động các Input Widget tương ứng cấu trúc cột học tập mẫu ban đầu
                for i, col in enumerate(saved_x_cols):
                    col_container = form_cols[i % 2]
                    with col_container:
                        if col in saved_num_feats:
                            # Cột số -> lấy giá trị trung vị, min, max làm gợi ý
                            min_v = float(df[col].min())
                            max_v = float(df[col].max())
                            mean_v = float(df[col].median())
                            input_data[col] = st.number_input(f"Nhập {col}", min_value=min_v, max_value=max_v, value=mean_v, help=f"Kiểu số. Khoảng [{min_v} - {max_v}]")
                        elif col in saved_cat_feats:
                            # Cột phân loại -> selectbox lấy danh mục giá trị thực tế duy nhất
                            unique_options = df[col].dropna().unique().tolist()
                            input_data[col] = st.selectbox(f"Chọn {col}", options=unique_options)
                
                submit_pred = st.form_submit_button("💥 Thực hiện Dự báo", type="primary", use_container_width=True)
                
            if submit_pred:
                # Chuyển đổi bản ghi nhập vào thành DataFrame đúng cấu trúc
                single_df = pd.DataFrame([input_data])
                
                # Thực thi Pipeline (Bao gồm khâu tiền xử lý tự động trong session state)
                prediction = pipeline.predict(single_df)[0]
                
                st.success("### 🎉 Kết quả dự báo của hệ thống AI:")
                if hasattr(pipeline['model'], "predict_proba") and 'classification' in st.session_state['metrics']['type']:
                    prob = pipeline.predict_proba(single_df)[0]
                    class_idx = list(pipeline['model'].classes_).index(prediction)
                    st.metric(label=f"Nhãn lớp dự báo ({y_column})", value=str(prediction))
                    st.info(f"Độ tin cậy xác suất tương ứng: **{prob[class_idx]*100:.2f}%**")
                else:
                    st.metric(label=f"Giá trị dự báo đầu ra ({y_column})", value=f"{prediction:,.4f}" if isinstance(prediction, (int, float)) else str(prediction))

        # ---------------------------------------------------------
        # CHẾ ĐỘ 2: TẢI BẢNG FILE KIỂM TRA HÀNG LOẠT
        # ---------------------------------------------------------
        else:
            st.write("### 📁 Tải tệp chứa danh sách dữ liệu mới cần dự báo")
            st.caption("⚠️ Yêu cầu: File tải lên bắt buộc phải chứa đầy đủ các cột thuộc tính đầu vào sau:")
            st.code(", ".join(saved_x_cols))
            
            batch_file = st.file_uploader("Chọn file dữ liệu kiểm tra mới", type=["csv", "xlsx"], key="batch_uploader")
            
            if batch_file is not None:
                # Đọc tệp dữ liệu test mới độc lập
                if batch_file.name.endswith('.csv'):
                    batch_df = pd.read_csv(batch_file)
                else:
                    batch_df = pd.read_excel(batch_file)
                    
                # Kiểm tra Schema khớp với tập Train ban đầu hay không
                missing_cols = [col for col in saved_x_cols if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"❌ Cấu trúc File không khớp! Tệp của bạn đang thiếu các cột bắt buộc sau: {missing_cols}")
                else:
                    with st.spinner("Đang xử lý dự báo hàng loạt..."):
                        # Trích xuất đúng tập tính năng
                        X_batch = batch_df[saved_x_cols]
                        
                        # Chạy mô hình
                        batch_preds = pipeline.predict(X_batch)
                        
                        # Chèn cột kết quả mới vào bảng kết quả hiển thị cho người dùng tải về
                        result_df = batch_df.copy()
                        result_df[f'AI_Predicted_{y_column}'] = batch_preds
                        
                        st.success("✅ Đã xử lý xong toàn bộ dữ liệu mẫu!")
                        st.write("#### 📋 Xem trước bảng dữ liệu kết quả:")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Chuyển đổi DataFrame thành CSV chuỗi bytes phục vụ tải xuống nhanh
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
                        
                        st.download_button(
                            label="📥 Tải xuống tệp kết quả dự báo toàn bộ (.CSV)",
                            data=csv_bytes,
                            file_name=f"AI_Predictions_Result_{batch_file.name.split('.')[0]}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
