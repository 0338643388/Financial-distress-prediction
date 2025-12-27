import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os 

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dự báo Kiệt quệ Tài chính Doanh nghiệp",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. CSS & GIAO DIỆN
# ---------------------------------------------------------
st.markdown("""
<style>
    /* 1. Màu tiêu đề chính (Tự thích nghi) */
    .main-header {
        font-size: 28px; 
        font-weight: 700; 
        color: var(--text-color); /* Tự đổi Đen/Trắng theo Theme */
    }

    /* 2. Thẻ Card thông minh */
    .card {
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px;
        /* Dùng màu nền phụ của Streamlit (Xám nhạt ở Light, Xám đậm ở Dark) */
        background-color: var(--secondary-background-color); 
        /* Bắt buộc màu chữ lấy theo màu hệ thống */
        color: var(--text-color);
        /* Thêm viền mờ để nổi bật card */
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* 3. Màu chữ cảnh báo (Giữ nguyên vì Đỏ/Xanh nổi trên cả 2 nền) */
    .risk-high {color: #DC2626; font-weight: bold; font-size: 24px;}
    .risk-low {color: #059669; font-weight: bold; font-size: 24px;}
    
    /* Đảm bảo các đoạn văn trong card cũng nhận màu hệ thống */
    .card p, .card b {
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. HÀM LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_prediction_model(model_path):
    # Kiểm tra sự tồn tại của file trước
    if not os.path.exists(model_path):
        st.error(f"LỖI: File '{model_path}' không tồn tại trong thư mục hiện tại: {os.getcwd()}")
        return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        # Hiển thị lỗi chi tiết nếu file có tồn tại nhưng load thất bại
        st.error(f"Lỗi khi giải nén mô hình: {e}")
        return None

MODEL_PATH = "best_rf_model.pkl" 
model = load_prediction_model(MODEL_PATH)

if model is None:
    st.warning(f"⚠️ Chưa tìm thấy file mô hình `{MODEL_PATH}`. Ứng dụng đang chạy ở chế độ DEMO giao diện.")

# ---------------------------------------------------------
# 4. SIDEBAR - NHẬP LIỆU
# ---------------------------------------------------------
st.sidebar.header("📝 Nhập liệu Báo cáo Tài chính")
st.sidebar.markdown("Nhập các giá trị thô, hệ thống sẽ tự tính toán các tỷ số.")

with st.sidebar.form("financial_input_form"):
    
    # --- Nhóm 1: Vĩ mô & Tăng trưởng ---
    st.markdown("### 1. Chỉ số Vĩ mô & Tăng trưởng")
    col1, col2 = st.columns(2)
    with col1:
        inflation_rate = st.number_input("Lạm phát (Inflation Rate %)", value=6.5, step=0.1)
        gta = st.number_input("Tăng trưởng Tài sản (GTA %)", value=2.0, step=0.1)
    with col2:
        gdp_rate = st.number_input("Tăng trưởng GDP (GDP Rate %)", value=3.0, step=0.1)
        gnr = st.number_input("Tăng trưởng Doanh thu (GNR %)", value=5.0, step=0.1)

    # --- Nhóm 2: Kết quả Kinh doanh ---
    st.markdown("### 2. Kết quả Kinh doanh (VNĐ)")
    revenue = st.number_input("Doanh thu thuần", value=100000.0, step=1000.0)
    ebit = st.number_input("EBIT (Lợi nhuận trước lãi & thuế)", value=-15000.0, step=500.0)
    net_income = st.number_input("Lợi nhuận sau thuế (Net Income)", value=-10000.0, step=500.0)
    interest_expense = st.number_input("Chi phí lãi vay", value=2000.0, min_value=1.0, step=100.0)

    # --- Nhóm 3: Bảng Cân đối Kế toán ---
    st.markdown("### 3. Tài sản & Nguồn vốn (VNĐ)")
    total_assets = st.number_input("Tổng Tài sản (Total Assets)", value=200000.0, min_value=1.0, step=1000.0)
    current_assets = st.number_input("Tài sản Ngắn hạn", value=80000.0, step=1000.0)
    inventory = st.number_input("Hàng tồn kho", value=30000.0, step=1000.0)
    cash = st.number_input("Tiền & Tương đương tiền", value=10000.0, step=1000.0)
    
    total_liabilities = st.number_input("Tổng Nợ phải trả (Total Liabilities)", value=100000.0, min_value=1.0, step=1000.0)
    current_liabilities = st.number_input("Nợ Ngắn hạn", value=60000.0, min_value=1.0, step=1000.0)

    submitted = st.form_submit_button("🚀 Dự báo Ngay")

# ---------------------------------------------------------
# 5. XỬ LÝ TÍNH TOÁN (FEATURE ENGINEERING)
# ---------------------------------------------------------
if submitted:
    # 1. Tính toán các tỷ số (Ratios) dựa trên công thức
    
    # An toàn: Tránh chia cho 0
    safe_ta = total_assets if total_assets != 0 else 1
    safe_rev = revenue if revenue != 0 else 1
    safe_tl = total_liabilities if total_liabilities != 0 else 1
    safe_cl = current_liabilities if current_liabilities != 0 else 1
    safe_int = interest_expense if interest_expense != 0 else 1

    features = {
        'Inflation_rate': inflation_rate/100,
        'GDP rate': gdp_rate/100,
        
        # Tỷ suất sinh lời
        'EBITTA': ebit / safe_ta,
        'ROA': net_income / safe_ta,
        'TAT': revenue / safe_ta,
        
        # Tỷ lệ thanh khoản
        'CLTA': current_liabilities / safe_ta,
        'CATL': current_assets / safe_tl,
        'CLTS': current_liabilities / safe_rev,
        'CLTL': current_liabilities / safe_tl,
        'ITA': inventory / safe_ta,
        
        # Đòn bẩy
        'DA': total_liabilities / safe_ta,
        'CCL': cash / safe_cl,
        
        # Tăng trưởng 
        'GTA': gta/100,
        'GNR': gnr/100
    }

    # Tạo DataFrame đúng thứ tự cột mà mô hình yêu cầu 
    input_df = pd.DataFrame([features])
    
    # Thứ tự cột chuẩn mà mô hình đã huấn luyện
    expected_cols = [
        'Inflation_rate', 'GDP rate', 'EBITTA', 'ROA', 'TAT', 'CLTA', 'CATL',
       'CLTS', 'CLTL', 'ITA', 'DA', 'CCL', 'GTA', 'GNR'
    ]
    input_df = input_df[expected_cols]

    # ---------------------------------------------------------
    # 6. HIỂN THỊ KẾT QUẢ
    # ---------------------------------------------------------
    col_left, col_right = st.columns([2, 1.2])

    with col_left:
        st.markdown("### 📊 Kết quả Dự báo")
        
        prediction = None
        proba = None
        
        if model:
            try:
                prediction = model.predict(input_df)[0]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)[0][1] # Xác suất lớp 1 (Kiệt quệ)
            except Exception as e:
                st.error(f"Lỗi khi dự báo: {str(e)}")
        else:
            # Mockup cho demo nếu không có model
            prediction = 1 if (features['EBITTA'] < 0.05 or features['DA'] > 0.8) else 0
            proba = 0.85 if prediction == 1 else 0.15

        # Hiển thị trạng thái
        if prediction == 1:
            st.markdown(f"""
            <div class='card' style='border-left: 5px solid #DC2626;'>
                <h2 class='risk-high'>⚠️ CẢNH BÁO: NGUY CƠ KIỆT QUỆ TÀI CHÍNH</h2>
                <p>Mô hình dự báo doanh nghiệp có rủi ro cao rơi vào tình trạng kiệt quệ tài chính.</p>
                <p>Xác suất rủi ro: <b>{proba*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(proba))
        else:
            st.markdown(f"""
            <div class='card' style='border-left: 5px solid #059669;'>
                <h2 class='risk-low'>✅ TÌNH TRẠNG: AN TOÀN / ỔN ĐỊNH</h2>
                <p>Mô hình dự báo doanh nghiệp đang ở trạng thái tài chính bình thường.</p>
                <p>Xác suất rủi ro: <b>{proba*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(proba))

        # Hiển thị các chỉ số tính toán được
        # st.markdown("#### 🔍 Các chỉ số tính toán từ dữ liệu đầu vào")
        # metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        # metrics_col1.metric("EBITTA (Hiệu quả HĐ)", f"{features['EBITTA']:.4f}")
        # metrics_col2.metric("ROA (Sinh lời TS)", f"{features['ROA']:.4f}")
        # metrics_col3.metric("DA (Đòn bẩy Nợ)", f"{features['DA']:.2f}")
        
        # metrics_col4, metrics_col5, metrics_col6 = st.columns(3)
        # metrics_col4.metric("TAT (Vòng quay TS)", f"{features['TAT']:.2f}")
        # metrics_col5.metric("CATL (Thanh khoản)", f"{features['CATL']:.2f}")
        
        with st.expander("Xem bảng dữ liệu chi tiết đầu vào mô hình"):
            st.dataframe(input_df)

    with col_right:
        st.markdown("### 💡 Phân tích & Khuyến nghị")
        st.markdown("<div class='block-left'>", unsafe_allow_html=True)
        
        recommendations = []
        
        # 1. EBITTA & ROA
        if features['EBITTA'] < 0.1 or features['ROA'] < 0.05:
            st.markdown("🔴 **Hiệu quả hoạt động thấp:**")
            st.write("- EBITTA/ROA thấp là nguyên nhân hàng đầu gây rủi ro. Cần rà soát chi phí vận hành (COGS) và biên lợi nhuận.")
        else:
            st.markdown("🟢 **Hiệu quả hoạt động tốt:** EBITTA và ROA ở mức an toàn.")

        # 2. Đòn bẩy (DA)
        if features['DA'] > 0.7:
             st.markdown("🔴 **Cấu trúc vốn rủi ro:**")
             st.write(f"- Tỷ lệ Nợ/Tài sản (DA) là {features['DA']:.2f} (cao). Cần cân nhắc giảm bớt nợ vay để giảm áp lực lãi suất.")

        # 3. Thanh khoản (CATL)
        if features['CATL'] < 1.0:
            st.markdown("🟠 **Thanh khoản hạn chế:**")
            st.write("- Tài sản ngắn hạn thấp hơn tổng nợ. Cần cải thiện dòng tiền lưu động.")
        elif features['CATL'] > 2.5:
            st.markdown("🟠 **Lưu ý về Tài sản ngắn hạn:**")
            st.write("- CATL khá cao. Hãy kiểm tra xem tài sản ngắn hạn có phải chủ yếu là hàng tồn kho khó bán hay không?")

        # 4. Hiệu suất (TAT)
        if features['TAT'] < 0.8:
            st.markdown("🟠 **Vòng quay tài sản thấp:**")
            st.write("- Doanh nghiệp chưa sử dụng tài sản hiệu quả để tạo doanh thu.")

        if prediction == 1:
            st.info("📌 **Hành động đề xuất:** Tập trung tối ưu hóa chi phí để tăng EBIT, và xem xét lại cấu trúc nợ ngắn hạn ngay lập tức.")
        
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👈 Vui lòng nhập dữ liệu tài chính ở thanh bên trái và nhấn **Dự báo Ngay**.")




