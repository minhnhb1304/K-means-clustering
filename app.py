import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from utils import BookClusteringAI

# ===========================
# CẤU HÌNH TRANG WEB
# ===========================
st.set_page_config(
    page_title="AI Phân Cụm & Dự Báo Sách",
    page_icon="📚",
    layout="wide"
)

# ===========================
# TIÊU ĐỀ CHÍNH
# ===========================
st.title("📚 AI Phân Cụm & Dự Báo Sách")
st.markdown("---")

# Khởi tạo AI instance (sử dụng session state để giữ nguyên qua các lần tương tác)
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = BookClusteringAI()

ai = st.session_state.ai_model

# ===========================
# SIDEBAR - PANEL CẤU HÌNH
# ===========================
st.sidebar.header("⚙️ Cấu Hình Phân Tích")

# 1. Upload file CSV
st.sidebar.subheader("1. Tải dữ liệu")
uploaded_file = st.sidebar.file_uploader(
    "Chọn file CSV",
    type=['csv'],
    help="Tải lên file dữ liệu sách của bạn"
)

# 2. Chọn số cụm K
st.sidebar.subheader("2. Số Cụm (K)")
k_clusters = st.sidebar.slider(
    "Chọn số cụm K:",
    min_value=2,
    max_value=10,
    value=4,
    help="Số lượng cụm bạn muốn phân chia dữ liệu (mặc định: 4)"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Gợi ý:** Sử dụng Elbow Method ở Tab Dashboard để xác định K tối ưu!")

# ===========================
# TABS - TAB 1: DASHBOARD, TAB 2: PREDICTION
# ===========================
tab1, tab2 = st.tabs(["📊 Dashboard Phân Tích", "🧠 AI Dự Báo & Tư Vấn"])

# ===========================
# TAB 1: DASHBOARD PHÂN TÍCH (TRAINING)
# ===========================
with tab1:
    if uploaded_file is not None:
        try:
            # Tải dữ liệu
            df = ai.load_data(uploaded_file)
            st.sidebar.success("✅ Tải file thành công!")
            
            # Kiểm tra các cột cần thiết
            required_cols = ['quantity', 'n_review', 'avg_rating']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Thiếu các cột: {', '.join(missing_cols)}")
                st.info("File CSV cần có các cột: quantity, n_review, avg_rating")
            else:
                # ===========================
                # SECTION 1: XEM TRƯỚC DỮ LIỆU
                # ===========================
                st.header("1️⃣ Xem Trước Dữ Liệu")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("5 Dòng Đầu Tiên")
                    st.dataframe(df.head(), use_container_width=True)
                
                with col2:
                    st.subheader("Thông Tin Tổng Quan")
                    st.write(f"**Tổng số dòng:** {df.shape[0]}")
                    st.write(f"**Tổng số cột:** {df.shape[1]}")
                    st.write(f"**Số giá trị thiếu:** {df.isnull().sum().sum()}")
                
                st.subheader("Thống Kê Mô Tả")
                st.dataframe(df.describe(), use_container_width=True)
                
                st.markdown("---")
                
                # ===========================
                # SECTION 2: TIỀN XỬ LÝ DỮ LIỆU
                # ===========================
                st.header("2️⃣ Tiền Xử Lý Dữ Liệu")
                
                # Tiền xử lý
                df_processed, X_scaled, scaler = ai.preprocess_data(df)
                
                rows_removed = df.shape[0] - df_processed.shape[0]
                st.success(f"✅ Đã loại bỏ {rows_removed} dòng có giá trị thiếu")
                st.write(f"**Số dòng còn lại:** {df_processed.shape[0]}")
                
                st.subheader("🔄 Chuẩn Hóa Dữ Liệu (StandardScaler)")
                st.info("ℹ️ **StandardScaler** chuyển đổi dữ liệu về mean=0 và std=1, giúp các thuật toán ML hoạt động tốt hơn")
                
                st.markdown("---")
                
                # ===========================
                # SECTION 3: ELBOW METHOD
                # ===========================
                st.header("3️⃣ Phương Pháp Elbow - Xác Định K Tối Ưu")
                
                st.write("**Elbow Method** giúp xác định số cụm tối ưu bằng cách tính toán Inertia (tổng khoảng cách bình phương)")
                
                # Tính toán phương pháp Elbow (đã cache)
                with st.spinner("🔄 Đang tính toán Inertia cho các giá trị K..."):
                    K_range, inertia_values, silhouette_scores = ai.calculate_elbow_method(X_scaled, k_range=(1, 11))
                
                # Vẽ biểu đồ Elbow
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(8, 5))
                    ax1.plot(K_range, inertia_values, 'bo-', linewidth=2, markersize=8)
                    ax1.set_xlabel('Số cụm K', fontsize=12)
                    ax1.set_ylabel('Inertia', fontsize=12)
                    ax1.set_title('Elbow Method - Xác định K tối ưu', fontsize=14, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    ax1.axvline(x=k_clusters, color='r', linestyle='--', label=f'K đã chọn = {k_clusters}')
                    ax1.legend()
                    st.pyplot(fig1)
                
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    ax2.plot(range(2, 11), silhouette_scores[1:], 'go-', linewidth=2, markersize=8)
                    ax2.set_xlabel('Số cụm K', fontsize=12)
                    ax2.set_ylabel('Silhouette Score', fontsize=12)
                    ax2.set_title('Silhouette Score - Đánh giá chất lượng phân cụm', fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    ax2.axvline(x=k_clusters, color='r', linestyle='--', label=f'K đã chọn = {k_clusters}')
                    ax2.legend()
                    st.pyplot(fig2)
                
                st.info("💡 **Cách đọc:** Điểm 'khuỷu tay' (elbow) trên đồ thị là K tối ưu. Silhouette Score cao hơn (càng gần 1) thì phân cụm tốt hơn.")
                
                st.markdown("---")
                
                # ===========================
                # SECTION 4: PHÂN CỤM VÀ TRỰC QUAN HÓA
                # ===========================
                st.header("4️⃣ Kết Quả Phân Cụm & Trực Quan Hóa")
                
                # Nút huấn luyện mô hình
                if st.button("🚀 Huấn Luyện Mô Hình", type="primary", use_container_width=True):
                    with st.spinner(f"🔄 Đang huấn luyện mô hình với K={k_clusters}..."):
                        kmeans_final, cluster_labels, df_with_clusters = ai.train_model(X_scaled, n_clusters=k_clusters)
                        st.session_state.df_with_clusters = df_with_clusters
                        st.session_state.scaler = scaler
                        st.success("✅ Mô hình đã được huấn luyện và lưu thành công!")
                
                # Hiển thị kết quả nếu đã huấn luyện
                if 'df_with_clusters' in st.session_state:
                    df_with_clusters = st.session_state.df_with_clusters
                    
                    # Thêm cột nhãn động vào dataframe để hiển thị
                    df_with_clusters['Nhãn Cụm'] = df_with_clusters['Cluster'].apply(
                        lambda x: ai.get_cluster_label_name(x)
                    )
                    
                    # Hiển thị thông tin về các cụm
                    st.subheader("Phân Bố Các Cụm")
                    cluster_counts = df_with_clusters['Cluster'].value_counts().sort_index()
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Tạo bảng với nhãn động
                        cluster_info = pd.DataFrame({
                            'Cụm': cluster_counts.index,
                            'Nhãn': [ai.get_cluster_label_name(cid) for cid in cluster_counts.index],
                            'Số lượng': cluster_counts.values,
                            'Phần trăm': [f"{(v/len(df_with_clusters)*100):.1f}%" for v in cluster_counts.values]
                        })
                        st.dataframe(cluster_info, use_container_width=True, hide_index=True)
                        
                        # Tính Silhouette Score
                        silhouette_avg = silhouette_score(X_scaled, df_with_clusters['Cluster'].astype(int))
                        st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                    
                    with col2:
                        # Biểu đồ phân bố cụm với nhãn động
                        fig_bar = px.bar(
                            x=cluster_info['Nhãn'],
                            y=cluster_info['Số lượng'],
                            labels={'x': 'Nhãn Cụm', 'y': 'Số lượng'},
                            title='Phân Bố Số Lượng Theo Nhãn Cụm',
                            color=cluster_info['Nhãn'],
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Trực quan hóa 2D với nhãn động
                    st.subheader("Trực Quan Hóa 2D - Scatter Plot")
                    
                    # Cho phép người dùng chọn trục X và Y
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_axis = st.selectbox(
                            "Chọn trục X:",
                            options=ai.features,
                            index=0
                        )
                    
                    with col2:
                        y_axis = st.selectbox(
                            "Chọn trục Y:",
                            options=ai.features,
                            index=1
                        )
                    
                    # Vẽ scatter plot với Plotly - sử dụng nhãn động
                    fig_scatter = px.scatter(
                        df_with_clusters,
                        x=x_axis,
                        y=y_axis,
                        color='Nhãn Cụm',  # Sử dụng nhãn động thay vì Cluster ID
                        title=f'Phân Cụm K-Means với Nhãn Động (K={k_clusters})',
                        hover_data=['Cluster', 'category'] if 'category' in df_with_clusters.columns else ['Cluster'],
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        width=900,
                        height=600
                    )
                    
                    # Thêm tâm các cụm
                    centers = st.session_state.scaler.inverse_transform(ai.model.cluster_centers_)
                    centers_df = pd.DataFrame(centers, columns=ai.features)
                    
                    fig_scatter.add_scatter(
                        x=centers_df[x_axis],
                        y=centers_df[y_axis],
                        mode='markers',
                        marker=dict(
                            size=20,
                            color='red',
                            symbol='x',
                            line=dict(width=2, color='black')
                        ),
                        name='Tâm cụm',
                        showlegend=True
                    )
                    
                    fig_scatter.update_layout(
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        font=dict(size=12),
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # ===========================
                    # THỐNG KÊ THEO CỤM
                    # ===========================
                    st.subheader("Thống Kê Chi Tiết Theo Cụm")
                    
                    # Tính toán thống kê theo cụm với nhãn
                    cluster_stats = ai.get_cluster_statistics(df_with_clusters)
                    st.dataframe(cluster_stats, use_container_width=True)
                    
                    # Tải xuống kết quả
                    st.markdown("---")
                    st.subheader("Tải Kết Quả")
                    csv_data = df_with_clusters.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Tải xuống dữ liệu đã phân cụm (CSV)",
                        data=csv_data,
                        file_name=f"clustered_data_k{k_clusters}.csv",
                        mime="text/csv",
                        help="Tải về file CSV chứa dữ liệu gốc kèm theo nhãn cụm"
                    )
                else:
                    st.info("👆 Nhấn nút 'Huấn Luyện Mô Hình' để bắt đầu phân cụm!")
        
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý file: {str(e)}")
            st.info("Vui lòng kiểm tra lại định dạng file CSV của bạn!")
    
    else:
        st.info("👈 Vui lòng tải lên file CSV từ sidebar để bắt đầu phân tích!")

# ===========================
# TAB 2: AI DỰ BÁO & TƯ VẤN (PREDICTION)
# ===========================
with tab2:
    st.header("Dự Báo & Tư Vấn Cho Sách Mới")
    st.write("Nhập thông tin cuốn sách mới để AI dự báo cụm và đưa ra lời khuyên kinh doanh.")
    
    # Kiểm tra xem mô hình đã được huấn luyện chưa
    if ai.model is None:
        # Thử tải mô hình đã lưu
        if not ai.load_saved_model():
            st.warning("⚠️ **Chưa có mô hình được huấn luyện!**")
            st.info("""
            **Hướng dẫn:**
            1. Chuyển sang Tab **"📊 Dashboard Phân Tích"**
            2. Tải lên file CSV dữ liệu sách
            3. Nhấn nút **"🚀 Huấn Luyện Mô Hình"**
            4. Quay lại Tab này để sử dụng tính năng dự báo
            """)
        else:
            st.success("✅ Đã tải mô hình đã lưu trước đó!")
    
    # Form nhập liệu
    st.subheader("Nhập Thông Tin Sách Mới")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        quantity = st.number_input(
            "Số lượng bán (Quantity):",
            min_value=0.0,
            value=100.0,
            step=1.0,
            help="Số lượng sách đã bán được"
        )
    
    with col2:
        n_review = st.number_input(
            "Số lượng đánh giá (n_review):",
            min_value=0.0,
            value=500.0,
            step=1.0,
            help="Tổng số lượt đánh giá"
        )
    
    with col3:
        rating = st.number_input(
            "Điểm đánh giá (avg_rating):",
            min_value=0.0,
            max_value=5.0,
            value=4.0,
            step=0.1,
            help="Điểm đánh giá trung bình (0-5)"
        )
    
    # Nút dự báo
    if st.button("Dự Báo & Tư Vấn", type="primary", use_container_width=True):
        with st.spinner("🔄 Đang phân tích và dự báo..."):
            result = ai.predict_new_book(quantity, n_review, rating)
            
            if 'error' in result:
                st.error(result['error'])
            else:
                st.success("✅ Dự báo hoàn tất!")
                
                # Hiển thị kết quả trong 2 cột
                col1, col2 = st.columns(2)
                
                # Cột A: Chiến lược Nhập hàng (Cho Manager)
                with col1:
                    st.markdown("### Chiến Lược Nhập Hàng (Cho Quản Lý)")
                    st.markdown(f"""
                    <div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3;'>
                        <h4 style='color: #1565C0; margin-top: 0;'>Cụm Dự Báo: <b>{result['cluster_label']}</b></h4>
                        <p style='font-size: 16px; color: #1F1F1F; line-height: 1.6;'>
                            <b>Lời Khuyên:</b><br>
                            {result['manager_advice']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Cột B: Gợi ý Marketing (Cho Marketing)
                with col2:
                    st.markdown("### 💡 Gợi ý Marketing")
                    st.markdown(f"""
                    <div style='background-color: #F3E5F5; padding: 20px; border-radius: 10px; border-left: 5px solid #9C27B0;'>
                        <h4 style='color: #6A1B9A; margin-top: 0;'>Chiến Lược Marketing & Bán Hàng</h4>
                        <p style='font-size: 16px; color: #1F1F1F; line-height: 1.6;'>
                            <b>Hành Động Marketing:</b><br>
                            {result['marketing_action']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Thông tin chi tiết
                st.markdown("---")
                st.subheader("Thông Tin Chi Tiết")
                
                # info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                info_col1, info_col2, info_col3, info_col4 = st.columns([1, 3, 1, 1])
                
                with info_col1:
                    st.metric("ID Cụm", result['cluster_id'])
                
                with info_col2:
                    st.metric("Nhãn Cụm", result['cluster_label'])
                
                with info_col3:
                    st.metric("Số Lượng Bán", f"{quantity:.0f}")
                
                with info_col4:
                    st.metric("Rating", f"{rating:.1f}")

# Footer
st.markdown("---")
# st.markdown("""
# <div style='text-align: center; color: #1F1F1F;'>
#     <p>Developed with ❤️ | Phân Cụm & Dự Đoán Xu Hướng Bán Sách</p>
# </div>
# """, unsafe_allow_html=True)
