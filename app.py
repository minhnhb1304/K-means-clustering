import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ===========================
# CẤU HÌNH TRANG WEB
# ===========================
st.set_page_config(
    page_title="K-Means Clustering App",
    page_icon="📊",
    layout="wide"
)

# ===========================
# TIÊU ĐỀ CHÍNH
# ===========================
st.title("📊 K-Means Clustering - Phân Tích Dữ Liệu Sách")
st.markdown("---")

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

# Kiểm tra xem file đã được upload chưa
if uploaded_file is not None:
    # Đọc dữ liệu từ file CSV
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Tải file thành công!")
        
        # 2. Chọn các đặc trưng số để phân cụm
        st.sidebar.subheader("2. Chọn Đặc Trưng")
        
        # Tự động phát hiện các cột số trong dataset
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Đặt mặc định là quantity, n_review, avg_rating (nếu có)
        default_features = [col for col in ['quantity', 'n_review', 'avg_rating'] 
                           if col in numeric_columns]
        
        selected_features = st.sidebar.multiselect(
            "Chọn các cột số để clustering:",
            options=numeric_columns,
            default=default_features if default_features else numeric_columns[:3],
            help="Chọn ít nhất 2 đặc trưng để phân cụm"
        )
        
        # 3. Chọn số cụm K
        st.sidebar.subheader("3. Số Cụm (K)")
        k_clusters = st.sidebar.slider(
            "Chọn số cụm K:",
            min_value=2,
            max_value=10,
            value=3,
            help="Số lượng cụm bạn muốn phân chia dữ liệu"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info("💡 **Gợi ý:** Sử dụng Elbow Method ở phần 3 để xác định K tối ưu!")
        
        # ===========================
        # MAIN CONTENT - NỘI DUNG CHÍNH
        # ===========================
        
        # Kiểm tra xem người dùng đã chọn đủ features chưa
        if len(selected_features) < 2:
            st.warning("⚠️ Vui lòng chọn ít nhất 2 đặc trưng số để thực hiện phân cụm!")
        else:
            # ===========================
            # SECTION 1: XEM TRƯỚC DỮ LIỆU
            # ===========================
            st.header("1️⃣ Xem Trước Dữ Liệu")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📋 5 Dòng Đầu Tiên")
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.subheader("📊 Thông Tin Tổng Quan")
                st.write(f"**Tổng số dòng:** {df.shape[0]}")
                st.write(f"**Tổng số cột:** {df.shape[1]}")
                st.write(f"**Số giá trị thiếu:** {df.isnull().sum().sum()}")
            
            st.subheader("📈 Thống Kê Mô Tả")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.markdown("---")
            
            # ===========================
            # SECTION 2: TIỀN XỬ LÝ DỮ LIỆU
            # ===========================
            st.header("2️⃣ Tiền Xử Lý Dữ Liệu")
            
            # Tạo bản copy để xử lý
            df_processed = df.copy()
            
            # Hiển thị số lượng giá trị thiếu trước khi xử lý
            missing_before = df_processed[selected_features].isnull().sum().sum()
            st.write(f"**Số giá trị thiếu trong các cột đã chọn:** {missing_before}")
            
            # Xử lý giá trị thiếu - loại bỏ các dòng có giá trị thiếu
            df_processed = df_processed.dropna(subset=selected_features)
            missing_after = df_processed[selected_features].isnull().sum().sum()
            
            st.success(f"✅ Đã loại bỏ {df.shape[0] - df_processed.shape[0]} dòng có giá trị thiếu")
            st.write(f"**Số dòng còn lại:** {df_processed.shape[0]}")
            
            # Chuẩn hóa dữ liệu bằng StandardScaler
            st.subheader("🔄 Chuẩn Hóa Dữ Liệu (StandardScaler)")
            
            # Tạo scaler và fit với dữ liệu
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_processed[selected_features])
            
            # Tạo DataFrame cho dữ liệu đã chuẩn hóa
            df_scaled = pd.DataFrame(
                X_scaled,
                columns=[f"{col}_scaled" for col in selected_features],
                index=df_processed.index
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dữ liệu gốc (5 dòng đầu):**")
                st.dataframe(df_processed[selected_features].head(), use_container_width=True)
            
            with col2:
                st.write("**Dữ liệu đã chuẩn hóa (5 dòng đầu):**")
                st.dataframe(df_scaled.head(), use_container_width=True)
            
            st.info("ℹ️ **StandardScaler** chuyển đổi dữ liệu về mean=0 và std=1, giúp các thuật toán ML hoạt động tốt hơn")
            
            st.markdown("---")
            
            # ===========================
            # SECTION 3: ELBOW METHOD
            # ===========================
            st.header("3️⃣ Phương Pháp Elbow - Xác Định K Tối Ưu")
            
            st.write("**Elbow Method** giúp xác định số cụm tối ưu bằng cách tính toán Inertia (tổng khoảng cách bình phương)")
            
            # Tính toán Inertia cho K từ 1 đến 10
            inertia_values = []
            silhouette_scores = []
            K_range = range(1, 11)
            
            with st.spinner("🔄 Đang tính toán Inertia cho các giá trị K..."):
                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    inertia_values.append(kmeans.inertia_)
                    
                    # Tính Silhouette Score (chỉ cho k >= 2)
                    if k >= 2:
                        score = silhouette_score(X_scaled, kmeans.labels_)
                        silhouette_scores.append(score)
                    else:
                        silhouette_scores.append(0)
            
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
            
            # Chạy KMeans với K đã chọn
            with st.spinner(f"🔄 Đang thực hiện phân cụm với K={k_clusters}..."):
                kmeans_final = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans_final.fit_predict(X_scaled)
            
            # Thêm nhãn cụm vào DataFrame
            df_processed['Cluster'] = cluster_labels
            df_processed['Cluster'] = df_processed['Cluster'].astype(str)
            
            # Hiển thị thông tin về các cụm
            st.subheader("📊 Phân Bố Các Cụm")
            cluster_counts = df_processed['Cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(
                    pd.DataFrame({
                        'Cụm': cluster_counts.index,
                        'Số lượng': cluster_counts.values,
                        'Phần trăm': [f"{(v/len(df_processed)*100):.1f}%" for v in cluster_counts.values]
                    }),
                    use_container_width=True
                )
                
                # Tính Silhouette Score
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
            
            with col2:
                # Biểu đồ phân bố cụm
                fig_bar = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={'x': 'Cụm', 'y': 'Số lượng'},
                    title='Phân Bố Số Lượng Theo Cụm',
                    color=cluster_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("---")
            
            # Trực quan hóa 2D
            st.subheader("📈 Trực Quan Hóa 2D - Scatter Plot")
            
            # Cho phép người dùng chọn trục X và Y
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox(
                    "Chọn trục X:",
                    options=selected_features,
                    index=0 if len(selected_features) > 0 else 0
                )
            
            with col2:
                y_axis = st.selectbox(
                    "Chọn trục Y:",
                    options=selected_features,
                    index=1 if len(selected_features) > 1 else 0
                )
            
            # Tạo hover data nếu có cột category
            hover_data_cols = []
            if 'category' in df_processed.columns:
                hover_data_cols.append('category')
            hover_data_cols.extend([col for col in selected_features if col not in [x_axis, y_axis]])
            
            # Vẽ scatter plot với Plotly
            fig_scatter = px.scatter(
                df_processed,
                x=x_axis,
                y=y_axis,
                color='Cluster',
                title=f'Phân Cụm K-Means (K={k_clusters})',
                hover_data=hover_data_cols,
                color_discrete_sequence=px.colors.qualitative.Set2,
                width=900,
                height=600
            )
            
            # Thêm tâm các cụm
            centers = scaler.inverse_transform(kmeans_final.cluster_centers_)
            centers_df = pd.DataFrame(centers, columns=selected_features)
            
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
            st.subheader("📋 Thống Kê Chi Tiết Theo Cụm")
            
            # Tính toán thống kê trung bình cho mỗi cụm
            cluster_stats = df_processed.groupby('Cluster')[selected_features].mean()
            cluster_stats = cluster_stats.round(2)
            
            st.dataframe(cluster_stats, use_container_width=True)
            
            # ===========================
            # TẢI DỮ LIỆU KẾT QUẢ
            # ===========================
            st.markdown("---")
            st.subheader("💾 Tải Kết Quả")
            
            # Chuẩn bị dữ liệu để tải xuống
            result_df = df_processed.copy()
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Tải xuống dữ liệu đã phân cụm (CSV)",
                data=csv_data,
                file_name=f"clustered_data_k{k_clusters}.csv",
                mime="text/csv",
                help="Tải về file CSV chứa dữ liệu gốc kèm theo nhãn cụm"
            )
            
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc file: {str(e)}")
        st.info("Vui lòng kiểm tra lại định dạng file CSV của bạn!")

else:
    # Hiển thị hướng dẫn khi chưa upload file
    st.info("👈 Vui lòng tải lên file CSV từ sidebar để bắt đầu phân tích!")
    
    st.markdown("""
    ## 📖 Hướng Dẫn Sử Dụng
    
    ### Bước 1: Tải dữ liệu
    - Click vào nút **"Browse files"** ở sidebar
    - Chọn file CSV chứa dữ liệu sách của bạn
    
    ### Bước 2: Chọn đặc trưng
    - Chọn các cột số bạn muốn sử dụng để phân cụm
    - Khuyến nghị: `quantity`, `n_review`, `avg_rating`
    
    ### Bước 3: Xác định số cụm
    - Sử dụng **Elbow Method** để tìm K tối ưu
    - Điều chỉnh slider để chọn số cụm K
    
    ### Bước 4: Phân tích kết quả
    - Xem trực quan hóa 2D của các cụm
    - Phân tích thống kê chi tiết theo từng cụm
    - Tải xuống kết quả để sử dụng sau
    
    ---
    
    ### 📊 Ví Dụ Dữ Liệu
    
    File CSV của bạn nên có định dạng như sau:
    
    | quantity | category | n_review | avg_rating |
    |----------|----------|----------|------------|
    | 150      | Fiction  | 2500     | 4.5        |
    | 200      | Science  | 1800     | 4.2        |
    | 80       | History  | 950      | 4.7        |
    
    ### 🎯 Yêu Cầu
    
    - File phải có định dạng `.csv`
    - Phải có ít nhất 2 cột số để thực hiện phân cụm
    - Dữ liệu nên được làm sạch trước khi upload (hoặc app sẽ tự động xử lý giá trị thiếu)
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Developed with ❤️ using Streamlit | K-Means Clustering Application</p>
    </div>
    """, unsafe_allow_html=True)

