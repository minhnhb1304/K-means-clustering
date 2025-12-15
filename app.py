import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Import các hàm logic từ utils
from utils import (
    load_data,
    get_numeric_columns,
    get_default_features,
    preprocess_data,
    calculate_elbow_method,
    train_kmeans,
    add_cluster_labels_to_df,
    calculate_cluster_statistics,
    calculate_global_averages,
    identify_trend_cluster,
    get_cluster_label,
    get_dominant_category,
    get_cluster_feature_stats,
    get_category_distribution,
    prepare_download_data
)

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
        df = load_data(uploaded_file)
        st.sidebar.success("✅ Tải file thành công!")
        
        # 2. Chọn các đặc trưng số để phân cụm
        st.sidebar.subheader("2. Chọn Đặc Trưng")
        
        # Tự động phát hiện các cột số trong dataset
        numeric_columns = get_numeric_columns(df)
        
        # Lấy các đặc trưng mặc định
        default_features = get_default_features(numeric_columns)
        
        selected_features = st.sidebar.multiselect(
            "Chọn các cột số để clustering:",
            options=numeric_columns,
            default=default_features,
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
            
            # Hiển thị số lượng giá trị thiếu trước khi xử lý
            missing_before = df[selected_features].isnull().sum().sum()
            st.write(f"**Số giá trị thiếu trong các cột đã chọn:** {missing_before}")
            
            # Tiền xử lý dữ liệu
            df_processed, X_scaled, scaler, df_scaled, rows_removed = preprocess_data(df, selected_features)
            
            st.success(f"✅ Đã loại bỏ {rows_removed} dòng có giá trị thiếu")
            st.write(f"**Số dòng còn lại:** {df_processed.shape[0]}")
            
            # Chuẩn hóa dữ liệu
            st.subheader("🔄 Chuẩn Hóa Dữ Liệu (StandardScaler)")
            
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
            
            # Tính toán phương pháp Elbow (đã cache)
            with st.spinner("🔄 Đang tính toán Inertia cho các giá trị K..."):
                K_range, inertia_values, silhouette_scores = calculate_elbow_method(X_scaled, k_range=(1, 11))
            
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
            
            # Huấn luyện KMeans
            with st.spinner(f"🔄 Đang thực hiện phân cụm với K={k_clusters}..."):
                kmeans_final, cluster_labels = train_kmeans(X_scaled, k_clusters)
            
            # Thêm nhãn cụm vào dataframe
            df_processed = add_cluster_labels_to_df(df_processed, cluster_labels)
            
            # Hiển thị thông tin về các cụm
            st.subheader("📊 Phân Bố Các Cụm")
            cluster_counts = df_processed['Cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(
                    pd.DataFrame({
                        'Cụm': cluster_counts.index,
                        'Số lượng phần tử': cluster_counts.values,
                        'Phần trăm': [f"{(v/len(df_processed)*100):.1f}%" for v in cluster_counts.values]
                    }),
                    use_container_width=True
                )
                
                # Tính Silhouette Score
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
            
            with col2:
                # Vẽ biểu đồ phân bố cụm
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
            
            # Cho phép người dùng chọn trục X và Y để vẽ
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
            
            # Tính toán thống kê theo cụm
            cluster_stats = calculate_cluster_statistics(df_processed, selected_features)
            st.dataframe(cluster_stats, use_container_width=True)
            
            # ===========================
            # SECTION 5: CLUSTER INTERPRETATION & AUTO-LABELING
            # ===========================
            st.markdown("---")
            st.header("5️⃣ Phân Tích & Gán Nhãn Tự Động Cho Từng Cụm")
            
            st.write("**Phân tích tự động** để hiểu đặc điểm của từng nhóm sách và gán nhãn phù hợp.")
            
            # Kiểm tra xem có đủ đặc trưng cần thiết không
            if 'quantity' in selected_features and 'avg_rating' in selected_features:
                # 1. Tính trung bình toàn cục
                global_avgs = calculate_global_averages(df_processed, ['quantity', 'avg_rating'])
                avg_qty_all = global_avgs['quantity']
                avg_rating_all = global_avgs['avg_rating']
                
                st.info(f"📊 **Chỉ Số Trung Bình Toàn Dataset:** Lượng bán TB = {avg_qty_all:.1f} | Rating TB = {avg_rating_all:.2f}")
                
                # 2. Xác định cụm xu hướng
                trend_cluster_id = identify_trend_cluster(df_processed, 'quantity')
                cluster_avg_qty = df_processed.groupby('Cluster')['quantity'].mean()
                
                st.markdown("---")
                
                # 3. Lặp qua từng cụm và áp dụng logic gán nhãn
                for cluster_id in sorted(df_processed['Cluster'].unique()):
                    cluster_data = df_processed[df_processed['Cluster'] == cluster_id]
                    
                    # Tính thống kê cho cụm
                    n_books = len(cluster_data)
                    mean_qty = cluster_data['quantity'].mean()
                    mean_rating = cluster_data['avg_rating'].mean()
                    
                    # Lấy nhãn và màu sắc
                    label, label_color = get_cluster_label(
                        cluster_id, trend_cluster_id, mean_qty, mean_rating,
                        avg_qty_all, avg_rating_all
                    )
                    
                    # Lấy thể loại chiếm ưu thế
                    dominant_category, dominant_count, category_info = get_dominant_category(cluster_data, 'category')
                    
                    # Hiển thị bằng Streamlit expander
                    with st.expander(f"**Cụm {cluster_id}:** {label}", expanded=(cluster_id == trend_cluster_id)):
                        st.markdown(f"### <span style='color:{label_color}'>{label}</span>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("📚 Số Lượng Sách", f"{n_books}")
                        
                        with col2:
                            delta_qty = mean_qty - avg_qty_all
                            st.metric(
                                "📦 Trung Bình Bán",
                                f"{mean_qty:.1f}",
                                delta=f"{delta_qty:+.1f} so với TB chung",
                                delta_color="normal"
                            )
                        
                        with col3:
                            delta_rating = mean_rating - avg_rating_all
                            st.metric(
                                "⭐ Trung Bình Rating",
                                f"{mean_rating:.2f}",
                                delta=f"{delta_rating:+.2f} so với TB chung",
                                delta_color="normal"
                            )
                        
                        st.markdown("**🏷️ Thể Loại Chủ Đạo:**")
                        st.markdown(f"<h4>{category_info}</h4>", unsafe_allow_html=True)
                        
                        # Thống kê bổ sung
                        st.markdown("**📊 Chi Tiết Các Chỉ Số:**")
                        stats_df = get_cluster_feature_stats(cluster_data, selected_features)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # 5. Phần kết luận cuối cùng
                st.markdown("---")
                st.markdown("## 🏆 KẾT LUẬN CUỐI CÙNG")
                
                # Phân tích cụm xu hướng
                trending_cluster_data = df_processed[df_processed['Cluster'] == trend_cluster_id]
                
                if 'category' in df_processed.columns:
                    trending_category_counts = trending_cluster_data['category'].value_counts()
                    top_category = trending_category_counts.index[0]
                    top_category_count = trending_category_counts.values[0]
                    total_in_trending = len(trending_cluster_data)
                    percentage = (top_category_count / total_in_trending) * 100
                    
                    # Tạo hộp kết luận đẹp mắt
                    st.markdown(f"""
                    <div style='background-color: #002147; padding: 20px; border-radius: 10px; border-left: 5px solid #ff9800;'>
                        <h3 style='color: #e65100; margin-top: 0;'>🔥 Phân Tích Xu Hướng Thị Trường</h3>
                        <p style='font-size: 18px; line-height: 1.6;'>
                            Dựa trên phân tích dữ liệu bán hàng, <b>Cụm {trend_cluster_id}</b> đã được xác định là 
                            <b style='color: #d32f2f;'>NHÓM XU HƯỚNG</b> với lượng bán trung bình cao nhất 
                            (<b>{cluster_avg_qty[trend_cluster_id]:.1f}</b> quyển/sách).
                        </p>
                        <p style='font-size: 20px; font-weight: bold; color: #1976d2; margin: 15px 0;'>
                            📚 Thể loại đang là XU THẾ SỐ 1 trên sàn là: <span style='color: #c62828;'>{top_category.upper()}</span>
                        </p>
                        <p style='font-size: 16px;'>
                            <b>Lý do:</b> Thể loại <b>{top_category}</b> chiếm <b style='color: #2e7d32;'>{percentage:.1f}%</b> 
                            ({top_category_count}/{total_in_trending} quyển) trong nhóm bán chạy nhất, 
                            với trung bình <b>{trending_cluster_data['quantity'].mean():.1f}</b> quyển được bán ra mỗi đầu sách.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.warning("⚠️ Không tìm thấy cột 'category' trong dataset để phân tích xu hướng thể loại.")
            else:
                st.warning("⚠️ Cần có cả 'quantity' và 'avg_rating' trong các đặc trưng đã chọn để thực hiện phân tích tự động.")
            
            # ===========================
            # TẢI DỮ LIỆU KẾT QUẢ
            # ===========================
            st.markdown("---")
            st.subheader("💾 Tải Kết Quả")
            
            # Chuẩn bị dữ liệu để tải xuống
            csv_data = prepare_download_data(df_processed)
            
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
    # Hiển thị hướng dẫn khi người dùng chưa upload file
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
    
    # Chân trang
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Developed with ❤️ | K-Means Clustering Application</p>
    </div>
    """, unsafe_allow_html=True)
