import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ===========================
# TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# ===========================

def load_data(uploaded_file):
    """
    Tải dữ liệu từ file CSV đã upload.
    
    Tham số:
        uploaded_file: Đối tượng UploadedFile của Streamlit
        
    Trả về:
        pd.DataFrame: DataFrame đã tải
    """
    return pd.read_csv(uploaded_file)


def get_numeric_columns(df):
    """
    Lấy tất cả các cột số từ dataframe.
    
    Tham số:
        df: pandas DataFrame
        
    Trả về:
        list: Danh sách tên các cột số
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_default_features(numeric_columns):
    """
    Lấy các đặc trưng mặc định để phân cụm (quantity, n_review, avg_rating nếu có).
    
    Tham số:
        numeric_columns: Danh sách tên các cột số
        
    Trả về:
        list: Danh sách tên các đặc trưng mặc định
    """
    default_features = [col for col in ['quantity', 'n_review', 'avg_rating'] 
                       if col in numeric_columns]
    return default_features if default_features else numeric_columns[:3]


def preprocess_data(df, selected_features):
    """
    Tiền xử lý dữ liệu: loại bỏ giá trị thiếu và chuẩn hóa đặc trưng.
    
    Tham số:
        df: pandas DataFrame
        selected_features: Danh sách tên đặc trưng để sử dụng
        
    Trả về:
        tuple: (df_processed, X_scaled, scaler, df_scaled, rows_removed)
    """
    # Tạo bản sao
    df_processed = df.copy()
    
    # Đếm số dòng trước khi xử lý
    rows_before = df_processed.shape[0]
    
    # Loại bỏ các dòng có giá trị thiếu trong các đặc trưng đã chọn
    df_processed = df_processed.dropna(subset=selected_features)
    
    # Tính số dòng đã loại bỏ
    rows_removed = rows_before - df_processed.shape[0]
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_processed[selected_features])
    
    # Tạo dataframe đã chuẩn hóa để hiển thị
    df_scaled = pd.DataFrame(
        X_scaled,
        columns=[f"{col}_scaled" for col in selected_features],
        index=df_processed.index
    )
    
    return df_processed, X_scaled, scaler, df_scaled, rows_removed


# ===========================
# PHƯƠNG PHÁP ELBOW VÀ ĐÁNH GIÁ
# ===========================

@st.cache_data
def calculate_elbow_method(X_scaled, k_range=(1, 11)):
    """
    Tính toán inertia và silhouette scores cho các giá trị K khác nhau.
    Được cache để cải thiện hiệu suất.
    
    Tham số:
        X_scaled: Ma trận đặc trưng đã chuẩn hóa
        k_range: Tuple của (min_k, max_k)
        
    Trả về:
        tuple: (K_range, inertia_values, silhouette_scores)
    """
    inertia_values = []
    silhouette_scores = []
    K_range = range(k_range[0], k_range[1])
    
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
    
    return K_range, inertia_values, silhouette_scores


# ===========================
# PHÂN CỤM K-MEANS
# ===========================

def train_kmeans(X_scaled, n_clusters):
    """
    Huấn luyện mô hình K-Means và trả về dự đoán.
    
    Tham số:
        X_scaled: Ma trận đặc trưng đã chuẩn hóa
        n_clusters: Số lượng cụm
        
    Trả về:
        tuple: (kmeans_model, cluster_labels)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    return kmeans, cluster_labels


def add_cluster_labels_to_df(df, cluster_labels):
    """
    Thêm nhãn cụm vào dataframe.
    
    Tham số:
        df: pandas DataFrame
        cluster_labels: Mảng các nhãn cụm
        
    Trả về:
        pd.DataFrame: DataFrame với cột 'Cluster'
    """
    df_copy = df.copy()
    df_copy['Cluster'] = cluster_labels
    df_copy['Cluster'] = df_copy['Cluster'].astype(str)
    return df_copy


def calculate_cluster_statistics(df_processed, selected_features):
    """
    Tính toán thống kê trung bình cho mỗi cụm.
    
    Tham số:
        df_processed: DataFrame có nhãn cụm
        selected_features: Danh sách đặc trưng để tính thống kê
        
    Trả về:
        pd.DataFrame: Thống kê theo từng cụm
    """
    cluster_stats = df_processed.groupby('Cluster')[selected_features].mean()
    cluster_stats = cluster_stats.round(2)
    return cluster_stats


# ===========================
# PHÂN TÍCH VÀ GÁN NHÃN CỤM
# ===========================

def calculate_global_averages(df, features):
    """
    Tính trung bình toàn cục cho các đặc trưng được chỉ định.
    
    Tham số:
        df: pandas DataFrame
        features: Danh sách tên đặc trưng
        
    Trả về:
        dict: Dictionary của feature: giá_trị_trung_bình
    """
    return {feature: df[feature].mean() for feature in features}


def identify_trend_cluster(df_processed, quantity_col='quantity'):
    """
    Xác định cụm có lượng bán trung bình cao nhất (cụm xu hướng).
    
    Tham số:
        df_processed: DataFrame có nhãn cụm
        quantity_col: Tên cột quantity
        
    Trả về:
        str: ID của cụm có lượng bán trung bình cao nhất
    """
    cluster_avg_qty = df_processed.groupby('Cluster')[quantity_col].mean()
    return cluster_avg_qty.idxmax()


def get_cluster_label(cluster_id, trend_cluster_id, mean_qty, mean_rating, 
                     avg_qty_all, avg_rating_all):
    """
    Áp dụng logic gán nhãn để xác định nhãn và màu cho cụm.
    
    Tham số:
        cluster_id: ID của cụm hiện tại
        trend_cluster_id: ID của cụm xu hướng
        mean_qty: Lượng bán trung bình của cụm hiện tại
        mean_rating: Rating trung bình của cụm hiện tại
        avg_qty_all: Lượng bán trung bình toàn cục
        avg_rating_all: Rating trung bình toàn cục
        
    Trả về:
        tuple: (label, label_color)
    """
    if cluster_id == trend_cluster_id:
        label = "🔥 NHÓM XU HƯỚNG (TRENDING - Bán Chạy Nhất)"
        label_color = "#ff4b4b"
    elif mean_qty < avg_qty_all and mean_rating >= avg_rating_all:
        label = "💎 NHÓM TIỀM NĂNG (Bán ít nhưng Rating rất cao)"
        label_color = "#00cc88"
    elif mean_qty < avg_qty_all and mean_rating < avg_rating_all:
        label = "⚠️ NHÓM CẦN CẢI THIỆN (Hiệu suất thấp)"
        label_color = "#ffa500"
    else:
        label = "📚 NHÓM PHỔ THÔNG (Bán ổn định)"
        label_color = "#0068c9"
    
    return label, label_color


def get_dominant_category(cluster_data, category_col='category'):
    """
    Tìm thể loại chiếm ưu thế trong một cụm.
    
    Tham số:
        cluster_data: DataFrame chứa dữ liệu cụm
        category_col: Tên cột category
        
    Trả về:
        tuple: (dominant_category, count, category_info_string) hoặc (None, 0, "N/A")
    """
    if category_col in cluster_data.columns:
        category_counts = cluster_data[category_col].value_counts()
        dominant_category = category_counts.index[0]
        dominant_count = category_counts.values[0]
        category_info = f"**{dominant_category}** ({dominant_count} sách)"
        return dominant_category, dominant_count, category_info
    else:
        return None, 0, "N/A (không có cột category)"


def get_cluster_feature_stats(cluster_data, selected_features):
    """
    Tính toán thống kê chi tiết cho các đặc trưng của cụm.
    
    Tham số:
        cluster_data: DataFrame chứa dữ liệu cụm
        selected_features: Danh sách tên đặc trưng
        
    Trả về:
        pd.DataFrame: DataFrame thống kê
    """
    stats_df = pd.DataFrame({
        'Chỉ Số': selected_features,
        'Giá Trị TB': [cluster_data[feat].mean() for feat in selected_features],
        'Min': [cluster_data[feat].min() for feat in selected_features],
        'Max': [cluster_data[feat].max() for feat in selected_features]
    })
    stats_df['Giá Trị TB'] = stats_df['Giá Trị TB'].round(2)
    stats_df['Min'] = stats_df['Min'].round(2)
    stats_df['Max'] = stats_df['Max'].round(2)
    return stats_df


def get_category_distribution(cluster_data, top_n=8, category_col='category'):
    """
    Lấy phân bố thể loại trong một cụm.
    
    Tham số:
        cluster_data: DataFrame chứa dữ liệu cụm
        top_n: Số lượng thể loại hàng đầu cần trả về
        category_col: Tên cột category
        
    Trả về:
        pd.Series: Số lượng theo thể loại (top_n thể loại)
    """
    if category_col in cluster_data.columns:
        category_counts = cluster_data[category_col].value_counts()
        return category_counts.head(min(top_n, len(category_counts)))
    return None


def prepare_download_data(df):
    """
    Chuẩn bị dữ liệu để tải xuống CSV.
    
    Tham số:
        df: pandas DataFrame
        
    Trả về:
        bytes: Dữ liệu CSV đã mã hóa
    """
    return df.to_csv(index=False).encode('utf-8')

