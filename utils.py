import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class BookClusteringAI:
    """
    Class AI để phân cụm và dự báo sách sử dụng K-Means Clustering.
    """
    
    def __init__(self):
        """Khởi tạo BookClusteringAI"""
        self.model = None
        self.scaler = None
        self.cluster_label_mapping = {}  # Mapping từ cluster_id -> label (Trend/Potential/Risk/Standard)
        self.features = ['quantity', 'n_review', 'avg_rating']
        self.model_path = 'kmeans_model.pkl'
        self.scaler_path = 'scaler.pkl'
        self.mapping_path = 'cluster_mapping.pkl'
        self.df_processed = None
        self.X_scaled = None
    
    def load_data(self, uploaded_file):
        return pd.read_csv(uploaded_file)
    
    def preprocess_data(self, df):
        # Tạo bản sao
        df_processed = df.copy()
        
        # Loại bỏ các dòng có giá trị thiếu trong các đặc trưng
        df_processed = df_processed.dropna(subset=self.features)
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_processed[self.features])
        
        # Lưu vào instance
        self.df_processed = df_processed
        self.X_scaled = X_scaled
        self.scaler = scaler
        
        return df_processed, X_scaled, scaler
    
    @st.cache_data
    def calculate_elbow_method(_self, X_scaled, k_range=(1, 11)):
        """
        Tính toán inertia và silhouette scores cho các giá trị K.
        Được cache để tối ưu hiệu suất.
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
    
    def train_model(self, X_scaled, n_clusters=4):
        # Huấn luyện mô hình K-Means và lưu vào file .pkl.

        # Huấn luyện mô hình
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Lưu mô hình và scaler
        self.model = kmeans
        joblib.dump(kmeans, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        # Thêm nhãn cụm vào dataframe
        df_with_clusters = self.df_processed.copy()
        df_with_clusters['Cluster'] = cluster_labels
        df_with_clusters['Cluster'] = df_with_clusters['Cluster'].astype(str)
        
        # Phân tích và gán nhãn động cho các cụm
        self._analyze_and_label_clusters(df_with_clusters)
        
        # Lưu mapping
        joblib.dump(self.cluster_label_mapping, self.mapping_path)
        
        return kmeans, cluster_labels, df_with_clusters
    
    def _analyze_and_label_clusters(self, df_with_clusters):
        # Phân tích các cụm và gán nhãn thông minh (Dynamic Labeling Logic).
        
        # Tính trung bình toàn cục
        avg_qty_all = df_with_clusters['quantity'].mean()
        avg_rating_all = df_with_clusters['avg_rating'].mean()
        
        # Tính trung bình theo cụm
        cluster_stats = df_with_clusters.groupby('Cluster').agg({
            'quantity': 'mean',
            'avg_rating': 'mean'
        })
        
        # Tìm cụm có lượng bán cao nhất -> Xu Hướng
        trend_cluster_id = cluster_stats['quantity'].idxmax()
        
        # Khởi tạo mapping
        self.cluster_label_mapping = {}
        
        # Phân tích từng cụm
        for cluster_id in sorted(df_with_clusters['Cluster'].unique()):
            mean_qty = cluster_stats.loc[cluster_id, 'quantity']
            mean_rating = cluster_stats.loc[cluster_id, 'avg_rating']
            
            # Logic gán nhãn 
            if cluster_id == trend_cluster_id:
                label = "🔥 Xu Hướng (Best-Seller)"
            elif mean_qty < avg_qty_all and mean_rating >= avg_rating_all:
                label = "💎 Tiềm Năng (Kén Khách)"
            elif mean_qty < avg_qty_all and mean_rating < avg_rating_all:
                label = "⚠️ Rủi Ro (Cần Cải Thiện)"
            else:
                label = "📚 Phổ Thông (Bán Ổn Định)"
            
            self.cluster_label_mapping[str(cluster_id)] = label
    
    def get_cluster_label_name(self, cluster_id):
        # Lấy tên nhãn của cụm từ cluster_id.

        cluster_id_str = str(cluster_id)
        return self.cluster_label_mapping.get(cluster_id_str, "Unknown")
    
    def load_saved_model(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path) and os.path.exists(self.mapping_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.cluster_label_mapping = joblib.load(self.mapping_path)
                return True
            return False
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình: {str(e)}")
            return False
    
    def predict_new_book(self, quantity, n_review, rating):
        # Kiểm tra xem mô hình đã được huấn luyện chưa
        if self.model is None or self.scaler is None:
            # Thử tải mô hình đã lưu
            if not self.load_saved_model():
                return {
                    'error': 'Mô hình chưa được huấn luyện. Vui lòng huấn luyện mô hình ở Tab Dashboard trước.'
                }
        
        # Chuẩn bị dữ liệu đầu vào (sử dụng DataFrame để tránh warning về feature names)
        input_data = pd.DataFrame({
            'quantity': [quantity],
            'n_review': [n_review],
            'avg_rating': [rating]
        })
        
        # Chuẩn hóa
        input_scaled = self.scaler.transform(input_data)
        
        # Dự báo
        cluster_id = self.model.predict(input_scaled)[0]
        cluster_id_str = str(cluster_id)
        
        # Lấy nhãn
        cluster_label = self.get_cluster_label_name(cluster_id_str)
        
        # Tạo lời khuyên dựa trên nhãn cụm
        advice = self._get_business_advice(cluster_label)
        
        return {
            'cluster_id': cluster_id_str,
            'cluster_label': cluster_label,
            'manager_advice': advice['manager'],
            'marketing_action': advice['marketing']
        }
    
    def _get_business_advice(self, cluster_label):
        # Lời khuyên kinh doanh dựa trên nhãn cụm.
        advice_map = {
            "🔥 Xu Hướng (Best-Seller)": {
                "manager": "Nhập số lượng lớn. Đảm bảo tồn kho > 500 cuốn.",
                "marketing": "Ưu tiên trưng bày tại trang chủ/kệ Hot. Chạy Ads ngân sách cao."
            },
            "💎 Tiềm Năng (Kén Khách)": {
                "manager": "Nhập số lượng vừa phải. Theo dõi kỹ review.",
                "marketing": "Viết content review sâu sắc. Target nhóm khách hàng chuyên biệt."
            },
            "⚠️ Rủi Ro (Cần Cải Thiện)": {
                "manager": "Hạn chế nhập thêm. Cân nhắc xả hàng.",
                "marketing": "Tạo Flash Sale giảm giá sâu để đẩy hàng tồn."
            },
            "📚 Phổ Thông (Bán Ổn Định)": {
                "manager": "Duy trì mức nhập trung bình.",
                "marketing": "Bán kèm combo khuyến mãi. Phù hợp bán trên sàn TMĐT."
            }
        }
        
        return advice_map.get(cluster_label, {
            "manager": "Chưa có dữ liệu đủ để đưa ra lời khuyên.",
            "marketing": "Chưa có gợi ý marketing cụ thể."
        })
    
    def get_cluster_statistics(self, df_with_clusters):
        # Tính toán thống kê theo cụm.

        stats = df_with_clusters.groupby('Cluster')[self.features].mean()
        stats = stats.round(2)
        
        # Thêm cột nhãn
        stats['Nhãn Cụm'] = [self.get_cluster_label_name(cluster_id) for cluster_id in stats.index]
        
        # Đổi tên cột sang tiếng Việt
        stats.columns = ['Số lượng bán TB', 'Số đánh giá TB', 'Rating TB', 'Nhãn Cụm']
        
        return stats
