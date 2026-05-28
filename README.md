# Dự Án Machine Learning: Hồi Quy và Phân Loại

Bộ sưu tập hai dự án Machine Learning hoàn chỉnh với phân tích dữ liệu, so sánh mô hình và tối ưu hóa siêu tham số.

## 📋 Mục Lục

1. [Dự Án 1: Hồi Quy Giá Nhà California](#dự-án-1-hồi-quy-giá-nhà-california)
2. [Dự Án 2: Phân Loại Sống Sót Titanic](#dự-án-2-phân-loại-sống-sót-titanic)
3. [Yêu Cầu Cài Đặt](#yêu-cầu-cài-đặt)
4. [Cấu Trúc Dự Án](#cấu-trúc-dự-án)

---

## 🏠 Dự Án 1: Hồi Quy Giá Nhà California

**File**: `California_Housing_Regression_EDA_CV_Tuning.ipynb`

**Mục tiêu**: Dự đoán giá nhà trung vị tại California

### Dữ Liệu
- **Kích thước**: 20.640 quan sát
- **Đặc trưng**: 8 biến số (Thu nhập, Tuổi nhà, Phòng, Phòng ngủ, Dân số, Vị trí...)
- **Biến dự đoán**: `MedHouseVal` (Giá nhà tính theo $100.000)

### Mô Hình So Sánh
- **Linear Regression** - Mô hình cơ sở
- **Gradient Boosting Regressor** - Mô hình nâng cao

### Các Bước Chính
1. Tải và khám phá dữ liệu
2. Xử lý giá trị thiếu và ngoại lệ
3. Chuẩn hóa đặc trưng
4. Huấn luyện mô hình với cross-validation
5. Tối ưu siêu tham số
6. Đánh giá: MAE, RMSE, R² Score

**Kết Quả**: Gradient Boosting cho kết quả tốt hơn (R² ≈ 0.84 vs 0.57)

---

## 🚢 Dự Án 2: Phân Loại Sống Sót Titanic

**File**: `Titanic_Survival_Classification_EDA_CV_Tuning.ipynb`

**Mục tiêu**: Dự đoán khả năng sống sót của hành khách Titanic

### Dữ Liệu
- **Kích thước**: 891 hành khách
- **Đặc trưng**: Hạng vé, Giới tính, Tuổi, Giá vé, Cảng lên tàu...
- **Biến dự đoán**: `Survived` (0 = Chết, 1 = Sống sót)

### Mô Hình So Sánh
- **Logistic Regression** - Mô hình cơ sở
- **Random Forest Classifier** - Mô hình nâng cao

### Các Bước Chính
1. Tải và khám phá dữ liệu
2. Xử lý giá trị thiếu
3. Mã hóa biến phân loại (One-Hot Encoding)
4. Chuẩn hóa đặc trưng
5. Huấn luyện mô hình
6. Tối ưu siêu tham số
7. Đánh giá: Accuracy, Precision, Recall, F1-Score

**Kết Quả**: Random Forest cho kết quả tốt hơn (Accuracy ≈ 82% vs 78%)

---

## 📥 Cách Tải Dữ Liệu

### Dữ Liệu California Housing
- **Tự động tải**: Dữ liệu được tải tự động từ scikit-learn trong notebook
- **Không cần tải thêm**: Không phải chuẩn bị file CSV

### Dữ Liệu Titanic
File `train.csv` và `test.csv` cần tải từ Kaggle:

**Cách 1: Tải từ Kaggle Website**
1. Truy cập: https://www.kaggle.com/competitions/titanic/data
2. Đăng nhập tài khoản Kaggle (hoặc tạo nếu chưa có)
3. Tải file `train.csv` và `test.csv`
4. Lưu vào thư mục dự án

**Cách 2: Tải qua Kaggle API** (nhanh hơn)
```bash
# Cài đặt kaggle
pip install kaggle

# Tải dữ liệu Titanic
kaggle competitions download -c titanic

# Giải nén
unzip titanic.zip
```

**Cách 3: Tải trực tiếp**
- Train: https://www.kaggle.com/api/v1/competitions/titanic/download/train.csv
- Test: https://www.kaggle.com/api/v1/competitions/titanic/download/test.csv

Sau khi tải, đặt file trong cùng thư mục với notebook Titanic

---

## 📥 Yêu Cầu Cài Đặt

### Thư Viện Cần Thiết
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Chạy Dự Án
1. Mở Jupyter Notebook
2. Chạy các cell theo thứ tự từ trên xuống dưới
3. Xem kết quả và biểu đồ

---

## 📂 Cấu Trúc Dự Án

```
Machine_Learning_Problem/
├── README.md
├── California_Housing_Regression_EDA_CV_Tuning.ipynb
├── Titanic_Survival_Classification_EDA_CV_Tuning.ipynb
├── train.csv
└── test.csv
```

---

## � Cách Hoạt Động

### Quy Trình Chung

Cả hai dự án đều theo quy trình Machine Learning tiêu chuẩn:

```
1. Tải Dữ Liệu
   ↓
2. Khám Phá Dữ Liệu (EDA)
   └─ Xem phân phối, mối quan hệ giữa biến
   ↓
3. Xử Lý Dữ Liệu
   └─ Xóa/điền giá trị thiếu
   └─ Loại bỏ ngoại lệ
   └─ Mã hóa biến phân loại (nếu có)
   ↓
4. Chuẩn Hóa Dữ Liệu
   └─ Đưa tất cả biến về cùng thang đo
   ↓
5. Chia Tập Dữ Liệu
   └─ 70-80% huấn luyện, 20-30% kiểm thử
   ↓
6. Huấn Luyện Mô Hình
   └─ Linear Regression / Logistic Regression
   └─ Gradient Boosting / Random Forest
   ↓
7. Tối Ưu Siêu Tham Số
   └─ Dùng cross-validation để tìm tham số tốt nhất
   ↓
8. Đánh Giá Mô Hình
   └─ Hồi quy: MAE, RMSE, R²
   └─ Phân loại: Accuracy, Precision, Recall, F1
   ↓
9. So Sánh & Trực Quan Hóa
   └─ Vẽ biểu đồ so sánh kết quả
```

### Các Kỹ Thuật Chính

**Xử Lý Dữ Liệu**
- Kiểm tra và xử lý giá trị thiếu
- Phát hiện và loại bỏ ngoại lệ
- Mã hóa biến phân loại (One-Hot Encoding)
- Biến đổi log cho biến lệch

**Huấn Luyện Mô Hình**
- Cross-validation (k-fold) để đánh giá ổn định
- Tối ưu hóa siêu tham số tự động
- So sánh 2 mô hình khác nhau

**Đánh Giá Kết Quả**
- Dùng nhiều chỉ số để đánh giá
- Vẽ biểu đồ trực quan
- Phân tích sai số chi tiết

---

## �🛠️ Công Nghệ Sử Dụng

| Thư Viện | Mục Đích |
|----------|---------|
| **NumPy** | Tính toán số học |
| **Pandas** | Xử lý dữ liệu bảng |
| **Matplotlib** | Vẽ biểu đồ |
| **Seaborn** | Trực quan hóa thống kê |
| **Scikit-learn** | Machine Learning |

---
