# Mô tả bài toán phân loại và hồi quy

Tài liệu này mô tả ngắn gọn hai dạng bài toán Machine Learning trong dự án: bài toán phân loại sống sót Titanic và bài toán hồi quy giá nhà California.

---

## Phần 1: Bài toán phân loại

### 1. Mục tiêu bài toán

Bài toán phân loại có mục tiêu dự đoán một nhãn rời rạc cho từng mẫu dữ liệu. Trong dự án này, bài toán phân loại được áp dụng cho bộ dữ liệu Titanic nhằm dự đoán một hành khách có sống sót hay không.

Biến cần dự đoán là `Survived`:

- `0`: Không sống sót
- `1`: Sống sót

### 2. Dữ liệu đầu vào

Dữ liệu đầu vào gồm thông tin của từng hành khách trên tàu Titanic, ví dụ:

- `Pclass`: Hạng vé của hành khách
- `Sex`: Giới tính
- `Age`: Tuổi
- `SibSp`: Số anh/chị/em hoặc vợ/chồng đi cùng
- `Parch`: Số cha/mẹ hoặc con đi cùng
- `Fare`: Giá vé
- `Embarked`: Cảng lên tàu

Các đặc trưng này được dùng để học mối quan hệ giữa thông tin hành khách và khả năng sống sót.

### 3. Đầu ra của mô hình

Đầu ra là một nhãn phân loại cho từng hành khách:

- Hành khách thuộc nhóm sống sót
- Hành khách thuộc nhóm không sống sót

Một số mô hình cũng có thể trả về xác suất sống sót, ví dụ `0.82` nghĩa là mô hình ước lượng hành khách có 82% khả năng sống sót.

### 4. Quy trình xử lý

Quy trình chính của bài toán phân loại gồm:

1. Đọc và khám phá dữ liệu.
2. Xử lý giá trị thiếu, ví dụ điền tuổi bị thiếu.
3. Mã hóa biến phân loại như `Sex` và `Embarked`.
4. Chuẩn hóa hoặc biến đổi dữ liệu nếu cần.
5. Chia dữ liệu thành tập huấn luyện và tập kiểm tra.
6. Huấn luyện mô hình phân loại.
7. Đánh giá kết quả dự đoán.

### 5. Mô hình sử dụng

Các mô hình phù hợp cho bài toán này gồm:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine

Trong dự án, Logistic Regression có thể được dùng làm mô hình cơ sở, còn Random Forest được dùng để so sánh với một mô hình mạnh hơn.

### 6. Chỉ số đánh giá

Vì đây là bài toán phân loại, các chỉ số đánh giá thường dùng gồm:

- Accuracy: Tỷ lệ dự đoán đúng trên toàn bộ mẫu.
- Precision: Trong các mẫu được dự đoán là sống sót, tỷ lệ dự đoán đúng là bao nhiêu.
- Recall: Trong các mẫu thật sự sống sót, mô hình tìm đúng được bao nhiêu.
- F1-score: Trung bình điều hòa giữa Precision và Recall.
- Confusion Matrix: Ma trận nhầm lẫn giữa nhãn thật và nhãn dự đoán.

### 7. Ý nghĩa thực tế

Bài toán này giúp minh họa cách xây dựng mô hình dự đoán nhãn rời rạc. Thay vì dự đoán một giá trị số liên tục, mô hình phải đưa ra quyết định mẫu thuộc lớp nào.

---

## Phần 2: Bài toán hồi quy

### 1. Mục tiêu bài toán

Bài toán hồi quy có mục tiêu dự đoán một giá trị số liên tục. Trong dự án này, bài toán hồi quy được áp dụng cho bộ dữ liệu California Housing nhằm dự đoán giá trị trung vị của nhà ở tại các khu vực tại California.

Biến cần dự đoán là `MedHouseVal`, biểu diễn giá trị nhà trung vị theo đơn vị 100.000 USD.

### 2. Dữ liệu đầu vào

Dữ liệu đầu vào gồm các đặc trưng mô tả khu vực dân cư, ví dụ:

- `MedInc`: Thu nhập trung vị của hộ gia đình
- `HouseAge`: Tuổi trung vị của nhà
- `AveRooms`: Số phòng trung bình
- `AveBedrms`: Số phòng ngủ trung bình
- `Population`: Dân số khu vực
- `AveOccup`: Số người trung bình trong mỗi hộ
- `Latitude`: Vĩ độ
- `Longitude`: Kinh độ

Các đặc trưng này được dùng để học mối quan hệ giữa điều kiện khu vực và giá trị nhà.

### 3. Đầu ra của mô hình

Đầu ra là một giá trị số liên tục, ví dụ:

- `2.45` nghĩa là giá trị nhà trung vị được dự đoán khoảng 245.000 USD.
- `3.10` nghĩa là giá trị nhà trung vị được dự đoán khoảng 310.000 USD.

Khác với phân loại, đầu ra của hồi quy không phải là nhãn lớp mà là một con số có thể thay đổi liên tục.

### 4. Quy trình xử lý

Quy trình chính của bài toán hồi quy gồm:

1. Đọc và khám phá dữ liệu.
2. Kiểm tra phân phối của các biến.
3. Xử lý giá trị thiếu và ngoại lệ nếu có.
4. Chuẩn hóa đặc trưng để đưa các biến về cùng thang đo.
5. Chia dữ liệu thành tập huấn luyện và tập kiểm tra.
6. Huấn luyện mô hình hồi quy.
7. Đánh giá sai số dự đoán.
8. So sánh kết quả giữa các mô hình.

### 5. Mô hình sử dụng

Các mô hình phù hợp cho bài toán hồi quy gồm:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Trong dự án, Linear Regression có thể được dùng làm mô hình cơ sở, còn Gradient Boosting Regressor được dùng để cải thiện chất lượng dự đoán.

### 6. Chỉ số đánh giá

Vì đây là bài toán hồi quy, các chỉ số đánh giá thường dùng gồm:

- MAE: Sai số tuyệt đối trung bình.
- MSE: Sai số bình phương trung bình.
- RMSE: Căn bậc hai của MSE, cùng đơn vị với biến cần dự đoán.
- R2 Score: Mức độ mô hình giải thích được sự biến thiên của dữ liệu.

Trong đó:

- MAE và RMSE càng thấp thì mô hình càng tốt.
- R2 Score càng gần `1` thì mô hình giải thích dữ liệu càng tốt.

### 7. Ý nghĩa thực tế

Bài toán này giúp minh họa cách xây dựng mô hình dự đoán giá trị số. Mô hình không chỉ phân nhóm dữ liệu mà còn ước lượng một đại lượng cụ thể, chẳng hạn giá nhà, doanh thu, nhiệt độ hoặc nhu cầu tiêu thụ.

---

## So sánh ngắn gọn

| Tiêu chí | Phân loại | Hồi quy |
| --- | --- | --- |
| Mục tiêu | Dự đoán nhãn/lớp | Dự đoán giá trị số liên tục |
| Ví dụ trong dự án | Dự đoán sống sót Titanic | Dự đoán giá nhà California |
| Biến đầu ra | `Survived` gồm `0` hoặc `1` | `MedHouseVal` là giá trị số |
| Mô hình tiêu biểu | Logistic Regression, Random Forest | Linear Regression, Gradient Boosting |
| Chỉ số đánh giá | Accuracy, Precision, Recall, F1-score | MAE, RMSE, R2 Score |

