# Logistic Regression from Scratch

## 1. Overview

Logistic Regression là một thuật toán học có giám sát được sử dụng phổ biến cho các bài toán phân loại nhị phân.

Trong dự án Spam Detection, Logistic Regression được sử dụng để phân loại email thành:

- Ham (0)
- Spam (1)

Khác với Linear Regression, Logistic Regression dự đoán xác suất một mẫu thuộc lớp Spam thông qua hàm Sigmoid.

---

## 2. Tại sao lựa chọn Logistic Regression?

Logistic Regression là một trong những thuật toán cơ bản nhưng hiệu quả đối với bài toán phân loại văn bản.

Lý do lựa chọn:

- Dễ triển khai từ đầu bằng NumPy.
- Tốc độ huấn luyện nhanh.
- Hoạt động tốt trên dữ liệu TF-IDF có số chiều lớn.
- Có khả năng sinh xác suất dự đoán.
- Dễ điều chỉnh ngưỡng phân loại (Threshold).
- Dễ giải thích kết quả hơn SVM.

Đối với bài toán Spam Detection, xác suất dự đoán là một ưu điểm quan trọng vì có thể điều chỉnh ngưỡng quyết định nhằm giảm số lượng email Ham bị phân loại nhầm thành Spam.

---

## 3. Logistic Regression hoạt động như thế nào?

Mô hình tính điểm tuyến tính:

```text
z = w.x + b
```

Trong đó:

- `w`: vector trọng số
- `b`: bias
- `x`: vector đặc trưng TF-IDF

Giá trị `z` có thể nằm trong khoảng:

```text
(-∞, +∞)
```

Do đó cần sử dụng hàm Sigmoid để chuyển thành xác suất.

---

## 4. Hàm Sigmoid

Hàm Sigmoid:

```text
Sigmoid(z)

=

1 / (1 + e^(-z))
```

Kết quả luôn nằm trong khoảng:

```text
(0, 1)
```

Ý nghĩa:

- Gần 0 → Ham
- Gần 1 → Spam

Ví dụ:

| z   | Sigmoid(z) |
| --- | ---------- |
| -5  | 0.0067     |
| 0   | 0.5        |
| 5   | 0.9933     |

Trong code:

```python
def compute_Sigmoid(x):
    return 1/(1 + np.exp(-x))
```

---

## 5. Xác suất dự đoán

Sau khi tính Sigmoid:

```text
P(Spam|X)
=
Sigmoid(w.x+b)
```

Ví dụ:

| Probability |
| ----------- |
| 0.95        |
| 0.80        |
| 0.15        |
| 0.03        |

Giá trị càng lớn thì khả năng email là Spam càng cao.

---

## 6. Binary Cross Entropy Loss

Để đánh giá mức độ sai lệch giữa dự đoán và nhãn thực tế, mô hình sử dụng Binary Cross Entropy.

Công thức:

```text
Loss

=

-mean(
y*log(y_pred)
+
(1-y)*log(1-y_pred)
)
```

Trong đó:

- `y`: nhãn thực tế
- `y_pred`: xác suất dự đoán

Đặc điểm:

- Dự đoán đúng → Loss nhỏ
- Dự đoán sai → Loss lớn

Trong code:

```python
def compute_loss(y_true, y_pred):
    y_pred = np.clip(
        y_pred,
        1e-10,
        1-1e-10
    )

    return -np.mean(
        y_true*np.log(y_pred)
        +
        (1-y_true)*np.log(1-y_pred)
    )
```

---

## 7. Gradient Descent

Mục tiêu của Logistic Regression là tìm bộ trọng số:

```text
w
b
```

sao cho Loss nhỏ nhất.

Sau mỗi vòng lặp:

```text
w = w - learning_rate * dw

b = b - learning_rate * db
```

Trong đó:

- `dw`: gradient của trọng số
- `db`: gradient của bias

Code:

```python
dw = (1 / n_samples) * np.dot(
    X.T,
    (y_pred - y)
)

db = (1 / n_samples) * np.sum(
    y_pred - y
)
```

---

## 8. Learning Rate

Learning Rate quyết định tốc độ cập nhật trọng số.

Trong dự án:

```python
learning_rate = 0.5
```

Nếu Learning Rate quá nhỏ:

```text
Huấn luyện rất chậm
```

Nếu Learning Rate quá lớn:

```text
Mô hình dễ dao động và khó hội tụ
```

---

## 9. Số vòng lặp (Iterations)

Mô hình được huấn luyện trong:

```python
n_iters = 2000
```

Mỗi vòng lặp thực hiện:

1. Tính z
2. Tính Sigmoid
3. Tính Loss
4. Tính Gradient
5. Cập nhật tham số

Loss được in ra sau mỗi 500 Epoch để theo dõi quá trình hội tụ.

---

## 10. Threshold

Sau khi có xác suất dự đoán:

```text
P(Spam|X)
```

cần xác định ngưỡng quyết định.

Thông thường:

```text
Threshold = 0.5
```

Tuy nhiên bộ dữ liệu Spam/Ham bị mất cân bằng nên dự án sử dụng:

```text
Threshold = 0.2
```

Code:

```python
return (y_pred > 0.2).astype(int)
```

Ý nghĩa:

Nếu:

```text
P(Spam|X) > 0.2
```

→ Spam

Ngược lại:

```text
P(Spam|X) <= 0.2
```

→ Ham

Việc giảm Threshold giúp mô hình phát hiện được nhiều email Spam hơn và cải thiện Recall của lớp Spam.

---

## 11. Dự đoán xác suất

Model hỗ trợ dự đoán xác suất:

```python
predict_proba()
```

Code:

```python
def predict_proba(self, X):
    return compute_Sigmoid(
        X @ self.w + self.b
    )
```

Kết quả:

| Email   | Probability |
| ------- | ----------- |
| Email A | 0.95        |
| Email B | 0.12        |
| Email C | 0.72        |

---

## 12. Đánh giá mô hình

Mô hình được đánh giá bằng:

- Accuracy
- Precision
- Recall
- F1-score

Trong đó:

### Precision

```text
TP / (TP + FP)
```

Đo tỷ lệ email dự đoán Spam thực sự là Spam.

### Recall

```text
TP / (TP + FN)
```

Đo khả năng phát hiện email Spam.

### F1-score

```text
2 * Precision * Recall
/
(Precision + Recall)
```

Là trung bình điều hòa giữa Precision và Recall.

---

## 13. Ưu điểm

- Dễ triển khai.
- Dễ giải thích.
- Huấn luyện nhanh.
- Hoạt động tốt với TF-IDF.
- Có khả năng dự đoán xác suất.
- Dễ điều chỉnh Threshold.

---

## 14. Nhược điểm

- Chỉ học được ranh giới tuyến tính.
- Nhạy cảm với dữ liệu mất cân bằng.
- Hiệu năng có thể thấp hơn các mô hình phức tạp hơn trong một số trường hợp.

---

## 15. Vai trò trong dự án

Trong dự án Spam Detection, Logistic Regression được xây dựng hoàn toàn bằng NumPy và sử dụng:

- TF-IDF Vectorization
- Sigmoid Function
- Binary Cross Entropy Loss
- Gradient Descent
- Threshold = 0.2

Kết quả được đánh giá bằng:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- TPR
- FPR

và được so sánh với Multinomial Naive Bayes và Linear SVM để lựa chọn mô hình phù hợp nhất cho hệ thống phân loại email Spam/Ham.
