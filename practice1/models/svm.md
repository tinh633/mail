# Linear SVM from Scratch

## 1. Overview

Linear SVM (Support Vector Machine) là thuật toán phân loại nhị phân có giám sát. Mục tiêu của SVM là tìm ra một siêu phẳng (hyperplane) có khả năng phân tách hai lớp dữ liệu với khoảng cách biên (margin) lớn nhất.

Trong bài toán Spam Detection, mô hình được sử dụng để phân loại email thành:

- Ham (0)
- Spam (1)

---

## 2. Hyperplane

SVM tìm siêu phẳng:

```text
w.x + b = 0
```

Trong đó:

- `w`: vector trọng số
- `b`: bias

Điểm dữ liệu được phân loại dựa trên:

```text
f(x) = w.x + b
```

Nếu:

```text
f(x) >= 0
```

→ Spam

Nếu:

```text
f(x) < 0
```

→ Ham

---

## 3. Margin

Hai đường biên của SVM:

```text
w.x + b = 1
```

và

```text
w.x + b = -1
```

Khoảng cách giữa hai đường này gọi là Margin.

Mục tiêu của SVM là tối đa hóa Margin để tăng khả năng tổng quát hóa của mô hình.

---

## 4. Chuyển đổi nhãn

SVM yêu cầu dữ liệu đầu ra thuộc:

```text
{-1, 1}
```

Do đó:

| Nhãn gốc | Nhãn SVM |
| -------- | -------- |
| 0        | -1       |
| 1        | 1        |

Code:

```python
y = np.where(y == 0, -1, 1)
```

---

## 5. Hinge Loss

Linear SVM sử dụng hàm mất mát:

```text
Loss = max(0, 1 - y*(w.x + b))
```

Ý nghĩa:

- Nếu điểm nằm ngoài margin và được phân loại đúng:

```text
y*(w.x + b) >= 1
```

Loss = 0

- Nếu điểm nằm trong margin hoặc phân loại sai:

```text
y*(w.x + b) < 1
```

Loss > 0

---

## 6. Regularization

Để tránh overfitting, SVM bổ sung Regularization:

```text
(lambda / 2) * ||w||²
```

Hàm mất mát cuối cùng:

```text
Loss =
Mean(Hinge Loss)
+
(lambda / 2) * ||w||²
```

Trong đó:

- `lambda`: hệ số điều chuẩn
- `||w||²`: tổng bình phương các trọng số

---

## 7. Class Weight

Dữ liệu Spam/Ham bị mất cân bằng.

Để giảm thiên lệch về lớp Ham, mô hình sử dụng Class Weight:

```text
weight_spam = N / (2 * N_spam)

weight_ham = N / (2 * N_ham)
```

Trong đó:

- `N`: tổng số mẫu
- `N_spam`: số email Spam
- `N_ham`: số email Ham

Nhờ đó các email Spam được chú ý nhiều hơn trong quá trình huấn luyện.

---

## 8. Gradient Descent

Sau mỗi Epoch:

```text
w = w - learning_rate * dw

b = b - learning_rate * db
```

Trong đó:

- `dw`: gradient của w
- `db`: gradient của b

Chỉ các điểm vi phạm Margin:

```text
y*(w.x + b) < 1
```

mới tham gia cập nhật trọng số.

---

## 9. Learning Rate Decay

Learning Rate được giảm dần theo thời gian:

```text
lr_epoch = lr / (1 + 0.001 * epoch)
```

Mục đích:

- Học nhanh ở giai đoạn đầu
- Hội tụ ổn định ở giai đoạn cuối
- Giảm dao động khi tối ưu

---

## 10. Decision Function

Sau khi huấn luyện:

```text
score = w.x + b
```

Code:

```python
def decision_function(self, X):
    return np.dot(X, self.w) + self.b
```

Score càng lớn thì khả năng thuộc lớp Spam càng cao.

---

## 11. Prediction

Linear SVM chuẩn sử dụng:

```text
sign(w.x + b)
```

Nếu:

```text
score >= 0
```

→ Spam (1)

Nếu:

```text
score < 0
```

→ Ham (0)

Code:

```python
def predict(self, X):
    scores = self.decision_function(X)
    return np.where(scores >= 0, 1, 0)
```

---

## 12. Ưu điểm

- Hiệu quả với dữ liệu nhiều chiều.
- Hoạt động tốt với TF-IDF.
- Chống Overfitting tốt nhờ Margin Maximization.
- Phù hợp cho bài toán phân loại văn bản.

---

## 13. Nhược điểm

- Không sinh xác suất trực tiếp.
- Khó diễn giải hơn Logistic Regression.
- Nhạy cảm với tham số Learning Rate và Lambda.
- Thời gian huấn luyện tăng khi dữ liệu lớn.

---

## 14. Vai trò trong dự án

Trong dự án Spam Detection, Linear SVM được xây dựng hoàn toàn bằng NumPy.

Mô hình sử dụng:

- TF-IDF Vectorization
- Hinge Loss
- Regularization
- Class Weight

Kết quả được đánh giá bằng:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- TPR
- FPR

và được so sánh với Logistic Regression và Naive Bayes để lựa chọn mô hình phù hợp nhất cho hệ thống phát hiện email Spam.
