# Multinomial Naive Bayes from Scratch

## 1. Overview

Multinomial Naive Bayes là một thuật toán phân loại xác suất dựa trên định lý Bayes. Thuật toán đặc biệt phù hợp với dữ liệu văn bản được biểu diễn dưới dạng Bag of Words hoặc TF-IDF.

Trong dự án Spam Detection, Multinomial Naive Bayes được sử dụng để phân loại email thành:

- Ham (0)
- Spam (1)

Mô hình dự đoán dựa trên xác suất xuất hiện của các từ trong từng lớp.

---

## Lý do lựa chọn Multinomial Naive Bayes

Naive Bayes không phải là một thuật toán duy nhất mà là một họ các thuật toán phân loại xác suất, trong đó phổ biến nhất gồm:

- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- Multinomial Naive Bayes

Mỗi biến thể được thiết kế cho những loại dữ liệu khác nhau.

### Gaussian Naive Bayes

Gaussian Naive Bayes giả định các đặc trưng đầu vào tuân theo phân phối chuẩn (Normal Distribution).

Thuật toán thường được sử dụng cho các dữ liệu số liên tục như:

- Chiều cao
- Cân nặng
- Tuổi
- Nhiệt độ

Tuy nhiên trong bài toán phân loại email Spam/Ham, dữ liệu đầu vào là các đặc trưng văn bản được biểu diễn bằng TF-IDF. Các giá trị này không tuân theo phân phối chuẩn nên Gaussian Naive Bayes không phải là lựa chọn phù hợp.

### Bernoulli Naive Bayes

Bernoulli Naive Bayes chỉ quan tâm một từ có xuất hiện hay không xuất hiện trong văn bản.

Ví dụ:

| Từ khóa | Giá trị |
| ------- | ------- |
| free    | 1       |
| money   | 0       |
| click   | 1       |

Mô hình không xét số lần xuất hiện của từ.

Điều này dẫn đến việc hai email:

```text
Free money now
```

và

```text
Free free free free money now
```

được xem gần như tương đương mặc dù email thứ hai có dấu hiệu Spam mạnh hơn.

### Multinomial Naive Bayes

Multinomial Naive Bayes sử dụng tần suất xuất hiện của từ hoặc trọng số TF-IDF trong văn bản.

Ví dụ:

| Từ khóa | Giá trị |
| ------- | ------- |
| free    | 5       |
| money   | 3       |
| click   | 2       |

Nhờ đó mô hình khai thác được nhiều thông tin hơn về nội dung email.

Trong bài toán Spam Detection, các từ như:

```text
free
win
money
offer
prize
click
```

thường xuất hiện với tần suất cao trong email Spam. Việc sử dụng tần suất xuất hiện của từ giúp mô hình phân biệt Spam và Ham hiệu quả hơn.

### Kết luận

Trong dự án này, dữ liệu sau khi tiền xử lý được chuyển thành vector TF-IDF. Đây là dạng biểu diễn phổ biến trong bài toán phân loại văn bản.

Do đó:

- Gaussian Naive Bayes không phù hợp vì dữ liệu không tuân theo phân phối chuẩn.
- Bernoulli Naive Bayes làm mất thông tin về mức độ xuất hiện của từ.
- Multinomial Naive Bayes được thiết kế riêng cho dữ liệu văn bản và hoạt động hiệu quả với Bag of Words cũng như TF-IDF.

Vì vậy nhóm lựa chọn Multinomial Naive Bayes để xây dựng mô hình phân loại email Spam/Ham.

## 2. Định lý Bayes

Naive Bayes dựa trên công thức:

```text
P(C|X) = P(X|C) * P(C) / P(X)
```

Trong đó:

- `P(C|X)`: xác suất email thuộc lớp C khi biết dữ liệu X
- `P(X|C)`: xác suất xuất hiện dữ liệu X trong lớp C
- `P(C)`: xác suất tiên nghiệm của lớp C (Prior)
- `P(X)`: xác suất xuất hiện dữ liệu X

Do `P(X)` giống nhau với mọi lớp nên có thể bỏ qua khi so sánh.

Khi đó:

```text
P(C|X) ∝ P(X|C) * P(C)
```

---

## 3. Giả định Naive

Từ "Naive" xuất phát từ giả định:

```text
Các đặc trưng độc lập với nhau khi biết lớp.
```

Ví dụ:

```text
"free money now"
```

Naive Bayes giả định:

```text
P(free, money, now | Spam)

=

P(free | Spam)
*
P(money | Spam)
*
P(now | Spam)
```

Mặc dù giả định này không hoàn toàn đúng trong thực tế, nhưng thường cho kết quả tốt trong bài toán phân loại văn bản.

---

## 4. Prior Probability

Prior là xác suất xuất hiện của từng lớp trong tập huấn luyện.

Công thức:

```text
P(C) = Số mẫu thuộc lớp C / Tổng số mẫu
```

Trong code:

```python
self.prior[c] = X_c.shape[0] / n_samples
```

Ví dụ:

| Class | Samples |
| ----- | ------- |
| Ham   | 4825    |
| Spam  | 747     |

Tổng:

```text
5572
```

Prior:

```text
P(Ham) = 4825 / 5572

P(Spam) = 747 / 5572
```

---

## 5. Word Frequency

Đối với mỗi lớp, mô hình đếm tổng số lần xuất hiện của từng từ.

Code:

```python
word_count = X_c.sum(axis=0)
```

Ví dụ:

| Word  | Count in Spam |
| ----- | ------------- |
| free  | 1200          |
| win   | 850           |
| money | 900           |

Các giá trị này được dùng để tính xác suất xuất hiện của từ trong từng lớp.

---

## 6. Laplace Smoothing

Nếu một từ chưa từng xuất hiện trong lớp nào đó:

```text
P(word|class) = 0
```

Khi nhân các xác suất:

```text
0 × bất kỳ số nào = 0
```

Toàn bộ xác suất sẽ bằng 0.

Để tránh hiện tượng này, sử dụng Laplace Smoothing:

```text
P(word|class)

=

(word_count + alpha)

/

(total_words + alpha * vocabulary_size)
```

Trong đó:

- `alpha`: hệ số smoothing
- `word_count`: số lần xuất hiện của từ
- `total_words`: tổng số từ trong lớp

Code:

```python
self.word_prob[c] = (
    word_count + self.alpha
) / (
    total_words + self.alpha * n_words
)
```

---

## 7. Vai trò của Alpha

Alpha là tham số điều chỉnh mức độ làm mượt.

Ví dụ:

```text
alpha = 0
```

Không smoothing.

```text
alpha = 1
```

Laplace Smoothing chuẩn.

Nếu alpha quá lớn:

```text
Mọi từ có xác suất gần giống nhau.
```

Nếu alpha quá nhỏ:

```text
Dễ bị ảnh hưởng bởi dữ liệu hiếm.
```

Trong dự án:

```python
khởi tạo alpha = 1.0
```

---

## 8. Log Probability

Khi số lượng từ lớn:

```text
0.0001 × 0.0002 × 0.0003 × ...
```

có thể gây tràn số hoặc underflow.

Do đó chuyển sang log:

```text
log(a × b)

=

log(a) + log(b)
```

Code:

```python
log_prior = np.log(
    self.prior[c]
)
```

và:

```python
log_likelihood = np.sum(
    x * np.log(
        self.word_prob[c] + 1e-10
    )
)
```

---

## 9. Tính điểm cho từng lớp

Đối với mỗi email:

```text
Score(class)

=

log(P(class))

+

Σ count(word) × log(P(word|class))
```

Code:

```python
scores[c] = (
    log_prior +
    log_likelihood
)
```

Điểm số càng cao thì email càng có khả năng thuộc lớp đó.

---

## 10. Dự đoán

Sau khi tính score cho tất cả các lớp:

```python
max(scores, key=scores.get)
```

Mô hình chọn lớp có điểm cao nhất.

Ví dụ:

| Class | Score  |
| ----- | ------ |
| Ham   | -235.6 |
| Spam  | -187.3 |

Kết quả:

```text
Spam
```

vì:

```text
-187.3 > -235.6
```

---

## 11. Độ phức tạp

### Huấn luyện

```text
O(N × D)
```

Trong đó:

- N: số mẫu
- D: số đặc trưng

### Dự đoán

```text
O(C × D)
```

Trong đó:

- C: số lớp
- D: số đặc trưng

Do đó Naive Bayes có tốc độ rất nhanh.

---

## 12. Ưu điểm

- Dễ triển khai.
- Tốc độ huấn luyện nhanh.
- Tốc độ dự đoán nhanh.
- Hoạt động tốt với dữ liệu văn bản.
- Không yêu cầu tài nguyên tính toán lớn.

---

## 13. Nhược điểm

- Giả định các đặc trưng độc lập.
- Khó mô hình hóa mối quan hệ giữa các từ.
- Độ chính xác thường thấp hơn Logistic Regression hoặc SVM trên các bài toán phức tạp.

---

## 14. Vai trò trong dự án

Trong dự án Spam Detection, Multinomial Naive Bayes được xây dựng hoàn toàn bằng NumPy và sử dụng:

- TF-IDF Vectorization
- Prior Probability
- Laplace Smoothing
- Log Probability

Kết quả của mô hình được đánh giá bằng:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- TPR
- FPR

Sau đó được so sánh với Logistic Regression và Linear SVM để lựa chọn mô hình phù hợp nhất cho hệ thống phân loại email Spam/Ham.
