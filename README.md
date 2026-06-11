# 🤖 Ứng dụng Web Học Máy từ Jupyter Notebook sang Streamlit

Ứng dụng web này được chuyển đổi tự động và chuẩn hóa cấu trúc từ quy trình mô hình huấn luyện trên Notebook (.ipynb). Giao diện phân chia theo cơ chế Zoning (Sidebar điều phối tham số đầu vào, khu vực chính hiển thị các Dashboard phân tích độc lập).

## 📌 Các Tính Năng Chính
- **Sidebar**: Tải file dữ liệu gốc, cấu hình linh hoạt bài toán (Phân loại/Hồi quy), tùy chỉnh các siêu tham số nâng cao của mô hình (như số cây `n_estimators`, độ sâu `max_depth`...) và kích hoạt chạy pipeline duy nhất chỉ với 1 click.
- **Tab 1 - Tổng quan dữ liệu**: Hiển thị nhanh kích thước tệp dữ liệu, thống kê mô tả phân phối tổng quát của tập các thuộc tính đặc trưng.
- **Tab 2 - Trực quan hóa**: Tự động nhận diện định dạng dữ liệu (Số/Phân loại) nhằm vẽ biểu đồ phân phối Histogram hoặc Bar Plot trực quan dạng lưới 2x2.
- **Tab 3 - Kết quả kiểm định**: Hiển thị các chỉ số đo lường hiệu năng cốt lõi (Accuracy, F1-Score, Ma trận nhầm lẫn đối với Phân loại; R², RMSE, MAE đối với Hồi quy).
- **Tab 4 - Sử dụng mô hình**: Chế độ dự báo kép trực tuyến: Nhập tay trực tiếp từng thuộc tính qua Form động hoặc đẩy File dữ liệu Test hàng loạt và xuất File kết quả cuối cùng (.CSV).

## 🛠️ Hướng dẫn Cài đặt & Khởi chạy

**Bước 1:** Tải mã nguồn ứng dụng về thư mục máy tính cá nhân.

**Bước 2:** Cài đặt các môi trường thư viện phụ thuộc bằng terminal:
```bash
pip install -r requirements.txt
