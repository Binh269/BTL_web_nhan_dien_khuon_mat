import os
from models import Student

# Thay đổi đường dẫn này thành thư mục chứa dataset của bạn
DATASET_DIR = "project_Binh\project_Binh\dataset"

for folder_name in os.listdir(DATASET_DIR):
    parts = folder_name.split("_")
    
    if len(parts) != 4:
        print(f"Bỏ qua thư mục không hợp lệ: {folder_name}")
        continue

    MSSV, name, class_name, khoa = parts

    # Thêm dữ liệu vào SQLite nếu chưa tồn tại
    student, created = Student.objects.get_or_create(
        MSSV=MSSV,
        defaults={"name": name, "class_name": class_name, "khoa": khoa}
    )

    if created:
        print(f"Thêm sinh viên: {folder_name}")
    else:
        print(f"Sinh viên đã tồn tại: {folder_name}")
