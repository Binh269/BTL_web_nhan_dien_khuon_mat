import base64
import datetime
from mailbox import mbox
import os
import sqlite3
import time
from tkinter import Image
from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.views import View
from .models import Attendance, In4SV, Model_Phong, Model_ThietBi, Student
from django.contrib import messages
import json
import numpy as np
# from .models import Student 
from django.core.files.base import ContentFile
# from .models import FaceModel
import torch
import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from django.views.decorators.csrf import csrf_exempt
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from scipy.spatial.distance import cosine
# from .models import Student, Attendance
import pyodbc
from datetime import datetime
from unidecode import unidecode
from collections import Counter
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.contrib.sessions.models import Session
from .traning import train_new_student


@login_required(login_url="login")  
def index(request):
    return render(request, "index.html")

@login_required(login_url="login")  
def Lich(request):
	return render(request, 'lich.html')


@login_required(login_url="login")  
def Diemdanh(request):
	return render(request, 'diemdanh.html')

@login_required(login_url="login")  
def thongke_view(request):
    return render(request, "thongke.html")

@login_required(login_url="login")  
def index(request):
	PhongHoc = Model_Phong.objects.all()
	SoPhongDangHoc = Model_Phong.objects.filter(trangThaiPhong='Phòng đang học').count()
	SoPhongDangTrong = Model_Phong.objects.filter(trangThaiPhong='Phòng không học').count()
	SoPhongDangSua = Model_Phong.objects.filter(trangThaiPhong='Phòng đang sửa chữa').count()
	return render(request, 'index.html', {'PhongHoc': PhongHoc,'DangHoc': SoPhongDangHoc, 'DangTrong': SoPhongDangTrong, 'DangSua': SoPhongDangSua})

@login_required(login_url="login")  
def Phong(request):
	if request.method == "POST":
		Loai_MuonThem = request.POST.get("Loai_MuonThem")
		if Loai_MuonThem == 'Phong':
			maPhong = request.POST.get("maPhong")
			tenPhong = request.POST.get("tenPhong")
			trangThai = request.POST.get("trangThai")
			print('Muốn thực hiện thêm phòng, lưu phòng')
			if Model_Phong.objects.filter(maPhong=maPhong).exists():
				print('Mã phòng đã tồn tại. Không thể thực hiện thêm')
				messages.error(request, 'Mã Phòng đã tồn tại! Không thể thêm')
				return redirect('/phong')  
			else:
				create_Phong = Model_Phong(maPhong=maPhong, tenPhong=tenPhong, trangThaiPhong=trangThai)
				create_Phong.save()
		if Loai_MuonThem == 'thietBi':
			maThietBi = request.POST.get("maThietBi")
			tenThietBi = request.POST.get("tenThietBi")
			trangThaiThietBi = request.POST.get("trangThaiThietBi")
			MaPhong = request.POST.get("MaPhong")
			phong_hoc = Model_Phong.objects.get(maPhong=MaPhong)
			print('Muốn thực hiện thêm thiết bị, lưu thiết bị')
			if Model_ThietBi.objects.filter(maThietBi=maThietBi).exists():
				print('Mã thiết bị đã tồn tại đã tồn tại. Không thể thực hiện thêm')
				messages.error(request, 'Mã Thiết bị đã tồn tại! Không thể thêm')
				return redirect('/phong')  
			else:
				create_ThietBi = Model_ThietBi(maThietBi=maThietBi, tenThietBi=tenThietBi, trangThaiThietBi=trangThaiThietBi, phongHoc=phong_hoc)
				create_ThietBi.save()
		return redirect('/phong')
	PhongHoc = Model_Phong.objects.all()
	ThietBi = Model_ThietBi.objects.all()
	return render(request, 'phong.html', {'PhongHocs': PhongHoc, 'ThietBi':ThietBi})

def Phong_Delete(request, id):
	if request.method == 'POST':
		id_delete = Model_Phong.objects.get(pk=id)
		id_delete.delete()
		return redirect('/phong')

def Phong_Update(request, id):
	if request.method == "POST":
		edit_maPhong = request.POST.get('edit_maPhong')
		edit_tenPhong = request.POST.get('edit_tenPhong')
		edit_trangThai = request.POST.get('edit_trangThai')
		if edit_maPhong and edit_tenPhong:
			save_info = Model_Phong(id=id, maPhong=edit_maPhong, tenPhong=edit_tenPhong, trangThaiPhong=edit_trangThai)
			save_info.save()
			return redirect('/phong')

def ThietBi_Delete(request, id):
	if request.method == 'POST':
		id_delete = Model_ThietBi.objects.get(pk=id)
		id_delete.delete()
		return redirect('/phong')

def ThietBi_Update(request, id):
	if request.method == "POST":
		edit_maThietBi = request.POST.get('edit_maThietBi')
		edit_tenThietBi = request.POST.get('edit_tenThietBi')
		edit_trangThaiThietBi = request.POST.get('edit_trangThaiThietBi')
		MaPhong = request.POST.get("maThietBiInput__")
		phong_hoc = Model_Phong.objects.get(maPhong=MaPhong)
		print('---->', phong_hoc)
		print(f'Mã thiết bị: {edit_maThietBi}, tên thiết bị {edit_tenThietBi}, trạng thái: {edit_trangThaiThietBi} và ID: {id}')
		if edit_maThietBi and edit_tenThietBi:
			save_info = Model_ThietBi(id=id, maThietBi=edit_maThietBi, tenThietBi=edit_tenThietBi, trangThaiThietBi=edit_trangThaiThietBi, phongHoc=phong_hoc)
			save_info.save()
			return redirect('/phong')


def connect_to_db():
    return sqlite3.connect("db.sqlite3")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(device=device)
model.load_state_dict(torch.load('Show/models/face_recognition_model1.pth', map_location=device))
EMBEDDINGS_PATH = "Show/models/embeddings_data.pth"
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
DATASET_DIR = r'dataset'
dataset_embeddings = []
dataset_labels = []
for label in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, label)
    if os.path.isdir(person_dir):
        student_name = label.strip()
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            face = cv2.imread(image_path)
            if face is not None:
                boxes, _ = mtcnn.detect(face)
                if boxes is not None:
                    for box in boxes:
                        face_crop = face[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                        face_tensor = transform(face_crop).unsqueeze(0).to(device)
                        embedding = model(face_tensor).detach().cpu().numpy().flatten()
                        dataset_embeddings.append(embedding)
                        dataset_labels.append(student_name)
dataset_embeddings = np.array(dataset_embeddings)
dataset_labels = np.array(dataset_labels)

from threading import Lock
detected_label = []
label_lock = Lock()

def detect_face():
    global detected_label,is_capturing
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            with label_lock:
                detected_label = None
            for box in boxes:
                face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                face_tensor = transform(face).unsqueeze(0).to(device)
                face_embedding = model(face_tensor).detach().cpu().numpy().flatten()
                distances = [cosine(face_embedding, stored_embedding) for stored_embedding in dataset_embeddings]
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                label = dataset_labels[min_distance_idx] if min_distance < 0.2 else "Unknown"
                with label_lock:  
                    detected_label = label if label != "Unknown" else None
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                print(detected_label)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


def video_feed(request):
    """ Trả về stream video cho Django """
    return StreamingHttpResponse(detect_face(), content_type='multipart/x-mixed-replace; boundary=frame')
@csrf_exempt  
def diemdanh(request):
    global detected_label
    print("Nhận được yêu cầu điểm danh")
    print("Giá trị detected_label trước khi kiểm tra:", detected_label)
    with label_lock:
        if detected_label:
            print("Sinh viên nhận diện: " + detected_label)
            mssv = detected_label.strip().split('_')[0]
            time.sleep(3)
            conn = connect_to_db()
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT name FROM thongtin WHERE mssv = ?", (mssv,))
                student = cursor.fetchone()
                print(student)
                if student:
                    name = student[0]  # Vì fetchone() trả về tuple (name,)
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Định dạng thời gian
                    print(current_time)

                    cursor.execute("""
                        INSERT INTO diemdanh (mssv, time, status)
                        VALUES (?, ?, ?)
                    """, (mssv, current_time, 'success'))
                    conn.commit()

                    return JsonResponse({
                        'success': True,
                        'mssv': mssv,
                        'name': name,
                        'time': current_time,
                        'status': 'success',
                        'message': 'Đã điểm danh'
                    })

                else:
                    return JsonResponse({'success': False, 'message': 'Không tìm thấy sinh viên'})

            except Exception as e:
                print(f"Lỗi khi truy vấn cơ sở dữ liệu: {e}")
                return JsonResponse({'status': 'fail', 'message': 'Lỗi hệ thống'})
            finally:
                cursor.close()
                conn.close()

        else:
            return JsonResponse({'status': 'fail', 'message': 'Đang nhận diện khuôn mặt'})

def diemdanh_list(request):
    conn = connect_to_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT diemdanh.mssv, thongtin.name, thongtin.lop, thongtin.Khoa, diemdanh.time
            FROM diemdanh
            JOIN thongtin ON diemdanh.mssv = thongtin.mssv
            ORDER BY diemdanh.id
        """)
        attendances = cursor.fetchall()

        data = [
            {
                'mssv': row[0],
                'name': row[1],
                'lop': row[2],
                'Khoa': row[3],
                'time': row[4] 
            }
            for row in attendances
        ]
        return JsonResponse(data, safe=False)

    except Exception as e:
        print(f"Lỗi khi truy vấn dữ liệu điểm danh: {e}")
        return JsonResponse({'status': 'fail', 'message': 'Lỗi khi lấy dữ liệu điểm danh'})

    finally:
        cursor.close()
        conn.close()


@csrf_exempt
def them_sv(request):
    if request.method == "POST":
        mssv = request.POST.get("mssv")
        ten = request.POST.get("name")
        lop = request.POST.get("lop")
        khoa = request.POST.get("khoa")
        images = request.FILES.getlist("images")

        if not mssv or not ten or not lop or not khoa:
            return JsonResponse({"error": "Vui lòng điền đầy đủ thông tin"}, status=400)

        if not images:
            return JsonResponse({"error": "Vui lòng chọn ít nhất một ảnh"}, status=400)

        try:
            conn = connect_to_db()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM thongtin WHERE mssv = ?", (mssv,))
            if cursor.fetchone()[0] > 0:
                return JsonResponse({"error": "Mã số sinh viên đã tồn tại"}, status=400)

            cursor.execute(
                "INSERT INTO thongtin (mssv, name, lop, Khoa) VALUES (?, ?, ?, ?)",
                (mssv, ten, lop, khoa),
            )
            conn.commit()

            folder_name = f"{mssv}_{unidecode(ten).replace(' ', '')}_{unidecode(lop).replace(' ', '')}_{unidecode(khoa).replace(' ', '')}"
            student_path = os.path.join(DATASET_DIR, folder_name)
            os.makedirs(student_path, exist_ok=True)
            if not os.access(student_path, os.W_OK):
                return JsonResponse({"error": "Không có quyền ghi vào thư mục dataset"}, status=500)

            for idx, image in enumerate(images):
                image_path = os.path.join(student_path, f"{idx+1}.jpg")
                with open(image_path, "wb") as f:
                    for chunk in image.chunks():
                        f.write(chunk)

            print(f"Đã lưu {len(images)} ảnh vào {student_path}")

            print(f"Gọi train_new_student với folder_name: {folder_name}")
            success, message = train_new_student(folder_name)
            print(f"train_new_student trả về: success={success}, message={message}")
            if not success:
                return JsonResponse({"error": message}, status=400)

            return JsonResponse({"message": "Thêm sinh viên và cập nhật model thành công"})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({"error": f"Lỗi hệ thống: {str(e)}"}, status=500)

        finally:
            cursor.close()
            conn.close()

    return JsonResponse({"error": "Phương thức không hợp lệ"}, status=405)
@csrf_exempt
def add_folder(request):
    if request.method == "POST":
        dataset_path = "dataset"

        if not os.path.exists(dataset_path):
            return JsonResponse({"error": "Thư mục dataset không tồn tại"}, status=400)

        conn = connect_to_db()
        cursor = conn.cursor()

        added_students = []
        errors = []

        try:
            # Duyệt tất cả thư mục con trong dataset
            for folder_name in os.listdir(dataset_path):
                folder_path = os.path.join(dataset_path, folder_name)
                if not os.path.isdir(folder_path):  
                    continue  # Bỏ qua nếu không phải thư mục

                parts = folder_name.split("_")
                if len(parts) < 4:
                    errors.append(f"Thư mục '{folder_name}' có tên không đúng định dạng")
                    continue

                mssv, ten, lop, khoa = parts[0], parts[1], parts[2], "_".join(parts[3:])

                # Kiểm tra MSSV đã tồn tại chưa
                cursor.execute("SELECT COUNT(*) FROM thongtin WHERE mssv = ?", (mssv,))
                if cursor.fetchone()[0] > 0:
                    errors.append(f"MSSV {mssv} đã tồn tại, bỏ qua")
                    continue

                # Thêm sinh viên vào bảng thongtin
                cursor.execute("INSERT INTO thongtin (mssv, name, lop, Khoa) VALUES (?, ?, ?, ?)", 
                               (mssv, ten, lop, khoa))
                conn.commit()
                added_students.append({"mssv": mssv, "name": ten, "lop": lop, "khoa": khoa})

            return JsonResponse({"message": "Quét thư mục hoàn tất!", "added": added_students, "errors": errors})

        except Exception as e:
            return JsonResponse({"error": f"Lỗi: {str(e)}"}, status=500)

        finally:
            cursor.close()
            conn.close()

    return JsonResponse({"error": "Yêu cầu không hợp lệ"}, status=400)

def diemdanh_thongke(request):
    conn = connect_to_db()
    cursor = conn.cursor()

    try:
        # Truy vấn dữ liệu từ SQL Server
        cursor.execute("""
            SELECT dd.mssv, sv.name, sv.lop, sv.Khoa, dd.time
            FROM diemdanh dd
            JOIN thongtin sv ON dd.mssv = sv.mssv
            ORDER BY dd.id
        """)
        attendances = cursor.fetchall()

        print(attendances)  # Debug dữ liệu truy vấn

        # Chuyển đổi dữ liệu từ tuple sang dictionary
        data = []
        for row in attendances:
            try:
                # Xử lý giá trị thời gian
                time_value = row[4]
                if isinstance(time_value, str):
                    formatted_time = time_value  # Nếu là string thì giữ nguyên
                elif isinstance(time_value, datetime):
                    formatted_time = time_value.strftime('%Y-%m-%d %H:%M:%S')  # Chuyển về định dạng chuẩn
                else:
                    formatted_time = str(time_value)  # Chuyển đổi thành chuỗi nếu là kiểu khác

                data.append({
                    'mssv': row[0],
                    'name': row[1],
                    'lop': row[2].strip(),  # Xóa khoảng trắng dư thừa
                    'Khoa': row[3],
                    'time': formatted_time
                })
            except Exception as e:
                print("Lỗi xử lý dữ liệu:", e)
                return JsonResponse({'status': 'fail', 'message': f'Lỗi dữ liệu: {e}'})

        # Đếm số lần điểm danh theo MSSV (Sinh viên)
        counter_mssv = Counter([item['mssv'] for item in data])

        # Đếm số lần điểm danh theo ngày (tách phần ngày từ 'time')
        counter_date = Counter([item['time'].split()[0] for item in data])

        # Chuẩn bị dữ liệu cho biểu đồ
        response_data = {
            "labels_mssv": list(counter_mssv.keys()),
            "counts_mssv": list(counter_mssv.values()),
            "labels_date": list(counter_date.keys()),
            "counts_date": list(counter_date.values()),
            "details": data  # Trả về danh sách chi tiết nếu cần
        }

        return JsonResponse(response_data, safe=False)

    except Exception as e:
        print(f"Lỗi khi truy vấn dữ liệu điểm danh: {e}")
        return JsonResponse({'status': 'fail', 'message': 'Lỗi khi lấy dữ liệu điểm danh'})

    finally:
        cursor.close()
        conn.close()

def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        if User.objects.filter(username=username).exists():
            messages.error(request, "Tên đăng nhập đã tồn tại!")
        else:
            user = User.objects.create_user(username=username, password=password)
            user.save()
            messages.success(request, "Đăng ký thành công! Hãy đăng nhập.")
            return redirect("login")  # Điều hướng về trang đăng nhập
    return render(request, "register.html")

def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            request.session["user_id"] = user.id  # Lưu user vào session
            return redirect("index")  # Điều hướng về trang chủ
        else:
            messages.error(request, "Sai tên đăng nhập hoặc mật khẩu!")
    return render(request, "login.html")

def logout_view(request):
    """Xóa session khi người dùng đăng xuất"""
    logout(request)  # 
    request.session.flush()  
    Session.objects.all().delete()
    return redirect('login')

def clear_all_sessions():
    """Xóa toàn bộ session khi server restart"""
    print("🔥 Server restart - Xóa tất cả session!")
    Session.objects.all().delete()

def manual_open(request):
    if request.method == "POST":
        mssv = request.POST.get("manual_mssv")
        username = request.POST.get("username")
        password = request.POST.get("password")

        # Xóa khoảng trắng, kiểm tra đầu vào
        if not mssv or not username or not password:
            return JsonResponse({"error": "Thiếu thông tin đăng nhập hoặc MSSV!"}, status=400)

        mssv = mssv.strip()  # Xóa khoảng trắng nếu có
        print(f"📌 MSSV nhận được từ request: '{mssv}'")  # Debug giá trị MSSV

        # Kiểm tra thông tin đăng nhập
        user = authenticate(username=username, password=password)
        if user is None:
            return JsonResponse({"error": "Tài khoản hoặc mật khẩu không chính xác!"}, status=400)

        conn = connect_to_db()
        cursor = conn.cursor()

        # Kiểm tra MSSV trong database
        cursor.execute("SELECT mssv FROM thongtin WHERE mssv = ?", (mssv,))
        sinh_vien = cursor.fetchone()
        print(f"📌 Kết quả truy vấn SQLite: {sinh_vien}")  # Debug kết quả

        if not sinh_vien:
            conn.close()
            return JsonResponse({"error": "MSSV không tồn tại trong hệ thống!"}, status=400)

        # Lưu điểm danh vào SQLite
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO diemdanh (mssv, time) VALUES (?, ?)", (mssv, time_now))

        conn.commit()
        conn.close()

        return JsonResponse({"message": "✅ Đã mở cửa thành công!"})

    return JsonResponse({"error": "Yêu cầu không hợp lệ!"}, status=400)


# import base64
# import datetime
# from mailbox import mbox
# import os
# import sqlite3
# import time
# from tkinter import Image
# from django.conf import settings
# from django.http import JsonResponse, StreamingHttpResponse
# from django.shortcuts import render, redirect
# from django.views import View
# from .models import Attendance, In4SV, Model_Phong, Model_ThietBi, Student
# from django.contrib import messages
# import json
# import numpy as np
# import torch
# import cv2
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from django.views.decorators.csrf import csrf_exempt
# from PIL import Image
# from scipy.spatial.distance import cosine
# import pyodbc
# from datetime import datetime
# from unidecode import unidecode
# from collections import Counter
# from django.contrib.auth.models import User
# from django.contrib.auth import login, authenticate
# from django.contrib.auth import logout
# from django.contrib.auth.decorators import login_required
# from django.contrib.sessions.models import Session
# from threading import Lock

# # Định nghĩa mô hình CNN sâu hơn với residual connections
# class DeepFaceRecognitionCNN(nn.Module):
#     def __init__(self):
#         super(DeepFaceRecognitionCNN, self).__init__()
        
#         # Block 1
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
        
#         # Block 2
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
        
#         # Block 3
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn6 = nn.BatchNorm2d(256)
        
#         # Block 4
#         self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn7 = nn.BatchNorm2d(512)
#         self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn8 = nn.BatchNorm2d(512)
        
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(512 * 10 * 10, 2048)  # Tăng số neuron để học đặc trưng sâu hơn
#         self.fc2 = nn.Linear(2048, 1024)
#         self.fc3 = nn.Linear(1024, 128)  # Embedding size vẫn là 128
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         # Block 1 với residual connection
#         identity = x
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.bn2(self.conv2(x))
#         x += identity  # Residual connection
#         x = F.relu(x)
#         x = self.pool(x)
        
#         # Block 2
#         identity = F.interpolate(self.conv3(x), size=x.shape[2:], mode='nearest')  # Điều chỉnh kích thước identity
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.bn4(self.conv4(x))
#         x += identity
#         x = F.relu(x)
#         x = self.pool(x)
        
#         # Block 3
#         identity = F.interpolate(self.conv5(x), size=x.shape[2:], mode='nearest')
#         x = F.relu(self.bn5(self.conv5(x)))
#         x = self.bn6(self.conv6(x))
#         x += identity
#         x = F.relu(x)
#         x = self.pool(x)
        
#         # Block 4
#         identity = F.interpolate(self.conv7(x), size=x.shape[2:], mode='nearest')
#         x = F.relu(self.bn7(self.conv7(x)))
#         x = self.bn8(self.conv8(x))
#         x += identity
#         x = F.relu(x)
#         x = self.pool(x)
        
#         x = x.view(-1, 512 * 10 * 10)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # Hàm tiền xử lý cải tiến
# def detect_and_crop_face(image):
#     # Cải thiện ảnh bằng histogram equalization
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
#     face_list = []
#     box_list = []
#     for (x, y, w, h) in faces:
#         face_crop = image[y:y+h, x:x+w]
#         face_crop = cv2.resize(face_crop, (160, 160))
#         face_crop = face_crop.astype(np.float32) / 127.5 - 1
#         face_list.append(face_crop)
#         box_list.append((x, y, x+w, y+h))
#     return face_list, box_list

# def preprocess_image(image):
#     return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

# # Khởi tạo mô hình và device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DeepFaceRecognitionCNN().to(device)
# MODEL_PATH = 'Show/models/deep_face_recognition_model.pth'  # Đổi tên file để tránh xung đột với mô hình cũ
# EMBEDDINGS_PATH = 'Show/models/deep_embeddings_data.pth'
# DATASET_DIR = r'dataset'

# # Tải mô hình từ file đã huấn luyện
# if os.path.exists(MODEL_PATH):
#     try:
#         model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#         print(f"Đã tải mô hình từ {MODEL_PATH}")
#     except RuntimeError as e:
#         print(f"Lỗi khi tải state_dict: {e}")
#         print("Vui lòng huấn luyện lại mô hình bằng train_face_model.py")
# else:
#     print(f"Không tìm thấy file {MODEL_PATH}. Vui lòng huấn luyện mô hình trước!")
#     raise FileNotFoundError("Model file not found. Please train the model first.")

# model.eval()

# # Tạo hoặc tải embedding cho tập dữ liệu
# dataset_embeddings = []
# dataset_labels = []

# if os.path.exists(EMBEDDINGS_PATH):
#     data = torch.load(EMBEDDINGS_PATH, map_location='cpu')
#     dataset_embeddings = data['embeddings']
#     dataset_labels = data['labels']
#     print(f"Đã tải {len(dataset_labels)} embedding từ {EMBEDDINGS_PATH}")
# else:
#     print(f"Không tìm thấy file embedding {EMBEDDINGS_PATH}. Tạo mới embedding từ dataset...")
#     for label in os.listdir(DATASET_DIR):
#         person_dir = os.path.join(DATASET_DIR, label)
#         if os.path.isdir(person_dir):
#             student_name = label.strip()
#             for image_name in os.listdir(person_dir):
#                 image_path = os.path.join(person_dir, image_name)
#                 image = cv2.imread(image_path)
#                 if image is not None:
#                     face_crops, _ = detect_and_crop_face(image)
#                     for face_crop in face_crops:
#                         face_tensor = preprocess_image(face_crop).to(device)
#                         with torch.no_grad():
#                             embedding = model(face_tensor).cpu().numpy().flatten()
#                         dataset_embeddings.append(embedding)
#                         dataset_labels.append(student_name)
#     dataset_embeddings = np.array(dataset_embeddings)
#     dataset_labels = np.array(dataset_labels)
#     torch.save({'embeddings': dataset_embeddings, 'labels': dataset_labels}, EMBEDDINGS_PATH)
#     print(f"Đã tạo và lưu {len(dataset_labels)} embedding vào {EMBEDDINGS_PATH}")

# # Nhận diện khuôn mặt trong video
# detected_labels = []
# label_lock = Lock()

# def detect_face():
#     global detected_labels
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue
        
#         face_crops, boxes = detect_and_crop_face(frame)
#         current_labels = []
        
#         if face_crops:
#             with label_lock:
#                 detected_labels = []
            
#             # Xử lý batch embedding để tăng tốc
#             face_tensors = torch.cat([preprocess_image(face_crop) for face_crop in face_crops], dim=0).to(device)
#             with torch.no_grad():
#                 face_embeddings = model(face_tensors).cpu().numpy()
            
#             for face_embedding, box in zip(face_embeddings, boxes):
#                 distances = [cosine(face_embedding, stored_embedding) for stored_embedding in dataset_embeddings]
#                 min_distance_idx = np.argmin(distances)
#                 min_distance = distances[min_distance_idx]
                
#                 # Ngưỡng động: dựa trên trung bình khoảng cách
#                 mean_distance = np.mean(distances)
#                 threshold = max(0.4, mean_distance * 0.8)  # Ngưỡng linh hoạt
#                 label = dataset_labels[min_distance_idx] if min_distance < threshold else "Unknown"
                
#                 print(f"Face at {box}: Min distance: {min_distance:.4f}, Mean distance: {mean_distance:.4f}, Threshold: {threshold:.4f}, Predicted label: {label}")
#                 for i, (dist, lbl) in enumerate(zip(distances, dataset_labels)):
#                     print(f"Distance to {lbl}: {dist:.4f}")

#                 current_labels.append(label if label != "Unknown" else None)
                
#                 x1, y1, x2, y2 = box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{label} ({min_distance:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#             with label_lock:
#                 detected_labels = current_labels
        
#         _, buffer = cv2.imencode('.jpg', frame)
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

#     cap.release()

# def video_feed(request):
#     return StreamingHttpResponse(detect_face(), content_type='multipart/x-mixed-replace; boundary=frame')

# # Các view khác (giữ nguyên như trước)
# @login_required(login_url="login")  
# def index(request):
#     PhongHoc = Model_Phong.objects.all()
#     SoPhongDangHoc = Model_Phong.objects.filter(trangThaiPhong='Phòng đang học').count()
#     SoPhongDangTrong = Model_Phong.objects.filter(trangThaiPhong='Phòng không học').count()
#     SoPhongDangSua = Model_Phong.objects.filter(trangThaiPhong='Phòng đang sửa chữa').count()
#     return render(request, 'index.html', {
#         'PhongHoc': PhongHoc,
#         'DangHoc': SoPhongDangHoc,
#         'DangTrong': SoPhongDangTrong,
#         'DangSua': SoPhongDangSua
#     })

# @login_required(login_url="login")  
# def Lich(request):
#     return render(request, 'lich.html')

# @login_required(login_url="login")  
# def Diemdanh(request):
#     return render(request, 'diemdanh.html')

# @login_required(login_url="login")  
# def thongke_view(request):
#     return render(request, "thongke.html")

# @login_required(login_url="login")  
# def Phong(request):
#     if request.method == "POST":
#         Loai_MuonThem = request.POST.get("Loai_MuonThem")
#         if Loai_MuonThem == 'Phong':
#             maPhong = request.POST.get("maPhong")
#             tenPhong = request.POST.get("tenPhong")
#             trangThai = request.POST.get("trangThai")
#             if Model_Phong.objects.filter(maPhong=maPhong).exists():
#                 messages.error(request, 'Mã Phòng đã tồn tại! Không thể thêm')
#                 return redirect('/phong')  
#             else:
#                 create_Phong = Model_Phong(maPhong=maPhong, tenPhong=tenPhong, trangThaiPhong=trangThai)
#                 create_Phong.save()
#         if Loai_MuonThem == 'thietBi':
#             maThietBi = request.POST.get("maThietBi")
#             tenThietBi = request.POST.get("tenThietBi")
#             trangThaiThietBi = request.POST.get("trangThaiThietBi")
#             MaPhong = request.POST.get("MaPhong")
#             phong_hoc = Model_Phong.objects.get(maPhong=MaPhong)
#             if Model_ThietBi.objects.filter(maThietBi=maThietBi).exists():
#                 messages.error(request, 'Mã Thiết bị đã tồn tại! Không thể thêm')
#                 return redirect('/phong')  
#             else:
#                 create_ThietBi = Model_ThietBi(maThietBi=maThietBi, tenThietBi=tenThietBi, trangThaiThietBi=trangThaiThietBi, phongHoc=phong_hoc)
#                 create_ThietBi.save()
#         return redirect('/phong')
#     PhongHoc = Model_Phong.objects.all()
#     ThietBi = Model_ThietBi.objects.all()
#     return render(request, 'phong.html', {'PhongHocs': PhongHoc, 'ThietBi': ThietBi})

# def Phong_Delete(request, id):
#     if request.method == 'POST':
#         id_delete = Model_Phong.objects.get(pk=id)
#         id_delete.delete()
#         return redirect('/phong')

# def Phong_Update(request, id):
#     if request.method == "POST":
#         edit_maPhong = request.POST.get('edit_maPhong')
#         edit_tenPhong = request.POST.get('edit_tenPhong')
#         edit_trangThai = request.POST.get('edit_trangThai')
#         if edit_maPhong and edit_tenPhong:
#             save_info = Model_Phong(id=id, maPhong=edit_maPhong, tenPhong=edit_tenPhong, trangThaiPhong=edit_trangThai)
#             save_info.save()
#             return redirect('/phong')

# def ThietBi_Delete(request, id):
#     if request.method == 'POST':
#         id_delete = Model_ThietBi.objects.get(pk=id)
#         id_delete.delete()
#         return redirect('/phong')

# def ThietBi_Update(request, id):
#     if request.method == "POST":
#         edit_maThietBi = request.POST.get('edit_maThietBi')
#         edit_tenThietBi = request.POST.get('edit_tenThietBi')
#         edit_trangThaiThietBi = request.POST.get('edit_trangThaiThietBi')
#         MaPhong = request.POST.get("maThietBiInput__")
#         phong_hoc = Model_Phong.objects.get(maPhong=MaPhong)
#         if edit_maThietBi and edit_tenThietBi:
#             save_info = Model_ThietBi(id=id, maThietBi=edit_maThietBi, tenThietBi=edit_tenThietBi, trangThaiThietBi=edit_trangThaiThietBi, phongHoc=phong_hoc)
#             save_info.save()
#             return redirect('/phong')

# def connect_to_db():
#     return sqlite3.connect("db.sqlite3")

# @csrf_exempt  
# def diemdanh(request):
#     global detected_labels
#     print("Nhận được yêu cầu điểm danh")
#     print("Giá trị detected_labels trước khi kiểm tra:", detected_labels)
#     with label_lock:
#         if detected_labels and any(label is not None for label in detected_labels):
#             recognized_labels = [label for label in detected_labels if label is not None]
#             responses = []
#             conn = connect_to_db()
#             cursor = conn.cursor()
#             try:
#                 for label in recognized_labels:
#                     print("Sinh viên nhận diện: " + label)
#                     mssv = label.strip().split('_')[0]
#                     time.sleep(5)
#                     cursor.execute("SELECT name FROM thongtin WHERE mssv = ?", (mssv,))
#                     student = cursor.fetchone()
#                     if student:
#                         name = student[0]
#                         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                         cursor.execute("INSERT INTO diemdanh (mssv, time, status) VALUES (?, ?, ?)", (mssv, current_time, 'success'))
#                         conn.commit()
#                         responses.append({
#                             'mssv': mssv,
#                             'name': name,
#                             'time': current_time,
#                             'status': 'success',
#                             'message': 'Đã điểm danh'
#                         })
#                     else:
#                         responses.append({'status': 'fail', 'message': f'Không tìm thấy sinh viên {mssv}'})
#                 return JsonResponse(responses, safe=False) if len(responses) > 1 else JsonResponse(responses[0])
#             except Exception as e:
#                 print(f"Lỗi khi truy vấn cơ sở dữ liệu: {e}")
#                 return JsonResponse({'status': 'fail', 'message': 'Lỗi hệ thống'})
#             finally:
#                 cursor.close()
#                 conn.close()
#         else:
#             print("Không nhận diện được khuôn mặt nào hoặc tất cả là Unknown")
#             return JsonResponse({'status': 'fail', 'message': 'Đang nhận diện khuôn mặt'})

# def diemdanh_list(request):
#     conn = connect_to_db()
#     cursor = conn.cursor()
#     try:
#         cursor.execute("""
#             SELECT diemdanh.mssv, thongtin.name, thongtin.lop, thongtin.Khoa, diemdanh.time
#             FROM diemdanh
#             JOIN thongtin ON diemdanh.mssv = thongtin.mssv
#             ORDER BY diemdanh.id
#         """)
#         attendances = cursor.fetchall()
#         data = [
#             {
#                 'mssv': row[0],
#                 'name': row[1],
#                 'lop': row[2],
#                 'Khoa': row[3],
#                 'time': row[4]
#             }
#             for row in attendances
#         ]
#         return JsonResponse(data, safe=False)
#     except Exception as e:
#         print(f"Lỗi khi truy vấn dữ liệu điểm danh: {e}")
#         return JsonResponse({'status': 'fail', 'message': 'Lỗi khi lấy dữ liệu điểm danh'})
#     finally:
#         cursor.close()
#         conn.close()

# @csrf_exempt
# def them_sv(request):
#     if request.method == "POST":
#         mssv = request.POST.get("mssv")
#         ten = request.POST.get("name")
#         lop = request.POST.get("lop")
#         khoa = request.POST.get("khoa")
#         images = request.FILES.getlist("images")

#         if not mssv or not ten or not lop or not khoa:
#             return JsonResponse({"error": "Vui lòng điền đầy đủ thông tin"}, status=400)

#         if not images:
#             return JsonResponse({"error": "Vui lòng chọn ít nhất một ảnh"}, status=400)

#         try:
#             conn = connect_to_db()
#             cursor = conn.cursor()
#             cursor.execute("SELECT COUNT(*) FROM thongtin WHERE mssv = ?", (mssv,))
#             if cursor.fetchone()[0] > 0:
#                 return JsonResponse({"error": "Mã số sinh viên đã tồn tại"}, status=400)

#             cursor.execute("INSERT INTO thongtin (mssv, name, lop, Khoa) VALUES (?, ?, ?, ?)", (mssv, ten, lop, khoa))
#             conn.commit()

#             folder_name = f"{mssv}_{unidecode(ten).replace(' ', '')}_{unidecode(lop).replace(' ', '')}_{unidecode(khoa).replace(' ', '')}"
#             student_path = os.path.join(DATASET_DIR, folder_name)
#             os.makedirs(student_path, exist_ok=True)

#             for idx, image in enumerate(images):
#                 image_path = os.path.join(student_path, f"{idx+1}.jpg")
#                 with open(image_path, "wb") as f:
#                     for chunk in image.chunks():
#                         f.write(chunk)

#             print(f"Đã lưu {len(images)} ảnh vào {student_path}")
#             print("Vui lòng chạy lại train_face_model.py để huấn luyện mô hình với dữ liệu mới!")

#             return JsonResponse({"message": "Thêm sinh viên thành công. Chạy lại train_face_model.py để cập nhật mô hình."})
#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)
#         finally:
#             cursor.close()
#             conn.close()

#     return JsonResponse({"error": "Phương thức không hợp lệ"}, status=405)

# @csrf_exempt
# def add_folder(request):
#     if request.method == "POST":
#         dataset_path = "dataset"
#         if not os.path.exists(dataset_path):
#             return JsonResponse({"error": "Thư mục dataset không tồn tại"}, status=400)

#         conn = connect_to_db()
#         cursor = conn.cursor()
#         added_students = []
#         errors = []

#         try:
#             for folder_name in os.listdir(dataset_path):
#                 folder_path = os.path.join(dataset_path, folder_name)
#                 if not os.path.isdir(folder_path):
#                     continue

#                 parts = folder_name.split("_")
#                 if len(parts) < 4:
#                     errors.append(f"Thư mục '{folder_name}' có tên không đúng định dạng")
#                     continue

#                 mssv, ten, lop, khoa = parts[0], parts[1], parts[2], "_".join(parts[3:])
#                 cursor.execute("SELECT COUNT(*) FROM thongtin WHERE mssv = ?", (mssv,))
#                 if cursor.fetchone()[0] > 0:
#                     errors.append(f"MSSV {mssv} đã tồn tại, bỏ qua")
#                     continue

#                 cursor.execute("INSERT INTO thongtin (mssv, name, lop, Khoa) VALUES (?, ?, ?, ?)", (mssv, ten, lop, khoa))
#                 conn.commit()
#                 added_students.append({"mssv": mssv, "name": ten, "lop": lop, "khoa": khoa})

#             return JsonResponse({"message": "Quét thư mục hoàn tất! Chạy lại train_face_model.py để huấn luyện mô hình.", "added": added_students, "errors": errors})
#         except Exception as e:
#             return JsonResponse({"error": f"Lỗi: {str(e)}"}, status=500)
#         finally:
#             cursor.close()
#             conn.close()

#     return JsonResponse({"error": "Yêu cầu không hợp lệ"}, status=400)

# def diemdanh_thongke(request):
#     conn = connect_to_db()
#     cursor = conn.cursor()
#     try:
#         cursor.execute("""
#             SELECT dd.mssv, sv.name, sv.lop, sv.Khoa, dd.time
#             FROM diemdanh dd
#             JOIN thongtin sv ON dd.mssv = sv.mssv
#             ORDER BY dd.id
#         """)
#         attendances = cursor.fetchall()

#         data = []
#         for row in attendances:
#             time_value = row[4]
#             if isinstance(time_value, str):
#                 formatted_time = time_value
#             elif isinstance(time_value, datetime):
#                 formatted_time = time_value.strftime('%Y-%m-%d %H:%M:%S')
#             else:
#                 formatted_time = str(time_value)

#             data.append({
#                 'mssv': row[0],
#                 'name': row[1],
#                 'lop': row[2].strip(),
#                 'Khoa': row[3],
#                 'time': formatted_time
#             })

#         counter_mssv = Counter([item['mssv'] for item in data])
#         counter_date = Counter([item['time'].split()[0] for item in data])

#         response_data = {
#             "labels_mssv": list(counter_mssv.keys()),
#             "counts_mssv": list(counter_mssv.values()),
#             "labels_date": list(counter_date.keys()),
#             "counts_date": list(counter_date.values()),
#             "details": data
#         }
#         return JsonResponse(response_data, safe=False)
#     except Exception as e:
#         print(f"Lỗi khi truy vấn dữ liệu điểm danh: {e}")
#         return JsonResponse({'status': 'fail', 'message': 'Lỗi khi lấy dữ liệu điểm danh'})
#     finally:
#         cursor.close()
#         conn.close()

# def register(request):
#     if request.method == "POST":
#         username = request.POST["username"]
#         password = request.POST["password"]
#         if User.objects.filter(username=username).exists():
#             messages.error(request, "Tên đăng nhập đã tồn tại!")
#         else:
#             user = User.objects.create_user(username=username, password=password)
#             user.save()
#             messages.success(request, "Đăng ký thành công! Hãy đăng nhập.")
#             return redirect("login")
#     return render(request, "register.html")

# def login_view(request):
#     if request.method == "POST":
#         username = request.POST["username"]
#         password = request.POST["password"]
#         user = authenticate(request, username=username, password=password)
#         if user:
#             login(request, user)
#             request.session["user_id"] = user.id
#             return redirect("index")
#         else:
#             messages.error(request, "Sai tên đăng nhập hoặc mật khẩu!")
#     return render(request, "login.html")

# def logout_view(request):
#     logout(request)
#     request.session.flush()
#     Session.objects.all().delete()
#     return redirect('login')

# def clear_all_sessions():
#     print("🔥 Server restart - Xóa tất cả session!")
#     Session.objects.all().delete()

# def manual_open(request):
#     if request.method == "POST":
#         mssv = request.POST.get("manual_mssv")
#         username = request.POST.get("username")
#         password = request.POST.get("password")

#         if not mssv or not username or not password:
#             return JsonResponse({"error": "Thiếu thông tin đăng nhập hoặc MSSV!"}, status=400)

#         mssv = mssv.strip()
#         user = authenticate(username=username, password=password)
#         if user is None:
#             return JsonResponse({"error": "Tài khoản hoặc mật khẩu không chính xác!"}, status=400)

#         conn = connect_to_db()
#         cursor = conn.cursor()
#         cursor.execute("SELECT mssv FROM thongtin WHERE mssv = ?", (mssv,))
#         sinh_vien = cursor.fetchone()

#         if not sinh_vien:
#             conn.close()
#             return JsonResponse({"error": "MSSV không tồn tại trong hệ thống!"}, status=400)

#         time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         cursor.execute("INSERT INTO diemdanh (mssv, time) VALUES (?, ?)", (mssv, time_now))
#         conn.commit()
#         conn.close()

#         return JsonResponse({"message": "✅ Đã mở cửa thành công!"})
#     return JsonResponse({"error": "Yêu cầu không hợp lệ!"}, status=400)