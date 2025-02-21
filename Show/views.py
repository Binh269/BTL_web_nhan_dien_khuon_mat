import base64
import datetime
from mailbox import mbox
import os
import sqlite3
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


def index(request):
	PhongHoc = Model_Phong.objects.all()
	SoPhongDangHoc = Model_Phong.objects.filter(trangThaiPhong='Phòng đang học').count()
	SoPhongDangTrong = Model_Phong.objects.filter(trangThaiPhong='Phòng không học').count()
	SoPhongDangSua = Model_Phong.objects.filter(trangThaiPhong='Phòng đang sửa chữa').count()
	return render(request, 'index.html', {'PhongHoc': PhongHoc,'DangHoc': SoPhongDangHoc, 'DangTrong': SoPhongDangTrong, 'DangSua': SoPhongDangSua})

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

def Lich(request):
	return render(request, 'lich.html')
def Diemdanh(request):
	return render(request, 'diemdanh.html')



def connect_to_db():
    return sqlite3.connect("db.sqlite3")

device = torch.device("cuda")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(device=device)
model.load_state_dict(torch.load('Show/models/face_recognition_model1.pth', map_location=device))
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
DATASET_DIR = r'E:\BTL_Nhan_Dien_Khuon_Mat\BT_IoT_CV\dataset'
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
detected_label = None

from threading import Lock
detected_label = []
label_lock = Lock()

def detect_face():
    global detected_label
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
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
                label = dataset_labels[min_distance_idx] if min_distance < 0.5 else "Unknown"
                with label_lock:  
                    detected_label = label if label != "Unknown" else None
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                print(detected_label)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
def video_feed(request):
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

            conn = connect_to_db()
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT name FROM thongtin WHERE mssv = ?", (mssv,))
                student = cursor.fetchone()
                
                if student:
                    name = student[0]  # Vì fetchone() trả về tuple (name,)
                    print(f"Đã tìm thấy sinh viên: {name}, MSSV: {mssv}")
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Định dạng thời gian

                    cursor.execute("""
                        INSERT INTO diemdanh (mssv, time, status)
                        VALUES (?, ?, ?)
                    """, (mssv, current_time, 'success'))
                    conn.commit()

                    return JsonResponse({
                        'mssv': mssv,
                        'name': name,
                        'time': current_time,
                        'status': 'success',
                        'message': 'Đã điểm danh'
                    })
                else:
                    print("Không tìm thấy sinh viên")
                    return JsonResponse({'status': 'fail', 'message': 'Không tìm thấy sinh viên'})

            except Exception as e:
                print(f"Lỗi khi truy vấn cơ sở dữ liệu: {e}")
                return JsonResponse({'status': 'fail', 'message': 'Lỗi hệ thống'})
            finally:
                cursor.close()
                conn.close()

        else:
            print("Không nhận diện được khuôn mặt")
            return JsonResponse({'status': 'fail', 'message': 'Không nhận diện được khuôn mặt'})

def diemdanh_list(request):
    conn = connect_to_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT diemdanh.mssv, thongtin.name, thongtin.lop, thongtin.Khoa, 
                   strftime('%d/%m/%Y %H:%M:%S', diemdanh.time) AS formatted_time
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
                'time': row[4]  # Đã được format đúng định dạng ngày giờ
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
    if request.method == 'POST':
        mssv = request.POST.get('mssv')
        ten = request.POST.get('name')
        lop = request.POST.get('lop')
        khoa = request.POST.get('khoa')

        if not mssv or not ten or not lop or not khoa:
            return JsonResponse({'error': 'Vui lòng điền đầy đủ thông tin'}, status=400)

        try:
            conn = connect_to_db()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM thongtin WHERE mssv = ?", (mssv,))
            if cursor.fetchone()[0] > 0:
                return JsonResponse({'error': 'Mã số sinh viên đã tồn tại'}, status=400)

            cursor.execute("INSERT INTO thongtin (mssv, name, lop, Khoa) VALUES (?, ?, ?, ?)", (mssv, ten, lop, khoa))
            conn.commit()

            return JsonResponse({'message': 'Thêm sinh viên thành công'})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

        finally:
            cursor.close()
            conn.close()

    return JsonResponse({'error': 'Phương thức không hợp lệ'}, status=405)

@csrf_exempt
def add_folder(request):
    if request.method == "POST":
        dataset_path = "E:/IOT/project_Binh/project_Binh/dataset"

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
    connection = connect_to_db()
    cursor = connection.cursor()

    try:
        # Truy vấn dữ liệu từ SQL Server
        cursor.execute("""
            SELECT dd.mssv, sv.name, sv.lop, sv.Khoa, dd.time
            FROM diemdanh dd
            JOIN thongtin sv ON dd.mssv = sv.mssv
            ORDER BY dd.id
        """)
        attendances = cursor.fetchall()

        # Chuyển đổi dữ liệu từ tuple sang dictionary
        data = [
            {
                'mssv': row[0],
                'name': row[1],
                'lop': row[2],
                'Khoa': row[3],
                'time': row[4].strftime('%d/%m/%Y %H:%M:%S')  # Định dạng ngày giờ
            }
            for row in attendances
        ]

        # Đếm số lần điểm danh theo MSSV (Sinh viên)
        counter_mssv = Counter([item['mssv'] for item in data])

        # Đếm số lần điểm danh theo ngày
        counter_date = Counter([item['time'].split()[0] for item in data])  # Lấy phần ngày (dd/mm/yyyy)

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
        connection.close()

    return JsonResponse(response_data)

def thongke_view(request):
    return render(request, "thongke.html")