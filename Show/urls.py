from django.urls import path
from .views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("register/", register, name="register"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    
    path('', index, name='index'),
    path('phong', Phong, name='phong'),
    path('phong_Delete/<int:id>/', Phong_Delete, name='phong_Delete'),
    path('phong_Update/<str:id>', Phong_Update, name='phong_Update'),
    path('thietbi_Delete/<int:id>/', ThietBi_Delete, name='thietbi_Delete'),
    path('thietbi_Update/<str:id>', ThietBi_Update, name='thietbi_Update'),
    path('lich', Lich, name='lich'),

    path('diemdanh/', Diemdanh, name='diemdanh'),
    path('video_feed/', video_feed, name='video_feed'),
    path('mark_attendance/', diemdanh, name='mark_attendance'),
    path('diemdanh_list/', diemdanh_list, name='diemdanh_list'),
    path('them_sv/', them_sv, name='them_sv'),
    path('add_folder/', add_folder, name='add_folder'),
    path("thongke/", thongke_view, name="thongke"),
    path("diemdanh_thongke/", diemdanh_thongke, name="diemdanh_thongke"),
    path("manual-open/", manual_open, name="manual_open"),

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
