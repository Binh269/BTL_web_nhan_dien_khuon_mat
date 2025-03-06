Cách chạy chương trình:
- Cài môi trường ảo và thư viện
- Tải model :https://drive.google.com/drive/folders/1rRLzNUOrqyXv1Io-m-SbXSAFdiRvWYW2?usp=sharing
- Thêm model vào project_Binh\project_Binh\Show\models
- Sau khi cài xong : cd .\project_Binh\
  +  Chạy trên locall : python .\manage.py runserver
  +  Chạy trên server
    *  thêm hostname vào ALLOWED_HOSTS = ['']  trong project_Binh\project_Binh\settings.py
    *  mở port cho laptop
    *  python .\manage.py runserver hostname:port
