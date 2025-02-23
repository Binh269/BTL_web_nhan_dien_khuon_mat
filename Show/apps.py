from django.apps import AppConfig


class ShowConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Show'
    # def ready(self):
    #     """ Khởi động MQTT khi Django chạy """
    #     # from .giaima import start_mqtt
    #     # start_mqtt()
    import Show.giaima
    Show.giaima.start_mqtt()


