from django.urls import path
from .views import ANPRViewcc,video_feed

urlpatterns = [
    path('video_feed/', video_feed, name='video_feed'),
    # path('anpr/', ANPRView.as_view(), name='anpr'),
    path('anprcc/', ANPRViewcc.as_view(), name='anprcc')
   
]
   
