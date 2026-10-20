from django.urls import path
from .views import AnalyzeView, JobView

urlpatterns = [path('api/v1/analyze', AnalyzeView.as_view()),
               path('api/v1/jobs/<str:job_id>', JobView.as_view())]
