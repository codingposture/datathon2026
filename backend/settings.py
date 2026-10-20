import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / 'backend/.env')

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', '')
API_KEY = os.environ.get('API_KEY', '')
if not SECRET_KEY or not API_KEY:
    raise ImproperlyConfigured('Run scripts/configure_local_backend.py to configure the local API key and Django secret.')
DEBUG = False
ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', '127.0.0.1,localhost,[::1]').split(',')
ROOT_URLCONF = 'backend.urls'
WSGI_APPLICATION = 'backend.wsgi.application'
INSTALLED_APPS = ['rest_framework']
MIDDLEWARE = ['django.middleware.security.SecurityMiddleware', 'django.middleware.common.CommonMiddleware']
DATABASES = {}
APPEND_SLASH = False
USE_TZ = True
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [],
    'DEFAULT_PERMISSION_CLASSES': [],
    'UNAUTHENTICATED_USER': None,
    'DEFAULT_RENDERER_CLASSES': ['rest_framework.renderers.JSONRenderer'],
    'DEFAULT_PARSER_CLASSES': ['rest_framework.parsers.MultiPartParser'],
}
FILE_UPLOAD_HANDLERS = ['backend.uploads.SizeLimitUploadHandler',
                        'django.core.files.uploadhandler.TemporaryFileUploadHandler']
DATA_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024
DATA_UPLOAD_MAX_NUMBER_FIELDS = 10
DATA_UPLOAD_MAX_NUMBER_FILES = 1
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_VIDEO_SECONDS = 300
MAX_VIDEO_PIXELS = 3840 * 2160
MAX_RESULT_BYTES = 16 * 1024 * 1024
JOB_TTL_SECONDS = 15 * 60
MAX_RETAINED_JOBS = 8
# Leave time for the frontend's next poll before its five-minute deadline.
ANALYSIS_TIMEOUT_SECONDS = min(295, float(os.environ.get('ANALYSIS_TIMEOUT_SECONDS', '295')))
INFERENCE_THREADS = max(1, int(os.environ.get('INFERENCE_THREADS', '2')))
MODEL_ARTIFACT_DIR = Path(os.environ.get('MODEL_ARTIFACT_DIR', str(BASE_DIR / 'artifacts/hybrid_v2'))).resolve()
if not Path(os.environ.get('MODEL_ARTIFACT_DIR', str(BASE_DIR / 'artifacts/hybrid_v2'))).is_absolute():
    MODEL_ARTIFACT_DIR = (BASE_DIR / os.environ['MODEL_ARTIFACT_DIR']).resolve()
if ANALYSIS_TIMEOUT_SECONDS <= 0:
    raise ImproperlyConfigured('ANALYSIS_TIMEOUT_SECONDS must be greater than zero.')
