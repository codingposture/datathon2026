import hmac
import tempfile
import time
from pathlib import Path

from django.conf import settings
from django.core.exceptions import RequestDataTooBig, TooManyFieldsSent, TooManyFilesSent
from rest_framework.exceptions import ParseError
from rest_framework.permissions import BasePermission
from rest_framework.response import Response
from rest_framework.views import APIView

from .jobs import get_manager
from .uploads import VideoUploadSerializer, UploadTooLarge, validate_video


class APIKeyPermission(BasePermission):
    message = 'Missing or invalid X-API-Key.'

    def has_permission(self, request, view):
        supplied = request.headers.get('X-API-Key', '')
        return bool(supplied) and hmac.compare_digest(supplied.encode(), settings.API_KEY.encode())


class AnalysisView(APIView):
    permission_classes = [APIKeyPermission]

    def handle_exception(self, error):
        # Multipart parsing can fail before request.FILES exists. Close partial uploads too.
        for handler in self.request.upload_handlers:
            handler.upload_interrupted()
        if isinstance(error, RequestDataTooBig):
            error = UploadTooLarge()
        elif isinstance(error, (TooManyFieldsSent, TooManyFilesSent)):
            error = ParseError('Send exactly one video in the file field.')
        return super().handle_exception(error)

    def finalize_response(self, request, response, *args, **kwargs):
        response = super().finalize_response(request, response, *args, **kwargs)
        response['Cache-Control'] = 'no-store'
        return response


class AnalyzeView(AnalysisView):
    def post(self, request):
        started = time.monotonic()
        manager = get_manager()
        manager.reserve()
        temporary, uploaded, handed_off = None, None, False
        try:
            length = request.META.get('CONTENT_LENGTH')
            if length and int(length) > settings.MAX_UPLOAD_BYTES + 1024 * 1024:
                raise UploadTooLarge()
            serializer = VideoUploadSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            uploaded = serializer.validated_data['file']
            temporary = tempfile.TemporaryDirectory(prefix='fortebank-')
            path = Path(temporary.name) / 'input.mov'
            with path.open('wb') as output:
                for chunk in uploaded.chunks():
                    output.write(chunk)
            validate_video(path)
            job_id = manager.start(temporary, started)
            handed_off = True
            return Response({'job_id': job_id}, status=202)
        finally:
            if uploaded is not None:
                uploaded.close()
            if not handed_off:
                if temporary is not None:
                    temporary.cleanup()
                manager.slot.release()


class JobView(AnalysisView):
    def get(self, request, job_id):
        result = get_manager().get(job_id)
        if result is None:
            return Response({'detail': 'This job was not found or has expired. Upload the video again.'}, status=404)
        return Response(result)
