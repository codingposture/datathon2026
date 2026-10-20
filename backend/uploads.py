from pathlib import Path
import json
import math
import subprocess

from django.conf import settings
from django.core.files.uploadhandler import FileUploadHandler
from rest_framework import serializers
from rest_framework.exceptions import APIException


class UploadTooLarge(APIException):
    status_code = 413
    default_detail = 'The video exceeds the 64 MiB upload limit.'


class InvalidVideo(APIException):
    status_code = 415
    default_detail = 'Upload a decodable QuickTime .mov file containing one video stream.'


class SizeLimitUploadHandler(FileUploadHandler):
    """Enforce the byte limit while streaming, even without an honest Content-Length."""
    def __init__(self, request=None):
        super().__init__(request)
        self.received = 0

    def receive_data_chunk(self, raw_data, start):
        self.received += len(raw_data)
        if self.received > settings.MAX_UPLOAD_BYTES:
            raise UploadTooLarge()
        return raw_data

    def file_complete(self, file_size):
        return None


class VideoUploadSerializer(serializers.Serializer):
    file = serializers.FileField(allow_empty_file=False)

    def validate_file(self, file):
        if Path(file.name).suffix.lower() != '.mov':
            raise InvalidVideo('Only .mov uploads are supported.')
        if file.size > settings.MAX_UPLOAD_BYTES:
            raise UploadTooLarge()
        return file


def validate_video(path):
    """Inspect actual container/stream metadata, not the client MIME type or filename."""
    try:
        probe = subprocess.run([
            'ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-f', 'mov',
            '-show_entries', 'format=duration:stream=codec_type,width,height,duration',
            '-of', 'json', str(path)], capture_output=True, timeout=10, check=False)
        if probe.returncode:
            raise InvalidVideo()
        data = json.loads(probe.stdout)
        streams = [s for s in data.get('streams', []) if s.get('codec_type') == 'video']
        if len(streams) != 1:
            raise InvalidVideo()
        stream = streams[0]
        duration = float(stream.get('duration', data.get('format', {}).get('duration', 0)))
        width, height = int(stream.get('width', 0)), int(stream.get('height', 0))
        if not math.isfinite(duration) or not 0 < duration <= settings.MAX_VIDEO_SECONDS:
            raise InvalidVideo('Video duration must be greater than zero and at most five minutes.')
        if min(width, height) <= 0 or width * height > settings.MAX_VIDEO_PIXELS:
            raise InvalidVideo('Video resolution must be at most 3840 × 2160 pixels (portrait supported).')
    except (ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as error:
        raise InvalidVideo() from error
