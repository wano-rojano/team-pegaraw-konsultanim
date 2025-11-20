"""Alibaba Cloud OSS configuration for Konsultanim."""
import os
import oss2
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class OSSConfig:
    """OSS configuration and connection manager."""
    
    def __init__(self):
        self.access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
        self.access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET')
        self.endpoint = os.getenv('OSS_ENDPOINT', 'oss-ap-southeast-5.aliyuncs.com')
        self.bucket_name = os.getenv('OSS_BUCKET_NAME', 'konsultanim-docs')
        
        if not self.access_key_id or not self.access_key_secret:
            raise ValueError(
                "OSS credentials not found. Set OSS_ACCESS_KEY_ID and "
                "OSS_ACCESS_KEY_SECRET environment variables."
            )
        
        self.auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)
        logger.info(f"OSS connected: {self.bucket_name} at {self.endpoint}")
    
    def list_pdfs(self, prefix: str = '') -> List[Dict]:
        """List all PDF files in a folder."""
        pdf_files = []
        try:
            for obj in oss2.ObjectIterator(self.bucket, prefix=prefix):
                if obj.key.endswith('.pdf'):
                    pdf_files.append({
                        'key': obj.key,
                        'size': obj.size,
                        'last_modified': obj.last_modified
                    })
            logger.info(f"Found {len(pdf_files)} PDFs in {prefix or 'root'}")
        except Exception as e:
            logger.error(f"Error listing PDFs in {prefix}: {e}")
        return pdf_files
    
    def download_to_temp(self, object_key: str, temp_path: str):
        """Download OSS object to temporary file."""
        try:
            self.bucket.get_object_to_file(object_key, temp_path)
            logger.debug(f"Downloaded {object_key} to {temp_path}")
        except Exception as e:
            logger.error(f"Error downloading {object_key}: {e}")
            raise
    
    def get_object_url(self, object_key: str, expires: int = 3600) -> str:
        """Generate signed URL for temporary access."""
        return self.bucket.sign_url('GET', object_key, expires)
    
    def upload_file(self, local_path: str, object_key: str):
        """Upload local file to OSS."""
        try:
            self.bucket.put_object_from_file(object_key, local_path)
            logger.info(f"Uploaded {local_path} to {object_key}")
        except Exception as e:
            logger.error(f"Error uploading {local_path}: {e}")
            raise