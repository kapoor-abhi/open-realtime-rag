#storage.py
"""
Object storage service backed by MinIO (local S3-compatible).
Drop-in replacement for the previous Cloudflare R2 implementation.
boto3 API is identical — only the endpoint_url and credentials change.
"""
import boto3
from botocore.exceptions import ClientError
import logging
from app.core.config import get_settings

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        self.settings = get_settings()
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.settings.MINIO_ENDPOINT_URL,
            aws_access_key_id=self.settings.MINIO_ACCESS_KEY_ID,
            aws_secret_access_key=self.settings.MINIO_SECRET_ACCESS_KEY,
            # MinIO does not use AWS regions but boto3 requires the field.
            region_name="us-east-1",
        )
        self.bucket = self.settings.MINIO_BUCKET_NAME
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Create the bucket if it does not already exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            error_code = int(e.response["Error"]["Code"])
            if error_code == 404:
                self.s3_client.create_bucket(Bucket=self.bucket)
                # Make the bucket publicly readable so stored image URLs work.
                self.s3_client.put_bucket_policy(
                    Bucket=self.bucket,
                    Policy=f'''{{
                        "Version": "2012-10-17",
                        "Statement": [{{
                            "Effect": "Allow",
                            "Principal": {{"AWS": ["*"]}},
                            "Action": ["s3:GetObject"],
                            "Resource": ["arn:aws:s3:::{self.bucket}/*"]
                        }}]
                    }}'''
                )
                logger.info(f"Created MinIO bucket: {self.bucket}")
            else:
                raise

    def upload_file(self, file_path: str, object_name: str) -> str:
        """
        Upload a local file to MinIO and return its public URL.
        The public URL format mirrors what Cloudflare R2 returned, so
        the rest of the codebase needs no changes.
        """
        self.s3_client.upload_file(file_path, self.bucket, object_name)
        public_url = self.settings.MINIO_PUBLIC_URL.rstrip("/")
        return f"{public_url}/{object_name}"

    def delete_file(self, object_name: str) -> bool:
        """Delete an object from MinIO. Returns True on success."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=object_name)
            return True
        except ClientError as e:
            logger.error(f"Failed to delete {object_name}: {e}")
            return False
