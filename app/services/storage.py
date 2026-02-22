#storage.py
import boto3
from app.core.config import get_settings

class StorageService:
    def __init__(self):
        self.settings = get_settings()
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=f"https://{self.settings.CLOUDFLARE_R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=self.settings.CLOUDFLARE_R2_ACCESS_KEY_ID,
            aws_secret_access_key=self.settings.CLOUDFLARE_R2_SECRET_ACCESS_KEY,
            region_name="auto"
        )

    def upload_file(self, file_path: str, object_name: str) -> str:
        self.s3_client.upload_file(file_path, self.settings.CLOUDFLARE_R2_BUCKET_NAME, object_name)
        public_url = self.settings.CLOUDFLARE_R2_PUBLIC_URL.rstrip("/")
        return f"{public_url}/{object_name}"