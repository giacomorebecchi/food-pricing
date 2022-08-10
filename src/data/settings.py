from functools import lru_cache
from typing import Dict

from pydantic import BaseSettings


class S3Settings(BaseSettings):
    ANON: bool = False
    AWS_ACCESS_KEY_ID: str = "aws_key_id"
    AWS_SECRET_ACCESS_KEY: str = "aws_secret_key"
    BUCKET: str = "food-delivery-crawler"
    S3_ENDPOINT: str = "http://localhost:9000"
    AWS_REGION: str = "eu-west-1"
    SIGNATURE_VERSION: str = "s3v4"

    class Config:
        env_file = ".env"
        env_prefix = ""

    def format_settings_s3fs(self) -> Dict:
        settings = {
            "client_kwargs": {
                "aws_access_key_id": self.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": self.AWS_SECRET_ACCESS_KEY,
                "endpoint_url": self.S3_ENDPOINT,
                "region_name": self.AWS_REGION,
            },
            "config_kwargs": {
                "signature_version": self.SIGNATURE_VERSION,
            },
        }
        return settings

    def format_settings_pyarrow_fs(self) -> Dict:
        settings = {
            "access_key": self.AWS_ACCESS_KEY_ID,
            "secret_key": self.AWS_SECRET_ACCESS_KEY,
            "anonymous": self.ANON,
            "region": self.AWS_REGION,
            "endpoint_override": self.S3_ENDPOINT,
        }
        return settings


@lru_cache()
def get_S3_settings() -> S3Settings:
    return S3Settings()


class NotifierSettings(BaseSettings):
    BOT_TOKEN: str
    CHAT_ID: int

    class Config:
        env_file = ".env"
        env_prefix = "TELEGRAM" + "_"


@lru_cache()
def get_notifier_settings() -> NotifierSettings:
    return NotifierSettings()
