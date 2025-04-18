#!/bin/bash

# 讀取 .env 檔案中的環境變數 (注意: 生產環境請考慮更安全的憑證管理方式)
# 如果使用 IAM Role for IoT Thing，則不需要這些環境變數
# 如果使用 settings.yaml 包含憑證，則也不需要這些環境變數
# export AWS_ACCESS_KEY=YOUR_ACCESS_KEY
# export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
# export S3_BUCKET=YOUR_S3_BUCKET
# export S3_REGION=YOUR_S3_REGION
# export IOT_CORE_ENDPOINT=YOUR_IOT_ENDPOINT
# export THING_NAME=YOUR_THING_NAME
# export IOT_CERT_PATH=path/to/your/certificate.pem.crt
# export IOT_PRI_KEY_PATH=path/to/your/private.pem.key
# export IOT_ROOT_CA_PATH=path/to/your/root-CA.crt

# 更改到專案根目錄
cd $(dirname "$0")

# 執行主程式
python3 main.py

# 清理或其他結束操作 (如果需要)