# ICAM 邊緣端智慧倉儲應用

這是一個運行在研華 ICAM-540 (搭載 Jetson 平台) 上的邊緣端應用程式，用於智慧倉儲場景中的物件偵測、事件觸發，並與 AWS 雲端服務進行邊雲協同。

## 專案概覽

應用程式利用 Jetson 的邊緣計算能力，執行輕量級 AI 模型進行即時影像分析，識別人員、貨物等。當偵測到符合預設規則的事件時（例如：偵測到人員、貨物異常），會捕獲相關影像並上傳到 AWS S3，同時透過 AWS IoT Core 發送結構化的事件通知給雲端後端。雲端後端（未包含在此專案代碼中）可以接收這些事件，調用更強大的雲端模型（如 AWS Nova-Lite, Claude）進行二次分析、生成報告、發送告警，並通過 IoT Core 向邊緣端發送控制命令。

**邊緣端的主要職責：**

*   從 ICAM 攝影機獲取影像。
*   利用 Jetson GPU 執行物件偵測等邊緣 AI 模型。
*   根據模型結果和預設規則判斷是否發生特定事件。
*   在事件發生時捕獲關鍵影像/短片。
*   將捕獲的檔案異步上傳到 AWS S3。
*   通過 AWS IoT Core 安全地將事件元數據（包含 S3 路徑）發送給雲端。
*   接收並處理來自雲端的控制命令（如更新配置、手動拍照）。

**雲端端的主要職責 (在此專案範圍之外):**

*   接收來自 IoT Core 的事件通知。
*   從 S3 下載相關影像/數據。
*   使用 AWS Nova-Lite, Claude 或其他模型進行深度分析、確認事件類型。
*   將分析結果存儲到數據庫。
*   生成報告、發送告警（通過 SNS 等）。
*   提供使用者介面進行監控、查詢和控制。
*   通過 IoT Core 向邊緣設備發送命令。

## 系統架構

```mermaid
graph TD
    A[ICAM Edge Device] -->|Capture Video| B(Image Preprocessing)
    B --> C(Edge AI Inference)
    C --> D(Detectors<br>Rule Logic)
    D --> E{Event Triggered?}
    E -- Yes --> F(Data Capture<br>Image/Clip)
    F --> G[S3 Uploader<br>Async Thread]
    G -->|Upload File| H[AWS S3]
    E -- Yes --> I(Event Manager<br>Cooldown/State)
    I --> J(Event Publisher)
    J -->|MQTT Event Msg| K[AWS IoT Core]
    K -->|Route Msg (Rules Engine)| L[AWS Cloud Backend<br>Lambda, Bedrock(Nova-Lite/Claude), SQS, DB, etc.]
    L -->|Cloud Commands| K
    K -->|MQTT Command Msg| A
    A --> B; % Loop back to start for next frame
    D -->|Metadata| I; % Detectors inform Event Manager
    F -->|S3 Path| J; % Capture Manager informs Event Publisher about S3 path
    G --> H; % S3 Uploader uploads to S3
    K --> L; % IoT Core sends events to cloud backend
    L --> K; % Cloud backend sends commands via IoT Core
```

## 先決條件

*   **硬體:** 研華 ICAM-540 或其他搭載 NVIDIA Jetson 平台的設備。
*   **作業系統:** JetPack SDK 安裝完成，包含 CUDA, cuDNN, TensorRT。
*   **軟體:**
    *   Python 3.6+
    *   安裝 `jetson.inference` 和 `jetson.utils` (隨 JetPack 提供或需額外安裝)。
*   **AWS 帳戶:** 擁有 AWS 帳戶並具備相應的權限。

## 安裝步驟

1.  **克隆專案:**
    ```bash
    git clone <您的 Git 倉庫 URL>
    cd icam_edge_app
    ```

2.  **安裝 Python 依賴:**
    ```bash
    pip install -r requirements.txt
    ```
    *注意：`jetson.inference` 和 `jetson.utils` 通常隨 JetPack 提供，不需要通過 pip 安裝。如果遇到 ImportError，請檢查您的 JetPack 安裝和環境變數。*

3.  **準備模型檔案:**
    將您為 Jetson 編譯或下載的邊緣 AI 模型檔案（如 `ssd-mobilenet-v2` 或您的 TensorRT 引擎檔案）放入 `models/` 資料夾。

## AWS 設定

您需要在 AWS 雲端建立必要的資源以實現邊緣與雲端的通訊。

1.  **建立 S3 儲存桶:**
    建立一個 S3 儲存桶用於存放邊緣設備上傳的影像或短片。記錄下儲存桶名稱和區域。

2.  **設定 AWS IoT Core:**
    *   進入 AWS IoT Core 控制台。
    *   在 "Manage" -> "Things" 下建立一個新的 Thing 來代表您的 ICAM 設備（例如命名為 `ICAM_001`）。
    *   為該 Thing 建立一個憑證 (Certificate) 和金鑰對 (Key Pair)。選擇 "One-click certificate creation" 或 "Generate certificate and keys".
    *   **下載**生成的證書 (`.pem.crt`)、私鑰 (`.pem.key`) 和根 CA 證書 (`root-CA.crt`，通常是 Amazon Root CA 1)。請務必妥善保管這些檔案，它們只會顯示一次！
    *   建立一個新的 IoT Policy，定義您的設備可以發布和訂閱哪些 MQTT Topic。以下是一個範例 Policy JSON，請根據您的 Topic 設定修改：
        ```json
        {
          "Version": "2012-10-17",
          "Statement": [
            {
              "Effect": "allow",
              "Action": [
                "iot:Connect"
              ],
              "Resource": "arn:aws:iot:YOUR_AWS_REGION:YOUR_AWS_ACCOUNT_ID:client/YOUR_THING_NAME"
            },
            {
              "Effect": "allow",
              "Action": [
                "iot:Publish"
              ],
              "Resource": "arn:aws:iot:YOUR_AWS_REGION:YOUR_AWS_ACCOUNT_ID:topic/icam/YOUR_THING_NAME/events"
            },
            {
              "Effect": "allow",
              "Action": [
                "iot:Subscribe"
              ],
              "Resource": "arn:aws:iot:YOUR_AWS_REGION:YOUR_AWS_ACCOUNT_ID:topicFilter/icam/YOUR_THING_NAME/commands"
            },
             {
              "Effect": "allow",
              "Action": [
                "iot:Receive"
              ],
              "Resource": "arn:aws:iot:YOUR_AWS_REGION:YOUR_AWS_ACCOUNT_ID:topic/icam/YOUR_THING_NAME/commands"
            }
          ]
        }
        ```
        將 `YOUR_AWS_REGION`, `YOUR_AWS_ACCOUNT_ID`, `YOUR_THING_NAME` 替換為您的實際值。
    *   將 Policy 附加到您剛才建立的證書上。
    *   找到您的 AWS IoT Core 的 Endpoint URL (在 Settings 或 "Interact" 選項卡下，格式通常是 `xxxxxxxxxxxxxx-ats.iot.your-region.amazonaws.com`)。

3.  **設定 AWS 憑證 (邊緣設備訪問 S3 和 IoT Core):**
    您的邊緣設備需要權限來將檔案上傳到 S3 以及連接和發布/訂閱 IoT Core。有多種方式提供憑證給 `boto3` 和 AWS IoT Device SDK：
    *   **推薦（生產環境）：** 使用 IAM Role for IoT Thing。這需要更複雜的設定，但設備本身不需要硬編碼或存儲長期有效的 Access Key/Secret Key。
    *   **開發/測試方便：**
        *   **方法 A: 環境變數:** 在運行腳本的環境中設置 `AWS_ACCESS_KEY_ID` 和 `AWS_SECRET_ACCESS_KEY`。這是您目前範例中使用的方式。
        *   **方法 B: AWS Credentials File:** 在設備上創建 `~/.aws/credentials` 文件。
        *   **方法 C: IAM User Access Key/Secret Key in `settings.yaml`:** 將憑證直接寫入配置文件。**非常不推薦用於生產環境！**
    *   **對於 IoT Core 的連接，您必須使用下載的證書和私鑰**，而不是 Access Key/Secret Key。

## 本地配置

編輯 `config/settings.yaml` 文件，填寫您的特定設定：

*   `camera`: 攝影機設備 ID, 解析度等。
*   `aws.region`: AWS 區域。
*   `aws.s3.bucket_name`: 您的 S3 儲存桶名稱。
*   `aws.s3.upload_folder`: S3 儲存桶內用於存放檔案的檔案夾。
*   `aws.iot.endpoint`: 您的 AWS IoT Core Endpoint URL。
*   `aws.iot.thing_name`: 您的 IoT Thing 名稱。
*   `aws.iot.cert_path`, `aws.iot.pri_key_path`, `aws.iot.root_ca_path`: 下載的證書檔案在設備上的路徑。建議將這些檔案放在一個安全的位置（例如專案目錄外的 `certs` 檔案夾），並確保其權限設定正確。
*   `models.object_detection.model_path`: 您在 `models/` 檔案夾中的模型檔案路徑或 `jetson.inference` 支持的模型名稱。
*   `models.object_detection.class_mapping`: 確認您的模型輸出類別 ID 與程式內部使用的類別名稱（如 "person", "cargo"）的映射關係。
*   `detectors`: 根據您的需求啟用或禁用特定的偵測器，並調整其設定（如冷卻時間）。
*   `display.enabled`: 控制是否在本地顯示影像（對於無顯示器的邊緣設備應設為 `false`）。

## 模型準備

為了在 Jetson 上獲得最佳性能，建議使用經過 NVIDIA TensorRT 優化的模型。`jetson.inference` 庫可以載入一些預訓練的 TensorRT 模型，您也可以使用 TensorRT builder 將您自己的模型轉換為 TensorRT 引擎。確保 `settings.yaml` 中的 `model_path` 指向正確的模型檔案或名稱。

## 運行應用程式

設置好所有配置和檔案後，運行 `run.sh` 腳本：

```bash
./run.sh
```

應用程式將啟動，從攝影機讀取影像，運行邊緣模型，偵測事件，並在滿足條件時將影像上傳到 S3 並發布 MQTT 訊息到 AWS IoT Core。

您可以通過查看控制台輸出（日誌）來監控應用程式的運行狀態。

## 專案結構說明

*   `config/`: 存放配置檔案。
*   `models/`: 存放邊緣端 AI 模型檔案。
*   `utils/`: 存放通用的輔助工具函數和類。
    *   `cuda_utils.py`: 處理 CUDA 影像數據的轉換。
    *   `image_utils.py`: 影像繪圖和處理功能。
    *   `s3_uploader.py`: 異步的 S3 檔案上傳執行緒。
*   `inference/`: 負責載入和執行邊緣 AI 模型推論。
    *   `model_manager.py`: 模型載入和管理。
    *   `inferencer.py`: 模型推論的基類和具體實現（如 `ObjectDetector`）。
*   `detectors/`: 存放不同偵測邏輯的模塊。每個文件代表一種事件或對象的偵測處理。
    *   `base_detector.py`: 所有偵測器的基類，提供基本結構和通用方法（如觸發事件）。
    *   `person_detector.py`: 處理人員偵測和相關事件邏輯。
    *   `cargo_detector.py`: 處理貨物偵測和相關事件邏輯。
    *   `...`: 可以根據需求添加更多偵測器。
*   `events/`: 管理邊緣事件的生命週期和發布。
    *   `event_types.py`: 定義事件類型列表。
    *   `event_manager.py`: 管理事件冷卻時間和觸發頻率。
    *   `event_publisher.py`: 格式化事件數據並通過 IoT 客戶端發布。
*   `iot_client/`: 封裝與 AWS IoT Core 的通訊邏輯。
    *   `aws_iot_client.py`: 連接管理、發布和訂閱。
*   `data_capture/`: 負責在事件觸發時捕獲當前影像或短片。
    *   `capture_manager.py`: 管理影像捕獲過程並將任務提交給 S3 上傳器。
*   `main.py`: 應用程式的主入口點，協調所有模塊的運行。
*   `requirements.txt`: Python 依賴列表。
*   `run.sh`: 運行應用程式的腳本。

## 擴展與定製

*   **添加新的偵測器:** 在 `detectors/` 目錄下創建新的 Python 文件，繼承 `BaseDetector`，並在 `main.py` 中根據設定初始化並添加到 `detectors` 列表中。
*   **集成更多模型:** 在 `inference/model_manager.py` 和 `inference/inferencer.py` 中添加對新模型類型（如分類、姿勢估計）的支持，並在需要這些模型的偵測器中引入並使用。
*   **增加複雜規則:** 在各個偵測器的 `process` 方法中實現更複雜的邏輯，例如結合多個幀的數據進行跟蹤，或者基於特定區域的規則。
*   **處理雲端命令:** 在 `main.py` 的 `handle_cloud_command` 函數中添加處理新的雲端命令類型。
*   **短片捕獲:** 修改 `data_capture/capture_manager.py` 添加短片錄製功能，並在事件觸發時協調錄製和上傳。

## 注意事項

*   **性能:** 在 Jetson 平台上，CPU 資源有限，而 GPU 是關鍵。確保 AI 模型推論在 GPU 上運行，並盡量減少 CPU 和 GPU 之間的數據拷貝。OpenCV 的一些操作（如色彩空間轉換）可能較慢，需要注意。異步處理（S3 上傳，IoT 通訊）對於保持主循環流暢至關重要。
*   **資源管理:** 監控 CPU、GPU、記憶體使用率，避免資源耗盡導致程式崩潰。注意釋放 OpenCV 的 VideoCapture 資源和 IoT 連接。
*   **錯誤處理:** 在各個模塊中加入足夠的錯誤處理邏輯（try-except），特別是對於硬體、模型推論和網絡操作。
*   **日誌記錄:** 利用 Python 的 `logging` 模塊，設置不同的日誌級別（DEBUG, INFO, WARNING, ERROR），方便調試和監控。
*   **憑證安全:** 在生產環境中，請務必使用更安全的 AWS 憑證管理方式，如 IAM Role for IoT Thing，而不是直接在設備上存儲 Access Key/Secret Key 或證書檔案。如果必須使用證書檔案，確保其文件權限設置為只有運行應用的用戶可以讀取。