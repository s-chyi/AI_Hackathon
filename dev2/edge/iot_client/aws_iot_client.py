# iot_client/aws_iot_client.py

from awsiot import mqtt_connection_builder
# 引入 QoS 枚舉和 CRT 錯誤類型
from awscrt.mqtt import QoS
import awscrt.exceptions

import json
import logging
import threading
import time # 添加 time 模組用於延遲或時間相關日誌
from concurrent.futures import Future
from typing import Dict, Any, Optional, Callable # 引入類型提示

# 配置 logging
logger = logging.getLogger(__name__)

# 定義連接超時和訂閱超時 (秒)
CONNECT_TIMEOUT_SEC = 10
SUBSCRIBE_TIMEOUT_SEC = 5

class AWSIoTClient:
    """
    處理與 AWS IoT Core 的 MQTT 連接和通訊。
    """
    def __init__(self, iot_settings: Dict[str, Any],
                 command_callback: Optional[Callable[[str, str], None]] = None,
                 result_callback: Optional[Callable[[str, str], None]] = None): # <-- 新增結果回調參數
        """
        初始化 AWS IoT 客戶端。
        Args:
            iot_settings (Dict[str, Any]): AWS IoT 相關設定。
            command_callback (Optional[Callable[[str, str], None]], optional): 收到命令訊息時調用的回調函數。
            result_callback (Optional[Callable[[str, str], None]], optional): 收到結果訊息時調用的回調函數。
        """
        self.iot_settings = iot_settings
        self.command_callback = command_callback
        self.result_callback = result_callback # 保存結果回調

        self.mqtt_connection = None
        self._is_connected = False
        self._connection_lock = threading.Lock()
        self._disconnect_requested = threading.Event()

        self._connect()

    def _connect(self):
        """
        建立到 AWS IoT Core 的 MQTT 連接。
        """
        # ... 連接建立邏輯 (保持不變) ...
        endpoint = self.iot_settings.get('endpoint')
        thing_name = self.iot_settings.get('thing_name')
        cert_path = self.iot_settings.get('cert_path')
        pri_key_path = self.iot_settings.get('pri_key_path')
        root_ca_path = self.iot_settings.get('root_ca_path')

        if not all([endpoint, thing_name, cert_path, pri_key_path, root_ca_path]):
             logger.error("AWS IoT 設定不完整。應用程式終止。")
             return

        logger.info(f"嘗試連接到 AWS IoT Core Endpoint: {endpoint}")
        # ... Debug 日誌 ...

        try:
            self.mqtt_connection = mqtt_connection_builder.mtls_from_path(
                endpoint=endpoint, cert_filepath=cert_path, pri_key_filepath=pri_key_path,
                ca_filepath=root_ca_path, client_id=thing_name,
                clean_session=False, keep_alive_secs=30
            )

            self.mqtt_connection.on_connection_interrupted = self._on_connection_interrupted
            self.mqtt_connection.on_connection_resumed = self._on_connection_resumed

            logger.debug("調用 mqtt_connection.connect()")
            connect_future = self.mqtt_connection.connect()
            logger.debug(f"等待連接完成 (最多 {CONNECT_TIMEOUT_SEC} 秒)...")
            connect_future.result(timeout=CONNECT_TIMEOUT_SEC)
            with self._connection_lock:
                self._is_connected = True
                self._disconnect_requested.clear()
            logger.info("成功連接到 AWS IoT Core!")

            # 訂閱命令 Topic (如果設定了回調)
            if self.command_callback:
                command_topic_format = self.iot_settings.get('command_topic')
                if not command_topic_format:
                     logger.warning("設定中未指定命令 Topic 格式，跳過訂閱命令。")
                else:
                    command_topic = command_topic_format.format(thing_name=thing_name)
                    logger.info(f"訂閱命令 Topic: {command_topic}")
                    subscribe_future, packet_id = self.mqtt_connection.subscribe(
                        topic=command_topic, qos=QoS.AT_LEAST_ONCE, callback=self._on_mqtt_message
                    )
                    logger.debug(f"等待訂閱完成 (最多 {SUBSCRIBE_TIMEOUT_SEC} 秒)...")
                    subscribe_result = subscribe_future.result(timeout=SUBSCRIBE_TIMEOUT_SEC)
                    logger.info(f"成功訂閱。Packet ID: {packet_id}, Result QoS: {subscribe_result.get('qos')}")

            # 新增：訂閱結果 Topic (如果設定了回調)
            if self.result_callback:
                result_topic_format = self.iot_settings.get('result_topic')
                if not result_topic_format:
                     logger.warning("設定中未指定結果 Topic 格式，跳過訂閱結果。")
                else:
                    result_topic = result_topic_format.format(thing_name=thing_name)
                    logger.info(f"訂閱結果 Topic: {result_topic}")
                    subscribe_future, packet_id = self.mqtt_connection.subscribe(
                        topic=result_topic, qos=QoS.AT_LEAST_ONCE, callback=self._on_mqtt_message # 使用同一個回調函數
                    )
                    logger.debug(f"等待訂閱完成 (最多 {SUBSCRIBE_TIMEOUT_SEC} 秒)...")
                    subscribe_result = subscribe_future.result(timeout=SUBSCRIBE_TIMEOUT_SEC)
                    logger.info(f"成功訂閱。Packet ID: {packet_id}, Result QoS: {subscribe_result.get('qos')}")


        except TimeoutError:
            logger.error(f"連接或訂閱 AWS IoT Core 超時 ({CONNECT_TIMEOUT_SEC}或{SUBSCRIBE_TIMEOUT_SEC}秒)。請檢查 endpoint 和網絡連接。", exc_info=True)
            with self._connection_lock:
                 self._is_connected = False
        except FileNotFoundError as e:
             logger.error(f"證書檔案找不到: {e}。請檢查 settings.yaml 中的證書路徑是否正確，以及檔案是否存在。", exc_info=True)
             with self._connection_lock:
                 self._is_connected = False
        except awscrt.exceptions.AwsCrtError as e:
            logger.error(f"AWS CRT 錯誤: {e}。這通常表示認證/授權失敗或證書問題。請檢查 Policy、證書狀態和檔案。", exc_info=True)
            with self._connection_lock:
                 self._is_connected = False
        except Exception as e:
            logger.error(f"連接或訂閱 AWS IoT Core 時發生意外錯誤: {e}", exc_info=True)
            with self._connection_lock:
                 self._is_connected = False

    def _on_mqtt_message(self, topic: str, payload: bytes, **kwargs):
        """
        內部回調函數，處理收到的 MQTT 訊息。
        根據 Topic 判斷是命令還是結果，並調用相應的回調函數。
        Args:
            topic (str): 收到訊息的 Topic。
            payload (bytes): 訊息的 Payload (Bytes 格式)。
        """
        logger.info(f"收到 MQTT 訊息 - Topic: {topic}")
        try:
            payload_str = payload.decode('utf-8')

            # 根據 Topic 判斷並調用回調
            command_topic_format = self.iot_settings.get('command_topic', '').format(thing_name=self.iot_settings.get('thing_name', ''))
            result_topic_format = self.iot_settings.get('result_topic', '').format(thing_name=self.iot_settings.get('thing_name', ''))

            if topic == command_topic_format and self.command_callback:
                 logger.debug("將訊息轉發給命令回調。")
                 self.command_callback(topic, payload_str)
            elif topic == result_topic_format and self.result_callback:
                 logger.debug("將訊息轉發給結果回調。")
                 self.result_callback(topic, payload_str)
            else:
                 logger.warning(f"收到未知 Topic 的訊息或未設定回調：{topic}")

        except Exception as e:
            logger.error(f"處理 MQTT 訊息或調用回調時發生錯誤: {e}", exc_info=True)

    def _on_connection_interrupted(self, connection, error, **kwargs):
        """
        連接中斷時的回調函數。
        SDK 會自動嘗試重連。
        """
        logger.warning(f"AWS IoT Core 連接中斷: {error}. SDK 將自動嘗試重連...")
        with self._connection_lock:
             self._is_connected = False
             # 如果不是應用程式主動請求斷開，可以記錄下來或觸發其他邏輯

    def _on_connection_resumed(self, connection, return_code, session_present, **kwargs):
        """
        連接恢復時的回調函數。
        """
        logger.info(f"AWS IoT Core 連接恢復成功。Return Code: {return_code}, Session Present: {session_present}")
        with self._connection_lock:
             self._is_connected = True
             self._disconnect_requested.clear() # 連接恢復，清除斷開請求標誌 (如果設置過)

        # 連接恢復後，如果 clean_session=False，SDK 會自動重新訂閱之前的 Topic
        # 如果 clean_session=True，則需要手動在這裡重新訂閱

    def publish_event(self, event_payload: Dict[str, Any]) -> Future:
        """
        將事件訊息發布到 AWS IoT Core 的事件 Topic。
        Args:
            event_payload (Dict[str, Any]): 包含事件數據的字典。
        Returns:
            Future: MQTT 發布操作的 Future 對象。可以選擇等待其結果。
        """
        with self._connection_lock:
            if not self._is_connected:
                # 如果連接斷開，記錄警告並返回一個失敗的 Future
                logger.warning("MQTT 連接未建立或已中斷，無法發布事件。")
                f = Future()
                f.set_exception(ConnectionError("MQTT connection is not available"))
                return f

        try:
            event_topic_format = self.iot_settings.get('event_topic')
            if not event_topic_format:
                 logger.error("設定中未指定事件 Topic 格式，無法發布事件。")
                 f = Future()
                 f.set_exception(ValueError("Event topic format is not configured"))
                 return f

            event_topic = event_topic_format.format(thing_name=self.iot_settings['thing_name'])
            payload_json = json.dumps(event_payload)
            # logger.debug(f"發布事件到 {event_topic}: {payload_json}") # DEBUG 級別輸出 Payload

            # 發布訊息
            publish_future = self.mqtt_connection.publish(
                topic=event_topic,
                payload=payload_json,
                # 修正：使用 QoS 枚舉
                qos=QoS.AT_LEAST_ONCE # 設置 QoS 等級為 1
            )
            logger.debug(f"已提交發布任務到 {event_topic}。")
            return publish_future

        except Exception as e:
            logger.error(f"發布 MQTT 事件時發生錯誤: {e}", exc_info=True)
            # 返回一個已完成的 Future，表示失敗
            f = Future()
            f.set_exception(e)
            return f

    def is_connected(self) -> bool:
        """
        檢查 MQTT 連接是否建立且處於活動狀態。
        """
        # 檢查內部標誌 _is_connected
        with self._connection_lock:
            return self._is_connected

        # 也可以更嚴格地檢查底層 SDK 的狀態 (如果需要)
        # if self.mqtt_connection:
        #     return self.mqtt_connection.is_connected()
        # return False


    def disconnect(self):
        """
        中斷與 AWS IoT Core 的連接。
        設置斷開請求標誌，並調用 SDK 的 disconnect 方法。
        """
        logger.info("請求中斷 AWS IoT Core 連接...")
        # 設置標誌表示是應用程式主動請求斷開
        self._disconnect_requested.set()

        if self.mqtt_connection:
            try:
                disconnect_future = self.mqtt_connection.disconnect()
                logger.debug(f"等待斷開連接完成 (最多 {CONNECT_TIMEOUT_SEC} 秒)...")
                disconnect_future.result(timeout=CONNECT_TIMEOUT_SEC) # 設置斷開超時時間
                logger.info("成功中斷 AWS IoT Core 連接。")
            except Exception as e:
                logger.error(f"中斷 AWS IoT Core 連接時發生錯誤: {e}", exc_info=True)
            finally:
                with self._connection_lock:
                     self._is_connected = False # 無論成功失敗，都將狀態設為 False
                self.mqtt_connection = None # 清空連接實例

    # 可以添加一個方法來檢查是否是應用程式主動斷開
    # def is_disconnect_requested(self) -> bool:
    #     """
    #     檢查是否是應用程式主動請求斷開連接。
    #     """
    #     return self._disconnect_requested.is_set()