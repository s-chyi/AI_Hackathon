# iot_client/aws_iot_client.py

from awsiot import mqtt_connection_builder
import json
import logging
import threading
from concurrent.futures import Future

# 配置 logging
logger = logging.getLogger(__name__)

class AWSIoTClient:
    """
    處理與 AWS IoT Core 的 MQTT 連接和通訊。
    """
    def __init__(self, iot_settings: dict, command_callback=None):
        """
        初始化 AWS IoT 客戶端。
        Args:
            iot_settings (dict): AWS IoT 相關設定，包含 endpoint, thing_name, cert_paths, topics 等。
            command_callback (callable): 收到命令訊息時調用的回調函數 (topic, payload) -> None。
        """
        self.iot_settings = iot_settings
        self.command_callback = command_callback

        self.mqtt_connection = None
        self._is_connected = False
        self._connection_lock = threading.Lock() # 用於保護連接狀態的鎖

        self._connect()

    def _connect(self):
        """
        建立到 AWS IoT Core 的 MQTT 連接。
        """
        logger.info(f"嘗試連接到 AWS IoT Core Endpoint: {self.iot_settings['endpoint']}")
        try:
            self.mqtt_connection = mqtt_connection_builder.mtls_from_path(
                endpoint=self.iot_settings['endpoint'],
                cert_filepath=self.iot_settings['cert_path'],
                pri_key_filepath=self.iot_settings['pri_key_path'],
                ca_filepath=self.iot_settings['root_ca_path'],
                client_id=self.iot_settings['thing_name'],
                clean_session=False, # 設置為 False，方便設備離線後重新連接時，IoT Core 保留一些狀態
                keep_alive_secs=30 # 設定心跳包間隔
            )

            # 設定連接、斷開和訊息的回調函數
            self.mqtt_connection.on_connection_interrupted = self._on_connection_interrupted
            self.mqtt_connection.on_connection_resumed = self._on_connection_resumed

            # 啟動連接
            connect_future = self.mqtt_connection.connect()

            # 非阻塞等待連接完成 (也可以在獨立執行緒中處理連接過程)
            # 這裡選擇在初始化時阻塞等待，如果連接失敗則直接報錯
            connect_future.result(timeout=10) # 設置連接超時時間
            self._is_connected = True
            logger.info("成功連接到 AWS IoT Core!")

            # 如果設定了命令回調，則訂閱命令 Topic
            if self.command_callback:
                command_topic = self.iot_settings['command_topic'].format(thing_name=self.iot_settings['thing_name'])
                logger.info(f"訂閱命令 Topic: {command_topic}")
                subscribe_future, packet_id = self.mqtt_connection.subscribe(
                    topic=command_topic,
                    qos=1, # 設置 QoS 等級
                    callback=self._on_mqtt_message # 收到訊息時調用內部回調
                )
                # 非阻塞等待訂閱完成
                subscribe_result = subscribe_future.result(timeout=5)
                logger.info(f"成功訂閱。Packet ID: {packet_id}, Result: {str(subscribe_result['qos'])}")

        except TimeoutError:
            logger.error("連接 AWS IoT Core 超時！請檢查 endpoint 和網絡連接。")
        except FileNotFoundError as e:
             logger.error(f"證書檔案找不到: {e}。請檢查路徑。")
        except Exception as e:
            logger.error(f"連接或訂閱 AWS IoT Core 時發生錯誤: {e}", exc_info=True)
            self._is_connected = False

    def _on_mqtt_message(self, topic, payload, **kwargs):
        """
        內部回調函數，處理收到的 MQTT 訊息。
        """
        logger.info(f"收到 MQTT 訊息 - Topic: {topic}")
        try:
            payload_str = payload.decode('utf-8')
            # logger.debug(f"Payload: {payload_str}") # 訊息較多時可能會影響性能，DEBUG 級別輸出

            # 如果設定了外部的回調函數，則調用它
            if self.command_callback:
                self.command_callback(topic, payload_str)

        except Exception as e:
            logger.error(f"處理 MQTT 訊息時發生錯誤: {e}", exc_info=True)

    def _on_connection_interrupted(self, connection, error, **kwargs):
        """
        連接中斷時的回調函數。
        """
        logger.warning(f"AWS IoT Core 連接中斷: {error}")
        with self._connection_lock:
             self._is_connected = False

    def _on_connection_resumed(self, connection, return_code, session_present, **kwargs):
        """
        連接恢復時的回調函數。
        """
        logger.info(f"AWS IoT Core 連接恢復。Return Code: {return_code}, Session Present: {session_present}")
        with self._connection_lock:
             self._is_connected = True

    def publish_event(self, event_payload: dict) -> Future:
        """
        將事件訊息發布到 AWS IoT Core 的事件 Topic。
        Args:
            event_payload (dict): 包含事件數據的字典。
        Returns:
            Future: MQTT 發布操作的 Future 對象。
        """
        with self._connection_lock:
            if not self._is_connected:
                logger.warning("MQTT 連接未建立或已中斷，無法發布事件。")
                # 返回一個已完成的 Future，表示失敗
                f = Future()
                f.set_exception(ConnectionError("MQTT connection is not available"))
                return f

        try:
            event_topic = self.iot_settings['event_topic'].format(thing_name=self.iot_settings['thing_name'])
            payload_json = json.dumps(event_payload)
            # logger.debug(f"發布事件到 {event_topic}: {payload_json}") # 訊息較多時，DEBUG 級別輸出

            # 發布訊息，並返回 Future 對象，可以選擇等待其結果或不等待
            publish_future = self.mqtt_connection.publish(
                topic=event_topic,
                payload=payload_json,
                qos=1 # 設置 QoS 等級
            )
            return publish_future

        except Exception as e:
            logger.error(f"發布 MQTT 事件時發生錯誤: {e}", exc_info=True)
            # 返回一個已完成的 Future，表示失敗
            f = Future()
            f.set_exception(e)
            return f

    def is_connected(self) -> bool:
        """
        檢查 MQTT 連接是否建立。
        """
        with self._connection_lock:
            return self._is_connected

    def disconnect(self):
        """
        中斷與 AWS IoT Core 的連接。
        """
        logger.info("請求中斷 AWS IoT Core 連接...")
        if self.mqtt_connection:
            try:
                disconnect_future = self.mqtt_connection.disconnect()
                disconnect_future.result(timeout=5) # 設置斷開超時時間
                logger.info("成功中斷 AWS IoT Core 連接。")
            except Exception as e:
                logger.error(f"中斷 AWS IoT Core 連接時發生錯誤: {e}")
            finally:
                with self._connection_lock:
                     self._is_connected = False