import sys
import os

# 假設 fall_detector.py 與 gui.py 在同一目錄下，或者 Project3/combine 已在 PYTHONPATH 中
# For example, if gui.py is in Project3/combine/ and fall_detector.py is also in Project3/combine/
try:
    from fall_detector import FallDetector
except ImportError:
    # 如果 FallDetector 不在標準路徑，嘗試調整 sys.path
    # 這假設 gui.py 在 Project3/combine/ 目錄下
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(current_dir) # 若 fall_detector 在 Project3/
    # sys.path.append(project_root) # 或 sys.path.append(current_dir)
    # from combine.fall_detector import FallDetector # 如果 fall_detector 在 combine 子目錄
    print("無法導入 FallDetector。請確保 fall_detector.py 在 PYTHONPATH 中或與 gui.py 在同一目錄下。", file=sys.stderr)
    sys.exit(1)


from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

import torch.multiprocessing as mp

# 確保 multiprocessing 的啟動方法設定正確，這對於 PyTorch 和 PyQt 同時使用很重要
# 必須在任何 multiprocessing 相關代碼之前，且最好在腳本的頂部
if __name__ == '__main__': # Guard for multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已經設定，可能會拋出 RuntimeError，可以忽略
        pass

# Top-level function for multiprocessing target to avoid PicklingError
def _run_fall_detector_process_target(detector_instance_for_process, queue):
    try:
        detector_instance_for_process.begin()
        queue.put("COMPLETED") # 通知主進程已完成
    except Exception as e_proc:
        # It's good practice to send the full traceback or more detailed error
        # import traceback
        # queue.put(f"ERROR: {str(e_proc)}\n{traceback.format_exc()}")
        queue.put(f"ERROR: {str(e_proc)}") # 通知主進程錯誤

class FallDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.fall_detector_process = None
        self.process_queue = None # Initialize queue attribute
        self.status_update_timer = QTimer(self) # 用於檢查進程狀態
        self.status_update_timer.timeout.connect(self.check_process_status)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('跌倒偵測系統')
        self.setGeometry(200, 200, 450, 250) # 調整視窗大小
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D30;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 15px; /* 調整 padding */
                font-size: 14px;
                font-weight: bold;
                min-height: 30px; /* 設定最小高度 */
            }
            QPushButton:hover {
                background-color: #1C97EA;
            }
            QPushButton:pressed {
                background-color: #00559E;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #AAAAAA;
            }
        """)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel('選擇偵測來源')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Arial', 16, QFont.Bold))
        title_label.setStyleSheet("color: #00AAFF; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.webcam_button = QPushButton('開啟 Webcam 偵測')
        self.webcam_button.clicked.connect(self.start_webcam_detection)
        button_layout.addWidget(self.webcam_button)

        self.stream_button = QPushButton('開啟串流偵測')
        self.stream_button.clicked.connect(self.start_stream_detection)
        button_layout.addWidget(self.stream_button)
        
        main_layout.addLayout(button_layout)

        self.status_label = QLabel('狀態：待命')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #CCCCCC; margin-top: 10px; font-size: 12px;")
        main_layout.addWidget(self.status_label)
        
        self.stop_button = QPushButton('停止偵測')
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F; /* 紅色 */
            }
            QPushButton:hover {
                background-color: #E57373;
            }
            QPushButton:pressed {
                background-color: #B71C1C;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #AAAAAA;
            }
        """)
        main_layout.addWidget(self.stop_button)

    def _start_detection_process(self, args_modifier_func):
        if self.fall_detector_process and self.fall_detector_process.is_alive():
            self.status_label.setText('狀態：偵測已在運行中！')
            self.status_label.setStyleSheet("color: #FFAA00; font-size: 12px;") # 黃色警告
            return

        try:
            detector_instance = FallDetector() 
            args = detector_instance.args 

            args_modifier_func(args)
            
            detector_instance.args = args
            
            # 使用 Queue 來接收來自子進程的消息
            self.process_queue = mp.Queue()
            # Pass the top-level function as the target
            self.fall_detector_process = mp.Process(target=_run_fall_detector_process_target, 
                                                    args=(detector_instance, self.process_queue))
            self.fall_detector_process.daemon = True 
            self.fall_detector_process.start()
            
            self.status_label.setText('狀態：偵測已啟動')
            self.status_label.setStyleSheet("color: #55FF55; font-size: 12px;") # 綠色
            self.webcam_button.setEnabled(False)
            self.stream_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_update_timer.start(1000) # 每秒檢查一次進程狀態

        except Exception as e:
            error_msg = f'啟動錯誤：{str(e)}'
            self.status_label.setText(error_msg)
            self.status_label.setStyleSheet("color: #FF5555; font-size: 12px;") # 紅色錯誤
            print(error_msg, file=sys.stderr)
            # import traceback
            # traceback.print_exc() # For more detailed error logging
            
            if self.fall_detector_process and self.fall_detector_process.is_alive():
                self.fall_detector_process.terminate()
                self.fall_detector_process.join()
            self.fall_detector_process = None
            if self.process_queue:
                self.process_queue.close() # Close the queue
                self.process_queue.join_thread() # Ensure queue feeder thread is joined
            self.process_queue = None
            self._reset_buttons_and_status()

    def start_webcam_detection(self):
        self.status_label.setText('狀態：正在啟動 Webcam 偵測...')
        self.status_label.setStyleSheet("color: #FFFF00; font-size: 12px;") # 黃色
        def modify_args_for_webcam(args):
            args.video = 0 
            args.num_cams = 1
            args.save_output = False 
            args.plot_graph = False  
            args.disable_cuda = False 
        self._start_detection_process(modify_args_for_webcam)

    def start_stream_detection(self):
        self.status_label.setText('狀態：正在啟動串流偵測...')
        self.status_label.setStyleSheet("color: #FFFF00; font-size: 12px;") # 黃色
        
        stream_url = "rtmp://140.116.215.49:1935/live" 
        
        def modify_args_for_stream(args):
            args.video = stream_url 
            args.num_cams = 1
            args.save_output = False
            args.plot_graph = False
            args.disable_cuda = False
        self._start_detection_process(modify_args_for_stream)

    def stop_detection(self):
        self.status_update_timer.stop()
        if self.fall_detector_process and self.fall_detector_process.is_alive():
            try:
                self.status_label.setText('狀態：正在停止偵測...')
                self.status_label.setStyleSheet("color: #FFAA00; font-size: 12px;") 
                QApplication.processEvents() 

                self.fall_detector_process.terminate() 
                self.fall_detector_process.join(timeout=5) 
                
                if self.fall_detector_process.is_alive():
                    print("警告：跌倒偵測進程未能優雅終止，可能需要強制結束。", file=sys.stderr)
                
                self.status_label.setText('狀態：偵測已停止')
                self.status_label.setStyleSheet("color: #CCCCCC; font-size: 12px;")
            except Exception as e:
                error_msg = f'停止時發生錯誤: {str(e)}'
                self.status_label.setText(error_msg)
                self.status_label.setStyleSheet("color: #FF5555; font-size: 12px;")
                print(error_msg, file=sys.stderr)
            finally:
                self.fall_detector_process = None
                if self.process_queue:
                    self.process_queue.close()
                    self.process_queue.join_thread()
                self.process_queue = None
                self._reset_buttons_and_status(final_status='狀態：偵測已停止')
        else:
            self._reset_buttons_and_status(final_status='狀態：沒有偵測正在運行')
            
    def _reset_buttons_and_status(self, final_status='狀態：待命'):
        self.webcam_button.setEnabled(True)
        self.stream_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText(final_status)
        if "錯誤" in final_status or "Error" in final_status.upper(): # Make error check case-insensitive
             self.status_label.setStyleSheet("color: #FF5555; font-size: 12px;")
        else:
            self.status_label.setStyleSheet("color: #CCCCCC; font-size: 12px;")


    def check_process_status(self):
        if self.fall_detector_process and not self.fall_detector_process.is_alive():
            self.status_update_timer.stop()
            exit_code = self.fall_detector_process.exitcode
            final_message = f'狀態：偵測已結束 (代碼: {exit_code})'
            
            if self.process_queue:
                try:
                    # Drain the queue to get the last message
                    last_message_from_proc = None
                    while not self.process_queue.empty():
                        last_message_from_proc = self.process_queue.get_nowait()

                    if last_message_from_proc:
                        if "ERROR:" in last_message_from_proc: # Check for "ERROR:" prefix
                            final_message = f'狀態：偵測錯誤 - {last_message_from_proc.split("ERROR:", 1)[1].strip()}'
                        elif "COMPLETED" in last_message_from_proc:
                            final_message = '狀態：偵測任務完成'
                except Exception: # Queue empty or other error
                    pass
                finally:
                    self.process_queue.close()
                    self.process_queue.join_thread()
                    self.process_queue = None


            print(f"偵測進程已結束，退出代碼: {exit_code}")
            self.fall_detector_process.join() # Ensure process is fully joined
            self.fall_detector_process = None
            self._reset_buttons_and_status(final_status=final_message)


    def closeEvent(self, event):
        self.stop_detection() 
        if 'torch' in sys.modules and hasattr(sys.modules['torch'], 'cuda') and sys.modules['torch'].cuda.is_available():
            try:
                sys.modules['torch'].cuda.empty_cache()
                print("CUDA cache cleared.")
            except Exception as e:
                print(f"Error clearing CUDA cache: {e}", file=sys.stderr)
        event.accept()


def main_gui():
    # freeze_support() is essential for Windows when creating executables
    # or running scripts that use multiprocessing with 'spawn' or 'forkserver'.
    # It should be called right after the if __name__ == '__main__': line.
    # mp.set_start_method is already handled at the top for __main__ guard.
    # No need to call freeze_support() here if it's in __main__ block.

    app = QApplication(sys.argv)
    
    window = FallDetectionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True) moved to top of script, guarded by if __name__ == '__main__'
    # This is a more robust placement.
    
    if sys.platform.startswith('win'):
         mp.freeze_support() # Call freeze_support() at the beginning of the __main__ block on Windows.

    # Path adjustments (if needed, uncomment and adapt)
    # current_script_path = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.abspath(os.path.join(current_script_path, '..')) # Assuming Project3 is parent of combine
    # if project_root not in sys.path:
    #    sys.path.insert(0, project_root)
    # if current_script_path not in sys.path: # If fall_detector is in the same dir as gui.py
    #    sys.path.insert(0, current_script_path)

    main_gui()
