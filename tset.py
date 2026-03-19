import sys
import serial
import serial.tools.list_ports
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QComboBox, QLabel, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt


# 串口通讯后台线程
class SerialWorker(QThread):
    # 定义信号：波形值, 心率, 血氧
    new_data = pyqtSignal(float, int, int)
    error = pyqtSignal(str)

    def __init__(self, port, baud):
        super().__init__()
        self.port = port
        self.baud = baud
        self.active = True

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.1)
            ser.flush()  # 清空缓存
            while self.active:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    # 期待格式: "波形值,心率,血氧"
                    try:
                        data = line.split(',')
                        if len(data) >= 3:
                            self.new_data.emit(float(data[0]), int(data[1]), int(data[2]))
                    except (ValueError, IndexError):
                        continue
            ser.close()
        except Exception as e:
            self.error.emit(str(e))


class OxygenMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("血氧信号实时监测 (PyQt6)")
        self.resize(1000, 600)

        self.buffer_size = 300  # 屏幕显示的点数
        self.data_x = []
        self.worker = None

        self.setup_ui()

    def setup_ui(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- 工具栏 ---
        top_bar = QHBoxLayout()
        self.combo_ports = QComboBox()
        self.btn_refresh = QPushButton("刷新")
        self.btn_connect = QPushButton("开启监测")

        self.refresh_ports()

        self.label_hr = QLabel("心率: -- BPM")
        self.label_spo2 = QLabel("血氧: -- %")
        # 字体加粗加大
        font = "font: bold 24px; color: {};"
        self.label_hr.setStyleSheet(font.format("#e74c3c"))
        self.label_spo2.setStyleSheet(font.format("#2ecc71"))

        top_bar.addWidget(QLabel("端口:"))
        top_bar.addWidget(self.combo_ports)
        top_bar.addWidget(self.btn_refresh)
        top_bar.addWidget(self.btn_connect)
        top_bar.addStretch()
        top_bar.addWidget(self.label_hr)
        top_bar.addWidget(self.label_spo2)

        # --- 绘图区 ---
        self.plot_view = pg.PlotWidget(title="PPG 脉搏波实时信号")
        self.plot_view.setBackground('w')  # 白色背景，更像医疗仪器
        self.plot_view.showGrid(x=True, y=True)
        self.curve = self.plot_view.plot(pen=pg.mkPen(color='b', width=2))

        layout.addLayout(top_bar)
        layout.addWidget(self.plot_view)

        # 绑定事件
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect.clicked.connect(self.handle_connection)

    def refresh_ports(self):
        self.combo_ports.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.combo_ports.addItems(ports)

    def handle_connection(self):
        if self.worker and self.worker.isRunning():
            self.worker.active = False
            self.btn_connect.setText("开启监测")
        else:
            port = self.combo_ports.currentText()
            if not port:
                QMessageBox.warning(self, "错误", "未找到有效串口！")
                return

            self.worker = SerialWorker(port, 115200)
            self.worker.new_data.connect(self.update_plot)
            self.worker.error.connect(lambda e: QMessageBox.critical(self, "串口错误", e))
            self.worker.start()
            self.btn_connect.setText("停止监测")

    def update_plot(self, val, hr, spo2):
        self.data_x.append(val)
        if len(self.data_x) > self.buffer_size:
            self.data_x.pop(0)

        self.curve.setData(self.data_x)
        self.label_hr.setText(f"心率: {hr} BPM")
        self.label_spo2.setText(f"血氧: {spo2} %")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OxygenMonitor()
    win.show()
    sys.exit(app.exec())