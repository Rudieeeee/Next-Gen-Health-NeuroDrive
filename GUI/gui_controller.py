# === Updated Main GUI Controller (decoupled & integrated eye-tracking + serial logic) ===

import sys
import os
import serial
import threading
import socket
import math
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QGridLayout, QStackedWidget
)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QPainter, QColor
from hexadecimal import decimal_to_hex  # make sure this module is available

SOCKET_PATH = '/tmp/eyetracker.sock'  # UNIX domain socket path

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSM Navigation")
        self.setGeometry(100, 100, 600, 600)

        self.stack = QStackedWidget()
        self.buttons = []
        self.current_index = 0
        self.debug = True

        self.fsm = StateMachine()
        self.calibration_screen = CalibrationScreen(self, self.fsm, debug=self.debug)
        self.stack.addWidget(self.calibration_screen)
        self.init_main_menu()
        self.init_detail_pages()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.stack)
        self.stack.setCurrentWidget(self.calibration_screen)

        self.serial_port = None
        if not self.debug:
            try:
                self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
                print("Serial port connected.")
            except serial.SerialException as e:
                print(f"Failed to connect to serial port: {e}")

            self.timer = QTimer()
            self.timer.timeout.connect(self.read_serial_data)
            self.timer.start(50)

            threading.Thread(target=self.start_eye_tracking_socket_server, daemon=True).start()
            self.launch_eye_tracker()

    def init_main_menu(self):
        self.menu = QWidget()
        grid = QGridLayout()
        grid.setSpacing(10)

        buttons = ['Fixed', 'Axis', 'Tracking', 'Setting']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for name, (row, col) in zip(buttons, positions):
            btn = QPushButton(name)
            btn.setStyleSheet("font-size: 20px;")
            btn.setMinimumSize(200, 200)
            btn.clicked.connect(lambda _, n=name: self.show_page(n))
            grid.addWidget(btn, row, col)
            self.buttons.append(btn)

        self.menu.setLayout(grid)
        self.stack.addWidget(self.menu)

    def init_detail_pages(self):
        self.pages = {}
        for name in ['Fixed', 'Axis', 'Tracking', 'Setting']:
            page = make_detail_screen(name, self.go_home)
            self.pages[name] = page
            self.stack.addWidget(page)

    def show_menu(self):
        self.stack.setCurrentWidget(self.menu)

    def show_page(self, name):
        self.fsm.transition("Tracking" if name == "Tracking" else "MainMenu")
        index = list(self.pages.keys()).index(name) + 1
        self.stack.setCurrentIndex(index)

    def go_home(self):
        self.fsm.transition("MainMenu")
        self.stack.setCurrentIndex(1)

    def read_serial_data(self):
        if self.serial_port and self.serial_port.in_waiting:
            try:
                line = self.serial_port.readline().decode('utf-8').strip()
                if line:
                    self.fsm.handle_emg(line, self)
            except Exception as e:
                print(f"[ERROR] Serial read: {e}")

    def send_to_canbus(self, hex_command):
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.write(bytes([hex_command, 0x64]))
                print(f"Sent to CANBUS: {hex(hex_command)}")
            except Exception as e:
                print(f"[ERROR] Serial send failed: {e}")

    def start_eye_tracking_socket_server(self):
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_socket.bind(SOCKET_PATH)
        server_socket.listen(1)
        print("[SOCKET] Listening for eye-tracking data...")
        while True:
            conn, _ = server_socket.accept()
            with conn:
                data = conn.recv(1024)
                if data:
                    try:
                        x_str, y_str = data.decode('utf-8').split(',')
                        xx, yy = int(x_str), int(y_str)
                        self.fsm.handle_eye_tracking(xx, yy, self)
                    except ValueError:
                        print("[SOCKET] Invalid data:", data)

    def launch_eye_tracker(self):
        subprocess.Popen(['python3', 'eye-tracking.py'])


def make_detail_screen(name, back_function):
    page = QWidget()
    layout = QVBoxLayout()
    label = QLabel(f"{name} Page")
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet("font-size: 24px;")
    back_btn = QPushButton("Back")
    back_btn.setFixedHeight(50)
    back_btn.clicked.connect(back_function)
    layout.addWidget(label)
    layout.addStretch()
    layout.addWidget(back_btn)
    page.setLayout(layout)
    return page


class StateMachine:
    def __init__(self):
        self.state = "Calibration"
        self.calibration_data = {}

    def transition(self, new_state):
        print(f"[FSM] Transition: {self.state} -> {new_state}")
        self.state = new_state

    def handle_emg(self, data, app):
        print(f"[EMG] Trigger received: {data}")
        if self.state == "MainMenu":
            if data == '1':
                app.buttons[app.current_index].click()
            elif data == '2':
                app.go_home()
        elif self.state == "Tracking":
            if data == '1':
                app.send_to_canbus(0x00)
            elif data == '2':
                app.send_to_canbus(0x00)
                self.transition("MainMenu")
                app.go_home()

    def handle_eye_tracking(self, xx, yy, app):
        if self.state == "MainMenu":
            self._update_selected_button((xx, yy), app)
        elif self.state == "Tracking":
            hex_x, _ = self._transform_and_encode((xx, yy))
            app.send_to_canbus(hex_x)

    def _update_selected_button(self, gaze, app):
        closest = self._closest_calibration_label(gaze)
        index_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        if closest in index_map:
            app.current_index = index_map[closest]

    def _closest_calibration_label(self, gaze):
        gx, gy = gaze
        closest = min(self.calibration_data.items(),
                      key=lambda item: math.hypot(gx - item[1][0], gy - item[1][1]),
                      default=(None, None))[0]
        return closest

    def _transform_and_encode(self, gaze):
        gx, gy = gaze
        ref = self.calibration_data.get('a')  # reference point
        if not ref:
            return 0x00, 0x00
        dx = gx - ref[0]
        dy = gy - ref[1]

        if dx < -20:
            tx = max(dx * 1.67, -100)
        elif dx > 20:
            tx = min(dx * 2.1, 100)
        else:
            tx = 0

        hex_x = int(decimal_to_hex(int(tx)), 16)  # Use helper
        hex_y = 0x64  # Placeholder
        return hex_x, hex_y


class CalibrationScreen(QWidget):
    def __init__(self, parent, fsm, debug=False):
        super().__init__()
        self.parent = parent
        self.fsm = fsm
        self.debug = debug
        self.calibration_index = 0
        self.calibration_order = ['a', 'b', 'c', 'd']
        self.calibration_points = {dot: None for dot in self.calibration_order}
        self.dot_positions = [(100, 100), (500, 100), (100, 400), (500, 400)]
        self.setFocusPolicy(Qt.StrongFocus)
        self.label = QLabel("Calibration Mode: Look at red dot and press SPACE", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addStretch()
        self.setLayout(layout)

    def set_calibration_point(self, coords):
        dot = self.calibration_order[self.calibration_index]
        self.calibration_points[dot] = coords
        print(f"[CALIB] Set {dot} to {coords}")
        self.calibration_index += 1
        if self.calibration_index >= len(self.calibration_order):
            self.fsm.calibration_data = self.calibration_points
            self.fsm.transition("MainMenu")
            self.parent.show_menu()
        else:
            self.update()

    def paintEvent(self, event):
        if self.calibration_index < len(self.dot_positions):
            painter = QPainter(self)
            painter.setBrush(QColor(255, 0, 0))
            x, y = self.dot_positions[self.calibration_index]
            painter.drawEllipse(QRect(x, y, 30, 30))

    def keyPressEvent(self, event):
        if self.debug and event.key() == Qt.Key_Space:
            fake_gaze = self.dot_positions[self.calibration_index]
            self.set_calibration_point(fake_gaze)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
