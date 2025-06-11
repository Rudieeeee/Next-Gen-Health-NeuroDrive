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
    QVBoxLayout, QGridLayout, QStackedWidget,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QPainter, QColor
from hexadecimal import decimal_to_hex 



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

        self.gaze_label = QLabel("Gaze: -, -")
        self.gaze_label.setStyleSheet("font-size: 14px; color: #333;")
        self.layout().addWidget(self.gaze_label)

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
        self.eye_tracker_process = None
        self.launch_eye_tracker()

    def init_main_menu(self):
        self.menu = QWidget()
        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        buttons = ['Forward', 'Axis', 'Tracking', 'Setting']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for name, (row, col) in zip(buttons, positions):
            btn = QPushButton(name)
            btn.setStyleSheet(""
                "QPushButton {"
                "    font-size: 24px;"
                "    background-color: #e0e0e0;"
                "    border-style: outset;"
                "border-width: 2px;"
                "border-color: black;      "
                ""
                "}"
                "QPushButton:hover, QPushButton:checked, QPushButton:focus {"
                "    background-color: #a0c4ff;"
                "    outline: none;            "
                "}"
            )
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.clicked.connect(lambda _, n=name: self.show_page(n))
            layout.addWidget(btn, row, col)
            self.buttons.append(btn)

        self.menu.setLayout(layout)
        self.stack.addWidget(self.menu)

    def init_detail_pages(self):
        self.pages = {}
        for name in ['Forward', 'Axis', 'Tracking', 'Setting']:
            page = make_detail_screen(name, self.go_home)
            self.pages[name] = page
            self.stack.addWidget(page)

    def show_menu(self):
        self.stack.setCurrentWidget(self.menu)

    def show_page(self, name):
        fsm_state_map = {
            "Forward": "Forward",
            "Axis": "Axis",
            "Tracking": "Tracking",
            "Setting": "Setting"
        }
        self.fsm.transition(fsm_state_map.get(name, "MainMenu"))
        self.stack.setCurrentWidget(self.pages[name])

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

    def send_to_canbus(self, current_label, hex_command=None):
        if not self.serial_port and not self.serial_port.is_open:
            return

        if current_label == "Tracking":
            byte_x = hex_command if hex_command is not None else 0x00
            byte_y = 0x64
        elif current_label == "Axis":
            byte_x = hex_command if hex_command is not None else 0x00
            byte_y = 0x00
        elif current_label == "Forward":
            byte_x = 0x00
            byte_y = 0x64
        elif current_label == "Setting":
            byte_x = 0x00
            byte_y = 0x00
        else:
            byte_x = 0x00
            byte_y = 0x00

        try:
            self.serial_port.write(bytes([byte_x, byte_y]))
            print(f"[SERIAL] Sent: [{hex(byte_x)}, {hex(byte_y)}]")
        except Exception as e:
            print(f"[ERROR] Serial send failed: {e}")


    def start_eye_tracking_socket_server(self):
        HOST = '127.0.0.1'
        PORT = 65432
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"[SOCKET] Listening for eye-tracking data on {HOST}:{PORT}...")
        while True:
            conn, _ = server_socket.accept()
            with conn:
                data = conn.recv(1024)
                if data:
                    msg = data.decode('utf-8').strip()
                    try:
                        dx_str, dy_str = msg.split(',')
                        dx, dy = int(dx_str), int(dy_str)
                        self.gaze_label.setText(f"Gaze deviation: {dx}, {dy}")
                        self.fsm.handle_eye_tracking(dx, dy, self)
                    except ValueError:
                        print("[SOCKET] Invalid data:", msg)


    def launch_eye_tracker(self):
        args = [sys.executable, 'GUI/eye-tracking-gui.py']
        if self.debug:
            args.append('--test')
        self.eye_tracker_process = subprocess.Popen(args)


    def closeEvent(self, event):
        if self.eye_tracker_process:
            print("[GUI] Terminating eye tracker subprocess...")
            self.eye_tracker_process.terminate()
            self.eye_tracker_process.wait()
        if self.serial_port and self.serial_port.in_waiting:
            serial_port.close()
        event.accept()


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

        if self.state == "Calibration" and data == '1':
            if hasattr(app, 'last_gaze'):
                app.calibration_screen.set_calibration_point(app.last_gaze)
        elif self.state == "MainMenu":
            if data == '1':
                app.buttons[app.current_index].click()
            elif data == '2':
                QApplication.quit()
        elif self.state == "Tracking" or self.state == "Axis":
            if data == '1':
                app.send_to_canbus('Tracking')
            elif data == '2':
                app.send_to_canbus('Tracking')
                self.transition("MainMenu")
                app.go_home()
        elif self.state == "Forward":
            if data == '1':
                app.send_to_canbus('Forward')
                self.transition("MainMenu")
                app.go_home()
        

        data = 0 #avoids emg from continuously triggering
        

    def handle_eye_tracking(self, dx, dy, app):
        app.last_gaze = (dx, dy)

        if self.state == "MainMenu":
            self._update_selected_button((dx, dy), app)
        elif self.state == "Tracking" or self.state == "Axis":
            hex_x, _ = self._transform_and_encode((dx, dy))
            app.send_to_canbus(self.state, hex_x)
        

    def _update_selected_button(self, gaze, app):
        closest = self._closest_calibration_label(gaze)
        index_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        if closest in index_map:
            app.current_index = index_map[closest]
            print(f"[FSM] Gaze selected button: {closest} (index {app.current_index})")

            # Visually highlight the selected button
            for i, btn in enumerate(app.buttons):
                if i == app.current_index:
                    btn.setChecked(True)
                    btn.setFocus(Qt.OtherFocusReason)  # This is the key line
                else:
                    btn.setChecked(False)

    def _closest_calibration_label(self, gaze):
        gx, gy = gaze
        closest = min(self.calibration_data.items(),
                      key=lambda item: math.hypot(gx - item[1][0], gy - item[1][1]),
                      default=(None, None))[0]
        return closest

    def _transform_and_encode(self, gaze):
        
        dx, dy = gaze

        if dx < -20:
            tx = max(dx * 1.67, -100)
        elif dx > 20:
            tx = min(dx * 2.1, 100)
        else:
            tx = 0

        hex_x = int(decimal_to_hex(int(-tx)), 16)  # Use helper
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
        self.setFocusPolicy(Qt.StrongFocus)
        self.label = QLabel("Calibration Mode: Look at red dot and press SPACE", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addStretch()
        self.setLayout(layout)

    def get_scaled_dot_positions(self):
        w = self.width()
        h = self.height()
        return [
            (int(w * 0.2), int(h * 0.2)),  # top-left
            (int(w * 0.8), int(h * 0.2)),  # top-right
            (int(w * 0.2), int(h * 0.8)),  # bottom-left
            (int(w * 0.8), int(h * 0.8)),  # bottom-right
        ]


    def set_calibration_point(self, coords):
        dot = self.calibration_order[self.calibration_index]
        self.calibration_points[dot] = coords
        print(f"[CALIB] Set {dot} to {coords}")
        self.calibration_index += 1
        if self.calibration_index >= len(self.calibration_order):
            self.fsm.calibration_data = self.calibration_points
            print(f"[CALIB] Final calibration data: {self.calibration_points}")
            self.fsm.transition("MainMenu")
            self.parent.show_menu()
        else:
            self.update()


    def paintEvent(self, event):
        dot_positions = self.get_scaled_dot_positions()
        if self.calibration_index < len(dot_positions):
            painter = QPainter(self)
            painter.setBrush(QColor(255, 0, 0))
            x, y = dot_positions[self.calibration_index]
            painter.drawEllipse(QRect(x, y, 30, 30))

    def keyPressEvent(self, event):
        if self.debug and event.key() == Qt.Key_Space:
            if hasattr(self.parent, 'last_gaze'):
                self.set_calibration_point(self.parent.last_gaze)
    



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
 