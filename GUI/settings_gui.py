import sys
import serial
import threading
import socket
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QGridLayout, QStackedWidget
)
from PyQt5.QtCore import Qt, QTimer


class StateMachine:
    def __init__(self):
        self.state = "MainMenu"

    def transition(self, new_state):
        print(f"[FSM] Transition: {self.state} -> {new_state}")
        self.state = new_state

    def handle_emg(self, cmd, app):
        if self.state == "MainMenu":
            if cmd == 1:
                app.buttons[app.current_index].click()
            elif cmd == 2:
                app.go_home()
        elif self.state == "Tracking":
            if cmd == 1:
                app.send_to_canbus(0x00)
            elif cmd == 2:
                app.send_to_canbus(0x00)
                self.transition("MainMenu")
                app.go_home()

    def handle_eye_tracking(self, xx, yy, app):
        if self.state == "MainMenu":
            app.eyetracking_menu_cursor(xx, yy)
        elif self.state == "Tracking":
            hex_command = self.lookup_hex_command(xx, yy)
            app.send_to_canbus(hex_command)

    def lookup_hex_command(self, xx, yy):
        if xx < 300 and yy < 300:
            return 0xA1
        elif xx >= 300 and yy < 300:
            return 0xA2
        elif xx < 300 and yy >= 300:
            return 0xA3
        else:
            return 0xA4


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


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSM Navigation")
        self.setGeometry(100, 100, 600, 600)

        self.stack = QStackedWidget()
        self.buttons = []
        self.current_index = 0

        self.init_main_menu()
        self.init_detail_pages()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.stack)

        self.fsm = StateMachine()
        self.debug = True

        self.emg_serial_port = None
        self.canbus_serial_port = None

        if self.debug:
            try:
                self.emg_serial_port = serial.Serial('COM3', 9600, timeout=0.1)
                print("EMG serial port connected.")
            except serial.SerialException as e:
                print(f"Failed to connect to EMG serial port: {e}")

            try:
                self.canbus_serial_port = serial.Serial('COM4', 9600, timeout=0.1)
                print("CANBUS serial port connected.")
            except serial.SerialException as e:
                print(f"Failed to connect to CANBUS serial port: {e}")

        if self.debug and self.emg_serial_port:
            self.timer = QTimer()
            self.timer.timeout.connect(self.read_emg_data)
            self.timer.start(100)

        if self.debug:
            threading.Thread(target=self.start_eye_tracking_socket, daemon=True).start()

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

    def read_emg_data(self):
        if self.emg_serial_port.in_waiting > 0:
            line = self.emg_serial_port.readline().decode('utf-8').strip()
            try:
                cmd = int(line)
                self.fsm.handle_emg(cmd, self)
            except ValueError:
                print(f"Invalid EMG data: {line}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            print("Exiting...")
            QApplication.quit()

    def show_page(self, name):
        if name == "Tracking":
            self.fsm.transition("Tracking")
        else:
            self.fsm.transition("MainMenu")
        index = list(self.pages.keys()).index(name) + 1
        self.stack.setCurrentIndex(index)

    def go_home(self):
        self.fsm.transition("MainMenu")
        self.stack.setCurrentIndex(0)

    def eyetracking_menu_cursor(self, xx, yy):
        if yy < 300:
            row = 0
        else:
            row = 1
        if xx < 300:
            col = 0
        else:
            col = 1
        self.current_index = row * 2 + col

    def start_eye_tracking_socket(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', 65432))
        server_socket.listen(1)
        print("Eye-tracking socket server listening on port 65432")

        while True:
            conn, addr = server_socket.accept()
            with conn:
                data = conn.recv(1024)
                if data:
                    coords = data.decode('utf-8').split(',')
                    try:
                        xx, yy = int(coords[0]), int(coords[1])
                        self.fsm.handle_eye_tracking(xx, yy, self)
                    except ValueError:
                        print("Invalid eye-tracking data:", data)

    def send_to_canbus(self, hex_command):
        if self.canbus_serial_port and self.canbus_serial_port.is_open:
            self.canbus_serial_port.write(bytes([hex_command]))
            print(f"Sent to CANBUS: {hex(hex_command)}")


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
