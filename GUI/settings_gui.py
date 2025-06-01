import sys
import serial
import threading
import socket
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QGridLayout, QStackedWidget
)
from PyQt5.QtCore import Qt, QTimer


# Function to create a screen with a back button
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
        self.setWindowTitle("Full-Screen Buttons Navigation")
        self.setGeometry(100, 100, 600, 600)

        self.stack = QStackedWidget()
        self.buttons = []
        self.current_index = 0

        self.init_main_menu()
        self.init_detail_pages()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.stack)

       #serial ports for arduino modules
        self.emg_serial_port = serial.Serial('COM3', 9600, timeout=0.1)      # EMG Arduino
        self.canbus_serial_port = serial.Serial('COM4', 9600, timeout=0.1)   # CANBUS Arduino

       #timer for reading EMG data
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_emg_data)
        self.timer.start(100)  # every 100ms

        #socket server for eye tracking
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

    # EMG data reader
    def read_emg_data(self):
        if self.emg_serial_port.in_waiting > 0:
            line = self.emg_serial_port.readline().decode('utf-8').strip()
            print(f"Received EMG: {line}")
            self.handle_emg_command(line)

    #reimplement this as EMG is used for button clicking not movement
    def handle_emg_command(self, cmd):
        rows, cols = 2, 2
        row = self.current_index // cols
        col = self.current_index % cols

        if cmd == "UP":
            row = (row - 1) % rows
        elif cmd == "DOWN":
            row = (row + 1) % rows
        elif cmd == "LEFT":
            col = (col - 1) % cols
        elif cmd == "RIGHT":
            col = (col + 1) % cols
        elif cmd == "CLICK":
            print("EMG CLICK: sending dummy hex 0xB1 to CANBUS")
            self.send_to_canbus(0xB1)
            self.buttons[self.current_index].click()
            return
        elif cmd == "BACK":
            print("EMG BACK: sending dummy hex 0xB2 to CANBUS")
            self.send_to_canbus(0xB2)
            self.go_home()
            return

        self.current_index = row * cols + col
        self.update_selection()

    
    def update_selection(self):
        for i, btn in enumerate(self.buttons):
            btn.setStyleSheet(
                "font-size: 20px; background-color: lightblue;" if i == self.current_index
                else "font-size: 20px;"
            )

    def show_page(self, name):
        index = list(self.pages.keys()).index(name) + 1
        self.stack.setCurrentIndex(index)

    def go_home(self):
        self.stack.setCurrentIndex(0)
        self.update_selection()

    #Socket server for eye-tracking data
    def start_eye_tracking_socket(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', 65432))
        server_socket.listen(1)
        print("Eye-tracking socket server listening on port 65432…")

        while True:
            conn, addr = server_socket.accept()
            with conn:
                data = conn.recv(1024)
                if data:
                    coords = data.decode('utf-8').split(',')
                    try:
                        xx, yy = int(coords[0]), int(coords[1])
                        self.handle_eye_tracking_data(xx, yy)
                    except ValueError:
                        print("Invalid data received:", data)

    # edit to only send hex commands when run in tracking mode 
    def handle_eye_tracking_data(self, xx, yy):
        print(f"Eye-tracking data: xx={xx}, yy={yy}")
        # Determine which button to highlight based on coordinates
        if yy < 300:
            row = 0
        else:
            row = 1
        if xx < 300:
            col = 0
        else:
            col = 1

        self.current_index = row * 2 + col
        self.update_selection()

        # Dummy hex commands for CANBUS: A1, A2, A3, A4
        hex_commands = [0xA1, 0xA2, 0xA3, 0xA4]
        hex_command = hex_commands[self.current_index]
        print(f"Sending hex to CANBUS: {hex(hex_command)}")
        self.send_to_canbus(hex_command)

    # Utility to send hex command to CANBUS Arduino
    def send_to_canbus(self, hex_command):
        if self.canbus_serial_port.is_open:
            self.canbus_serial_port.write(bytes([hex_command]))


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
