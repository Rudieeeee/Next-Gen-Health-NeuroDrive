import sys

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QGridLayout, QStackedWidget
)
from PyQt5.QtCore import Qt

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
        self.init_main_menu()
        self.init_detail_pages()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.stack)

    def init_main_menu(self):
        menu = QWidget()
        grid = QGridLayout()
        grid.setSpacing(10)

        buttons = ['Forward', 'Axis', 'Tracking', 'Setting']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for name, (row, col) in zip(buttons, positions):
            btn = QPushButton(name)
            btn.setStyleSheet("font-size: 20px;")
            btn.setMinimumSize(200, 200)
            btn.clicked.connect(lambda _, n=name: self.show_page(n))
            grid.addWidget(btn, row, col)

        menu.setLayout(grid)
        self.stack.addWidget(menu)  # This is index 0

    def init_detail_pages(self):
        self.pages = {}
        for name in ['Forward', 'Axis', 'Tracking', 'Setting']:
            page = make_detail_screen(name, self.go_home)
            self.pages[name] = page
            self.stack.addWidget(page)  # index 1, 2, 3, 4...

    def show_page(self, name):
        index = list(self.pages.keys()).index(name) + 1  # +1 because index 0 is the menu
        self.stack.setCurrentIndex(index)

    def go_home(self):
        self.stack.setCurrentIndex(0)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
