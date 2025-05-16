from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QCheckBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, pyqtSlot
import sys

def main():
    print("main start")
    print(vars(args))
    if args.subcommand is None:
        print("subcommand is None")
        raise ValueError("ERROR: specify the experiment type")
    if args.cuda and not torch.cuda.is_available():
        print("cuda not available")
        raise ValueError("ERROR: cuda is not available, try running on CPU")
    print("run_demo start")

if __name__ == '__main__':
    main()