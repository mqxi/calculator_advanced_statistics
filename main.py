import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.controllers.main_controller import MainController

def main():
    controller = MainController()
    controller.run()

if __name__ == "__main__":
    main()