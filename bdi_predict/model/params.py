import os

BASE_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":
    print(BASE_PROJECT_PATH)
    