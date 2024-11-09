
from torch import cuda


def main():
    print("Cuda available: ", cuda.is_available())
    print("Cuda number of devices: ", cuda.device_count())
    print("Cuda device: ", cuda.get_device_name(cuda.current_device()))


if __name__ == "__main__":
    main()