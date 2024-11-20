from controller.core import Executor
from dao.loader import load_config


def start(conf_file: str):
    executor = Executor(load_config(conf_file=conf_file))
    executor.exec()
    executor.show()
if __name__ == '__main__':
    start(conf_file='config/config.json')
