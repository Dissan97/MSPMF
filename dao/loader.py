import json


def load_config(conf_file: str) -> dict:
    ret = {}
    try:
        with open(conf_file) as file:
            ret = json.load(file)
    except FileNotFoundError:
        print(f"Configuration file: {conf_file} not found.")

    return ret


