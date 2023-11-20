import argparse

def get_parse():
    parser = argparse.ArgumentParser(description="Nailfold Raw Dataset Process", add_help=True)

    parser.add_argument("--input", default=None, type=str, help="dataset path")
    parser.add_argument("--output", default=None, type=str, help="dataset root path")
    
    args = parser.parse_args()
    return args