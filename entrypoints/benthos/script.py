import argparse


def f(msg: str):
    with open(f"{msg}.txt", "w") as f:
        f.write("hello!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('message',
                        type=str,
                        help='message to parse')

    args = parser.parse_args()

    f(args.message)
