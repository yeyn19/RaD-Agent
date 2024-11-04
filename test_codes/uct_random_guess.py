import random
from tqdm import tqdm


def random_guess(data):
    result = 0
    total = 0
    for i in tqdm(range(10000)):
        total += 1
        result += random_guess_uct(data)
    print(result/total)

def random_guess_uct(data):
    now_position = 0
    for i in range(10):
        if random.random() < data[now_position]:
            now_position += 1
        if now_position == len(data):
            return 1
    return 0


if __name__ == "__main__":
    random_guess([1/24, 1/12,1/4])