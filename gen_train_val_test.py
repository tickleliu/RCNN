
import random
train = 0.75
val = 0
test = 0.25
with open('output.txt', 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)
    print(len(lines))
    with open('fine_tune_list.txt', 'w') as train_file:
        for line in lines[0: round(len(lines) * train)]:
            train_file.write(line)
    with open('test_list.txt', 'w') as test_file:
        for line in lines[round(len(lines) * train): -1]:
            test_file.write(line)



