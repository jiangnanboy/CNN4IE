import os
import csv

def process_train(train_source, train_target, save_path):
    with open(train_source, 'r', encoding='utf-8') as source_read, open(train_target, 'r', encoding='utf-8') as target_read:
        with open(save_path, 'w', encoding='utf-8', newline='') as csv_write:
            header = ['label', 'source', 'target']
            csv_writer = csv.writer(csv_write)
            csv_writer.writerow(header)
            for source, target in zip(source_read, target_read):
                csv_writer.writerow([None, source, target])
    print("done!")

if __name__ == '__main__':
    source_path = os.path.join(os.getcwd(), 'source.txt')
    target_path = os.path.join(os.getcwd(), 'target.txt')
    save_path = os.path.join(os.getcwd(), 'train.csv')
    process_train(source_path, target_path, save_path)