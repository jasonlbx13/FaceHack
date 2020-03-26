import os
import argparse

parser = argparse.ArgumentParser(description='rename pictures')
parser.add_argument('--file_dic', type=str,default="./workspace/data_src/aligned", help="image file direction")
parser.add_argument('--num', type=int, default="5", help="digit num")
args = parser.parse_args()

def main(args):
    img_name_book = sorted(os.listdir(args.file_dic))

    for i, name in enumerate(img_name_book):
        os.rename('{}/{}'.format(args.file_dic, name), '{}/a{}.jpg'.format(args.file_dic, str(i).zfill(args.num)))

    for i, name in enumerate(img_name_book):
        os.rename('{}/a{}.jpg'.format(args.file_dic, str(i).zfill(args.num)), '{}/{}.jpg'.format(args.file_dic, str(i).zfill(args.num)))

    print ('done')

if __name__ == '__main__':

    main(args)