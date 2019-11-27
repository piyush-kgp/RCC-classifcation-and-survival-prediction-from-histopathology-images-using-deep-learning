
from PIL import Image
import glob
import threading
import argparse

parser = argparse.ArgumentParser(description='Process args for Image Testing')
parser.add_argument("--root_dir", type=str, required=True, help="Root directory")


def test(fp):
    try:
        _  = Image.open(fp).load()
    except:
        print("Found bad file {}".format(fp), flush=True)

def main():
    args = parser.parse_args()
    root_dir = args.root_dir
    files = glob.glob("{}/*/*/*/*.png".format(root_dir))

    threads=[]
    for fp in files:
        t = threading.Thread(target=test, args=(fp,))
        threads.append(t)

    thread_groups = [threads[i*10:(i+1)*10] for i in range(len(threads)//10+1)]
    for grp in thread_groups:
        for t in grp:
            t.start()
        for t in grp:
            t.join()

if __name__=="__main__":
    main()
