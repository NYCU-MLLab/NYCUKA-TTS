from onedrivedownloader import download
import argparse

def onedrivedownload(config):
    download(config.url, filename=config.filename, unzip=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-u","--url")
    parser.add_argument("-f","--filename")
    args = parser.parse_args()
    onedrivedownload(args)