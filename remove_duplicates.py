import os
import configparser
import uuid

if __name__=='__main__':

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_remv = path['dir_remv']

    files = ([file for root, dirs, files in os.walk(dir_remv)
        for file in files if file.endswith('.json') ])
    for f in files:
        s = set()
        tmp_file=str(uuid.uuid4())
        out=open(tmp_file, 'w')
        with open(dir_remv+f,'r') as data_file:
            print("processing file: "+f)
            for line in data_file:
                if line not in s:
                    out.write(line)
                    s.add(line)
        data_file.close()
        os.rename(tmp_file,dir_remv+f)
