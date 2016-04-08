import os
import sys
import datetime
import dabase

class ReadFile:

    def __init__(self, dir_in, dir_out):
        self.dir_in = dir_in
        self.dir_out = dir_out

    def run(self):
        fnames = ([file for root, dirs, files in os.walk(self.dir_in)
            for file in files if file.endswith('.txt') or file.endswith('.TXT')  ])
        self.save_db(fnames)


    def save_file(self, fnames):
        name = self.dir_in.split("/")[-2]
        f =  open(dir_in+name+".txt", 'a+')
        for file in fnames:
            if file.startswith('LP'):
                with open(self.dir_in+"HE"+file[2:]) as data_file:
                    lines = data_file.readlines()
                    date = lines[2]
                    hour = lines[3]
                with open(self.dir_in+file) as fl:
                    for line in fl:
                        ln = line[17:93]
                        ln = ln.replace("Solidaried", "SDD       ")
                        ln = ln.replace("<------>", "Ausente ")
                        f.write(ln+" "+hour+" "+date+"\n")
        f.close()

    def save_db(self, fnames):

        name = ""
        presenca = ""
        partido = ""
        estado=""
        for file in fnames:
            if file.startswith('LP'):
                with open(self.dir_in+"HE"+file[2:]) as data_file:
                    lines = data_file.readlines()
                    date = datetime.datetime.strptime(lines[2][0:10], '%d/%m/%Y')
                    hour = lines[3]
                with open(self.dir_in+file) as fl:
                    for line in fl:
                        line = line.replace("Solidaried", "SDD       ")
                        line = line.replace("<------>", "Ausente ")
                        line[17:93]
                        name = line[17:56].decode('latin-1').encode("utf-8")
                        presenca = line[57:65]
                        partido = line[66:75]
                        estado = line[76:100].decode('latin-1').encode("utf-8")
 
                        vt = dabase.Votacao(nome=name, presenca=presenca, partido=partido, estado=estado, data=date, hora=hour)
                        vt.save()
            


if __name__=='__main__':

    dir_in = "/Users/lucasso/Downloads/Outubro15/"
    dir_out = "/Users/lucasso/Downloads/Dezembro14/"
    rf = ReadFile(dir_in, dir_out)
    rf.run()