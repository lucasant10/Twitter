import os
import xlrd

class Profile:

    def __init__(self, dir_out, excel_path, sheet_name):
        self.dir_out = dir_out
        self.excel_path = excel_path
        self.sheet_name = sheet_name

    def condicao(self, condicao):
        cond = -1
        if(condicao == "Candidatos à reeleição, que foram reeleitos"):
            cond = 0
        elif (condicao == "Candidatos à reeleição, que não foram reeleitos"):
            cond = 1
        elif (condicao == "Novos parlamentares"):
            cond = 2
        return cond 


    def save_profile(self):
        xls = xlrd.open_workbook(self.excel_path)
        sheet = xls.sheet_by_name(self.sheet_name)
        for i in range(len(sheet.col(0))):
            nome = sheet.cell_value(rowx= i, colx=4)
            partido = sheet.cell_value(rowx= i, colx=6)
            uf = sheet.cell_value(rowx= i, colx=11)
            votos = sheet.cell_value(rowx= i, colx=14)
            condic = self.condicao(sheet.cell_value(rowx= i, colx=20))
            perfil = sheet.cell_value(rowx= i, colx=36)

            path = dir_out+nome+"/"
            os.mkdir(path)
            f =  open(path+nome+".csv", 'a+')
            f.write("Nome"+";"+"Partido"+";"+"UF"+";"+"Votos"+";"+"Condicao"+";"+"Perfil"+"\n")
            f.write(nome+";"+partido+";"+uf+";"+str(votos)+";"+str(condic)+";"+perfil+"\n")
            f.close()


if __name__=='__main__':

    dir_out = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/profiles/"
    excel_path = "/Users/lucasso/Dropbox/Twitter_Marcelo/Arquivo Principal da Pesquisa - Quatro Etapas.xls"
    sheet_name = "pedro"
    pt = Profile(dir_out, excel_path, sheet_name )
    pt.save_profile()