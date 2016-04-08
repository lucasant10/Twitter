import peewee
from playhouse.db_url import connect
import numpy

#db = peewee.MySQLDatabase("parlamentares", host="localhost", user="root", passwd="", port="3306")
db = connect('mysql://root:@localhost:3306/parlamentares')

class Votacao(peewee.Model):

    nome = peewee.CharField()
    presenca = peewee.CharField()
    partido = peewee.CharField()
    estado = peewee.CharField()
    hora = peewee.TimeField(formats=['%H:%M:%S'])
    data = peewee.DateField(formats=['%d/%m/%Y'])


    class Meta:
        database = db

class Twitter(peewee.Model):

    user_name = peewee.CharField()
    id_parlamentar = peewee.IntegerField()
    condicao = peewee.CharField()
    tweet_hora = peewee.TimeField(formats=['%H:%M:%S'])
    tweet_data = peewee.DateField(formats=['%d/%m/%Y'])


    class Meta:
        database = db


def create_tables():
    db.connect()
    db.create_tables([Votacao,Twitter])


def create_corr():
    db.connect()
    presente = []
    num_tw = []
    tw_un=[]
    for tw in Twitter.raw("select distinct user_name from Twitter where condicao like 'Novos parlamentares'"):
        tw_un.append(tw.user_name.lower().replace(" ",""))

    for voto in Votacao.select().where(Votacao.data.between('2013-10-04','2015-10-04')):

        if voto.nome.lower().replace(" ","")  in tw_un:
            if voto.presenca=="Presente": presente.append(1)
            else: presente.append(0)
            num_tw.append(Twitter.select().where(Twitter.user_name==voto.nome, Twitter.tweet_data==voto.data).count())

    print numpy.corrcoef(presente,num_tw)
    db.close()

