import mongoengine
import dabase
import datetime
import time
import re
#db = peewee.MySQLDatabase("parlamentares", host="localhost", user="root", passwd="", port="3306")
db = mongoengine.connect('parlamentares')
dm = dabase.connect('mysql://root:@localhost:3306/parlamentares')


class Votacao(mongoengine.Document):

    nome = mongoengine.StringField()
    presenca = mongoengine.StringField()
    partido = mongoengine.StringField()
    estado = mongoengine.StringField()
    hora = mongoengine.ComplexDateTimeField()
    data = mongoengine.ComplexDateTimeField()

class Twitter(mongoengine.Document):

    user_name = mongoengine.StringField()
    id_parlamentar = mongoengine.IntField()
    condicao = mongoengine.StringField()
    tweet_hora = mongoengine.ComplexDateTimeField() 
    tweet_data = mongoengine.ComplexDateTimeField()



def create_corr():
    
    config.gThingCollection.find({"name":{"$regex":regex, "$options":"i"}})
    #all_tw = Twitter.objects(tweet_data__gt=datetime.datetime(2014,10,04),tweet_data__lt=datetime.datetime(2015,10,04))
    for voto in Votacao.objects():
        print(voto.data)
        print(voto.nome)
        regex = ".*" + filter + ".*";
        print(Twitter.objects( user_name=re.compile(regex, re.IGNORECASE)voto.nome).count())
        time.sleep(0.1)

def povoar():
    dm.connect()
 #   for voto in dabase.Votacao.select(): 
 #       vt = Votacao()
 #      vt.nome = voto.nome
 #       vt.presenca = voto.presenca
  #      vt.partido = voto.partido
 #       vt.hora = voto.hora
 #       vt.data = voto.data
 #       vt.save() 
    for tweet in dabase.Twitter.select():
        tw = Twitter()
        tw.user_name = tweet.user_name
        tw.id_parlamentar = tweet.id_parlamentar
        tw.condicao = tweet.condicao
        if (tweet.tweet_hora != None):
            tw.tweet_data = tweet.tweet_data
        else:
          tw.tweet_data = datetime.datetime(0001,01,01)
        if (tweet.tweet_hora != None):
            tw.tweet_hora =  tweet.tweet_hora 
        else:
            tw.tweet_hora = datetime.datetime(0001,01,01)
        tw.save()