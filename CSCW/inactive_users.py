import pymongo

class Inactive_Users():

    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017")
        self.db = self.client.twitterdb


    def inactive_users(self):
        users = self.db.tweets.aggregate(
               [{'$match':{'created_at':  {'$gte': 1380596400000, '$lt': 1443668400000},'cond_55': {'$exists': True}}},
                   {'$group': {'_id': "$user_id", 'count':{'$sum':1}}}] )
        tmp =list()
        for u in users:
            if u["count"] <= 87:
               tmp.append(u['_id'])
        return tmp


