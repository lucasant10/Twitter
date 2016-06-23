from text_processor import TextProcessor
import json
from scipy.spatial import distance
import networkx as nx
from scipy import stats


def days2time(days):
    #1380844800000  = 04/10/2013, 86400000 = 1 day 
    return 1380844800000+(days*86400000)




      
if __name__=='__main__':

    dir_out = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/plot/grafos/"
    name = "74173_DeputadoEduardoCunha"
    filedir = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/"+name+".json"
    lamb_dir = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/lambdas/lambdats/"+name+"_wsize7.dat"
    tp = TextProcessor()

    with open(filedir) as data_file:
        doc_set = list()
        doc_tw = set()
        dc =set()
        weeks = list()
        dist = list()
        lamb = list()
        inicial = 1
        final = 603
        for line in data_file:
            tweet = json.loads(line)
            created = int(tweet['created_at'])
            if(days2time(inicial) <= created < days2time(final)):
                doc_tw.add(tweet['text'])
                doc_set.append(tweet)
        texts = tp.text_process(doc_tw)        
        corpus, dic = tp.create_corpus(texts)
        ldamodel = tp.generate_lda(corpus, dic, 5)
        #ldamodel = tp.generate_hdp(corpus, dic)
        print(tp.print_topics(ldamodel))

        with open(lamb_dir) as l_file:
            for line in l_file:
                i = int(line.split('|')[2])
                w = int(line.split('|')[0])
                lamb.append(w)
                for s in range(i-1): 
                    lamb.append(w)

        for k in range(inicial, final, 7):
            doc = set()
            for tw in doc_set:
                if(days2time(k) <= tw['created_at'] < days2time(k+7)):
                    doc.add(tw['text'])
            documents = tp.text_process(doc)
            documents = [' '.join(w) for w in documents]        
            bow = ldamodel.id2word.doc2bow(documents)
            lda_vector = ldamodel[bow]
            vector = [v[1] for v in lda_vector]
            weeks.append(vector)
        for i in range(len(weeks)-1):
            dist.append(round(distance.euclidean(weeks[i],weeks[i+1]),2))


        s_lamb = nx.Graph()
        d_lamb = nx.Graph()
        s_ks = list()
        d_ks = list()
        print(lamb)
        for k in range(inicial, final-7, 7):
            l = lamb[(k//7)] 
            if(lamb[(k//7)+1] == l ):  
                s_lamb.add_edge(k, k+7, weight= dist[(k//7)])
                s_ks.append(dist[k//7])
            else:
                d_lamb.add_edge(k, k+7, weight= dist[(k//7)])
                d_ks.append(dist[k//7])

        f =  open(dir_out+name+".txt", 'w+')
        f.write(str(stats.ks_2samp(s_ks,d_ks)))
        f.close()


        """
        #nx.write_gml(s_lamb,dir_out+name+"_mesmo.gml")
        #nx.write_gml(d_lamb,dir_out+name+"_diferente.gml")

        pos=nx.spring_layout(s_lamb)
        weights = nx.get_edge_attributes(s_lamb,'weight')
        nx.draw_networkx_nodes(s_lamb,pos,node_size=150)
        nx.draw_networkx_edge_labels(s_lamb,pos,edge_labels=weights)
        nx.draw_networkx_edges(s_lamb,pos)
        plt.savefig(dir_out+"s_lambda.png") 

        #       plt.scatter(s_ks, stats.norm.cdf(s_ks), s=50, zorder=1)
        plt.scatter(d_ks, stats.norm.cdf(d_ks), s=50, zorder=1, color='g')

        rv = stats.norm()
        pL = np.linspace(-1, 1)
        plt.plot(pL, rv.cdf(pL), '--')

        # vertical lines
        dL = list()
        for x,y0 in zip(s_ks, stats.norm.cdf(s_ks)):
            y1 = rv.cdf(x)
            dL.append(abs(y1-y0))
            plt.plot((x,x),(y0,y1),
                color='r', zorder=0)
        for x,y0 in zip(d_ks, stats.norm.cdf(d_ks)):
            y1 = rv.cdf(x)
            dL.append(abs(y1-y0))
            plt.plot((x,x),(y0,y1),
                color='r', zorder=0)

        plt.savefig(dir_out+name+".png")
        """


