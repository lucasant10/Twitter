from collections import Counter
import os
import pickle
import numpy as np
import math

class WordTable():

    def read_counters(self, dir_in):

        counter_list = list()
        tot_counter = Counter()
        pck = ([file for root, dirs, files in os.walk(dir_in)
            for file in files if file.endswith('.pck') ])

        for i,counter_file in enumerate(pck):
            print("processando o arquivo: "+counter_file+"\n")
            with open(dir_in+counter_file, 'rb') as data_file:
                tw_counter = pickle.load(data_file)
                tot_counter += tw_counter
                counter_list.append(tw_counter)
        return tot_counter, counter_list, pck

    
    def tf(self, word, w_counter):
        return (w_counter[word] / float(sum(w_counter.values())))

    def n_containing(self, word, doc_counter):
        count = 0
        for document in doc_counter:
            if document[word] > 0:
                count += 1
        return count


    def idf(self, word, doc_counter):
        return (math.log(len(doc_counter) / float(self.n_containing(word, doc_counter))))

    def tfidf( self, word, w_counter, doc_counter):
        return (self.tf(word, w_counter) * self.idf(word, doc_counter))

    @staticmethod
    def entropy( word, tot_counter, counter_list):
        print("processing entropy...")
        ent = 0
        for counter in counter_list:
            prob = counter[word]/tot_counter[word]
            ent += prob * (-np.log2(prob+1e-100))
        return ent
    
    @staticmethod
    def ratio( word, dep_counter, randon_counter):
        print("processing ratio...")
        w_ratio = dep_counter[word]/(randon_counter[word]+1)
       
        return w_ratio
    
    @staticmethod
    def loadCounter(file_path):
        print("processing loadCounter...")
        with open(file_path, 'rb') as handle:
            t_count = pickle.load(handle)
        return t_count

    @staticmethod
    def num_word_rep(word, counter_list):
        print("processing num_word_rep...")
        n_dep = 0
        for counter in counter_list:
                if word in counter:
                    n_dep += 1
        return n_dep

    def table(self, tot_counter, counter_list, radon_counter):
        out = "Word|Count|#dep_w|TfIdf|Ratio|Entropy|\n\n"
        for k,v in tot_counter.items():
            print("processing word: "+k)
            out += k+"|"+str(v)+"|"+str(self.num_word_rep(k,counter_list))+"|"+str(self.tfidf(k,tot_counter,counter_list))+"|"
            out += str(self.ratio(k,tot_counter,radon_counter))+"|"+str(self.entropy(k, tot_counter,counter_list ))+"|\n"
        
        print("saving file ...")
        with open(dir_out+"tabela.txt", "w") as f:
         f.write(out)
         f.close()
    
    def table_parl_tfidf(self, tot_counter, counter_list, par_list):
        head = "Palavra"
        out = str()
    
        for k,v in tot_counter.items():
            print("processing word: "+k)
            out+= "\n"+k+"|"
            for counter in counter_list:
                if len(counter)!=0:
                    out+= str(self.tfidf(k,counter,counter_list))+"|"
                else:
                    out+="|-|"
        for parl in par_list:
            head += "|"+parl.split('_')[1]+"|"
        print("saving file ...")
        with open(dir_out+"tabela_parl_tfidf.txt", "w") as f:
         f.write(head+out)
         f.close()



            


if __name__=='__main__':

    dir_in = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/Tf-Idf/Counter_dep/"
    dir_out = "/Users/lucasso/Documents/"
    file_path = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/plot/Counter_pt-br.pck"
    word_table = WordTable()
    tot_counter, counter_list, pck = word_table.read_counters(dir_in)
    radon_counter = word_table.loadCounter(file_path)
    #word_table.table(tot_counter, counter_list, radon_counter)
    word_table.table_parl_tfidf(tot_counter, counter_list, pck)




