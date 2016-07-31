import json
from PtBrTwitter import PtBrTwitter


if __name__=='__main__':

    dir_in = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/"
    dir_out = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/plot/"
    ptbr = PtBrTwitter(dir_in,dir_out)
    round_c = ptbr.loadCounter("Counter_pt-br.pck")
    dep_c = ptbr.loadCounter("Counter_Dep.pck")

    dict_term = dict()
    for k, v in dep_c.items():
        if k in round_c:
            dict_term[k] = v/round_c[k]
        else:
            dict_term[k] = v/0.1
    sort = sorted(dict_term.items(), key=lambda x: x[1], reverse=True)
    with open(dir_out+"ratio.txt", "w") as f: f.write(json.dumps(sort))