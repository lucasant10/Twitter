class F_map:

    fmap = {
        # classification method
        'CNN': 0, 'LSTM': 1, 'FAST_TEXT': 2,
        # word_embeddings
        'cbow_s100.txt': 0, 'glove_s100.txt': 1, 'skip_s100.txt': 2, 'cbow_s300.txt': 3, 'glove_s300.txt': 4,
        # embedding_size
        100: 0, 300: 1,
        # sample dimension
        100: 0, 1000: 1, 2000: 2,
        # dispersion
        'random': 0, 'few_months': 1, 'few_parls': 2,
        # political condition
        ('novos', 'reeleitos'): '1, 1, 0', ('reeleitos', 'nao_eleitos'): '0, 1, 1', ('novos','nao_eleitos'): '1, 0, 1',
        # data label
        'politics': 0, 'non_politics': 1, 'all': 2

    }

    @staticmethod
    def get_id(feature):
        return F_map.fmap[feature]
