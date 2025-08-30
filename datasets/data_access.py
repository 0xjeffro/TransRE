import pickle


# ATIS
TRAIN_DATA_DIR = 'ATIS/data/raw_data/ms-cntk-atis/atis.train.pkl'
TEST_DATA_DIR = 'ATIS/data/raw_data/ms-cntk-atis/atis.test.pkl'

def load_atis_ds(filename, log=True):
    with open(filename, 'rb') as stream:
        ds, dicts = pickle.load(stream)
    if log:
        print('Done  loading: ', filename)
        print('      samples: {:4d}'.format(len(ds['query'])))
        print('   vocab_size: {:4d}'.format(len(dicts['token_ids'])))
        print('   slot count: {:4d}'.format(len(dicts['slot_ids'])))
        print(' intent count: {:4d}'.format(len(dicts['intent_ids'])))
    return ds, dicts

def load_atis_train_ds(log=True):
    return load_atis_ds(TRAIN_DATA_DIR, log=log)

def load_atis_test_ds(log=True):
    return load_atis_ds(TEST_DATA_DIR, log=log)

def get_atis_token2idx(log=False):
    ds, dicts = load_atis_train_ds(log=log)
    return dicts['token_ids']

def get_atis_idx2token(log=False):
    ds, dicts = load_atis_train_ds(log)
    return {v: k for k, v in dicts['token_ids'].items()}

def get_atis_label2idx(log=False):
    ds, dicts = load_atis_train_ds(log)
    return dicts['intent_ids']

def get_atis_idx2label(log=False):
    ds, dicts = load_atis_train_ds(log)
    return {v: k for k, v in dicts['intent_ids'].items()}

def print_atis_label_statistics():
    print('ATIS Label Statistics:')
    print('Total number of Train samples:', len(load_atis_train_ds(log=False)[0]['query']))
    print('Total number of Test samples:', len(load_atis_test_ds(log=False)[0]['query']))
    print('Total number of labels:', len(get_atis_label2idx(log=False)))
    print('Total number of tokens:', len(get_atis_token2idx(log=False)))
    # 输出每个label的样本数量
    ds_train, dict_train = load_atis_train_ds(log=False)
    ds_test, _ = load_atis_test_ds(log=False)
    idx2label = get_atis_idx2label(log=False)
    intent_counts_train, intent_counts_test = {}, {}
    for intent_id in ds_train['intent_labels']:
        intent = idx2label[intent_id[0]]
        if intent not in intent_counts_train:
            intent_counts_train[intent] = 0
        intent_counts_train[intent] += 1

    for intent_id in ds_test['intent_labels']:
        intent = idx2label[intent_id[0]]
        if intent not in intent_counts_test:
            intent_counts_test[intent] = 0
        intent_counts_test[intent] += 1

    for intent, intent_id in dict_train['intent_ids'].items():
        train_count = intent_counts_train.get(intent, 0)
        test_count = intent_counts_test.get(intent, 0)
        print(f'{intent_id}: \t  Label: {intent} \t Train/Test Count: {train_count}/{test_count}')

    print('Total number of Train samples:', sum(intent_counts_train.values()))
    print('Total number of Test samples:', sum(intent_counts_test.values()))


## QC Dataset
QC_TRAIN_DATA_DIR = 'QC/train_5500.label'
QC_TEST_DATA_DIR = 'QC/TREC_10.label'


if __name__ == '__main__':
    print_atis_label_statistics()
