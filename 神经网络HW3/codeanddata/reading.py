import pickle

'''
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print('\n')

for item in list(data.keys()):
    print(item)
    print(data[item]['data'].shape)
    print(data[item]['label'].shape)
    print('\n')
'''
def load_data():
    with open('data.pkl', 'rb') as f:
        data0 = pickle.load(f)
    print(data0.keys())
    print('\n')

    for item in list(data0.keys()):
        print(item)
        print(data0[item]['data'].shape)
        print(data0[item]['label'].shape)
        print('\n')
    return data0

def data_transfer(data, item):
    data_tar_domain = []
    label_tar_domain = []
    data_src_domain = []
    label_src_domain = []
    for item1 in list(data.keys()):
        if item1 != item :
            data_src_domain.extend(data[item1]['data'])
            label_src_domain.extend(data[item1]['label'])
            data_tar_domain.extend(data[item]['data'])
            label_tar_domain.extend(data[item]['label'])
    return data_src_domain, label_src_domain,data_tar_domain,label_tar_domain

def data_transfer_svm(data, item):
    data_tar_domain = data[item]['data']
    label_tar_domain = data[item]['label']
    data_src_domain = []
    label_src_domain = []
    for item1 in list(data.keys()):
        if item1 != item :
            data_src_domain.extend(data[item1]['data'])
            label_src_domain.extend(data[item1]['label'])            
    return data_src_domain, label_src_domain,data_tar_domain,label_tar_domain
        