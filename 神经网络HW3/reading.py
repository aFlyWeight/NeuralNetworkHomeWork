import pickle

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print('\n')

for item in list(data.keys()):
    print(item)
    print(data[item]['data'].shape)
    print(data[item]['label'].shape)
    print('\n')
    
def load_data():
    with open('data.pkl', 'rb') as f:
        data0 = pickle.load(f)
    return data0
