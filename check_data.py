with open('/home/huydang/project/post_asr_normalize/punctuation_tf2/Vietnamese_newspapers/load_data.pkl','rb') as f:
    import pickle
    ls = pickle.load(f)

import matplotlib.pyplot as plt
ls_length = []

for l in ls:
    # print(l)
    # print(len(l['words']))
    if '<PAD>' in l['words']:
        a = l['words'][:l['words'].index('<PAD>')]
    else: a = l['words']
    # if '0' in l['labels']:
    #     b = l['labels'][:l['labels'].index('0')]
    # else:  b  = l['labels']
    # print(a)
    # print(b)
    ls_length.append(len(a))
    # input()
print(len(ls_length))

plt.hist(ls_length, bins = 20)
plt.show()
plt.savefig('books_read.png')