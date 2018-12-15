from collections import OrderedDict
n=int(input())
d={}
e={}
d=OrderedDict()
for i in range(n):
    a=input().split()
    key=" ".join(a[:len(a)-1:1])
    if key not in d:
        d[key]=int(a[len(a)-1])
    else:
        d[key]+=int(a[len(a)-1])    
    
    

for k in d:
    print(k,d[k])

