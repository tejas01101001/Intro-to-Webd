a=input().split()
for x in a:
    b=[]
    for i in range(len(x)-1,2,-3):
        b.append(x[i:i-3:-1][::-1])
    n=int(len(x)%3)
    if n==0:
        b.append(x[0:3:1])
    else:
        b.append(x[0:n:1])
    b=b[::-1]
    b=",".join(b)
    print(b)

