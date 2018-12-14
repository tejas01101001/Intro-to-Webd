n = int(input())
for i in range(n):
    s = input()
    a=s.split(" ")
    b=[]
    for x in a:
        b.append(x[0])
    
    b="".join(b)
    print(b)

