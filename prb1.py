n=int(input())
a=0;
b=1;
print(a)
print(b)
for i in range (3,n+1,1):
	t=a+b
	a=b
	b=t
	print(t)
