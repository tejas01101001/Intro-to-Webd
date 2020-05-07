cmd prompt use
py name.py #to execute

(1)print
print("My	first	program")
input()-take input	from	console 
print()-output	to	console 
input()#always give string input
myVariable	=	value	or	expression

(2)Typecasting
n=int(input())
#string converted to int
a	=	'5.82'
b	=	float(a)
#string converted to float
a	=	3 
b	=	str(a)
#int coverted to string

(3)Joining strings
a	=	"one" 
b	=	"two" 
c	=	"three" 
d	=	a+b+c 
e	=	a+"	"+b 
print(d) 
print(e) 
print(e+","+c)

OUTPUT
onetwothree 
one	two 
one	two,three

Note:, puts extra space in print while +doesnot

(4)How	to	avoid	print()	coming	on	next line	automatically
print("line1",end="") 
print("line2",end="@") 
print("line3",end="\n") 
print("line4",end="	") 
print("line5",end=":")

OUTPUT
line1line2@line3
line4 line5:

(5)if elif else
if condition:
	code
elif condition:
	code
.
.
.
else:
	default code

(6)for loop
for countervariable in range(x,y,z):
	code
x=start from
y=end before
z=increment counter by

if z not written assume it to be 1
if x and z both not written 
z=1 and x=0

NOTE: z can be -ve	

for i in range(10,0,-2):
	print(i,end=" ")

output
10 8 6 4 2 

(7)while loop
while condition:				
	statement1				
	statement2
 """
 for multi line comments
 we use triple quotes
 """
(8)Arrays(lists)
#list of int
runs=[67,82,25,10]
#list of strings
names=["daman","stokes","god"]
#lists can be printed directly
runs.append(69)#given item inserted at the end
print(runs)
print(runs[2])
print(runs[-3])
print(runs[-1])
runs.insert(1,111)# 111 inserted at index 1
print(runs)

OUTPUT
[67,82,25,10,69]
25#2 since the first element is index 0
25#-3
69#-1 is the last element
[67,111,82,25,10,69]

(9)Deleting items from list
runs=[56,67,89,34]
del(runs[1])
print(runs)

output
[56,89,34]

(10)Remove a particular element from the list
runs=[56,67,89,34]
runs.remove(67)#remove the first 67 frm the list
print(runs)

(11)Merging lists
a=[2,3,4,5,6]
b=[3,5,8]
c=a+b#we can add multiple lists
print(c)

output
[2,3,4,5,6,3,5,8]

(12)Extending lists
a=[4,8,2,1]
b=[7,2]
#add elements of b to a
a.extend(b)
print(a)#same as a=a+b

output 
[4,8,2,1,7,2]

(13)strings
s[2]='a' kind of statements are not allowed
we can split string into lists on the basis of
some string.

s="this is a sentence"
a=s.split(" ")#split string at every space
print(a)

output 
['this','is','a','sentence']

join is just opp of split().It takes a list of strings and converts it into a string while joining the elements with a given string.

a=["one","two","three"]
s="/".join(a)
print(s)

output
one/two/three

(14)length,sum
a=[2,3,4,2]
b="python"

print(len(a)) #4
print(len(b)) #6
print(sum(a)) #11

(15)minimum
list:smallest element
string:smallest char
list of strings:alphabetically smallest string

a=[3,6,1,0]
b=["phy","chem","maths"]
s="python"

print(min(a)) #0
print(min(b)) #chem
print(min(s)) #h

NOTE:max similar to min


(16)sort
a=[3,6,1,0]
b=["one","two","three"]

a.sort()
b.sort()

print(a)
print(b)

output
[0,1,3,6]
['one','three','two']# dictionary order

(17)find
s="python"

if "h" in s:
	print("present")
else:
	print("not present")

(18)functions
def fac(n):
    p=1
    for i in range (2,n+1):
        p*=i
    return p
k=int(input("enter the number:"))
print(fac(k))

(19)Modules and import
Module is a python file containing func which
we want to use in another python file/program.
(a)Inbuilt Module

import math
b=math.sqrt(float(input()))
print(b)

from math import sin,ceil,floor
#now instead of math.sin we can directly use sin
a=1.123
b=sin(a)
c=ceil(a)
d=floor(a)
print(b)
print(c)
print(d)

NOTE:from math import *
above used to import all func of the module

(b)Making our own module
make a file say "tejas.py" write some func in it
import the above made file and use those func

from tejas import *
code

(20)Mapping
To apply a func on each item in a list,we will
have to go through the list and call that func
on the element.
SHORTCUT is to use map(func,list variable)
also we need to convert result back to list form
so we use list().

def cube(n):
	return n*n*n
a=[3,1,5,6,2]
b=map(cube,a)
b=list(b)
print(b)	

# classic
a=input().split()
b=list(map(int,a))#int() function used
print(a)
print(b)

input
3 4 1 -5 6 

output
['3','4','1','-5','6']
[3,4,1,-5,6]

(21)Logical expressions
and,or,not

(22)Array/String splicing
a=[-34,-20,-3,2,9,94,482,43]
b="this is a sentence"

c=a[2:7:2]#[-3,9,482]
d=b[3:15:3]#ss n

in general a[x:y:z]
x=start index(inclusive)
y=end index(exclusive)
z=jump of index(can be -ve)

if only[x:y]
z by default taken as 1

if x is omitted it is taken as 0 or len(a)#-ve
if y is omitted it is taken as len(a) or 0#-ve

a=[-34,-20,-3,2,9,94,482,3693]
b="this is a sentence"

c=a[::-1](very useful to check palindromes)
#[3693,482,94,9,2,-3,-20,-34]

d=b[:3:-3]# eeeai

(23)List comprehensions
#alternate way to access the elements of a list
for item in a:
	print(item)

To extract even nos from a list

a=[5,49,20,28,95,68,394]

b=[]#empty list	
for val in a:
	if val%2==0:
		b.append(val)
print(b)		

#using comprehensions
#syntax
newlist=[expression(item)for item in oldlist if condition(item)]

a=[5,49,20,28,95,68,394]
b=[val for val in a if val&2==0]
print(b)

ANOTHER use
#fill list with first 10 powers of 2
a=[2**p for p in range(1,11)]
print(a)

(24)Initialize a list
a=[1]*101
#we need a[100],list has 101 elements

(25)Tuples
used to represent objects with 2 or more
attributes:Eg point in 3D space(x,y,z)

(26)zip function
the shortest list will determine how many
elements are to be considered to form the 
Tuples

a=[5,6,23,7,8,9]
b=[4,1,553,1]
c=[4,9,23,1,5,6,78,8]
print(zip(a,b,c))
#[(5,4,4),(6,1,9),(23,553,23),(7,1,5)]

(27)Matrices
a=[[3,-5,-7,9],
   [13,0,-2,1],
   [8,9, -7,1]]
#No of rows    
r=len(a)
#No of columns 
c=len(a[0])

b=[[0,5,6,7],
   [4,6,7,23],
   [6,23,1,2]]

ans=[]
for i in range(r):
	ans.append([])
	for j in range(c):
		ans[i].append(a[i][j]+b[i][j])

print(ans)

using comprehensions
ans=[]
for i in range(r):
	ans.append([x+y for(x,y) in zip(a[i],b[i])])
print(ans)	

(28)sets 
>set,tuple,list can be converted by using list(),
tuple() and set() func

rolls=set({2,4,7,2})
print(rolls)#{2,4,7}
#duplicate values will be removed
rolls=set() #make empty set

Sets donot have index to access element.Instead,
to go through its elements,

for x in rolls:
	print(x)

>add(),remove(),len() functions have their usual
meaning
>clear() func removes all elements from set.
This func is also available for lists
>searching is also as usual

>LOGICAL operations on set can be done
using |,&,^,-
>similarly sets can be compared

rolls1={1,3,5,6}
rolls2={1,5}
rolls3={2,4}

print(rolls1^rolls2) #{3,6}
if rolls2<=rolls1:
	print("rolls2 is subset of rolls1")

if rolls1.isdisjoint(rolls3):
	print("disjoint")

(29)dictionary
in arrays numbers are used for indexing.
Dictionary can be thought as an array where
anything can be used for indexing.

stud={}
stud["name"]="tejas"
stud["year"]=1
stud["dept"]="CSE"
stud["courses"]=["cs101","mal02"]

print(stud)
print(stud["courses"])

for k in stud:
	print(k)

iterating through the dictionary
above gives all the keys(k)(index names) of the 
dictionary

for k in stud:
	print(k,"=",stud[k])

to see the value corresponding to the key above
or below can be used

for k,v in stud.items():
	print(k,"=",v)

>Deleting key from dictionary
if "year"in stud:
	del stud["year"]

ANOTHER way to creating a dictionary

gloss={
	"word1":"meaning1",
	"word2":"meaning2",
	"word3":"meaning3"
}

print(gloss)
print(gloss["word3"])

(30)Array of dictionary
list of dictionaries
dictionary of dictionaries

(31)comprehensions with dictionary
data=[{},{},...]

all dictionaries with key name

names=[x["name"] for x in data]
print(names)

(32)Determining data type of variables
pass var in type() to get type
to check var of certain type use
isinstance()

a=9
b=[1,2,4]
print(type(a))
print(isinstance(b,list))

output
<class 'int'>
True

(33)Files
>open() takes 2 parameters:filename and mode 
"r"-Read-Default value. Opens a file for reading, error if the file does not exist
"a"-Append-Opens a file for appending, creates the file if it does not exist
"w"-Write-Opens a file for writing, creates the file if it does not exist
"x"-Create-Creates the specified file, returns an error if the file exists

>Asume we have the following file, located in the same folder as Python:
>After modifying the file it must be closed using close()
















































