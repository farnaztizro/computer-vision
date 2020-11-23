def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return(fib(n-1) + fib(n-2))        

#print("here it is", fib(5))

#for giving an input
a = input("enter a number > ")
print(fib(int(a)))