import statistics as st
import random

def f(data, m=1):
    new = []
    mean = st.mean(data)
    std = st.pstdev(data)
    for x in data:
        if abs(x - mean) <= m * std:
            new.append(x)
    return new
    
data = [random.randrange(50,60) for _ in range(10)] +[70, 10]
print(data)
print(st.mean(data))
print(st.pstdev(data))
print(f(data))