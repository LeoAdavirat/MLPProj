A = """2
10 10
3
1 2 3"""
B = A.split('\n')
B = [B[i] for i in range(1, len(B), 2)]
B = [sum(int(j) for j in i.split()) for i in B]
print(B)