# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 01:38:32 2021

@author: abdka
"""

class Test():
    def __init__(self, val):
        self.val = val
    
    def __str__(self):
        return str(self.val)
    
    def __repr__(self):
        return str(self.val)
    
    def __lt__(self, other):
        return self.val < other.val
        



t1 = Test(1)
t2 = Test(2)
t3 = Test(9)

t4 = Test(4)
t5 = Test(5)
t6 = Test(6)
t7 = Test(7)

arr1 = [t3, t2, t1]

arr2 = [t6, t5, t4, t7]

print(arr1)
print(arr2)

arr1 = sorted(arr1, key=lambda x: x.val, reverse=True)
arr2 = sorted(arr2, key=lambda x: x.val, reverse=True)


print(arr1)
print(arr2)

i = 0
j = 0
while i < 3 and j < len(arr2):
    if arr1[i].val < arr2[j].val:
        arr1[i] = arr2[j]
        i += 1
        j += 1
    else:
        i += 1


print(arr1)
print(arr2)
        
        

