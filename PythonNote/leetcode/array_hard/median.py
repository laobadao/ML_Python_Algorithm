"""
4. Median of Two Sorted Arrays

There are two sorted arrays nums1 and nums2 of size m and n respectively.

给出的 两个数组已经是排序好的数组

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

Example 1:
nums1 = [1, 3]
nums2 = [2]

The median is 2.0

Example 2:
nums1 = [1, 2]
nums2 = [3, 4]

The median is (2 + 3)/2 = 2.5

中位数（又称中值，英语：Median），统计学中的专有名词，代表一个样本、
种群或概率分布中的一个数值，其可将数值集合划分为相等的上下两部分。

对于有限的数集，可以通过把所有观察值高低排序后找出正中间的一个作为中位数。
如果观察值有偶数个，通常取最中间的两个数值的平均数作为中位数。
"""

"""

      left_A             |        right_A
A[0], A[1], ..., A[i-1]  |  A[i], A[i+1], ..., A[m-1]
Since A has m elements, so there are m+1 kinds of cutting( i = 0 ~ m ).
 And we know: len(left_A) = i, len(right_A) = m - i . 
 Note: when i = 0 , left_A is empty, and when i = m , right_A is empty.
 
 eg:
 m = 5 则一共有 6 种 切法 下面 | 代表一种切法 left_A 有 0 个 到 m 个 元素 共 m+1 种可能
 
| 12 | 22 | 34 | 56 | 67 |

Put left_A and left_B into one set, and put right_A and right_B into another set. Let's name them left_part and right_part :

      left_part          |        right_part
A[0], A[1], ..., A[i-1]  |  A[i], A[i+1], ..., A[m-1]
B[0], B[1], ..., B[j-1]  |  B[j], B[j+1], ..., B[n-1]

If we can ensure:

1) len(left_part) == len(right_part)
2) max(left_part) <= min(right_part)

then we divide all elements in {A, B} into two parts with equal length,
 and one part is always greater than the other.
 Then median = (max(left_part) + min(right_part))/2.

To ensure these two conditions, we just need to ensure:

(1) i + j == m - i + n - j (or: m - i + n - j + 1)
    if n >= m, we just need to set: i = 0 ~ m, j = (m + n + 1)/2 - i
    
    equal:
    i + j = m - i + n - j + 1
       2j = m + n + 1 - 2i
        j = (m + n + 1)/2 -i
        
(2) B[j-1] <= A[i] and A[i-1] <= B[j]

ps.2 Why n >= m? Because I have to make sure j is non-nagative since 0 <= i <= m and j = (m + n + 1)/2 - i. 
If n < m , then j may be nagative, that will lead to wrong result.

推导：
    j = (m + n + 1)/2 - i （因为 0 <= i <= m 假设 i= m）
    j = (m + n + 1)/2 - i <=(m + n + 1)/2 - m <= (n-m)/2 + 1/2 ( n < m ,n - m < 0 ) 如果 n - m < -1 则 j < 0
    
    eg： m = 5; n =  2; i = 5
    j = ((m + n + 1)/2 - i)
      = (5 + 2 + 1)/2 - 5 = -1 (error)
      要保证 j >= 0 [0,n ] 区间

m = 8 ; i = 4
n = 4 ; j = 2

i + j = m - i + n - j

4 + 2 = 8 - 4 + 4- 2 = 6

Thank @Quentin.chen , him pointed out that: i < m ==> j > 0 and i > 0 ==> j < n . Because:

m <= n, i < m ==> j = (m+n+1)/2 - i > (m+n+1)/2 - m >= (2*m+1)/2 - m >= 0    
m <= n, i > 0 ==> j = (m+n+1)/2 - i < (m+n+1)/2 <= (2 * n + 1)/2 <= n

首先：j = (m + n + 1)/2 - i  //(因为 i < m ,将 i 替换为 m ， 把 (m + n + 1)/2 看做为 A ：A - i > A - m ；因为 i < m)

      j = (m + n + 1)/2 - i > ( m + n + 1)/2 - m // (m+n+1)/2 - 2m/2 = (2*m+1)/2 - m = 1/2 > 0
      
      所以 i < m ==> j > 0
      

"""

"""
A,B 两个有序的数组

:return  中位数
"""
import numpy as np


def findMedian(A, B):
    m = len(A)
    n = len(B)

    if m > n:
        # A = np.array([2, 3, 4, 5, 6])
        # B = np.array([2, 3, 9])
        # A: [2 3 9] B: [2 3 4 5 6] m: 3 n: 5
        # 互换，也就是 确保 n >= m
        A, B, m, n = B, A, n, m
  # print('A:', A, 'B:', B, 'm:', m, 'n:', n)
    if n < 0:
        # 程序员明确的触发异常，即 raise 语句 ValueError 传入无效参数
        raise ValueError




if __name__ == '__main__':
    A = np.array([2, 3, 4, 5, 6])
    B = np.array([2, 3, 9])
    findMedian(A, B)
