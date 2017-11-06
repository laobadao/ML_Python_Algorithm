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

then we divide all elements in {A, B} into two parts with equal length, and one part is always greater than the other. Then median = (max(left_part) + min(right_part))/2.

To ensure these two conditions, we just need to ensure:

(1) i + j == m - i + n - j (or: m - i + n - j + 1)
    if n >= m, we just need to set: i = 0 ~ m, j = (m + n + 1)/2 - i
    
    equal:
    i + j = m - i + n - j + 1
       2j = m + n + 1 - 2i
        j = (m + n + 1)/2 -i
        
(2) B[j-1] <= A[i] and A[i-1] <= B[j]


m = 8 ; i = 4
n = 4 ; j = 2

i + j = m - i + n - j

4 + 2 = 8 - 4 + 4- 2 =6

"""