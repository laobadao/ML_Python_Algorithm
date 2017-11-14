"""
 [10. Regular Expression Matching]

 https://leetcode.com/problems/regular-expression-matching/

 Question：正则表达式匹配
 Method：DP - dynamic programming 动态规划

Implement regular expression matching with support for '.' and '*'.
实现正则表达式，支持'.' and '*'

'.' Matches any single character. '.' 点匹配任何字母
'*' Matches zero or more of the preceding element. 匹配 0个 或更多元素

The matching should cover the entire input string (not partial).

The function prototype should be:
bool isMatch(const char *s, const char *p)
（字符串，匹配规则）
Some examples:
isMatch(“aa”,”a”) → false “aa” 俩字母 ，”a”一个字母 aa 不符合 a 这个正则表达式
isMatch(“aa”,”aa”) → true
isMatch(“aaa”,”aa”) → false
isMatch(“aa”, “a*”) → true “aa” 符合 a* ,开头字母是 a 后面的字母任意
isMatch(“aa”, “.*”) → true
isMatch(“ab”, “.*”) → true
isMatch(“aab”, “c*a*b”) → true ？？？

为什么 isMatch(“aab”, “c*a*b”) → true ？

因为：'*' Matches zero or more of the preceding element. 匹配 0个 或更多元素

So for this testcase "c* a* b" could be 0 c 2 a and 1 b, it matched aab

"c* a* b"  ，这么看 * 代表0 个或多个，所以就是 0个 或多个 c ,0个 或多个 a ,一个b

所以 aab 满足，0个 c,2 个 a,一个 b


"""

"""
part1:

Without a Kleene star'*', our solution would look like this:
没有 * ，不含通配符 * 的情况下
"""


def match(text, pattern):
    if not pattern:
        return not text
    # pattern[0] pattern 里面第 0 个元素在 {text[0], '.'} 这个字典中，
    # {text[0], '.'} 这个字典包含 text 里 第 0 个元素 和 '.'
    # bool(text) 1. 若 text = "abba"  print('bool(text)'True
    # text = ""  print('bool(text)') False
    first_match = bool(text) and pattern[0] in {text[0], '.'}
    # match() 方法递归循环的 去做判断，每一个字母的匹配，返回的是布尔 Boolean 类型
    # text[1:] 代表 从 索引 1 开始包含索引1 的元素 ，也就是不包含索引0 的元素
    return first_match and match(text[1:], pattern[1:])


def learn_bool():
    text = "abba"
    print('bool(text)', bool(text))  # True
    text = ""
    print('bool(text)', bool(text))  # False
    return bool(text)
