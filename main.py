#Merge Sorted Array
class Solution(object):
    def merge(self, nums1, m, nums2, n):
       
        nums3 = []
        k = l = i = 0
        while k < m and l < n:
            if nums1[k] <= nums2[l]:
                nums3.append(nums1[k])
                k+=1
            else:
                nums3.append(nums2[l])
                l+=1
            
        while m > k:
            nums3.append(nums1[k])
            k+=1
        while n > l:
            nums3.append(nums2[l])
            l+=1
        
        for x in range(m+n):
            nums1[x] = nums3[x]
#Remove Element
class Solution(object):
    def removeElement(self, nums, val):
        k = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[k]=nums[i]
                k+=1
        return k
        
#Remove Duplicates from Sorted Array
class Solution(object):
    def removeDuplicates(self, nums):
        k = 0
        for i in range(len(nums)-1):
            if(nums[i] != nums[i+1]):
                nums[k+1] = nums[i+1]
                k += 1
        return k+1

#Remove Duplicates from Sorted Array II
class Solution(object):
    def removeDuplicates(self, nums):
        
        if len(nums)<=2:
            return len(nums)

        k = 2
        for i in range(2,len(nums)):
            if nums[i] != nums[k-2]:
                nums[k] = nums[i]
                k += 1

        return k
    
#Rolling Dice
import random
noOfDice = int(input("Number of Dice you want to roll?"))

while True:
    choice = input("Roll the dice? (y/n): ").lower()

    if choice == 'y':
        print("(",end='', flush=True)
        for i in range(0,noOfDice):
            
            a = random.randint(1, 6)
            print(f"{a},",end='', flush=True)
        print(")")
    elif choice == 'n':
        print("Thanks for playing!")
        break
        
    else:
        print("Invalid Choice!")

#Rotate Array
class Solution(object):
    def rotate(self, nums, k):
        n = len(nums)
        k = k % n
        rotated = [0] * n

        for i in range(n):
            rotated[(i + k) % n] = nums[i]
        
        for i in range(n):
            nums[i] = rotated[i]


#Best Time to Buy and Sell Stock
class Solution(object):
    def maxProfit(self, prices):
        
        buy = prices[0]
        profit = 0
        for i in range(1,len(prices)):
            if(prices[i] > buy):
                profit = max(profit,(prices[i] - buy))
            else:
                buy = prices[i]
        
        return profit

#Best Time to Buy and Sell Stock II
class Solution(object):
    def maxProfit(self, prices):
       
        buy = prices[0]
        profit = 0
        for i in range(1,len(prices)):
            if prices[i]>buy:
                profit  = profit + (prices[i] - buy)
                buy = prices[i]
            else:
                buy = prices[i]
        return profit
    
#Jump Game
class Solution(object):
    def canJump(self, nums):
        farthest_reachable = 0 
        for i in range(len(nums)):
            if i > farthest_reachable:  
                return False
            farthest_reachable = max(farthest_reachable, i + nums[i])
            if farthest_reachable >= len(nums)-1:
                return True

#Jump Game 2
class Solution(object):
    def jump(self, nums):
        far_most = 0
        jump = 0
        current_reachable = 0
        if len(nums) == 1:
            return 0
        for i in range(len(nums)):
            far_most = max(far_most,nums[i] + 1)

            if i == current_reachable:
                jump +=1
                current_reachable = far_most

                if current_reachable >= len(nums)-1:
                    break
        return jump
            
        
                
#Insert Delete GetRandom O(1)
import random
class RandomizedSet(object):

    def __init__(self):
        self.nums = []  
        self.num_map = {}

    def insert(self, val):
        if val in self.nums:
            return False
        else:
            self.num_map[val] = len(self.nums)
            self.nums.append(val)
            return True

    def remove(self, val):
        if val in self.nums:
            last_element = self.nums[-1]
            index_of_val = self.num_map[val]
            self.nums[index_of_val] = last_element
            self.num_map[last_element] = index_of_val
            self.nums.pop()
            del self.num_map[val]
            return True
    def getRandom(self):
        return random.choice(self.nums)
        


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()


#Product of Array Except Self
class Solution(object):
    def productExceptSelf(self, nums):
        n = len(nums)
        prefix = [0] * n
        postfix = [0] * n

        prefix[0] = 1
        postfix[n-1] = 1
        for i in range(1, n):
            prefix[i] = prefix[i - 1] * nums[i - 1]
        
        for i in range(n-2, -1,-1):
           postfix[i] = postfix[i + 1] * nums[i + 1]
        
        result = [0] * n
        for i in range(n):
            result[i] = prefix[i] * postfix[i]
        
        return result

#134. Gas Station
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        
        start_index = 0
        tank = 0
        total_gas = 0
        total_cost = 0

        for i in range(len(gas)):
            total_gas = total_gas + gas[i]
            total_cost = total_cost + cost[i]

            tank = tank - cost[i] + gas [i]

            if tank < 0:
                start_index = i+1
                tank = 0
        if total_gas < total_cost:
            return -1
        return start_index
    

#135. Candy
class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
       """
        candies = [1] * len(ratings)
        for i in range(1,len(ratings)):
            if ratings[i] > ratings[i-1]:
                candies[i] = candies[i-1] + 1


        for i in range(len(ratings)-2, -1, -1):
            if ratings[i] > ratings[i+1] and candies[i]<=candies[i+1]:
                candies[i] = candies[i+1]+1

        return sum(candies)


#Trapping Rain Water
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if not height:
            return 0
        left_max = [0]*len(height)
        right_max = [0]*len(height)
        
        left_max[0]=height[0]
        for i in range(1,len(height)-1):
            left_max[i] = max(left_max[i-1],height[i])

        right_max[len(height)-1]=height[len(height)-1]
        for i in range(len(height)-2,-1,-1):
            right_max[i] = max(right_max[i+1],height[i])

        water =0
        for i in range(len(height)):
            water += max(0, min(left_max[i], right_max[i]) - height[i])
        

        
        return water
    

#58. Length of Last Word
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        length = 0
        last_word_length = length
        for i in range(len(s)):
            if s[i] == ' ':
                length = 0
            else:
                length += 1
                last_word_length = length
        return last_word_length

#14. Longest Common Prefix
class Solution(object):
    def longestCommonPrefix(self, strs):

        """
        :type strs: List[str]
        :rtype: str
        """
        strs.sort()
        first_word = strs[0]
        last_word = strs[len(strs)-1]
        word = ""
        for i in range(len(first_word)):
            if i < len(last_word) and first_word[i] == last_word[i]:
                word += first_word[i]
            else:   
                break
        return word

#13. Roman to Integer
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        roman_map = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
            
        }
    
        total = 0
        n = len(s)
        
        for i in range(n):
            if i < n - 1 and roman_map[s[i]] < roman_map[s[i + 1]]:
                total -= roman_map[s[i]]
            else:
                total += roman_map[s[i]]
        
        return total

#28. Find the Index of the First Occurrence in a String
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        i = 0
        j = 0
        while True:
            if haystack[i] == needle[j]:
                i += 1
                j += 1
            else:
                if j > 0:
                    i = i - j + 1
                    j = 0
                else:
                    i += 1
            if j == len(needle):
                return i-j
            if i == len(haystack):
                return -1
        return -1

#6. Zigzag Conversion
class Solution(object):
    def convert(self, s, numRows):
        if numRows == 1 or numRows >= len(s):
            return s

        # Create a list to store strings for each row
        rows = [''] * numRows
        current_row = 0
        going_down = False

        for char in s:
            rows[current_row] += char
            
            if current_row == 0 or current_row == numRows - 1:
                going_down = not going_down
            
            if going_down:
                current_row += 1
            else:
                current_row -= 1

        return ''.join(rows)
        

        
#1963. Minimum Number of Swaps to Make the String Balanced
class Solution(object):
    def minSwaps(self, s):
        """
        :type s: str
        :rtype: int
        """
        unmatched_close = 0
        max_unmatched_close = 0

        for i in range(len(s)):
            if s[i] == "]":
                unmatched_close += 1  
            else:
                unmatched_close -= 1  
            max_unmatched_close = max(max_unmatched_close, unmatched_close)
        return (max_unmatched_close + 1) // 2

#392. Is Subsequence
class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) == 0:
            return True
        s_iterate = 0
        for i in range(len(t)):
            if s[s_iterate] == t[i]:
                s_iterate += 1

                if s_iterate == len(s):
                    return True
        
        return False

#125. Valid Palindrome
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        clean_text = (''.join(char for char in s if char.isalnum())).lower()
        start = 0
        end = len(clean_text)-1

        while start < end:
            if clean_text[start] == clean_text[end]:
                start += 1
                end -= 1
            else:
                return False
        return True

#217. Contains Duplicate
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        occurrences = {}
        for num in nums:
            if num in occurrences:
                occurrences[num] += 1  
            else:
                occurrences[num] = 1  

        for count in occurrences.values():
            if count > 1:
                return True  

        return False 
        
#242. Valid Anagram

class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        sort1 = ''.join(sorted(s))
        sort2 = ''.join(sorted(t))
        print(sort1)
        print(sort2)

        if sort1==sort2:
            return True
        else:
            return False