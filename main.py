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

#1. Two Sum
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashmap = {}  
        
        for i, num in enumerate(nums):
            complement = target - num
            if complement in hashmap:
                return [hashmap[complement], i]
            hashmap[num] = i

#49. Group Anagrams
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        anagram_map = {}

        for word in strs:
            sorted_key = ''.join(sorted(word))
            if sorted_key in anagram_map:
                anagram_map[sorted_key].append(word) 
            else:
                anagram_map[sorted_key] = [word]
        return list(anagram_map.values()) 

#347. Top K Frequent Elements
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        number_of_occurence = {}

        for num in nums:
            if num in number_of_occurence:
                number_of_occurence[num] += 1
            else:
                number_of_occurence[num] = 1

        most_frequent = heapq.nlargest(k, number_of_occurence.keys(),key=number_of_occurence.get)

        return most_frequent


#128. Longest Consecutive Sequence
class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0

        nums.sort()  
        longest = 1
        current_streak = 1

        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:  
                continue
            elif nums[i] == nums[i - 1] + 1:  
                current_streak += 1
            else:
                longest = max(longest, current_streak)
                current_streak = 1  

        return max(longest, current_streak)


#36. Valid Sudoku
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        for row in board:
            seen = set()  
            for value in row:
                if value != ".":  
                    if value in seen:  
                        return False
                    seen.add(value)

            seen.clear()
        for col in range(9):
            seen = set()  
            for row in range(9):
                value = board[row][col]
                if value != ".":  
                    if value in seen:  
                        return False
                    seen.add(value) 
            seen.clear()
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                seen = set() 
                for i in range(3):
                    for j in range(3):
                        value = board[box_row + i][box_col + j]
                        if value != ".":
                            if value in seen:  
                                return False
                            seen.add(value)  
        return True


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

#167. Two Sum II - Input Array Is Sorted
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        i = 0
        j = len(numbers)-1

        while (i<j):
            if numbers[i] + numbers[j] == target:
                return [i+1,j+1]
            elif numbers[i] + numbers[j] <target:
                i += 1;
            else:
                j -= 1;
            
# Skip duplicates for left and right
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        result = []
        
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:  
                continue
                
            left, right = i + 1, len(nums) - 1
            
            while left < right:
                three_sum = nums[i] + nums[left] + nums[right]
                
                if three_sum == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                elif three_sum < 0:
                    left += 1
                else:
                    right -= 1
        
        return result


#11. Container With Most Water

class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if not height:
            return 0
        
        start = 0
        end = len(height) - 1
        max_area = 0
        
        while start < end:
            width = end - start
            length = min(height[start], height[end])
            area = width * length
            max_area = max(max_area, area)
            
            if height[start] < height[end]:
                start += 1
            else:
                end -= 1
                
        return max_area



#20. Valid Parentheses
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = deque()

        for char in s:
            if char == "[" or char == "(" or char == "{":
                stack.append(char)
            elif char == "]":
                if not stack or stack[-1] != "[":
                    return False
                stack.pop()
            elif char == ")":
                if not stack or stack[-1] != "(":
                    return False
                stack.pop()
            elif char == "}":
                if not stack or stack[-1] != "{":
                    return False
                stack.pop()

        if len(stack) == 0:
            return True
        else:
            return False

#155. Min Stack
class MinStack(object):

    def __init__(self):
        self.stack = []        
        self.min_stack = []

    def push(self, val):
        """
        :type val: int
        :rtype: None
        """
        self.stack.append(val)
        

    def pop(self):
        """
        :rtype: None
        """
        if self.stack:
            self.stack.pop()

    def top(self):
        """
        :rtype: int
        """
        if self.stack:
            return self.stack[-1]
        return None

    def getMin(self):
        """
        :rtype: int
        """
        if self.stack:
            return min(self.stack) 


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


#150. Evaluate Reverse Polish Notation


class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []
        for token in tokens:
            if token in {"+", "-", "*", "/"}:
                b = stack.pop()
                a = stack.pop()
                
                if token == "+":
                    c = a + b
                elif token == "-":
                    c = a - b
                elif token == "*":
                    c = a * b
                elif token == "/":
                    # Perform integer division that truncates towards zero
                    c = int(a / b) if a * b >= 0 else -(abs(a) // abs(b))
                    
                stack.append(c)
            else:
                # Convert the token to an integer before appending
                stack.append(int(token))
                
        return stack[0]


#22. Generate Parentheses

class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        stack = []
        result = []

        def backtract(openN, closeN):
            
            if openN == closeN == n:
                result.append("".join(stack));
                return
            if openN < n:
                stack.append("(")
                backtract(openN+1,closeN)
                stack.pop()
            if closeN < openN:
                stack.append(")")
                backtract(openN,closeN+1)
                stack.pop()
    
        backtract(0, 0)
        return result


#739. Daily Temperatures

class Solution(object):
    def dailyTemperatures(self, temperatures):
        """
        :type temperatures: List[int]
        :rtype: List[int]
        """
        answer = [0] * len(temperatures)
        stack = []  
        for i, temp in enumerate(temperatures):
        
            while stack and temperatures[stack[-1]] < temp:
                index = stack.pop()
                answer[index] = i - index  
            stack.append(i)
        
        return answer





#853. Car Fleet
class Solution(object):
    def carFleet(self, target, position, speed):
        """
        :type target: int
        :type position: List[int]
        :type speed: List[int]
        :rtype: int
        """
        sorted_cars = sorted(zip(position, speed), key=lambda x: x[0],reverse=True)
        stack = []

        for pos, spd in sorted_cars:
            time_to_reach = (target - pos) / float(spd)
            
            if not stack or time_to_reach > stack[-1]:
                stack.append(time_to_reach)

        return len(stack)



#704. Binary Search
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        start = 0
        end = len(nums)-1
        while start<=end:
            middle = end - start /2
            if nums[middle] == target:
                return middle
            elif nums[middle]>target:
                end = middle - 1
            else:
                start = middle + 1
        return -1


#74. Search a 2D Matrix
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        rows = len(matrix)          
        cols = len(matrix[0])
        start = 0
        end = rows*cols
        print(end)
        for i in range(rows*cols):
            mid = (end + start)/2
            row = mid // cols
            col = mid % cols
            
            if matrix[row][col] == target:
                return True
            elif target > matrix[row][col]:
                start = mid
            else:
                end = mid
        return False


#875. Koko Eating Bananas

class Solution(object):
    def minEatingSpeed(self, piles, h):
        """
        :type piles: List[int]
        :type h: int
        :rtype: int
        """
        end = max(piles)
        start = 1
        result = end
        while start <= end:
            mid = (start + end) // 2
            print(mid)
            total_hour = 0

            for pile in piles:

                total_hour += math.ceil(float(pile)/mid)
                
            if total_hour <= h:
                result = min(result,mid)  # Update the result
                end = mid - 1  # Check for smaller k
            else:
                start = mid + 1

        return result



#153. Find Minimum in Rotated Sorted Array
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        start = 0
        end = len(nums) - 1

        while start < end:  
            if nums[start] < nums[end]:
                return nums[start]
            mid = (start + end) // 2

            if mid > 0 and nums[mid] < nums[mid - 1]:
                return nums[mid]
            if mid < len(nums) - 1 and nums[mid] > nums[mid + 1]:
                return nums[mid + 1]

            if nums[mid] >= nums[start]:
                start = mid + 1
            else:
                end = mid - 1

        return nums[start]  



#33. Search in Rotated Sorted Array

class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        start = 0
        end = len(nums)-1
        while start <= end:
            mid = (start + end)//2

            if nums[mid] == target:
                return mid
            
            if nums[start]<=nums[mid]:
                if nums[start] <= target < nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            else:
                if nums[mid] < target <=nums[end]:
                    start = mid + 1
                else:
                    end = mid - 1
        return -1


#3. Longest Substring Without Repeating Characters
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        k = 0 
        l = 0 
        char_index = {}  
        max_length = 0  

        while l < len(s):
            if s[l] not in char_index or char_index[s[l]] < k:
                max_length = max(max_length, l - k + 1)
            else:
                k = char_index[s[l]] + 1 
            char_index[s[l]] = l
            l += 1  

        return max_length

#424. Longest Repeating Character Replacement

class Solution:
    def characterReplacement(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        char_count = {}  # Dictionary to store character counts in the current window
        start = 0  # Left pointer of the window
        max_freq = 0  # Maximum frequency of any character in the window
        max_length = 0  # Length of the longest valid substring

        for end in range(len(s)):
            char_count[s[end]] = char_count.get(s[end], 0) + 1
            max_freq = max(max_freq, char_count[s[end]])

            # If the number of characters to replace exceeds k, shrink the window
            while (end - start + 1) - max_freq > k:
                char_count[s[start]] -= 1
                start += 1

            # Update the maximum length of the window
            max_length = max(max_length, end - start + 1)

        return max_length


#567. Permutation in String
class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        s1_count = {}
        for char in s1:
            s1_count[char] = s1_count.get(char, 0) + 1

        s2_count = {}
        k = 0
        for l in range(len(s2)):
            s2_count[s2[l]] = s2_count.get(s2[l], 0) + 1
        
            if l-k+1 > len(s1):
                if s2_count[s2[k]] == 1:
                    del s2_count[s2[k]]
                else:
                    s2_count[s2[k]] -= 1
                k += 1
            if s1_count == s2_count:
                return True
        return False

#76. Minimum Window Substring
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if t == "":
            return ""
        t_count={}
        window={}
        for char in t:
            t_count[char] = t_count.get(char, 0) + 1
        
        have = 0
        need = len(t_count)
        res = [-1,-1]
        res_len = float("infinity")
        l = 0
        for r in range(len(s)):
            current = s[r]
            window[current] = window.get(current, 0) + 1

            if current in t_count and window[current] == t_count[current]:
                have +=1
            
            while have == need:
                if (r - l + 1) < res_len:
                    res = [l,r]
                    res_len =   (r - l + 1)
                window[s[l]] -= 1
                if s[l] in t_count and window[s[l]] < t_count[s[l]]:
                    have -=1
                l += 1
        l, r = res
        return s[l : r+1] if res_len != float("infinity") else ""



#2109. Adding Spaces to a String
class Solution(object):
    def addSpaces(self, s, spaces):
        """
        :type s: str
        :type spaces: List[int]
        :rtype: str
        """
        result = []
        space_index = 0
        for i in range(len(s)):
            if space_index < len(spaces) and i == spaces[space_index]:
                result.append(" ")
                space_index += 1  
            result.append(s[i])

        return "".join(result)


        # updated_string=""
        # i = 0
        # for j in range(len(s)):
        #     if j == spaces[i]:
        #         updated_string += " " + s[j]
        #         if len(spaces) > i+1:
        #             i += 1
        #             print(i)
        #     else:
        #         updated_string += s[j]
        # return updated_string



#4. Median of Two Sorted Arrays
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        
        m, n = len(nums1), len(nums2)
        start, end = 0, m
        
        while start <= end:
            partitionX = (start + end) // 2
            partitionY = (m + n + 1) // 2 - partitionX
            
            maxLeftX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]
            minRightX = float('inf') if partitionX == m else nums1[partitionX]
            
            maxLeftY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]
            minRightY = float('inf') if partitionY == n else nums2[partitionY]
            
            if maxLeftX <= minRightY and maxLeftY <= minRightX:
                if (m + n) % 2 == 0:
                    return (max(maxLeftX, maxLeftY) + min(minRightX, minRightY)) / 2.0
                else:
                    return max(maxLeftX, maxLeftY)
            elif maxLeftX > minRightY:
                end = partitionX - 1
            else:
                start = partitionX + 1
        
        raise ValueError("Input arrays are not sorted!")




#2337. Move Pieces to Obtain a String
class Solution(object):
    def canChange(self, start, target):
        """
        :type start: str
        :type target: str
        :rtype: bool
        """

        
        first = 0
        second = 0
        n = len(start)
        while first < n or second < n:
            while first < n and start[first] == '_':
                first += 1
            while second < n and target[second] == '_' :
                second += 1  

            if (first < n) != (second < n):
                return False
            
            if first < n and second < n:
                if start[first] != target[second]:
                    return False
                if start[first] == 'L' and first < second:
                    return False
                if start[first] == 'R' and first > second:
                    return False
            
            first += 1
            second += 1

        return True
            


#50. Pow(x, n)
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        positive_n = abs(n)
        result = 1.0
        while positive_n > 0:
            if positive_n % 2 == 1:
                result = result*x
            x = x*x
            positive_n = positive_n // 2

        if n < 0:
            return 1 / result
        return result



#21. Merge Two Sorted Lists

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        dummy = ListNode(-1)
        merged_list = dummy

        while list1 and list2:
            if list1.val < list2.val:
                merged_list.next = list1
                list1 = list1.next
            else:
                merged_list.next = list2
                list2 = list2.next

            merged_list = merged_list.next

        while list1:
            merged_list.next = list1
            list1 = list1.next
            merged_list = merged_list.next

        while list2:
            merged_list.next = list2
            list2 = list2.next
            merged_list = merged_list.next
        
        return dummy.next



#206. Reverse Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        current = head
        prev = None  
        while current:
            temp = current.next
            current.next = prev
            prev = current
            current = temp
        return prev

#143. Reorder List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reorderList(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: None Do not return anything, modify head in-place instead.
        """
        half = head
        end = head.next
        while end and end.next:
            half = half.next
            end = end.next.next

        current = half.next
        prev = None  
        half.next = None
        while current:
            temp = current.next
            current.next = prev
            prev = current
            current = temp

        mid = prev
        start = head
        while mid:
            midTemp = mid.next
            startTemp = start.next
            start.next = mid
            mid.next = startTemp
            start = startTemp
            mid = midTemp


# Remove Nth Node From End of List

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: Optional[ListNode]
        :type n: int
        :rtype: Optional[ListNode]
        """
        dummy = ListNode(0)
        dummy.next = head
        fast = dummy
        slow = dummy

        for _ in range(n + 1):
            fast = fast.next
        
        while fast:
            slow = slow.next
            fast = fast.next
        
        slow.next = slow.next.next
        return dummy.next


#2. Add Two Numbers

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: Optional[ListNode]
        :type l2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        dummy = ListNode(0)
        current = dummy
        carry = 0
        while l1 or l2:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
                
            total = val1 + val2 + carry
            carry = total // 10
            value = total % 10

            current.next = ListNode(value)
            current = current.next 
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        if carry > 0:
            current.next = ListNode(carry)     

        return dummy.next


#141. Linked List Cycle

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        
        slow = head
        fast = head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

            if slow == fast:
                return True
        return False

#287. Find the Duplicate Number
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        slow = nums[0]
        fast = nums[0]

        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]

            if slow == fast:
                break
        slow = nums[0]
        while slow!=fast:
            slow = nums[slow]
            fast = nums[fast]

        return slow
           


#234. Palindrome Linked Lists

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: bool
        """
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        current = slow
        prev = None
        while current:
            temp = current.next
            current.next = prev
            prev = current
            current = temp
        
        start = head
        while prev:
            if start.val != prev.val:
                return False
            
            start = start.next
            prev = prev.next

        return True

#226. Invert Binary Tree

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def invertTree(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: Optional[TreeNode]
        """
        if not root:
            return None

        root.left, root.right = root.right, root.left

        self.invertTree(root.left)
        self.invertTree(root.right)

        return root


#104. Maximum Depth of Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if root is None:
            return 0

        lHeight = self.maxDepth(root.left)
        rHeight = self.maxDepth(root.right)
        return max(lHeight, rHeight) + 1


#543. Diameter of Binary Tree

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        self.diameter = 0  

        def height(node):
            if not node:
                return 0  

            left_height = height(node.left)
            right_height = height(node.right)

            self.diameter = max(self.diameter, left_height + right_height)

            return max(left_height, right_height) + 1

        height(root)  
        return self.diameter
        


#110. Balanced Binary Tree

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        def height(root):
            if not root:
                return 0
            left_subtree = height(root.left)
            right_subtree = height(root.right)

            if left_subtree == -1 or right_subtree == -1:
                return -1
            elif abs(right_subtree - left_subtree) > 1:
                return -1
            else:
                return max(right_subtree,left_subtree) + 1

        return height(root) != -1
    

#100. Same Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: Optional[TreeNode]
        :type q: Optional[TreeNode]
        :rtype: bool
        """
        if not p and not q:
            return True
        
        if not p or not q or p.val != q.val:
            return False
        
        return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)

            
#572. Subtree of Another Tree

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSubtree(self, root, subRoot):
        """
        :type root: Optional[TreeNode]
        :type subRoot: Optional[TreeNode]
        :rtype: bool
        """
        if not root:
            return False
        if self.isSameTree(root, subRoot): 
            return True

        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
    def isSameTree(self, p, q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        
#572. Subtree of Another Tree

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSubtree(self, root, subRoot):
        """
        :type root: Optional[TreeNode]
        :type subRoot: Optional[TreeNode]
        :rtype: bool
        """
        if not root:
            return False
        if self.isSameTree(root, subRoot): 
            return True

        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
    def isSameTree(self, p, q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        

#916. Word Subsets

class Solution(object):
    def wordSubsets(self, words1, words2):
        """
        :type words1: List[str]
        :type words2: List[str]
        :rtype: List[str]
        """
        def mergeFrequency(words):
            merged = Counter()
            for word in words:
                freq = Counter(word)
                for char in freq:
                    merged[char] = max(merged[char], freq[char])
            return merged

        words2_freq = mergeFrequency(words2)
        result = []
        for word in words1:
            word_freq = mergeFrequency(word)
            if all(word_freq[char] >= words2_freq[char] for char in words2_freq):
                result.append(word)
                
        return result



#2657. Find the Prefix Common Array of Two Arrays

class Solution(object):
    def findThePrefixCommonArray(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: List[int]
        """
        n = len(A)
        freq = {}
        C = []
        for i in range(n):

            freq[A[i]] = freq.get(A[i], 0) + 1
            freq[B[i]] = freq.get(B[i], 0) + 1
                
            common_count = sum(1 for value in freq.values() if value > 1)

            C.append(common_count)

        return C

#35. Search Insert Position
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if target < nums[0]:
            return 0
        
        if len(nums) == 1 and target <= nums[0]:
            return 0
        elif len(nums) == 1 and target > nums[0]:
            return 1

        for i in range(len(nums)-1):
            if nums[i] == target:
                return i
            elif nums[i] < target <= nums[i+1]:
                return i+1
            
        return len(nums)

#66. Plus One
class Solution(object):
    def plusOne(self, digits):
        n = len(digits)
        digits[n-1] += 1
        for i in range(n-1, -1, -1):
            if digits[i] == 10:
                digits[i] = 0
                if i-1 >= 0:
                    digits[i-1] += 1
                else:
                    digits.append(0)
                    digits[i] += 1
        return digits

        

#1780. Check if Number is a Sum of Powers of Three

class Solution(object):
    def checkPowersOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        while n > 0:
            if n % 3 > 1:  
                return False
            n //= 3  
        return True


#2965. Find Missing and Repeated Values
class Solution(object):
    def findMissingAndRepeatedValues(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: List[int]
        """
        n = len(grid)
        freq = {}

        for row in grid:
            for num in row:
                freq[num] = freq.get(num, 0) + 1

        for num in range(1, n * n + 1):
            if num not in freq:
                missing  = num
            elif freq[num] == 2:
                repeat = num
        
        return [repeat,missing]
    


#3191. Minimum Operations to Make Binary Array Elements Equal to One I

class Solution(object):
    def minOperations(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        operations = 0

        for i in range(n - 2):  
            if nums[i] == 0:
                nums[i] ^= 1
                nums[i + 1] ^= 1
                nums[i + 2] ^= 1
                operations += 1

        if nums[-1] == 0 or nums[-2] == 0:
            return -1

        return operations
