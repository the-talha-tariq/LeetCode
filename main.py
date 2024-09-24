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