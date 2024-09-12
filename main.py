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