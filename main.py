print("Hello")
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
