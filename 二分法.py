class solution:
    def search(self,nums:list[int],target:int)-> int:
        if not nums:
            return -1
        l,r=0,len(nums)-1
        while l<=r:
            mid=(l+r)//2
            if nums[mid]==target:
                return mid
            if nums[0]<=nums[mid]:
                if num[0]<=target<nums[mid]:
                    r=mid-1
                else:
                    l=mid+1