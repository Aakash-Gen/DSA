#pragma GCC optimize("Ofast") 
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,fma") 
#pragma GCC optimize("unroll-loops") 
#include <bits/stdc++.h>   
#include <complex> 
#include <queue> 
#include <set> 
#include <unordered_set> 
#include <list> 
#include <chrono> 
#include <random> 
#include <iostream> 
#include <algorithm> 
#include <cmath> 
#include <string> 
#include <vector> 
#include <map> 
#include <unordered_map> 
#include <stack> 
#include <iomanip> 
#include <fstream> 

using namespace std; 
   
typedef long long ll; 
typedef long double ld; 
typedef pair<int,int> p32; 
typedef pair<ll,ll> p64; 
typedef pair<double,double> pdd; 
typedef vector<ll> v64; 
typedef vector<int> v32; 
typedef vector<vector<int> > vv32; 
typedef vector<vector<ll> > vv64; 
typedef vector<vector<p64> > vvp64; 
typedef vector<p64> vp64; 
typedef vector<p32> vp32; 
ll MOD = 998244353; 
double eps = 1e-12; 
#define forn(i,e) for(ll i = 0; i < e; i++) 
#define forsn(i,s,e) for(ll i = s; i < e; i++) 
#define rforn(i,s) for(ll i = s; i >= 0; i--) 
#define rforsn(i,s,e) for(ll i = s; i >= e; i--) 
#define ln "\n" 
#define dbg(x) cout<<#x<<" = "<<x<<ln 
#define mp make_pair 
#define pb push_back 
#define fi first 
#define se second 
#define INF 2e18 
#define fast_cin() ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL) 
#define all(x) (x).begin(), (x).end() 
#define sz(x) ((ll)(x).size()) 



// single element in a sorted array - there only one element which appears once find that
int singleElem(vector<int> &nums){
    int n = nums.size();
    int l=0;
    int r =n-2;
    while(l<=r){
        int mid = (l+r)/2;
        if(nums[mid]==nums[mid^1]){   // mid^1 gives mid-1 index if mid is odd and mid+1 if mid is even
            l= mid+1;
        } else {
            r = mid-1;
        }
    }
    return nums[l];
}
//  time complexity - O(logn) whereas brute force of xoring all elements will have time complexity of O(n)


// rotated sorted array - there is a sorted array which is rotated at some pivot find the index of the element target in the array
int rotatedSortedArray(vector<int> &nums, int target){
    int n = nums.size();
    int l =0;
    int r  = n-1;
    while(l<=r){
        int mid = (l+r)/2;
        if(nums[mid]==target){
            return mid;
        }
        if(nums[l]<=nums[mid]){
            if(nums[l]<=target && nums[mid]>=target){
                r = mid-1;
            } else {
                l = mid+1;
            }
        } else {
            if(nums[r]>=target && nums[mid]<=target){
                l = mid+1;
            } else {
                r = mid-1;
            }
        }
    }
    return -1;
}



// rotated sorted array 2 - return true if the target is in this rotated sorted array and false if not . There can be duplicates as well
bool rotatedSortedArray2(vector<int> &nums, int target){
    int n = nums.size();
    int left =0;
    int right = n-1;
    while(left<=right){
        int mid = (left+right)/2;
        if(nums[mid]==target){
            return true;
        }
        if(nums[left]==nums[mid] && nums[right]==nums[mid]){
            left++;
            right--;
        }
        if(nums[left]<=nums[mid]){
            if(target>=nums[left] && target<=nums[mid]){
                right =  mid-1;
            } else {
                left = mid + 1;
            }
        } else {
            if(nums[mid]<=target && target<=nums[right]){
                left = mid +1;
            } else {
                right = mid-1;
            }
        }
    }
    return false;
}


// search insert position - return the index if the target is found and if not return the index where it should be inserted
int searchInsertPosition(vector<int> &nums, int target){
    int n = nums.size();
    int left =0;
    int right = n-1;
    while(left<=right){
        int mid = (left+right)/2;
        if(nums[mid]==target) return mid;
        else if(nums[mid]<target){
            left = mid+1;
        } else {
            right = mid -1;
        }
    }
    return left;
}


// find first and last occurence of the element in a sorted array
vector<int>searchRange(vector<int> &nums, int target){
    int n = nums.size();
    int left =0;
    int right = n-1;
    int start =-1;
    while(left<=right){
        int mid = (left+right)/2;
        if(nums[mid]==target){
            start = mid;
            right = mid-1;
        }
        else if (nums[mid]<target){
            left = mid+1;
        } else {
            right = mid-1;
        }
    }
    left = 0;
    right = n-1;
    end =-1;
    while(left<=right){
        int mid = (left+right)/2;
        if(nums[mid]==target){
            end = mid;
            left = mid+1;
        }
        else if (nums[mid]<target){
            left = mid+1;
        } else {
            right = mid-1;
        }
    }

    return {start,end};
}


// find minimum in rotated sorted array
int minRotatedSortedArray(vector<int> &nums){
    int n = nums.size();
    int left =0;
    int right = n-1;
    int ans = INT_MAX;
    while(left<=right){
        int mid = (left + right)/2;
        if(nums[left]<=nums[mid]){
            ans = min(ans,nums[left]);
            left = mid +1;
        } else {
            ans = min(ans,nums[mid]);
            right = mid-1;
        }
    }
    return ans;
}


// find peak element - find the peak element index in the given array(not sorted) . peak element is the one that is greater than its neighbours
int peakElement(vector<int> &nums){
    int n = nums.size();
    if(nums[0]>nums[1]) return 0;
    if(nums[n-1]>nums[n-2]) return n-1;
    if(n==1) return 0;
    int left =0;
    int right = n-1;
    while(left<=right){
        int mid = (left+right)/2;
        if(nums[mid]>nums[mid-1] && nums[mid]>nums[mid+1]){
            return mid;
        }
        else if(nums[mid]>nums[mid+1]){
            right = mid-1;
        } else {
            left = mid+1;
        }
    }
    return -1;
}