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


// fibonacci number - where the value of nth number is given by sum of n-1th and n-2th number
int fib(int n){
    // vector<int> dp(n+1,0);
    // dp[0] =0;
    // dp[1] = 1;
    // for(int i=2;i<=n;i++){
    //     dp[i] = dp[i-1] + dp[i-2];
    // }   
    // return dp[n];

    int prev2 =0;
    int prev1 = 1;
    int curr = 0;
    for(int i=2;i<=n;i++){
        curr = prev2 + prev1;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}

// min cost to climb stairs - where we are given costs array and we can either start at index 0 or index 1 and move either 1 step or 2 steps
int minCostToClimbStairs(vector<int> &costs){
    int n = costs.size();
    int dp =0;
    int dp1 =0;
    int dp2 =0;
    for(int i=2;i<=n;i++){
        int step1 = dp1 + costs[i-1];
        int step2 = dp2 + costs[i-2];
        dp = min(step1,step2);
        dp2 = dp1;
        dp1 = dp;
    }
    return dp1;
}


// climbing stairs - where we ar given n steps and we can either climb one or two steps and we have to find ways in which we can climb the stairs
// this is same as fibonacci number
int climbingStairs(int n){
    if(n==0 || n==1) return 1;
    int prev2 =1;
    int prev1 =1;
    for(int i=2;i<=n;i++){
        int curr = prev2 + prev1;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}


// Pascal's triangle 2 - here we are given rowIndex and we need to output the row at the index which appears in the pascal triangle
vector<int> pascalTriangle2(int rowIndex){
    vector<int> ans(rowIndex+1);
    if(rowIndex==0) return {1};
    ans[0]=1;
    int prev =1;
    for(int i=1;i<=rowIndex;i++){
        ans[i] = prev * (rowIndex-i+1)/i;  // derived from binomial formula
        prev = ans[i];
    }
    return ans;
}


// Maximum repeating substring - where we need to find the maximum repitions of the word given in the string given.
string maxRepeatingSubstring(string s, string word){
    int count =0;
    string temp = word;
    while(s.find(temp)!=string::npos){
        count++;
        temp+=word;
    }
    return count;
}


// all pssoble binary trees - where we are given the no. of nodes and we need to find all the possible full binary trees we can form
vector<TreeNode*> allPossibleBinaryTrees(int n){
    unordered_map<int,vector<TreeNode*>> dp;
    return solve(n);
}
vector<TreeNode*> solve(int n, unordered_map<int,vector<TreeNode*>> &dp){
    vector<TreeNode*>temp;
    if(n%2==0) return temp;
    if(n==1){
        TreeNode* node = new TreeNode(0);
        temp.push_back(node);
        return temp;
    }
    if(dp.find(n)!=dp.end()) return dp[n];
    for(int i=1;i<n;i++){
        vector<TreeNode*> left = solve(i);
        vector<TreeNode*> right = solve(n-i-1);
        for(auto &l: left){
            for(auto &r:right){
                TreeNode* root = new TreeNode(0);
                root->left = l;
                root->right = r;
                temp.push_back(root);
            }
        }
    }
    return dp[n] = temp;
}


// count sorted vowel strings - where we are given n and we need too find no. of all lexicographic strings made of vowels with size n
int countSortedVowelStrings(int n){
    int a=1, e=1, i=1, o=1, u=1;
    n= n-1;
    while(n--){
        o+=u;
        i+=o;
        e+=i;
        a+=e;
    }
    return a+e+i+o+u;
}


// partition arr for max sum - here we are given an array and a value k .we can partition the array into contigious subarrays of len at most k and then can change the value of all elements of that array to the max value of the array
// memoization
int solve(vector<int> &arr, vector<int> &dp, int index, int n, int k){
    if(index==n) return 0;
    if(dp[index]!=-1) return dp[index];
    int len =0;
    int maxi = INT_MIN;
    int ans = INT_MIN;
    for(int i = index;i<min(index+k,n);i++){
        len++;
        maxi = max(maxi,arr[i]);
        int sum = len*maxi + solve(arr,dp,i+1,n,k);
        ans = max(ans,sum);
    }
    return dp[index] = ans;
}
int partitionArrayForMaxSum(vector<int> &arr, int k){
    int n = arr.size();
    vector<int> dp(n,-1);
    return solve(arr,dp,0,n,k);
}

// tabulation
int partitionArrayForMaxSum(vector<int> &arr, int k){
    int n = arr.size();
    vector<int> dp(n+1,0);
    for(int i=n-1;i>=0;i--){
        int len =0;
        int maxi = INT_MIN;
        int ans = INT_MIN;
        for(int j = i;j<min(i+k,n);j++){
            len++;
            maxi = max(maxi,arr[j]);
            int sum = len*maxi + dp[j+1];
            ans = max(ans,sum);
        }
        dp[i] = ans;
    }
    return dp[0];
}


// egg drop with 2 eggs and n floors - here i need to find the minimum no of moves to find the floor at or below which the egg will not break and above which it will break
int twoEggDrop(int n){
    if(n==0) return 0;
    if(n==1) return 1;
    vector<int> dp(n+1);
    dp[0] =0;
    dp[1] =1;
    dp[2] =2;
    for(int i=3;i<=n;i++){
        int best = n // as the worst case scenario is i check every floor
        for(int j=1;j<=i;j++){
            int breaks = 1 + j -1; // if i drop it at jth floor and it breaks the i will need 1 + (j-1) attempts to find safe floor
            int survives = 1 + dp[i-j]
            best = min(best, max(breaks,survive));
        }
        dp[i] = best;
    }
    return dp[n];
}


// geek Jump - here we can go from to nth floor by jumping either one step or 2 steps and energy is lost every time we jump which is given is heights array
// memoization
int solve(int n, vector<int> &height, int index, vector<int> &dp){
    if(index==0) return 0;
    if(dp[index]!=-1) return dp[index];

    int step1 = solve(n,height,index-1,dp) + abs(height[index]-height[index-1]);
    int step2 = INT_MAX;
    if(index>1){
        step2 = solve(n,height,index-2,dp) + abs(height[index]-height[index-2]);
    }
    return dp[index] = min(step1,step2);
}
int geekJump(int n, vector<int> &height){
    vector<int> dp(n,-1);
    return solve(n,height,n-1,dp);
}
//tabulation
int geekJump(int n, vector<int> &heights){
    vector<int> dp(n,0);  // as we need to  go to n-1th floor
    dp[0] =0;
    for(int i=1;i<n;i++){
        int step1 = abs(heights[i]-heights[i-1]) + dp[i-1];
        int step2 =0;
        if(i>1){
            step2 = abs(heights[i]-heights[i-2]) + dp[i-2];
        }
        dp[i] = min(step1,step2);
    }
    return dp[n-1];
}


// house robber - here we are given a list of houses and we cannot rob adjacent houses. what is the max amount we can rob
// memoization
int solve(vector<int> &houses, vector<int> &dp, int n, int index){
    if(index < 0) return 0;
    if(dp[index]!=-1) return dp[index];

    int notTake = solve(houses,dp,n,index-1);
    int take = solve(houses,dp,n,index-2) + nums[i];
    return dp[index] = max(take,notTake);
}
int houseRobber(vector<int> &houses){
    int n = houses.size();
    vector<int> dp(nums.size(),-1);
    return solve(houses,dp,n,n-1);
}
// tabulation
int houseRobber(vector<int> &houses){
    int n = houses.size();
    vector<int> dp(nums.size(),-1);
    dp[0] = nums[0];
    dp[1] = max(nums[0],nums[1]);
    for(int i=2;i<n;i++){
        int notTake = dp[i-1];
        int take = dp[i-2] + houses[i];
        dp[i] = max(notTake,take);
    }
    return dp[n-1];
}



// house robber2 - its same as house robber but the houses are placed in a circle and the first and last house are adjacent to each other
int solve(vector<int> &houses,vector<int> &dp, int start, int end){
    if(end<start) return 0;
    if(dp[end]!=-1) return dp[end];

    int notTake = solve(houses,dp,start,end-1);
    int take = solve(houses,dp,start,end-2) + houses[end];
    return dp[end] = max(take,notTake);
}
int houseRobber2(vector<int> &houses){
    int n = houses.size();
    if(n==0) return 0;
    if(n==1) return nums[0];
    vector<int> dp1(n,-1);
    vector<int>dp2(n,-1);
    int ans1 = solve(houses,dp1,0,n-2);
    int ans2  = solve(houses,dp2,1,n-1);
    return max(ans1,ans2);
}

//tabulation
int rob(vector<int> &nums){
    int n = nums.size();
    if(n==0) return 0;
    if(n==1) return nums[0];
    if(n==2) return max(nums[0],nums[1]);
    vector<int> dp1(n,-1);
    vector<int>dp2(n,-1);
    dp1[0] = nums[0];
    dp1[1] = max(nums[0],nums[1]);
    for(int i=2 ;i<n-1;i++){
        int notTake = dp1[i-1];
        int take = nums[i] + dp1[i-2];
        dp1[i] = max(take,notTake);
    }

    dp2[1] = nums[1];
    dp2[2] = max(nums[1],nums[2]);
    for(int i=3;i<n;i++){
        int notTake = dp2[i-1];
        int take = dp2[i-2] + nums[i];
        dp2[i] = max(take,notTake);
    }

    return max(dp1[n-2],dp2[n-1]);
}


// unique paths - there is a m x n grid and a robot is at at 0,0 . tell the no. of possible unique paths robot can take to reach last cell
int solve(int row, int col, vector<vector<int>> &dp){
    if(row == 0 && col==0) return 1;
    if(row<0 || col<0 ) return 0;
    if(dp[row][col]!=-1) return dp[row][col];

    int up = solve(row-1,col,dp);
    int left = solve(row,col-1,dp);

    return dp[row][col] = up + left;
}
int uniquePaths(int m, int n){
    vector<vector<int>> dp(m,vector<int>(n,-1));
    return solve(m-1,n-1,dp);
}

// tabulation 
int uniquePaths(int m,int n){
    vector<vector<int>> dp(m,vector<int>(n,0));
    dp[0][0] = 1;
    for(int row=0;row<m;row++){
        for(int col=0;col<n;col++){
            if(row==0 && col==0) continue;
            int up =0;
            if(row>0) up = dp[row-1][col];
            int left =0;
            if(col>0) left = dp[row][col-1];
            dp[row][col] = up + left;
        }
    }
    return dp[m-1][n-1];
}



// unique paths 2 - same as unique paths but there is an obstacle in the matrix marked as 1 and blank space is marked as 0
int uniquePaths2(vector<vector<int>> grid){
    int m = grid.size();
    int n = gird[0].size();
    vector<vector<int>> dp(m,vector<int>(n,-1));
    dp[0][0] = 1;
    for(int row =0;row<m;row++){
        for(int col=0;col<n;col++){
            if(row==0 && col==0) continue;
            if(grid[row][col]==1) dp[row][col] =0;
            int up=0;
            if(row>0) up = dp[row-1][col];
            int left =0;
            if(col>0) left = dp[row][col-1];
            dp[row][col] = left + up;
        }
    }
    dp[m-1][n-1]; 
}


// minimum path sum - we are given a grid and we need to find minimum path sum from start to end
int solve(vector<vector<int>> &grid, int row, int col, vector<vector<int>> &dp){
    if(row==0 && col==0) return grid[0][0];
    if(row<0 || col<0) return INT_MAX;
    if(dp[row][col]!=-1) return dp[row][col];
    int left = grid[row][col] + solve(grid,row,col-1,dp);
    int up = grid[row][col] + solve(grid,row-1,col,dp);
    return dp[row][col] = min(left,up);
}
int minimumPathSum(vector<vector<int>> &grid){
    int m = grid.size();
    int n = grid[0].size();
    vector<vector<int>> dp(m,vector<int>(n,-1));
    return solve(grid,m-1,n-1,dp);
}
// tabulation
int minimumPathSum(vector<vector<int>> &grid){
    int m = grid.size();
    int n = grid[0].size();
    vector<vector<int>> dp(m,vector<int>(n,0));
    dp[0][0] = grid[0][0];
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(i==0 && j==0) continue;
            int down = INT_MAX;
            if(i>0){
                down = grid[i][j] + dp[i-1][j];
            }
            int right =INT_MAX;
            if(j>0){
                right = grid[i][j] + dp[i][j-1];
            }
            dp[i][j] = min(down,right);
        }
    }
    return dp[m-1][n-1];
}


// triangle sum - same as minimum path sum but instead of grid we have a triangle
int solve(vector<vector<int>> &triangle, int row, int col, int n, vector<vector<int>> &dp){
    if(row==n) return 0;

    if(dp[row][col]!=-1) return dp[row][col];
    int left = triangle[row][col] + solve(triangle,row+1,col,n,dp);
    int right = triangle[row][col] + solve(triangle,row+1,col+1,n,dp);
    return dp[row][col] = min(left,right);
}
int minTriangle(vector<vector<int>> &triangle){
    int n = triangle.size();
    vector<vector<int>> dp(n,vector<int>(n,-1));
    return solve(triangle,0,0,n,dp);
}

//Tabulation (space optimized)
int minTriangle(vector<vector<int>> &triangle){
    int n = triangle.size();
    vector<int> dp(n,0);
    for(int i=0;i<n;i++){
        dp[i] = triangle[n-1][i];
    }

    for(int i=n-2;i>=0;i--){
        for(int j=0;j<=i;j++){
            dp[j] = triangle[i][j] + min(dp[j],dp[j+1]);
        }
    }
    return dp[0];
}


// minimum falling path sum - where we are given a grid and i need to find the path such that when i choose an elemnet from one row next elem would be from next row either down left or right
int solve(vector<vector<int>> &grid, vector<vector<int>> &dp, int row, int col){
    int m = grid.size();
    int n = grid[0].size();
    if(row==m) return 0;
    if(col<0 || col>n-1) return 1e9 + 7;
    if(dp[row][col]!=-1) dp[row][col];

    int down = grid[row][col] + solve(grid,dp,row+1,col);
    int right = grid[row][col] + solve(grid,dp,row+1,col+1);
    int left = grid[row][col] + solve(grid,dp,row+1,col-1);

    return dp[row][col]  = min(down,min(left,right));
}
int minimumFallingPathSum(vector<vector<int>> &grid){
    int m = grid.size();
    int n = grid[0].size();
    vector<vector<int>> dp(m,vector<int>(n,-1));
    int ans = INT_MAX;
    for(int i=0;i<n;i++){
        int sum = solve(grid,dp,0,i);
        ans = min(ans,sum);
    }
    return ans;
}



// subset sum - check if there exists a subset in the array which totals to sum  . subset doesnt mean subsequence
bool solve(vector<int> &arr, int sum, int n, vector<vector<int>> &dp, int index){
    if(sum==0) return true;
    if(index==n || sum<0) return false;
    if(dp[index][sum]!=-1) return dp[index][sum];

    bool notTake = solve(arr,sum,n,dp,index+1);
    
    bool take = false;
    if(sum>=arr[index]){
         take = solve(arr,sum-arr[index],n,dp,index+1);
    }
    
    return dp[index][sum] = take || notTake;
}
bool subsetSum(vector<int> &arr, int sum){
    int n = arr.size();
    vector<vector<int>> dp(n,vector<int>(sum+1,-1));
    return solve(arr,sum,n,dp,0);
}
// tabulation
bool subsetSum(vector<int> &arr, int sum){
    int n = arr.size();
    vector<vector<int>> dp(n+1,vector<int>(sum+1,false));
    for(int i=0;i<=n;i++){
        dp[i][0] = true;
    }
    for(int i=1;i<=n;i++){
        for(int j=1;j<=sum;j++){
            bool notTake = dp[i-1][j];
            bool take = false;
            if(j>=arr[i]){
                take = dp[i-1][j-arr[i]];
            }
            dp[i][j] = take || notTake;
        }
    }
    return dp[n][sum];
}


// partition equal subset sum - check if the array can be divided into 2 subsets with equal sum
bool solve(vector<int> &arr, int sum, vector<vector<int>> &dp, int index){
    if(sum==0) return true;
    if(index==arr.size() || sum<0) return false;
    if(dp[index][sum]!=-1) return dp[index][sum];

    bool notTake = solve(arr,sum,dp,index+1);
    bool take = false;
    if(sum>=arr[index]){
        take = solve(arr,sum-arr[index],dp,index+1);
    }
    return dp[index][sum] = take || notTake;
}
bool partitonSubsetSum(vector<int> &arr){
    int n = arr.size();
    int sum =0;
    for(int i=0;i<n;i++){
        sum+=arr[i];
    }
    if(sum%2==1) return false;
    sum = sum/2;
    vector<vector<int>> dp(n,vector<int>(sum+1,-1));
    return solve(arr,sum,dp,0);
}


// coin change - we are given coins array and total amount , we have to return min no. denominations required to makke up the amount wehre we can use each denomination any no of times
int solve(vector<int> &coins, int amount, vector<vector<int>> dp,int index){
    int n = coins.size();
    if(amount==0) return 0;
    if(index==n || amount<0) return 1e9;
    if(dp[index][amount]!=-1) return dp[index][amount];
    int notTake = 0 + solve(coins,amount,dp,index+1);
    int take = 1e9;
    if(amount>=coins[index]){
        take = 1 + solve(coins,amount- coins[index],dp,index);
    }

    return min(take,notTake);
}
int coinChange(vector<int> &coins, int amount){
    int n = coins.size();
    vector<vector<int>> dp(n,vector<int>(amount+1,-1));
    int ans = solve(coins,amount,dp,0);
    if(ans>=1e9) return -1;
    return ans;
}
// tabulation
int coinChange(vector<int> &coins, int amount){
    int n = coins.size();
    vector<vector<int>> dp(n,vector<int>(amount+1,-1));
    for(int i=0;i<=amount;i++){
        if(i%coins[0]==0){
            dp[0][i] = i/coins[0];
        } else {
            dp[0][i] = 1e9;
        }
    }

    for(int i=1;i<n;i++){
        for(int j=0;j<=amount;j++){
            int notTake = dp[i-1][j];
            int take = 1e9;
            if(j>=coins[i]){
                take = 1 + dp[i][j-coins[i]];
            }
            dp[i][j] = min(take,notTake);
        }
    }
    int ans = dp[n-1][amount];
    if(ans>=1e9) return -1;
    return ans;
}


// coin change2 - we are given coins array and amount and we need to tell the no. of combinations that make up the amount
int solve(vector<int> &coins, int amount, vector<vector<int>> &dp, int n, int index){
    if(amount==0) return 1;
    if(index==n || amount<0) return 0;
    if(dp[index][amount]!=-1) return dp[index][amount];

    int notTake = solve(coins,amount,dp,n,index+1);
    int take =0;
    if(amount>=coins[index]){
        take = solve(coins,amount-coins[index],dp,n,index+1);
    }
    return dp[index][amount] = take + notTake;
}
int coinChange2(vector<int> &coins, int amount){
    int n = coins.size();
    vector<vector<int>> dp(n,vector<int>(amount+1,-1));
    return solve(coins,amount,dp,n,0);
}


// target sum - we are given nums array and a target and we can either use + or - sign before each number to total to target. return total no. of ways to do so
int solve(vector<int> &nums, int target, vector<vector<int>> &dp, int n, int index, int sum){
    if(index==n){
        if(target==0) return 1;
        else return 0;
    }
    if(target+sum<0 || target+sum>=2*sum+1) return 0;
    if(dp[index][target+sum]!=-1) return dp[index][target+sum];

    int pos = solve(nums,target-nums[index],dp,n,index+1,sum);
    int neg = solve(nums,target+nums[index],dp,n,index+1,sum);

    return dp[index][target+sum] = pos + neg;
}
int findTargetSumWays(vector<int> &nums, int target){
    int n = nums.size();
    int sum = accumulate(nums.begin(),nums.end(),0);
    vector<vector<int>> dp(n,vector<int>(2*sum+1,-1));
    return solve(nums,target,dp,n,0,sum);
}


// longest common subsequence - where we are given two strings and we need to find the length of the longest common subsequence
int solve(string text1, string text2, vector<vector<int>> &dp, int i, int j){
    if(i<0 || j<0) return 0;
    if(dp[i][j]!=-1) return dp[i][j];
    if(text1[i]==text2[j]){
        dp[i][j] = 1 + solve(text1,text2,dp,i-1,j-1);
    } else {
        dp[i][j] = max(solve(text1,text2,dp,i-1,j),solve(text1,text2,dp,i,j-1));
    }
    return dp[i][j];
}
int longestCommonSubsequence(string text1, string text2){
    int m = text1.size();
    int n = text2.size();
    vector<vector<int>> dp(m,vector<int>(n,-1));
    return solve(text1,text2,dp,m-1,n-1);
}
// tabulation
int longestCommonSubsequence(string text1, string text2){
    int m = text1.size();
    int n = text2.size();
    vector<vector<int>> dp(m+1,vector<int>(n+1,0));
    for(int i=1;i<=m;i++){
        for(int j=1;j<=n;j++){
            if(text1[i]==text2[j]){
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j],dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}
// space optimization
int longestCommonSubsequence(string text1, string text2){
    int m = text1.size();
    int n = text2.size();
    vector<int> dp(n+1,0);
    vector<int> curr(n+1,0);
    for(int i=1;i<=m;i++){
        for(int j=1;j<=n;j++){
            if(text1[i-1]==text2[j-1]){
                curr[j] = dp[j-1] + 1;
            } else {
                curr[j] = max(dp[j],curr[j-1]);
            }
        }
        dp = curr;
    }
    return dp[n];
}



// longest palindromic subsequence - where we have one string and we need to find the length of the longest palindrome subsequence
int longestPalindromeSubsequence(string s){
    string t = s;
    int n = s.size();
    reverse(t.begin(),t.end());
    vector<int> dp(n+1,0);
    vector<int> dp2(n+1,0);
    for(int i=1;i<=n;i++){
        for(int j=1;j<=n;j++){
            if(s[i-1]==t[j-1]){
                dp2[j] = 1 + dp[j-1];
            } else {
                dp2[j] = max(dp[j],dp2[j-1]);
            }
        }
        dp = dp2;
    }
    return dp[n];
}


// minimum insertion steps to make a string palindrome 
int minStepsPalindrome(string s){
    int n = s.size();
    string t = s;
    reverse(t.begin(),t.end());
    vector<int> dp(n+1,0);
    vector<int> curr(n+1,0);
    for(int i=1;i<=n;i++){
        for(int j=1;j<=n;j++){
            if(s[i-1]==t[j-1]){
                curr[j] = 1 + dp[j-1];
            } else {
                curr[j] = max(dp[j],curr[j-1]);
            }
        }
        dp =curr;
    }
    return n - dp[n];
}


// delete operations for 2 strings - where we ar given 2 strings and we have to tell no. of steps reqd to make the words same if we can delete one char at a time
// just find the lCS and then delete all the chars
int delStrings(string s, string t){
    int m = s.size();
    int n = t.size();
    vector<int> dp(n+1,0);
    vector<int> curr(n+1,0); // they are both n+1 because thats the size of the inner loop
    for(int i=1;i<=m;i++){
        for(int j=1;j<=n;j++){
            if(s[i-1]==t[j-1]){
                curr[j] = 1 + dp[j-1];
            } else {
                curr[j] = max(dp[j],curr[j-1]);
            }
        }
        dp = curr;
    }
    return m + n - 2*dp[n];
}



// shortest common supersequence - we are given 2 strings and we need to find shortest string which has both of them as subsequences
string shortestCommonSupersequence(string s, string t){
    int m = s.size();
    int n = t.size();
    vector<vector<int>> dp(m+1,vector<int>(n+1,0));
    for(int i=1;i<=m;i++){
        for(int j=1;j<=n;j++){
            if(s[i-1]==t[j-1]){
                dp[i][j] = 1 + dp[i-1][j-1];
            } else {
                dp[i][j] = max(dp[i-1][j] + dp[i][j-1]);
            }
        }
    }
    int i = m;
    int j = n;
    string ans = "";
    while(i>0 && j>0){
        if(s[i-1]==t[j-1]){
            ans+=s[i-1];
            i--;
            j--;
        } else if(dp[i-1][j]>dp[i][j-1]){
            ans += s[i-1];
            i--;
        } else {
            ans+= t[j-1];
            j--;
        }
    }
    while(i>0){
        ans +=s[i-1];
        i--;
    }
    while(j>0){
        ans+=t[j-1];
        j--;
    }
    reverse(ans.begin(),ans.end());
    return ans;
}


// distinct subsequences - here i am given 2 strings s and t and i need to find all the distinct subsequences of s which equals t
int distinctSubsequences(string s, string t){
    int m = s.size();
    int n = t.size();
    vector<vector<int>> dp(m+1,vector<int>(n+1,0));
    for(int i=0;i<=m;i++){
        dp[i][0] = 1; // it means that when size of t is 0 the empty substring is also a subsequence of any prefix of s
    }
    for(int i=1;i<=m;i++){
        for(int j=1;j<=n;j++){
            if(s[i-1]==t[j-1]){
                dp[i][j] = (dp[i-1][j] + dp[i-1][j-1])%mod;  // it means that when chars are equal i hace two options either include it or not , if i do i move diagonally in the dp matrix if not i move up
            } else {
                dp[i][j] = dp[i-1][j];
            }
        }
    }
    return dp[m][n];
}



// edit distance - here we are given 2 strings and we need to find the no. of operations required to make them equal , these operations are insert, delete and replace
int editDistance(string s, string t){
    int m = s.size();
    int n = t.size();
    vector<vector<int>> dp(m+1,vector<int>(n+1,0));
    for(int i=0;i<=m;i++){
        dp[i][0] = i;  // if t is empty no. of deletions required are i for ith index
    }
    for(int j=0;j<=n;j++){
        dp[0][j] = j;   
    }
    for(int i=1;i<=m;i++){
        for(intj=1;j<=n;j++){
            if(s[i-1]==t[j-1]){
                dp[i][j] = dp[i-1][j-1];   // if equal no operation required
            } else {
                dp[i][j] = min(dp[i-1][j-1],min(dp[i][j-1],dp[i-1][j]));  // if not equal , compute all replace ,delete and insert and then find the min of all three
            }
        }
    }
    return dp[m][n];
}



// wildcard matching - we have 2 strings s and pattern (p) , we have to see if s and p match, if p[i] = '?' it means it can match one char and if p[i] = '*', it can match any no. of chars
bool helper(string p, int index){
    for(int i=0;i<=index;i++){
        if(p[i]!='*') return false;
    }
    return true;
}
bool wildcardMatching(string s, string p){
    int m = s.size();
    int n = p.size();
    vector<vector<int>> dp(m+1,vector<int>(n+1,false));
    dp[0][0] = true;
    for(int i=1;i<=m;i++){
        dp[i][0] = false;
    }
    for(int j=1;j<=n;j++){
        dp[0][j] = helper(p,j-1);
    }
    for(int i=1;i<=m;i++){
        for(int j=1;j<=n;j++){
            if(s[i-1]==p[j-1] || p[j-1]=='?'){
                dp[i][j] = dp[i-1][j-1];
            } else if(p[j-1]=='*'){
                dp[i][j] = dp[i-1][j] || dp[i][j-1];
            } else {
                dp[i][j] = false;
            }
        }
    }
    return dp[m][n];
}

// Best time to buy and sell stock 1- here i am given prices array and i can do 1 transaction and i need to maximize the profit 
int buyAndSell1(vector<int> &prices){
    int n = prices.size();
    int mini = prices[0];
    int profit =0;
    for(int i=0;i<n;i++){
        int cost = prices[i] - mini;
        profit = max(cost,profit);
        mini = min(mini,prices[i]);
    }
    return profit;
}

// best time to buy and sell 2 - here i can do as many transactions and i need to maximize the profit
int solve(vector<int> &prices,vector<vector<int>> &dp, int index, int buy){
    if(index== prices.size()) return 0;
    if(dp[index][buy]!=-1) return dp[index][buy];
    int profit =0;
    if(buy==0){
        profit = max(0+solve(prices,dp,index+1,0),-prices[index] + solve(prices,dp,index+1,1));
    }
    if(buy==1){
        profit = max(0+solve(prices,dp,index+1,1),prices[index] + solve(prices,dp,index+1,0));
    }
    return dp[index][buy] = profit;
}
int buyAndSell2(vector<int> &prices){
    int n = prices.size();
    vector<vector<int>> dp(n,vector<int>(2,-1));
    return solve(prices,dp,0,0);
}
// tabulation
int buyAndSell2(vector<int> &prices){
    int n = prices.size();
    vector<int> dp(2,0);
    vector<int> curr(2,0);
    dp[0] = dp[1] =0; // base case when index is at n
    int profit =0;
    for(int i=n-1;i>=0;i--){
        for(int j=0j<2;j++){
            if(j==0){
                profit = max(dp[0],-prices[i]+dp[1]);
            } 
            if(j==1){
                profit = max(dp[1],prices[i]+ dp[0]);
            }
            curr[j] = profit;
        }
        dp = curr;
    }
    return dp[0];
}


// buy and sell stock 3 - where you can do atmost 2 transactions
int solve(vector<int> &prices, vector<vector<vector<int>>> &dp, int index, int buy, int cap){
    if(index==prices.size()) return 0;
    if(cap==0) return 0;
    if(dp[index][buy][cap]!=-1) return dp[index][buy][cap];

    int profit =0;
    if(buy==0){
        profit = max(0+solve(prices,dp,index+1,0,cap),-prices[index]+ solve(prices,dp,index+1,1,cap));
    }
    if(buy==1){
        profit = max(0+solve(prices,dp,index+1,1,cap),prices[index]+ solve(prices,dp,index+1,0,cap-1));
    }
    return dp[index][buy][cap] = profit;
}
int buyAndSell3(vector<int> &prices){
    int n = prices.size();
    vector<vector<vector<int>>> dp(n,vector<vector<int>>(2,vector<int>(3,-1)));
    return solve(prices,dp,0,0,2);
}
// tabulation
int buyAndSell3(vector<int> &prices){
    int n = prices.size();
    vector<vector<int>> dp(2,vector<int>(3,0));
    vector<vector<int>> curr(2,vector<int>(3,0));
    dp[0][0] = 0;
    dp[1][0] = 0;
    int profit =0;
    for(int i=n-1;i>=0;i--){
        for(int j=0;j<2;j++){
            for(int cap=1;cap<=2;cap++){
                if(j==0){
                    profit = max(dp[0][cap],-prices[i] + dp[1][cap]);
                }
                if(j==1){
                    profit = max(dp[1][cap],prices[i] + dp[0][cap-1]);
                }
                curr[j][cap] = profit;
            }
        }
        dp = curr;
    }
    return dp[0][2];
}



// buy and sell stock 4 - where you can buy and sell at most k times
int buyAndSell4(vector<int> &prices, int k){
    int n = prices.size();
    vector<vector<int>> dp(2,vector<int>(k+1,0));
    vector<vector<int>> cur(2,vector<int>(k+1,0));
    int profit =0;
    
    for(int i=n-1;i>=0;i--){
        for(int j=0;j<2;j++){
            for(int cap =1;cap<=k;cap++){
                if(j==0){
                    profit = max(dp[0][cap],-prices[i]+dp[1][cap]);
                }
                if(j==1){
                    profit = max(dp[1][cap],prices[i]+dp[0][cap-1]);
                }
                curr[j][cap] = profit;
            }
        }
        dp = curr;
    }
    return dp[0][k];
}


// buy and sell stock with a cooldown - can do as many transactions but there will be cooldown period of one day after selling a stock
int buyAndSell5(vector<int> &prices){
    int n = prices.size();
    vector<int> dp1(2,0);
    vector<int> dp2(2,0);
    vector<int> curr(2,0);
    int profit =0;
    for(int i=n-1;i>=0;i--){
        for(int j=0;j<2;j++){
            if(j==0){
                profit = max(dp1[0],-prices[i] + dp1[1]);
            }
            if(j==1){
                profit = max(dp1[1],prices[i] + dp2[0]);
            }
            curr[j] = profit;
        }
        dp2 = dp1;
        dp1 = curr;
    }
    return dp1[0];
}


// buy and sell stock with a transaction fee - where can do as many transactions but there will be a transaction fee for each transactio
int buyAndSell6(vector<int> &prices, int fee){
    int n = prices.size();
    vector<int> dp(2,0);
    vector<int> curr(2,0);

    int profit =0;
    for(int i=n-1;i>=0;i--){
        for(int j=0;j<2;j++){
            if(j==0){
                profit = max(dp[0],-prices[i] + dp[1]);
            }
            if(j==1){
                profit = max(dp[1], prices[i] - fee + dp[0]);
            }
            curr[j] = profit;
        }
        dp = curr;
    }
    return dp[0];
}


// longest increasing subsequence - given an array nums , return the longest increasing subsequence
// memoization 
int LengthOflis(vector<int> &nums){
    int n = nums.size();
    vector<vector<int>> dp(n,vector<int>(n+1,-1));
    return solve(nums,0,-1,dp);
}
int solve(vector<int> &nums, int index, int prev, vector<vector<int>> &dp){
    int n = nums.size();
    if(index==n) return 0;
    if(dp[index][prev+1]!=-1) return dp[index][prev+1];

    int notTake = 0 + solve(nums,index+1,prev,dp);
    int take =0;
    if(prev==-1 || nums[index]>nums[prev]){
        take = 1 + solve(nums,index+1,index,dp);
    }
    return dp[index][prev+1] = max(take,notTake);
}
// tabulation
int LengthOfLis(vector<int> &nums){
    int n = nums.size();
    vector<vector<int>> dp(n+1,vector<int>(n+1,0));
    for(int index= n-1;index>=0;index--){
        for(int prev = index-1;prev>=-1;prev--){
            int notTake = 0 + dp[index+1][prev+1];
            int take = 0;
            if(prev==-1 || nums[index]>nums[prev]){
                take = 1 + dp[index+1][index+1];
            }
            dp[index][prev+1] = max(take,notTake);
        }
    }
    return dp[0][0];
}
// space optimization
int LengthOfLis(vector<int> &nums){
    int n = nums.size();
    vector<int> dp(n+1,0);
    vector<int> curr(n+1,0);
    for(int index= n-1;index>=0;index--){
        for(int prev = index-1;prev>=-1;prev--){
            int notTake = 0 + dp[prev+1];
            int take = 0;
            if(prev==-1 || nums[index]>nums[prev]){
                take = 1 + dp[index+1]
            }
            curr[prev+1] = max(take,notTake);
        }
        dp = curr;
    }
    return dp[0][0];
}
// binary search - O(nlogn)
int LengthOfLis(vector<int> &nums){
    int n = nums.size();
    vector<int> temp;
    temp.push_back(nums[0]);
    int cnt =0;
    for(int i=1;i<n;i++){
        if(nums[i]>temp.back()){
            temp.push_back(nums[i]);
            cnt++;
        } else {
            int index  = lower_bound(temp.begin(),temp.end(),nums[i] - temp.begin());  // uses binary search to find the index of the element which is equal to nums[i] or just greater than that.
            temp[index] = nums[i];
        }
    }
    return cnt;
}


// print longest increasing subsequence
vector<int> printLis(vector<int> &nums){
    int n = nums.size();
    vector<int> dp(n+1,0);  // for storing the length of lis which ends at that particular index
    vector<int> hash(n); // for storing the indexes of the prev element in the LIS

    int maxi = INT_MIN;  // for storing the value of Lis of the whole array
    int lastIndex = 0;  // it stores the lastIndex of the Lis 
    for(int i=0;i<n;i++){
        hash[i] = i;
        for(int j=0;j<i;j++){
            if(nums[i]>nums[j] && 1+ dp[j]>dp[i]){
                dp[i] = 1 + dp[j];
                hash[i] = j;
            }
        }
        if(dp[i]>maxi){
            maxi = dp[i];
            lastIndex = i;
        }
    }
    vector<int> ans;
    ans.push_back(nums[lastIndex]);
    while(hash[lastIndex]!=lastIndex){
        lastIndex = hash[lastIndex];
        ans.push_back(nums[lastIndex]);
    }
    reverse(ans.begin(),ans.end());
    return ans;
}


// largest divisible subset - here we are give nums and we need to return the largest subset where every pair satisfies nums[i]%nums[j] = 0
vector<int> largestDivisibleSubset(vector<int> &nums){
    int n = nums.size();
    vector<int> dp(n,1)
    sort(nums.begin(),nums.end());
    vector<int> hash(n);
    int lastIndex = 0;
    int maxi = INT_MIN;
    for(int i=0;i<n;i++){
        for(int j=0;j<i;j++){
            if(nums[i]%nums[j]==0 && 1+dp[j]>dp[i]){
                dp[i] = 1 + dp[j];
                hash[i] = j;
            }
        }
        if(dp[i]> maxi){
            maxi = dp[i];
            lastIndex = i;
        }
    }

    vector<int> ans;
    ans.push_back(nums[lastIndex]);
    while(hash[lastIndex]!=lastIndex){
        lastIndex = hash[lastIndex];
        ans.push_back(lastIndex);
    }
    return ans;
}


// min cost to cut a stick - we are given n which is the length of the stick and the cuts array and we need to minimize the cost of cutting where the cost to make a cut is the length of the stick
int minCostToCutStick(int n, vector<int> &cuts){
    sort(cuts.begin(),cuts.end());
    int c = cuts.size();
    cuts.push_back(n);
    cuts.insert(cuts.begin(),0);
    vector<vector<int>> dp(c+1,vector<int>(c+1,-1));
    return solve(1,c,dp,cuts);
}
int solve(int i, int j, vector<vector<int>> &dp, vector<int> &cuts){
    if(i>j) return 0;
    if(dp[i][j]!=-1) return dp[i][j];

    int ans = INT_MAX;
    for(int ind = i;ind<=j;ind++){
        int cost = cuts[j+1] - cuts[i-1] + solve(i,ind-1,dp,cuts) + solve(ind+1,j,cuts,dp);
        ans = min(cost,ans);
    }
    return dp[i][j] = ans;
}


int burstBalloons(vector<int> &nums){
    int n = nums.size();
    nums.push_back(1);
    nums.insert(nums.begin(),1);
    vector<vector<int>> dp(n+2,vector<int>(n+2,-1));
    return solve(1,n,dp,nums);
}
int solve(int i, int j, vector<vector<int>> &dp, vector<int> &nums){
    if(i>j) return 0;
    if(dp[i][j]!=-1) return dp[i][j];

    int ans = INT_MIN;
    for(int ind=i;ind<=j;ind++){
        int sum = solve(i,ind-1,dp,nums) + solve(ind+1,j,dp,nums) + nums[ind-1]*nums[ind]*nums[ind+1];
        ans = min(ans,sum);
    }
    return ans;
}