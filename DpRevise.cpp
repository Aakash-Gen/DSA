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



int fibonacci(int n){
    if(n==0) return 0;
    if(n==1) return 1;
    int prev2 = 0;
    int prev1 = 1;
    for(int i=2;i<=n;i++){
        int curr = prev2 + prev1;
        prev2 = prev1;
        prev1 = curr;
    }
    return curr;
}


int costStairs(vector<int> cost){
    int n = cost.size();
    int dp1 = 0;
    int dp2 = 0;
    int dp =0;
    for(int i=2;i<n;i++){
        int step1 = dp1 + cost[i-1];
        int step2 = dp2 + cost[i-2];
        dp = min(dp1,dp2);
        dp2 = dp1;
        dp1 = dp;
    }
    return dp;
}


vector<int> pascal(int n){
    vector<int> ans(n+1);
    if(n==0) return {1};
    ans[0] = 1;
    int prev = 1;
    for(int i=i;i<=n;i++){
        ans[i] = prev* (n-i+1)/i;
    }
    return ans;
}


int cnt(string s, string word){
    string temp = word;
    int cnt =0;
    while(s.find(temp)!=string::npos){
        cnt++;
        temp+=word;
    }
    return cnt;
}


vector<TreeNode*> fbt(int n){
    unordered_map<int,vector<TreeNode*>> mpp;
    return solve(n,mpp);
}
vector<TreeNode*> solve(int n , unordered_map<int,vector<TreeNode*>> &mpp){
    vector<TreeNode*> temp;
    if(n%2==0) return temp;
    if(n==1) {
        TreeNode* root = new TreeNode(0);
        temp.push_back(root);
        return temp;
    }
    if(mpp.find(n)!=mpp.end()){
        return mpp[n];
    }
    for(int i =1;i<n;i++){
        TreeNode* left = solve(i,mpp);
        TreeNode* right = solve(n-i-1,mpp);
        for(auto &l: left){
            for(auto &r: right){
                TreeNode* node = new TreeNode(0);
                node->left = l;
                node->right = r;
                temp.push_back(node);
            }
        }
    }
    return mpp[n];
}

int solve(vector<int> &dp, vector<int> &arr, int k, int index){
    if(index==arr.size()) return 0;
    if(dp[index]!=-1) return dp[index];

    int len = 0;
    int maxi = INT_MIN;
    int ans = INT_MIN;
    for(int i=index;i<min(index+k,n);i++){
        len++;
        maxi = max(maxi,arr[i]);
        int sum = len*maxi + solve(dp,arr,k,i+1);
        ans = min(ans,sum);
    }
    return dp[index] = ans;
}
int maxSum(vector<int> &arr, int k){
    int n = arr.size();
    vector<int> dp(n,-1);
    return solve(dp,arr,k,0);
}


int eggDrop(int n){
    if(n==0) return 0;
    if(n==1) return 1;
    vector<int> dp(n+1);
    dp[0] = 0;
    dp[1] = 1;
    dp[2] = 2;
    for(int i=3;i<=n;i++){
        int temp = n;
        for(int j=1;j<=i;i++){
            int breaks = j;
            int survives = dp[i-j] + 1;
            temp = min(temp,max(breaks,survives));
        }
        dp[i] = temp;
    }
    return dp[n];
}

int rob(vector<int> &nums){
    int n= nums.size();
    vector<int> dp(n,-1);
    return solve(nums,dp,0);
}
int solve(vector<int> &nums, vector<int> &dp, int index){
    if(index>=nums.size()) return 0;
    if(dp[index]!=-1) return dp[index];

    int notRob = solve(nums,dp,index+1);
    int rob = nums[index] + solve(nums,dp,index+2);
    return dp[index] = max(rob,notRob);
}
// tab
int rob(vector<int> &nums){
    int n = nums.size();
    vector<int> dp(n+1,0);
    dp[n] = 0;
    for(int i=n-1;i>=0;i--){
        int notRob = dp[i+1];
        int rob = nums[i];
        if(i<n-2){
            rob += dp[i+2];
        }
        dp[i] = max(rob,notRob);
    }
    return dp[0];
}

int rob2(vector<int> &nums){
    int n = nums.size();
    if(n==1) return nums[0];
    vector<int> dp1(n,-1);
    vector<int> dp2(n,-1);
    int index1 = solve(dp,nums,0,n-2);
    int index2 = solve(dp,nums,1,n-1);
    return max(index1,index2);
}
int solve(vector<int> &dp,vector<int> &nums, int index, int end){
    if(index>end) return 0;
    if(dp[index]!=-1) return dp[index];

    int notRob = solve(dp,nums,index+1,end);
    int rob = nums[index] + solve(dp,nums,index+2,end);

    return dp[index] = max(rob,notRob);
}


int paths(vector<vector<int>> &grid){
    int m = grid.size();
    int n = grid[0].size();
    vector<vector<int>> dp(m,vector<int>(n,-1));
    return solve(grid,dp,0,0);
}
int solve(vector<vector<int>> &grid, vector<vector<int>> &dp, int row, int col){
    if(roww== m-1 && col==n-1) return 1;
    if(row>=m || col>=n) return 0;
    if(dp[row][col]!=-1) return dp[row][col];

    int down = solve(grid,dp,row+1,col);  
    int right = solve(grid,dp,row,col+1);  

    return dp[row][col] = right + down;
}
// tab
int paths(vector<vector<int>> &grid){
    int m = grid.size();
    int n = grid[0].size();
    vector<vector<int>> dp(m,vector<int>(n,-1));
    dp[m-1][n-1] = 1;
    for(int j=0;j<n;j++){
        dp[m-1][j] = 1;
    }
    for(int i=0;i<m;i++){
        dp[i][n-1] = 1;
    }
    for(int i=m-2;i>=0;i--){
        for(int j=n-2;j>=0;j--){
            int down = dp[i+1][j];
            int right = dp[i][j+1];
            dp[i][j] = down + right;
        }
    }
    return dp[0][0];
}


int paths2(vector<vector<int>> &grid){
    int m = grid.size();
    int n = grid[0].size();
    vector<vector<int>> dp(m,vector<int>(n,-1));
    return solve(grid,dp,0,0);
}
int solve(vector<vector<int>> &grid, vector<vector<int>> &dp,int row, int col){
    int m= grid.size();
    int n = grid[0].size();
    if(row>=m || col>=n || grid[row][col]==1) return 0;
    if(row==m-1 && col==n-1) return 1;
    if(dp[row][col]!=-1) return dp[row][col];

    int down = solve(grid,dp,row+1,col);
    int right = solve(grid,dp,row,col+1);
    return dp[row][col] = down + right;
}
// tab
 int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid){
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        vector<vector<int>> dp(m,vector<int>(n));
        for(int row=0;row<m;row++){
            for(int col=0;col<n;col++){
                if(obstacleGrid[row][col]==1) dp[row][col]=0;
                else if(row==0 && col==0) dp[row][col]=1;
                else {
                    int left =0, up =0;
                    if(row>0) up = dp[row-1][col];
                    if(col>0) left = dp[row][col-1];
                    dp[row][col] = left + up;
                }
                
            }
        }
        return dp[m-1][n-1];
    } 


int minSum(vector<vector<int>> &grid){
    int m = grid.size();
    int n = grid[0].size();

    vector<vector<int>> dp(m,vector<int>(n,-1));
    int ans = INT_MAX;
    for(int i=0;i<n;i++){
        int sum = solve(grid,dp,0,i);
        ans= min(ans,sum);
    }
    return ans;
}   
int solve(vector<vector<int>> &grid, vector<vector<int>> &dp, int row, int col){
    int m = grid.size();
    int n = grid[0].size();
    if(row==m) return 0;
    if(col<0 || col>=n) return 1e9;
    if(dp[row][col]!=-1) return dp[row][col];

    int down = grid[row][col] + solve(grid,dp,row+1,col);
    int left = grid[row][col] + solve(grid,dp,row+1,col-1);
    int right = grid[row][col] + solve(grid,dp,row+1,col+1);

    return dp[row][col] = min({down,left,right});
}
// tab
int minSum(vector<vector<int>> &grid){
    int m = grid.size();
    int n = grid[0].size();

    vector<vector<int>> dp(m,vector<int>(n,-1));
    for(int i=0;i<n;i++){
        dp[m-1][i] = grid[m-1][i];
    }
    for(int i=m-2;i>=0;i--){
        for(int j=0;j<n;j++){
            int down = dp[i+1][j];
            int left = INT_MAX, right = INT_MAX;
            if(j>0){
                left = dp[i+1][j-1];
            }
            if(j<n-1){
                right = dp[i+1][j+1];
            }
            dp[i][j] = matrix[i][j] + min({left,down,right});
        }
    }
    int ans = INT_MAX;
    for(int i=0;i<n;i++){
        ans = min(ans,dp[0][i]);
    }
    return ans;
}   


int coinChange(vector<int> &nums,int amount){
    int n = nums.size();
    vector<vector<int>> dp(n,vecor<int>(amount+1,-1));
    return solve(nums,dp,amount,0);
}
int solve(vector<int> &nums, vector<vector<int>> &dp,int amount, int index){
    int n = nums.size();
    if(amount==0) return 0;
    if(index==n) return 1e9;
    if(dp[index][amount]!=-1) return dp[index][amount];

    int notTake = solve(nums,dp,amount,index+1);
    int take = 0;
    if(amount>=nums[index]){
        take = 1 + solve(nums,dp,amount-nums[index],index);
    }
    return dp[index][amount] = min(take,notTake);
}