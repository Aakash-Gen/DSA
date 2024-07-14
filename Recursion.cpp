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




// print all subsequences with a sum =k
vector<vector<int>> subsequenceSum(vector<int> arr,int sum){
    vector<vector<int>> ans;
    vector<int> temp;
    int s =0;
    solve(0,arr,sum,ans,s,temp);
    return ans;
}
void solve(int index , vector<int> &arr, int sum, vector<vector<int>> &ans,int s,vector<int> &temp){

    if(index==arr.size()){
        if(s==sum){
            ans.push_back(temp);
            return;
        }
    }
    
    temp.push_back(arr[index]);
    s+=arr[index];
    solve(index+1,arr,sum,ans,s);
    temp.pop_back();
    s-=arr[index];
    solve(index+1,arr,sum,ans,s);
}


// print any subsequence with sum =k
vector<int> anySubsequenceSum(vector<int> arr,int sum){
    vector<int> temp;
    int s =0;
    solve(0,arr,sum,s,temp);
    return temp;
}
bool solve(int index, vector<int> &arr, int sum, int s, vector<int> &temp){
    if(index==arr.size()){
        if(s==sum){
            return true;
        }
    } else {
        return false;
    }

    temp.push_back(arr[index]);
    s+=arr[index];
    if(solve(index+1,arr,sum,s,temp)==true) return true;
    s-=arr[index];
    temp.pop_back();
    if(solve(index+1,arr,sum,s,temp)==true) return true;
    return false;
}


// count the subsequences with sum = k
int countSubsequenceSum(vector<int> arr, int sum){
    int s=0;
    return solve(0,arr,sum,s);
}

int solve(int index, vector<int> &arr, int sum, int s){
    if(index==arr.size()){
        if(s==sum){
            return 1;
        } else {
            return 0;
        }
    }
    s+=arr[index];
    int l = solve(index+1,arr,sum,s);
    s-=arr[index];
    int r = solve(index+1,arr,sum,s);
    return l+r;
}



// subsets - where we have unique numbers and we we want to print all the combinations
vector<vector<int>> subsets(vector<int> &nums){
    vector<vector<int>> ans;
    vector<int> temp;
    solve(0,nums,temp,ans);
    return ans;
}
void solve(int index, vector<int> &nums, vector<int> &temp, vector<vector<int>> &ans){
    if(index==nums.size()){
        ans.push_back(temp);
        return;
    }
    temp.push_back(nums[index]);
    solve(index+1,nums,temp,ans);
    temp.pop_back();
    solve(index+1,nums,temp,ans);
}


// subsets2 - where we can have duplicate numbers in the given array and we want to print all the combinations making sure there are non duplicate temps in the ans
vector<vector<int>> subsets2(vector<int> &nums){
    vector<vector<int>> ans;
    sort(nums.begin(),nums.end());
    vector<int> temp;
    solve(0,nums,temp,ans);
    return ans;
}
void solve(int index, vector<int> &nums, vector<int> &temp, vector<vector<int>> &ans){
    if(index==nums.size()){
        return;
    }
    for(int i=index,i<nums.size();i++){
        if(i>index && nums[i]==nums[i-1]) continue;
        temp.push_back(nums[i]);
        ans.push_back(temp);
        solve(i+1,nums,temp,ans);
        temp.pop_back();
    }
}

// combination sum - where we have unique numbers and we need all the combinations whose sum is equal i target and here we can use any number any number of times
vector<vector<int>> combinationSum(vector<int> &nums, int target){
    vector<vector<int>> ans;
    vector<int> temp;
    solve(0,nums,temp,ans,target);
    return ans;
}
void solve(int index, vector<int> &nums, vector<int> &temp, vector<vector<int>> &ans, int target){
    if(index==nums.size()){
        if(target==0){
            ans.push_back(temp);
            return;
        }
    }
    if(nums[index]<=target){
        temp.push_back(nums[index]);
        solve(index,nums,temp,ans,target-nums[index]);
        temp.pop_back();
    }
    solve(index+1,nums,temp,ans,target);
}


// combination sum2 - where we can have dupicate number in a given array but we can only use one number once
vector<vector<int>> combinationSum2(vector<int> &nums){
    vector<vector<int>> ans;
    sort(nums.begin(),nums.end());
    int s=0;
    vector<int> temp;
    solve(0,nums,temp,ans,target);
    return ans;
}
void solve(int index, vector<int> &nums, vector<int> &temp, vector<vector<int>> &ans, int target){
    if(target==0){   // here we have not use index==nums.size() check bec it is not necessary we are gettiing target at end as we are breaking out of for loop before
        ans.push_back(temp);
        return;
    }
    for(int i=index,i<nums.size();i++){
        if(i>index && nums[i]==nums[i-1]) continue;
        if(nums[i]>target) break;
        temp.push_back(nums[i]);
        solve(i+1,nums,temp,ans,target-nums[i]);
        temp.pop_back();
    }
}


// combination sum3 - where we are given k and n where k is the size of the valid combination and n is the sum of all elememts of valid combination and numbers from only 1 to 9 are used
vector<vector<int>> combinationSum3(int n , int k){
    vector<vector<int>> ans;
    int s=0;
    vector<int> temp;
    solve(1,k,temp,ans,n,s);
    return ans;
}
void solve(int index, int k, vector<int> &temp, vector<vector<int>> &ans, int n, int s){
    if(s==n && temp.size()==k){
        ans.push_back(temp);
        return;
    }
    if(s>n || temp.size() > k || index>9) return;
    
    s+= index;
    temp.push_back(index);
    solve(index+1,k,temp,ans,n,s);
    s-=index;
    temp.pop_back();
    solve(index+1,k,temp,ans,n,s);
}


// combination sum4 - here we are given array of distinct integers and the value target and we need to find the no. of combinations which form the target
int solve(vector<int> &nums, vector<int> &dp, int target){
    if(target==0) return 1;
    if(target<0) return 0;
    if(dp[target]!=-1) return dp[target];
    int ans =0;
    for(int i=0;i<nums.size();i++){
        ans+= solve(nums,dp,target-nums[i]);
    }
    return dp[target] = ans;
}
int combinationSum4(vector<int> &nums, int target){
    int n = nums.size();
    vector<int> dp(target+1,-1);
    return solve(nums,dp,target);
}


// combinations - where we are give k which is size of the combination and n where we can take numbers from 1 to 9 , return all possible combinations
vector<vector<int>> combination(int n , int k){
    vector<vector<int>> ans;
    vector<int> temp;
    solve(1,k,temp,ans,n);
    return ans;
}
void solve(int index, int k, vector<int> &temp, vector<vector<int>> &ans, int n){
    if(temp.size()==k){
        ans.push_back(temp);
        return;
    }
    for(int i=index;i<=n;i++){
        temp.push_back(i);
        solve(index+1,k,temp,ans,n);  
        temp.pop_back();
    }
}


// permutations - where we have to find all the permutations of a given array where each permutation has same size as that of the array
vector<vector<int>> permutations(vector<int> &nums){
    vector<vector<int>> ans;
    solve(0,nums,ans);
    return ans;
}
void solve(int index, int k, vector<int> &temp, vector<vector<int>> &ans, int n){
    if(index==nums.size()){
        ans.push_back(nums);
        return;
    }
    for(int i=index,i<nums.size();i++){
        swap(nums[i],nums[index]);
        solve(index+1,nums,ans);  // to generate all permutations we do index+1 rather than i+1
        swap(nums[i],[nums[index]]);
    }
}


// permutations2 - here there are dublicates and find all the premutations but they should be distinct
void solve(int index, vector<vector<int>> &ans, vector<int> &temp, vector<int> &nums){
    if(index==nums.size()){
        ans.push_back(temp);
        return;
    }
    unordered_map<int,int> mpp;
    for(int i=index;i<nums.size();i++){
        if(mpp.find(nums[i]!=mpp.end())){
            continue;
        }
        mpp[nums[i]]++;
        swap(nums[i],nums[index]);
        solve(index+1,ans,temp,nums);
        swap(nums[i],nums[index]);
    }
}
vector<vector<int>> permutations(vector<int> &nums){
    int n = nums.size();
    vector<vector<int>> ans;
    vector<int> temp;
    solve(0,ans,temp,nums);
    return ans;
}


// n queens - we need to return all the possible ways we can place n queens on a chessboard of size nxn so that they dont cut each other
vector<vector<string>> nQueens(int n){
    vector<vector<string>> ans;
    vector<string> temp(n);
    string s(n,'.');
    for(int i=0;i<n;i++){
        temp[i] = s;
    }
    solve(ans,temp,n,0);
    return ans;
}
void solve(vector<vector<string>> &ans, vector<string> &temp, int n, int col){
    if(col==n){
        ans.push_back(temp);
        return;
    }
    for(int row=0;row<n;row++){
        if(isSafe(temp,n,col,row)){
            temp[row][col] = 'Q';
            solve(ans,temp,n,col+1);
            temp[row][col] = '.';
        }
    }
}
bool isSafe(vector<string> &temp, int n, int col, int row){
    int row1 = row;
    int col1 = col;
    while(row1>=0 && col1>=0){
        if(temp[row1][col1]=='Q') return false;
        row1--;
        col1--;
    }
    row1= row;
    col1 = col;
    while(col1>=0){
        if(temp[row][col]=='Q') return false;
        col1--;
    }
    row1= row;
    col1 = col;
    while(row1<n && col1>=0){
        if(temp[row1][col1]=='Q') return false;
        row1++;
        col1--;
    }
    return true;
}

// alternate using formula for leftrow, upper and lower diagonal
vector<vector<string>> nQueen(int n){
    vector<vector<string>> ans;
    vector<string> temp(n);
    string s(n,'.');
    for(int i=0;i<n;i++){
        temp[i] = s;
    }
    vector<int> left(n,0);
    vector<int> upperDiag(2*n-1,0);
    vector<int> lowerDiag(2*n-1,0);
    solve(ans,temp,n,0,left,upperDiag,lowerDiag);
    return ans;
}
void solve(vector<vector<string>> &ans, vector<string> &temp, int n, int col, vector<int> &left, vector<int> &upperDiag, vector<int> &lowerDiag){
    if(col==n){
        ans.push_back(temp);
        return;
    }
    for(int row=0;row<n;row++){
        if(left[row] == 0 && upperDiag[row+col]==0 && lowerDiag[n-1 + col-row]==0){
            temp[row][col]= 'Q';
            left[row] =1;
            upperDiag[row+col] = 1;
            lowerDiag[n-1+col-row] =1;
            solve(ans,temp,n,col+1,left,upperDiag,lowerDiag);
            temp[row][col]= '.';
            left[row] =0;
            upperDiag[row+col] = 0;
            lowerDiag[n-1+col-row] =0;
            
        }
    }
}


// sudoku solver 
void sudoku(vector<vector<char>> &board){
    return solve(board);
}
void solve(vector<vector<char>> &board){
    for(int i=0;i<board.size();i++){
        for(int j=0;j<board[0].size();j++){
            if(board[i][j]== '.'){
                for(int i='1';k<='9';k++){
                    if(isPossible(board,i,j,k)){
                        board[i][j] = k;
                        if(solve(board)==true) return true;
                        else board[i][j] = '.';
                    }
                }
                return false;
            }
        }
    }
    return true;
}
bool isPossible(vector<vector<char>> &board, int row, int col, int k){
    for(int i=0;i<9;i++){
        if(board[i][col]==k) return false;
        if(board[row][i]==k) return false;
        if(board[3*(row/3)+ i/3][3*(col/3)+ i%3]==k) return false;
    }
    return true;
}


// palindrome partitioning - given a sting s, patrition s such that every substring is a palindrome, return all such partitions
vector<vector<string>> palindromePartition(string s){
    int n = s.size();
    vector<vector<string>> ans;
    vector<string> temp;
    solve(ans,temp,s,0);
    return ans;
}
void solve(vector<vector<string>> &ans, vector<string> &temp, string s, int index){
    if(index==s.size()){
        ans.push_back(temp);
        return;
    }
    for(int i=index;i<s.size();i++){
        if(isPalindrome(s,i,index)){
            temp.push_back(s.substr(index,i-index+1));
            solve(ans,temp,s,i+1);
            temp.pop_back();
        }
    }
}
bool isPalindrome(string s, int end, int start){
    while(start<=end){
        if(s[start]==s[end]){
            end--;
            start++;
        } else return false;
    }
    return true;
}


// word break - given a string s and word dict - a dictionary of words , return true if s can be formed from the words , you can use same word any no. of times
bool wordBreak(string s, vector<string> &wordDict){
    unordered_map<int,bool> mpp;
    return solve(s,wordDict,mpp,0);
}
bool solve(string s, vector<string> &wordDict, unordered_map<int,bool> &mpp,int index){
    if(index==s.size()){
        return true;
    }
    if(mpp.find(index)!=mpp.end()) return mpp[index];
    for(int i=index;i<s.size();i++){
        string word = s.substr(index,i-index+1);
        if(find(wordDict.begin(),wordDict.end(),word)!=wordDict.end()){
            if(solve(s,wordDict,mpp,i+1)==true) return mpp[index] =true;
        }
    }
    return mpp[index];
}


// word search - return true if the word exists in the matrix and false otherwise
bool wordSearch(vector<vector<char>> &board, string s){
    int m = board.size();
    int n = board[0].size();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(board[i][j]==s[0] && solve(0,board,i,j,s)) return true;
        }
    }
    return false;
}
bool solve(int index, vector<vector<char>> &board, int row, int col,string s){
    int m = board.size();
    int n = board[0].size();
    if(index==s.size()) return true;
    if(row<0 || col<0 || row>=m || col>=n || board[row][col]=='*' || board[row][col]!=s[index]) return false;

    board[row][col] = '*';
    bool found = solve(index+1,board,row+1,col,s) || solve(index+1,board,row-1,col,s) || solve(index+1,board,row,col+1,s) || solve(index+1,board,row,col-1,s);
    board[row][col] = s[index];
    return found;
}


// binary tree paths - given the root of th ebinary tree, return all root to leaf paths in any order
vector<string> binaryTreePaths(TreeNode* root){
    vector<strin> ans;
    vector<int> temp;
    solve(ans,temp,root);
    return ans;
}
void solve(vector<string> &ans, vector<int> &temp, TreeNode* root){
    if(root==NULL) return;
    temp.push_back(root->val);
    if(root->left==NULL && root->right==NULL){
        string s = "";
        for(int i=0;i<temp.size();i++){
            s+= to_string(temp[i]);
            if(i<temp.size()-1){
                s+= "->";
            }
        }
    } else {
        solve(ans,temp, root->left);
        solve(ans,temp,root->right);
    }
    temp.pop_back();
}


vector<string> binaryWatch(int turnedOn){
    vector<string> ans;
    pair<int,int> time;
    vector<int> hours = {1,2,4,8};
    vector<int> minutes = {1,2,4,8,16,32};
    solve(ans,time,turnedOn,hours,minutes,0);
    return ans;
}
void solve(vector<string> &ans,pair<int,int> &time, int turnedOn, vector<int> &hours, vector<int> &minutes, int index){
    if(turnedOn==0){
        ans.push_back(to_string(time.first) + (time.second<10 ? ":0" : ":") + to_string(time.second));
        return;
    }
    for(int i=index;i<hours.size()+minutes.size();i++){
        if(i<hours.size()){
            time.first+=hours[i];
            if(time.first<12){
                solve(ans,time,turnedOn-1,hours,minutes,i+1);
            }
            time.first-=hours[i];
        } else {
            time.second+= minutes[i-hours.size()];
            if(time.second<60){
                solve(ans,time,turnedOn-1,hours,minutes,i+1);
            }
            time.seconnd-=minutes[i-hours.size()];
        }
    }
}