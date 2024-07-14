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
   



// zigzag level order traversal
vector<vector<int>> zigzagLevelOrder(TreeNode* root){
    vector<vector<int>> ans;
    if(root==NULL) return ans;
    queue<TreeNode*>q;
    q.push(root);
    bool leftToright= true;
    while(!q.empty()){
        int size = q.size();
        vector<int>level(size);
        for(int i=0;i<size;i++){
            TreeNode* node= q.front();
            q.pop();
            int index = leftToright? i: (size-i-1);
            level[index]=node->val;
            if(node->left!=NULL){
                q.push(node->left);
            }
            if(node->right!=NULL){
                q.push(node->right);
            }
        }
        ans.push_back(level);
        leftToright=!leftToright;
    }
    return ans;
}


// height of the tree
int maxDepth(TreeNode* root){
    if(root==NULL) return 0;
    int lh= maxDepth(root->left);
    int rh= maxDepth(root->right);
    return 1+max(lh,rh);
}


// diameter - the max distance between two nodes of the tree
int diameter(TreeNode* root){
    int diameter =0;
    height(root,diameter);
    return diameter;
} 
int height(TreeNode* root,int &diameter){
    if(root==NULL) return 0;
    int lh=height(root->left,diameter);
    int rh= height(root->right,diamter);
    diameter= max(diameter,(lh+rh));
    return 1+max(lh,rh);
}


// balanced Tree - the one where the right and left subtrees dont have a height difference of more than 1
bool balancedTree(TreeNode* root){
    return height(root)!=-1;
}
int height(TreeNode* root){
    if(root==NULL) return 0;
    int lh= height(root->left);
    if(lh==-1) return -1;
    int rh = height(root->right);
    if(rh==-1) return -1;
    if(abs(rh-lh)>1) return -1;
    return 1+max(lh,rh);
}


// same tree- tree which are identical
bool sameTree(TreeNode* root1, TreeNode* root2){
    if(root1==NULL && root2==NULL) return true;
    if(root1==NULL || root2==NULL) return false;
    return root1->val==root2->val && sameTree(root1->left,root2->left) && sameTree(root1->right,root2->right);
}


// right side view of the tree
vector<int> rightSideView(TreeNode* root){
    vector<int> ans;
    recursive(root,ans,0);
    return ans;
}
void recursive(TreeNode* root, vector<int> &ans, int level){
    if(root==NULL) return;
    if(level==ans.size()){
        ans.push_back(root->val);
    }
    recursive(root->right,ans,level+1);
    recursive(root->left,ans,level+1);
}


// symmetric tree
bool symmetric(TreeNode* root){
    if(root==NULL) return true;
    return recursive(root->left,root->right);
}
bool recursive(TreeNode* root1,TreeNode* root2){
    if(root1==NULL && root2==NULL) return true;
    if(root1==NULL || root2==NULL) return false;
    return root1->val==root2->val && recursive(root1->left,root2->right) && recursive(root1->right,root2->left);
}


// max path sum
int maxPathSum(TreeNode* root){
    int maxi= INT_MIN;
    findMaxPath(root,maxi);
    return maxi;
}
int findMaxPath(TreeNode* root, int &maxi){
    if(root==NULL) return 0;
    int leftSum= max(0,findMaxPath(root->left,maxi));
    int rightSum = max(0,findMaxPath(root->right,maxi));
    maxi= max(maxi,(leftSum+rightSum+root->val));
    return max(leftSum,rightSum) + root->val;
}


// vertical order traversal - column wise traversal from top to bottom
vector<vector<int>> verticalOrderTraversal(TreeNode* root){
    queue<pair<TreeNode*,pair<int,int>>> q;
    map<int,map<int,multiset<int>>> mpp;
    q.push({root,{0,0}});
    while(!q.empty()){
        auto temp =q.front();
        q.pop();
        TreeNode* node = temp.first;
        int x = temp.second.first;
        int y = temp.second.second;
        map[x,y].insert(node->val);
        if(node->left!=NULL){
            q.push({node->left,{x-1,y+1}});
        }
        if(node->right!=NULL){
            q.push({node->right,{x+1,y+1}});
        }
    }
    vector<vector<int>> ans;
    for(auto &i: mpp){
        vector<int> col;
        for(auto &j: i.second){
            col.insert(col.end(),j.second.begin(),j.second.end());
        }
        ans.push_back(col);
    }
    return ans;
}


// lowest common ancestor of two given nodes
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q){
    if(root==NULL || root==p || root==q) return root;
    TreeNode* left = lowestCommonAncestor(root->left,p,q);
    TreeNode* right = lowestCommonAncestor(root->right,p,q);
    if(left==NULL) return right;
    else if (right==NULL) return left;
    else return root;
}


// search in a BST
TreeNode* search(TreeNode* root, int val){
    while(root!=NULL && root->val!=NULL){
        root = val<root->val ? root->left : root->right;
    }
    return root;
}


// ceil in a bst - closest integer greater than or equal to the given number
int ceilOfBst(TreeNode* root, int x){
    int ceil =-1;
    while(root!=NULL){
        if(root->val==x){
        ceil=x;
        return ceil;
        }
        if(x>root->val){
            root=root->right;
        } else {
            ceil= root->val;
            root= root->left;
        }
    }
    return ceil;
}


// floor of a bst - closest integer smaller than or equal to the given number
int floorOfbst(TreeNode* root,int x){
    int floor=-1;
    while(root!=NULL){
        if(root->val==x){
        floor= x;
        return floor;
        }
        if(x<root->val){
            root=root->left;
        } else {
            floor = root->val;
            root = root->right;
        }
    }
    return floor;
}


// insert a node in bst
TreeNode* insertInBst(TreeNode* root, int key){
    if(root==NULL){
        return new TreeNode(val);
    }
    while(root!=NULL){
        if(key>root->val){
            if(root->right!=NULL){
                root=root->right;
            } else {
                root->right = new TreeNode(val);
                break;
            }
        } else {
            if(root->left!=NULL){
                root=root->left;
            } else {
                root->left = new TreeNode(val);
                break;
            }
        }
    }
    return root;
}