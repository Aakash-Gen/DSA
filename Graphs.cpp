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




// bfs transversal
vector<int> bfsOfGraph(int V, vector<int> adj[]){
    queue<int> q;
    vector<int> vis(V,0);
    vis[0] =1;
    q.push(0);
    vector<int> bfs;
    while(!q.empty()){
        int temp = q.front();
        q.pop();
        bfs.push_back(temp);
        for(auto it:adj[temp]){
            if(vis[it]==0){
                q.push(it);
                vis[it]=1;
            }
        }
    }
    return bfs;
}


// dfs traversal
void dfsSolve(vector<int> adj[],vector<int> &vis,vector<int> &dfs, int start){
    vis[start] =1;
    dfs.push_back(start);
    for(auto it: adj[start]){
        if(vis[it]==0){
            dfsSolve(adj,vis,dfs,it);
        }
    }
}
vector<int> dfsOfGraph(int V, vector<int> adj[]){
    vector<int> vis(v,0);
    vector<int> dfs;
    int start =0;
    return dfsSolve(adj,vis,dfs,start);
    return dfs;
}


// no. of provinces - count the no. of provices lets say a and b are connected so they are onr province
void dfs(vector<int> &vis, vector<int> adj[], int start){
    vis[start] =1;
    for(auto it: adj[start]){
        if(vis[it]==0){
            dfs(vis,adj,it);
        }
    }
}
int countProvinces(vector<vector<int>> &mat){
    int V = mat.size();
    vector<int> adj[V];
    for(int i=0;i<V;i++){
        for(int j=0;j<V;j++){
            if(mat[i][j]==1){
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }
    
    vector<int> vis(V,0);
    int start =0;
    int count =0;
    for(int i=0;i<V;i++){
        if(vis[i]==0){
            count++;
            dfs(vis,adj,start);
        }
    }
    return count;
}



// there is grid of oranges where 2 means rotten, 1 means fresh and 0 mean empty, give total minutes where all fresh oranges becomes rotten when connected 4 directionally with a rotten orange
int rottenOranges(vector<vector<int>> &grid){
    int m = grid.size();
    int n = grid[0].size();
    queue<pair<int,int>> q;
    vector<vector<int>> vis(m,vector<int> (n,0));
    int tot =0;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(grid[i][j]==2){
                vis[i][j]=1;
                q.push({i,j});
            }
            if(grid[i][j]!=0){
                tot++;
            }
        }
    }

    int cnt =0;
    int ans =0;
    vector<pair<int,int>> dir = {{1,0},{0,1},{-1,0},{0,-1}};
    while(!q.empty()){
        int size = q.size();
        cnt+= size;
        while(size--){
            auto temp = q.front();
            q.pop();
            int row = temp.first;
            int col = temp.second;
            for(int i=0;i<4;i++){
                int nx = row+ dir[i].first;
                int ny = col + dir[i].second;
                if(nx>=0 && ny>=0 && nx<m && ny<n && vis[nx][ny]==0 && grid[nx][ny]==1){
                    vis[nx][ny] = 1;
                    q.push({nx,ny});
                }
            }
        }
        if(!q.empty()) ans++;
    }
    return cnt==tot? ans: -1;
}



// flood fill - here there is an image matrix and we are given initial position and its color and we need to change all the cells with that color connected repeteadly 4 directionally to the new color and return the new image
void dfs(vector<vector<int>> &image, int iniColor, int color, vector<vector<int>> &vis, int sr, int sc){
    int m = image.size();
    int n = image[0].size();
    vector<pair<int,int>> dir = {{1,0},{0,1},{-1,0},{0,-1}};
    for(int i=0;i<4;i++){
        int row = sr + dir[i].first;
        int col = sc + dir[i].second;
        if(row>=0 && col>=0 && row<m && col<n && vis[row][col]==0 && image[row][col]==iniColor){
            image[row][col] =color;
            vis[row][col] =1;
            dfs(image,iniColor, color, vis,row,col);
        }
    }
}
vector<vector<int>> floodFill(vector<vector<int>> &image, int sr, int sc, int color){
    int m = image.size();
    int n = image[0].size();
    vector<vector<int>> vis(m,vector<int> (n,0));
    int iniColor = image[sr][sc];
    vis[sr][sc]=1;
    image[sr][sc] = color;
    dfs(image,iniColor,color,vis,sr,sc);
    return image;
}




// surrounding regions
void dfs(vector<vector<char>> &board, vector<vector<int>> &vis, int row, int col){
    int m = board.size();
    int n = board[0].size();
    vis[row][col]=1;
    vector<pair<int,int>> dir = {{1,0},{0,1},{-1,0},{0,-1}};
    for(int i=0;i<4;i++){
        int nx = row + dir[i].first;
        int ny = col + dir[i].second;
        if(nx>=0 && ny>=0 && nx<m && ny<n && vis[nx][ny]==0 && board[nx][ny]=='O'){
            dfs(board,vis,nx,ny);
        }
    }
}
vector<vector<char>> surroundedRegions(vector<vector<char>> &board){
    int m = board.size();
    int n = board[0].size();
    vector<vector<int>> vis(m,vector<int> (n,0));
    for(int i=0;i<n;i++){
        if(board[0][i]=='O' && vis[0][i]==0){
            dfs(board,vis,0,i);
        }
        if(board[m-1][i]=='O' && vis[m-1][i]==0){
            dfs(board,vis,m-1,i);
        }
    }
    for(int i=0;i<m;i++){
        if(board[i][0]=='O' && vis[i][0]==0){
            dfs(board,vis,i,0);
        }
        if(board[i][n-1]=='O' && vis[i][n-1]==0){
            dfs(board,vis,i,n-1);
        }
    }
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(vis[i][j]!=1){
                board[i][j]='X';
            }
        }
    }
    return board;
}


// wordladder - we are given a starting word , ending word and wordlist , we need to find min no. of steps to reach endWord such that we can change the startword into one of the words from the wordlist and it is considered one step
int wordLadder(string beginWord, string endWord, vector<string> &wordList){
    unordered_set<string> st(wordList.begin(),wordList.end());
    queue<pair<string,int>> q;
    st.erase(beginWord);
    q.push({beginWord,1});
    while(!q.empty()){
        auto temp = q.front();
        q.pop();
        string s = temp.first;
        int steps = temp.second;
        if(s==endWord) return steps;
        for(int i=0;i<s.size();i++){
            char original = s[i];
            for(char ch='a';ch<='z';ch++){
                s[i] = ch;
                if(st.find(s)!=st.end()){
                    st.erase(s);
                    q.push({s,steps+1});
                }
            }
            s[i] = original;
        }
    }
    return 0;
}


// bipartite graph - the graph is bipartite if the nodes of a graph can be coloured in a way such that no 2 adjacent nodes have same colour
bool solve(vector<vector<int>> &graph, vector<int> &color, int col, int start){
    color[start] = col;
    for(auto it: graph[start]){
        if(color[it]==-1){
            if(dfs(graph,color,!col,it)==false) return false;
        } else if ( color[it]==col) return false;
    }
    return true;
}
bool isBipartite(vector<vector<int>> &graph){
    int V = graph.size();
    vector<int> color(V,-1);
    for(int i=0;i<V;i++){
        if(color[i]==-1){
            if(dfs(graph,color,0,i)==false) return false;
        }
    }
    return truel
}



// topo sort - it is used for a directed acyclic graph only and it says that if there is a edge between u and v directed towards v, then u comes before v in the sorting
void dfs(vector<int> adj[], stack<int> &st, vector<int> &vis, int node){
    vis[node]=1;
    for(auto it: adj[node]){
        if(!vis[it]){
            dfs(adj,st,vis,it);
        }
    }
    st.push(node);
}
vector<int> topoSort(int V, vector<int> adj[]){
    stack<int> st;
    vector<int> vis(V,0)
    for(int i=0;i<V;i++){
        if(!vis[i]){
            dfs(adj,st,vis,i);
        }
    }
    vector<int> topo(V,0);
    while(!st.empty()){
        int temp = st.top();
        st.pop();
        topo.push_back(temp);
    }
    return topo;
}



// it is just like topo sort and is used for DAG graph but here we use bfs instead of dfs and use the concept of indegree which is the no. of edges coming inside a node 
vector<int> kahnAlgo(int V, vector<int> adj[]){
    vector<int> indegree(V,0);
    for(int i=0;i<V;i++){
        for(auto it: adj[i]){
            indegree[it]++;
        }
    }

    queue<int> q;
    for(int i=0;i<V;i++){
        if(indegree[i]==0){
            q.push(i);
        }
    }

    vector<int> topo;
    while(!q.empty()){
        int temp = q.front();
        q.pop();
        topo.push_back(temp);
        for(auto it: adj[temp]){
            indegree[it]--;
            if(indegree==0){
                q.push(it);
            }
        }
    }   
    return topo;
}



// course schedule 1 - here we are given prerequisites array in the form of [ai,bi] where we need to take bi first to take , so we need to tell if we can complete all courses or not
// we can use kahn's algorithm as if there is cycle present we wont be able to do all courses

// course schedule 2 - here instead of doing the count push the elements in an array just like kahn's algo and return the ordering
bool courseSchedule(int numCourses, vector<vector<int>> &prerequisites){
    int V = numCourses;
    vector<int> indegree(V,0);
    vector<int> adj[V];
    for(auto it: prerequisites){
        adj[it[1]].push_back(it[0]);
    }

    for(int i=0;i<V;i++){
        for(auto it: adj[i]){
            indegree[it]++;
        }
    }
    queue<int> q;
    for(int i=0;i<V;i++){
        if(indegree[i]==0){
            q.push(i);
        }
    }
    int cnt =0;
    while(!q.empty()){
        int temp = q.front();
        q.pop();
        cnt++;
        for(auto it: adj[temp]){
            indegre[it]--;
            if(indegree[it]==0){
                q.push(it);
            }
        }
    }

    return cnt==V? true: false;
}



// all paths from source to target - we are given a DAG and we need to find all paths from 0 to n-1
// In DAG we generally use toposort but only when we need to find one way , here as we need all possible ways , we will use dfs to find it
// as here if you notice i am passing vis and temp not as reference , i dont need to temp.pop_back() when backtrack , if i pass it by reference then i would need to do that
void dfs(vector<int> adj[], vector<int> vis, vector<vector<int>> &ans, vector<int> temp, int start){
    int n = vis.size();
    vis[start] = 1;
    temp.push_back(start);
    if(start ==n-1){
        ans.push_back(temp);
    }
    for(auto it: adj[start]){
        if(!vis[it]){
            dfs(adj,vis,ans,temp,it);
        }
    }
}
vector<vector<int>> allPathsFromSourceToTarget(vector<vector<int>> &graph){
    int V = graph.size();
    vector<int> adj[V];
    for(int i=0;i<V;i++){
        for(auto it: graph[i]){
            adj[i].push_back(it);
        }
    }
    vector<int> vis(V,0);
    vector<vector<int>> ans;
    vector<int> temp;
    dfs(adj,vis,ans,temp,0);
    return ans;
}


// to check if a directed graph is cyclic or not - we can use kahn's algo even tho its for acyclic, when we form the resultant ans , if its size is equal to total vertices then its acyclic otherwise not
bool isDirectedGraphCyclic(int V, vector<int> adj[]){
    vector<int> indegree(V,0);
    for(int i=0;i<V;i++){
        for(auto it: adj[i]){
            indegree[it]++;
        }
    }
    queue<int> q;
    for(int i=0;i<V;i++){
        if(indegree[i]==0){
            q.push(i);
        }
    }
    vector<int> ans;
    while(!q.empty()){
        int temp = q.front();
        q.pop();
        ans.push_back(temp);
        for(auto it: adj[temp]){
            indegree[it]--;
            if(indegree[it]==0){
                q.push(it);
            }
        }
    }

    return ans.size()==V? false: true;
}



// here we need to find all safe nodes 
// terminal nodes - which have no outgoing edges
// safe nodes - if every possible path from this node ends at a terminal node
vector<int> findallSafeStates(vector<vector<int>> &graph){
    int V = graph.size();
    vector<int> adj[V];
    vector<int> indegree(V,0);  
    for(int i=0;i<V;i++){    // here i have reversed the adj list as now when we find nodes with indegree 0 , those will be the terminal nodes
        for(auto it: graph[i]){
            adj[it].push_back(i);
            indegree[i]++;
        }
    }
    queue<int> q;
    for(int i=0;i<V;i++){
        if(indegree[i]==0){
            q.push(i);
        }
    }
    vector<int> ans;
    while(!q.empty()){
        int temp = q.front();
        q.pop();
        ans.push_back(temp);
        for(auto it: adj[temp]){
            indegree[it]--;
            if(indegree[it]==0){
                q.push(it);
            }
        }
    }
    sort(ans.begin(),ans.end());
    return ans;
}



// shortest path in undirected graph - we need to find the min distance from a src node to all other nodes
vector<int> shortestPathUndirected(vector<vector<int>> &edges, int n, int m, int src){
    vector<int> adj[n];
    for(auto it: edges){
        adj[it[0]].push_back(it[1]);
        adj[it[1]].push_back(it[0]);
    }
    queue<int> q;
    q.push(src);
    vector<int> dist(n,1e9);
    dist[src] =0;
    while(!q.empty()){
        int temp = q.front();
        q.pop();
        for(auto it: adj[temp]){
            if(dist[temp]+1<dist[it]){
                dist[it] = 1 + dist[temp];
                q.push(it);
            }
        }
    }
    for(int i=0;i<n;i++){
        if(dist[i]==1e9){
            dist[i] = -1;
        }
    }
    return dist;
}



// shortest path in DAG - find shortest path from src to all other nodes 
// we use topo sort here as it is a DAG
vector<int> shortestPathDag(vector<vector<int>> &edges, int N, int M){
    vector<pair<int,int>> adj[N];
    vector<int> indegree(N,0);
    for(auto it: edges){
        adj[it[0]].push_back({it[1],it[2]});
        indegree[it[1]]++;
    }
    queue<int> q;
    vector<int> dist(N,1e9);
    
    for(int i=0;i<N;i++){
        if(indegree[i]==0){
            q.push(i);
        }
    }
    dist[0] = 0;
    while(!q.empty()){
        int node = q.front();
        q.pop();
        for(auto it: adj[node]){
            int nextNode = it.first;
            int wt = it.second;
            if(dist[node]+wt< dist[nextNode]){
                dist[nextNode] = dist[node] + wt;
            }
            indegree[nextNode]--;
            if(indegree[nextNode]==0){
                q.push(nextNode);
            }
        }
    }
    for(int i=0;i<N;i++){
        if(dist[i]==1e9){
            dist[i]=-1;
        }
    }
    return dist;
}



// dijkstra algo - here we are adjacency list where ith element of the list contains node which it is connected to and their weight
// it is used both for directed and undirected graph
// it doesnt work for negative weights and negative cycle
vector<int> dijkstra(int V, vector<vector<int>> &adj[], int s){
    vector<int> dist(V,1e9);
    set<pair<int,int>> st;
    st.insert({0,s});
    dist[s] =0;
    while(!st.empty()){
        auto temp = *(st.begin());
        int wt = temp.first;
        int node = temp.second;
        st.erase(temp);
        for(auto it: adj[node]){
            int nextNode = it[0];
            int nextWt = it[1];
            if(dist[node]+nextWt< dist[nextNode]){
                if(dist[nextNode]!=1e9){
                    st.erase({dist[nextNode],nextNode});
                }
                dist[nextNode] = dist[node] + nextWt;
                st.insert({dist[nextNode],nextNode});
            }
        }
    }
    return dist;
}



// bellman ford algo - it is used like dikstra to find shortest path from one src to all other nodes
// this can be used for negative weights as well and can be used to detect negative cycle
// it is generally used for directed graph but to use it for undirected as well we need to consider an undirected path as 2 directed paths
vector<int> bellmanFord(int V, vector<vector<int>> &edges, int S){
    vector<int> dist(V,1e9);
    dist[S] = 0;
    for(int i=0;i<V-1;i++){
        for(auto it: edges){
            int u = it[0];
            int v = it[1];
            int wt = it[2];
            if(dist[u]!=1e9 && dist[u]+wt<dist[v]){
                dist[v] = dist[u] + wt;
            }
        }
    }
    for(auto it: edges){    // if we find a shorter distance here it means it is a negative cycle as if it was not we would have find shortest distances in above iterations
        int u = it[0];
        int v = it[1];
        int wt = it[2];
        if(dist[u]!=1e9 && dist[u]+wt<dist[v]){
            return {-1};
        }
    }
    return dist;
}



// floyd warshal - it used to find distances between every pair of vertices , the answer is stored in the form of matrix
// it can detect negative cycle as well
vector<vector<int>> floydWarshal(vector<vector<int>> &matrix){
    int n = matrix.size();
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(matrix[i][j]==-1) matrix[i][j] = 1e9;
        }
    }

    for(int k=0;k<n;k++){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j]);
            }
        }
    }

    bool hasNegativeCycle = false;
    for(int i=0;i<n;i++){
        if(matrix[i][i]<0){
            hasNegativeCycle = true;
            break;
        }
    }

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(matrix[i][j]==1e9) matrix[i][j] = -1;
        }
    }
    return matrix;
}



// shortest path in binary matrix - we need to find shortest path from top left to bottom right if we can move 8 directionally and path is calculated by no. of cells we visit and we can only move along 0's
// here if you note we have unweighted edges so no need to use dijkstra and no need to use set or priority queue
int shortestPathBinaryMatrix(vector<vector<int>> &grid){
    int m = grid.size();
    int n = grid[0].size();
    queue<pair<int,int>> q;
    q.push({0,0});
    if(grid[0][0]==1 || grid[m-1][n-1]==1) return -1;
    vector<vector<int>> dist(m,vector<int> (n,1e9));
    vector<pair<int,int>> dir = {{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,-1},{1,-1},{-1,1}};
    dist[0][0] = 1;
    while(!q.empty()){
        auto temp = q.front();
        q.pop();
        int row = temp.first;
        int col = temp.second;
        for(int i=0;i<8;i++){
            int x = row + dir[i].first;
            int y = col + dir[i].second;
            if(x>=0 && y>=0 && x<m && y<n && grid[x][y]==0){
                if(dist[row][col]+1<dist[x][y]){
                    dist[x][y] = 1 + dist[row][col];
                    q.push({x,y});
                }
            }
        }
    }
    if(dist[m-1][n-1]==1e9) return -1;
    return dist[m-1][n-1];
}



// path with minimum effort - here we are given 2d array of heights where each cell represents height if that cell and we can move 4 directionally . we need to find minimum path effort
// path effort - it is max absolute difference in heights of consecutive cells
// using set
int minimumEffortPath(vector<vector<int>> &heights){
    int m = heights.size();
    int n = heights[0].size();
    set<pair<int,int>> st;
    vector<vector<int>> dist(m,vector<int>(n,1e9));
    dist[0][0] = 0;
    st.insert({0,0});
    vector<pair<int,int>> dir = {{1,0},{0,1},{-1,0},{0,-1}};
    while(!st.empty()){
        auto temp = *(st.begin());
        st.erase(temp);
        int row = temp.first;
        int col = temp.second;
        for(int i=0;i<4;i++){
            int x = row + dir[i].first;
            int y = col + dir[i].second;
            if(x>=0 && y>=0 && x<m && y<n){
                int newEffort = max(dist[row][col],abs(heights[row][col]-heights[x][y]));
                if(newEffort< dist[x][y]){
                    dist[x][y] = newEffort;
                    st.insert({x,y});
                }
            }
        }
    }
    return dist[m-1][n-1];
}
// using priority queue - in priority queue i need to insert effort also in pq so as to get the right order
int minimumEffortPath(vector<vector<int>>& heights) {
    int m = heights.size();
    int n = heights[0].size();
    priority_queue<pair<int,pair<int,int>>,vector<pair<int,pair<int,int>>>,greater<pair<int,pair<int,int>>>> pq;
    vector<vector<int>> dist(m,vector<int>(n,1e9));
    dist[0][0] =0;
    pq.push({0,{0,0}});
    vector<pair<int,int>> dir = {{0,1},{1,0},{-1,0},{0,-1}};
    while(!pq.empty()){
        auto it = pq.top();
        int diff = it.first;
        int row = it.second.first;
        int col = it.second.second;
        pq.pop();
        for(auto it:dir){
            int x = row + it.first;
            int y = col + it.second;
            if(x>=0 && y>=0 && x<m && y<n){
                int newEffort = max(abs(heights[row][col]-heights[x][y]),diff);
                if(newEffort<dist[x][y]){
                    dist[x][y] = newEffort;
                    pq.push({dist[x][y],{x,y}});
                }
            }
        }
    }
    return dist[m-1][n-1];
}