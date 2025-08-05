def maximize_min_value_knapsack(C, val, wt):
    n = len(val)
    # 初始化动态规划表格
    dp = [[0 for _ in range(C + 1)] for _ in range(n + 1)]
    # 存储每个容量下选择的物品索引
    selection = [[() for _ in range(C + 1)] for _ in range(n + 1)]

    # 动态规划填表
    for i in range(1, n + 1):
        for j in range(1, C + 1):
            # 不携带当前物品
            dp[i][j] = dp[i - 1][j]
            selection[i][j] = selection[i - 1][j]

            # 尝试携带当前物品，如果不超过容量
            if j >= wt[i - 1]:
                # 如果携带当前物品能得到更大的价值
                if dp[i - 1][j - wt[i - 1]] + val[i - 1] > dp[i][j]:
                    dp[i][j] = dp[i - 1][j - wt[i - 1]] + val[i - 1]
                    # 记录选择的物品索引
                    selection[i][j] = selection[i - 1][j - wt[i - 1]] + (i - 1,)

    # 回溯找到选择的物品
    chosen_items = []
    current_capacity = C
    for i in range(n, 0, -1):
        if dp[i][current_capacity] != dp[i - 1][current_capacity]:
            chosen_items.append(i - 1)
            current_capacity -= wt[i - 1]

    # 返回最大价值和选择的物品索引
    return dp[n][C], chosen_items[::-1]  # 逆序，因为我们是从后向前回溯的

# 示例
C = 10  # 背包容量
val = [6, 5, 3, 2, 7]  # 物品价值数组
wt = [3, 2, 1, 1, 4]  # 物品重量数组

max_value, chosen_items = maximize_min_value_knapsack(C, val, wt)
print("Maximum Value:", max_value)
print("Chosen Items Indexes:", chosen_items)