# project-ml-intern-5
系统学习 树模型（决策树 → 随机森林 → GBDT/LightGBM） 的原理与实践，最后整理成完整的 baseline 实验。
# Week 5: Tree Models Learning & Practice

## 目标
系统学习并实践 **树模型家族**（Decision Tree、Random Forest、GBDT/LightGBM），理解理论基础、掌握模型训练与评估方法，并完成 baseline 实验。

---

## 每日学习安排

### 周一：决策树理论 + 阅读笔记
**任务操作：**
1. 新建笔记 `week5_day1_notes.md`，写上今日目标：理解信息增益和基尼系数。
2. 阅读 sklearn 文档关于 DecisionTreeClassifier 的章节，重点理解“分裂准则”。
3. 将文档中分裂准则摘录成 3 条要点写入笔记。
4. 查找一篇短博客（“信息增益 vs Gini”），用一个小表格做手算示例，把计算过程写入笔记。

**产出：**
- `week5_day1_notes.md`（含 3 个要点 + 手算示例）

---

### 周二：决策树实践
**任务操作：**
1. 打开 `week5_trees_baseline.ipynb` 并加载数据。
2. 用 sklearn 训练一棵决策树模型。
3. 绘制树结构（或使用文本输出查看树的分裂节点）。
4. 在 notebook 中记录 3 条观察：
   - 哪个特征最先分裂？
   - 分裂阈值大致是多少？
   - 子树的分裂结构情况。

**产出：**
- notebook 中的树图或文本树结构
- 3 条观察结论

---

### 周三：随机森林理论 + 实践
**任务操作：**
1. 写短笔记：解释 bagging 的作用，以及为什么能降低模型方差（2–3 句）。
2. 在 notebook 中训练随机森林模型。
3. 使用 5-fold 交叉验证，记录平均分。
4. 保存训练好的模型文件（以便后续加载使用）。

**产出：**
- CV 分数记录（CSV）
- 保存的模型文件

---

### 周四：GBDT 理论入门
**任务操作：**
1. 在笔记中写 GBDT 基本思想，包括：
   - 残差拟合
   - 逐步加模型
   - 学习率作用
2. 安装 LightGBM（本地或 Colab）。
3. 阅读 LightGBM 快速开始文档。
4. 在 notebook 中写出 LightGBM 默认训练流程示例（仅记录操作步骤即可）。

**产出：**
- `week5_day4_notes.md`
- LightGBM 训练示例流程记录

---

### 周五：LightGBM baseline 训练
**任务操作：**
1. 用 LightGBM 的 sklearn 接口训练 baseline 模型。
2. 使用 5-fold 交叉验证记录每折分数。
3. 与随机森林 CV score 做对比，并写一段简短结论：
   - 哪个模型表现更好
   - 可能原因
4. 将训练日志保存到 `experiments/baseline_scores.csv`（包含模型名、折数、分数）。

**产出：**
- `baseline_scores.csv`
- notebook 更新

---

### 周六：整理 + 提交
**任务操作：**
1. 整理本周代码和文件夹，形成统一目录结构：
