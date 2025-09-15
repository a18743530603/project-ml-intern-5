# std::vector

```c++
用于表示动态数组。
#include <vector>

std::vector<int> v;        // 空的整型向量
std::vector<double> d(5);  // 初始化一个包含5个double元素的向量，初始值为0.0
std::vector<bool> b(10, true);  // 10个true的布尔向量
| 操作               | 说明                   |
| ---------------- | -------------------- |
| `push_back(val)` | 在末尾添加一个元素            |
| `pop_back()`     | 删除末尾元素               |
| `size()`         | 当前元素数量               |
| `resize(n)`      | 修改大小（扩大会填充默认值）       |
| `clear()`        | 清空所有元素               |
| `[] / at(i)`     | 下标访问第i个元素（`at`有范围检查） |
| `begin()/end()`  | 支持迭代器遍历              |
eg:
#include <iostream>
#include <vector>

int main() {
    std::vector<int> nums;

    nums.push_back(10);
    nums.push_back(20);
    nums.push_back(30);

    for (size_t i = 0; i < nums.size(); ++i) {
        std::cout << nums[i] << " ";
    }
    // 输出：10 20 30
}
| 特性    | 说明                         |
| ----- | -------------------------- |
| 动态增长  | 不需提前指定长度                   |
| 顺序存储  | 内存连续，支持快速随机访问              |
| 泛型支持  | 可以存储任意类型（`std::vector<T>`） |
| STL兼容 | 支持算法和迭代器                   |
std::vector<float> maxWeight(noCatAtts_, -std::numeric_limits<float>::max());
逐部分解释如下：

部分代码	含义
std::vector<float>	创建一个 float 类型的向量（即数组）
maxWeight(...)	向量的变量名叫 maxWeight
noCatAtts_	向量的大小，由属性数决定，即有多少个属性
-std::numeric_limits<float>::max()	给每个元素初始赋值为 最小可能的浮点数，即 -3.4e38 左右
```

# 析构函数

```c++
在对象生命周期结束时，自动释放资源，做清理工作。
~TAN(void);
当你使用一个对象（比如 TAN tanLearner; 或 TAN* ptr = new TAN(...);），这个对象最终会被销毁（作用域结束或通过 delete 手动释放）。在这个时刻，C++ 会自动调用析构函数，完成以下工作：

释放内存（比如 new 分配的内容）

关闭文件/释放句柄

解除指针引用，避免内存泄漏

清空缓存、保存状态等
    
    如果你不写析构函数，编译器会生成一个默认析构函数，会：

自动调用每个成员变量的析构函数（比如 vector 会释放自己占用的内存）

但不会释放裸指针成员，比如 instanceStream_，所以你就可能发生内存泄漏！
```

```c++

void TAN::classify(const instance &inst, std::vector<double> &classDist) {

	for (CatValue y = 0; y < noClasses_; y++) {
		classDist[y] = xxyDist_.xyCounts.p(y)* (std::numeric_limits<double>::max() / 2.0);
	}

	if(loo_==true)
    {
        for (CategoricalAttribute xIndex = 0; xIndex < optAttIndex_; xIndex++) {

            const CategoricalAttribute x1=orderedAtts_[xIndex];
            const CategoricalAttribute parent = parents_[x1];

            if (parent == NOPARENT) {
                for (CatValue y = 0; y < noClasses_; y++) {
                    classDist[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
                }
            } else {
                for (CatValue y = 0; y < noClasses_; y++) {
                    classDist[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent,
                            inst.getCatVal(parent), y);
                }
            }
        }

    }
	else

    {
        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
            const CategoricalAttribute parent = parents_[x1];

			//if(active_[x1]==false)//如果该属性在训练时被禁用（例如冗余或缺失），则跳过。
				//continue;

            if (parent == NOPARENT) {//这表示属性 x1 是树根节点，它没有父亲（除了类别变量 Y）。
                for (CatValue y = 0; y < noClasses_; y++) {
                	classDist[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
                	                }
            } else {//这表示属性 x1 有一个父属性（在 TAN 树结构中建立的）。
                for (CatValue y = 0; y < noClasses_; y++) {
                	classDist[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent,inst.getCatVal(parent), y);
                }
            }
        }
    }


	normalise(classDist);
}
```



```
总体假设与原则（必读）

假设：零基础起步，每周可投入 15–20 小时；若时间更少需延长到 8–9 个月，更多则可加速。
产出导向：每月必须有可运行产物（Colab/Notebooks + README + demo）。
复盘节奏：每周末 30–60 分钟周报；每月末 3–5 分钟 demo 录屏并存档。
工具约定：GitHub（项目托管）、固定 random_state、简单实验记录表（CSV/Notion）。
6 个月（24 周）总览（模块化分配）

月1（周1–4）：编程与数学基础（Python、numpy、pandas、线代/概率入门） + 入门 ML（线性/逻辑回归）
月2（周5–8）：经典 ML 与实战（决策树、RF、GBDT 基础、sklearn pipeline） + 入门表格项目
月3（周9–12）：模型工程化与调参（CV、Optuna、特征工程、项目 v1）
月4（周13–16）：算法工程核心（从零实现关键算法、复杂度与向量化、profile 与加速）
月5（周17–20）：部署与在线推理（FastAPI、Docker、ONNX、benchmark）、工程化 v2
月6（周21–24）：面试冲刺（算法题、系统题、mock interview）、投递与项目打磨
每月详解与每周重点（精简版）

月1（周1–4）—— 编程 + 数学 + ML 入门
目标：能读写数据、用 numpy/pandas 做基本处理，理解线性/逻辑回归原理并手写实现。
每周示例任务：

Python + numpy/pandas（4–5h/w）：数据读写、groupby、merge、缺失处理、简单可视化。
数学（3h/w）：向量/矩阵运算、标量导数、基础概率（条件概率、期望/方差）。
ML 基础（3–4h/w）：损失函数、梯度下降、交叉验证概念；手写线性/逻辑回归实现。
交付物：1 个入门 notebook（数据清洗 + 手写回归）+ 1 分钟 demo 视频。
月2（周5–8）—— 经典 ML 与实战
目标：掌握树模型与集成方法，学会用 sklearn pipeline 与 CV，完成第一个表格项目 baseline。
每周示例任务：

学理论与使用（4–5h/w）：决策树、随机森林、LightGBM/XGBoost，理解分裂准则/正则化。
Pipeline/调参（4h/w）：ColumnTransformer、K-fold、Grid/Random/Optuna 基本用法。
小项目实践（3–4h/w）：选一数据集（Kaggle 中级），搭建 baseline，做 CV 报告。
交付物：表格项目 v0（notebook + README + CV 报告 + baseline model）。
月3（周9–12）—— 深化特征工程与模型稳定性
目标：把表格项目提升到 v1（更稳健的特征、调参、解释性）。
每周示例任务：

高级特征工程（4–5h/w）：目标编码、时间特征、聚合特征、缺失模式探索。
稳定性与解释（3–4h/w）：不同 seed/split 测试、SHAP 基本使用、误差来源分析。
调参与 stacking（3–4h/w）：Optuna 进阶、简单 stacking/blend。
交付物：表格项目 v1（可复现 pipeline + model.pkl + SHAP 分析 + 2 分钟 demo）。
月4（周13–16）—— 算法实现与性能优化（核心）
目标：能从零实现或理解关键算法组件，并做工程化优化（向量化/numba/C++ 可选）。
每周示例任务：

算法实现（5–6h/w）：从零实现决策树分裂或简化 GBDT 的关键步骤，写清复杂度分析。
性能剖析（3–4h/w）：用 cProfile/line_profiler 找瓶颈，应用向量化/Numba 加速并记录 speedup。
单元测试与 benchmark（2–3h/w）：写测试、benchmark 脚本与对比报告。
交付物：算法实现项目（README + benchmark 报表 + 加速代码）。
月5（周17–20）—— 部署与在线推理
目标：把至少一个项目做成可部署服务，并能展示延迟/吞吐指标。
每周示例任务：

API 与容器（4–5h/w）：FastAPI 编写预测接口，加入输入校验与异常处理。
导出与加速（3–4h/w）：ONNX/torchscript 导出，测单次/并发延迟。
CI/监控与文档（2–3h/w）：Dockerfile、GitHub Actions 简单流水线、demo curl。
交付物：可启动的 Docker image + API demo + 延迟 benchmark 报表 + README。
月6（周21–24）—— 面试冲刺与投递
目标：系统化准备算法题、ML 算法问答与系统设计，完成简历与项目打磨并广撒投递/内推。
每周示例任务：

算法刷题（6–8h/w）：集中做中等题并练习白板表达（数组/树/DP/二分/贪心）。
ML 算法 & 系统题（3–4h/w）：GBDT 原理、损失/梯度推导、在线预测系统设计题。
Mock & 简历（2–3h/w）：每周至少 1 次 mock（coding 或 ML 面试），完善简历与项目页。
交付物：简历 + 2 个 polished projects + demo 视频 + 若干 mock 记录。
每周与每月复盘模板（必须坚持）

每周复盘（30–60 分钟）：本周完成、未完成、最重要的 3 个卡点、下周具体计划（任务与时间）。
每月 demo（3–5 分钟录屏）：展示项目进展、关键指标、下月目标与遇到的最大问题。
错题本与卡点库：所有算法题与实现 bug 都要记录并分类复习。
量化 KPIs（6 个月目标）

项目：至少 2 个高质量项目（1 个表格工程化 + 1 个算法实现或部署 demo）。
算法题：目标 ~150–200 题（中等为主）；若时间紧张至少保证每周 8–10 题。
Mock：至少 12 次 mock（coding + ML/system design + behavior）。
部署：至少 1 个项目能在 20 分钟内由他人按 README 启动并调用接口。
推荐资源（速查）

入门：Python + pandas 官方教程、Kaggle 入门课程。
ML：Hands-On ML (Géron)、scikit-learn 文档。
算法刷题：LeetCode（中等题集）、《算法导论》节选（选读）。
部署/工程：FastAPI 文档、Docker 入门、ONNX 教程。
性能：Numba 文档、cProfile/line_profiler 使用教程。
```

总体说明（必读）

- 每周计划总时长：约15–16小时（周一–周六每天2.5h，周日复盘1h）。如果你想把工作日压缩或分散告诉我，我可以调整。
- 输出导向：每周必须在 GitHub 建立或更新对应 notebook（Colab-ready），并把关键结论写进 README。第4周末产出 1 个短 demo（3分钟以内）。
- 工具准备（第1天完成）：GitHub 账号、Colab/GDrive、Python 环境（推荐 Anaconda 或直接 Colab）、建议安装 pandas/numpy/matplotlib/seaborn/scikit-learn。

第1周（目标：搭建环境、巩固 Python 与 Numpy 基础）

- 周一（2.5h）
  - 环境与工具：注册 GitHub，创建仓库（project-ml-intern-1）；打开 Colab 并新建 notebook（命名 week1_py_numpy）。
  - 学习：Python 基础速览（变量、list/tuple/dict/set、函数、列表推导）。资源：官方 Python tutorial 或 Automate the Boring Stuff 前几章。
  - 小练习：写 8 个小函数（反转字符串、统计词频、两个列表取交集、判断回文等）。
- 周二（2.5h）
  - 学习：numpy 基础（ndarray、广播、向量化操作、索引与切片、常用函数）。资料：numpy quickstart 或相关官方教程。
  - 练习：实现向量化版的均值和标准差计算；比较 numpy 与纯 Python 循环时间。
- 周三（2.5h）
  - 学习：pandas 基础（Series/DataFrame 创建、读取 CSV、head/tail、info、describe、索引/筛选）。
  - 实践：读取一个小数据集（例如 titanic.csv），完成简单的 info/describe、缺失值统计、head/tail、value_counts。将操作放入 week1_py_numpy notebook。
- 周四（2.5h）
  - 学习：pandas 数据清洗基本操作（dropna、fillna、astype、merge/join、groupby 简单聚合）。
  - 练习：在 Titanic 数据上分别做年龄缺失填充策略实验（均值、中位数、用“Unknown”类别），记录结果与代码。
- 周五（2.5h）
  - 学习：数据可视化基础（matplotlib/seaborn：hist、boxplot、scatter、countplot）。
  - 实践：在 Titanic 上画几个图（年龄分布、票价分布、不同舱位生存率对比），在 notebook 中写简短解释。
- 周六（2.5h）
  - 综合练习：把本周所有代码整理成一个清晰的 Colab notebook（week1_py_numpy），加入“目的/数据来源/运行指令”。
  - Git 提交：push 到 GitHub（初始 commit、README 简短说明如何运行）。
- 周日（1h）—— 周复盘
  - 写周报（完成项、未完成、最主要卡点 3 条、下周计划），上传到仓库或 Notion。

第2周（目标：强化 Pandas、做完整 EDA、学习基础统计与分组特征）

- 周一（2.5h）
  - 学习：统计基础（均值、中位数、方差、协方差、相关系数、置信区间的概念）。资源：Khan Academy 概念，或《统计学入门》速读。
  - 练习：在 Titanic 数据上计算不同列之间的相关性矩阵，并可视化热力图。
- 周二（2.5h）
  - 学习：更复杂的 pandas 操作（透视表 pivot_table、时间序列基础 if 有、apply/lambda 优化注意点）。
  - 练习：做按舱位和性别分组的生存率透视表，并观察差异。
- 周三（2.5h）
  - 学习：特征工程入门（类别编码基本方法：one-hot、label encoding、target encoding 概念）。
  - 练习：在 Titanic 上尝试两种编码方案并比较简单模型表现（占位，不做深入调参）。
- 周四（2.5h）
  - 学习：机器学习入门概念（监督/无监督、训练/验证/测试、过拟合/欠拟合）。资源：吴恩达机器学习课程入门章节或 Géron 前几章。
  - 实践：用 scikit-learn 做一个简单的 LogisticRegression（train/test split），计算 accuracy、precision、recall。
- 周五（2.5h）
  - 学习：交叉验证基本概念（K-fold）与 sklearn 的 cross_val_score 用法。
  - 练习：用 K-fold 对 Titanic 做 5-fold CV，把结果记录在 notebook。
- 周六（2.5h）
  - 集成任务：将本周结果整理到一个名为 week2_eda_feature notebook，写清楚每个实验结论。
  - 将 notebook 与数据预处理脚本放入 GitHub，更新 README。
- 周日（1h）复盘
  - 写周报，记录 3 个要复习的点（例如 groupby 性能、交叉验证思路、编码方法）。

第3周（目标：理解并手写线性回归与逻辑回归，学习损失与优化）

- 周一（2.5h）
  - 学习：线性回归理论（最小二乘、MSE、解析解与梯度下降的差异）。
  - 实践：用 numpy 从零实现简单线性回归（batch gradient descent），并在 synthetic data 上测试。
- 周二（2.5h）
  - 学习：多元线性回归、正则化（Ridge/Lasso）基本概念及其作用。
  - 实践：实现带 L2 正则化的梯度下降并与 sklearn.linear_model.Ridge 比较系数与误差。
- 周三（2.5h）
  - 学习：逻辑回归（sigmoid、交叉熵 loss、概率输出、LR 与线性回归区别）。
  - 实践：用 numpy 从零实现 logistic regression（可用 batch gradient descent），在 Titanic 或小数据上观察输出概率。
- 周四（2.5h）
  - 学习：优化技巧（学习率调节、batch/minibatch、标准化/归一化的重要性、数值稳定性技巧如 log-sum-exp）。
  - 练习：在手写模型上测试不同学习率与归一化策略的训练曲线。
- 周五（2.5h）
  - 学习：模型评估扩展（ROC/AUC、PR 曲线、混淆矩阵），以及不平衡数据处理基本方法。
  - 实践：绘制 Logistic 回归在不同阈值的 ROC 曲线并计算 AUC。
- 周六（2.5h）
  - 整合任务：把手写线性/逻辑回归放入 week3_lr notebook，加入可复现的训练示例与 plot（loss vs epoch）。
  - push 到 GitHub，更新 README，注明如何运行（Colab 链接）。
- 周日（1h）复盘
  - 总结手写模型遇到的 3 个数值/收敛问题与解决方式，写入错题本。

第4周（目标：端到端小项目：数据清洗→特征→模型→评估→README + 3分钟 demo）

- 周一（2.5h）
  - 任务：选定第一个端到端小项目数据集（建议：Titanic 或其他 Kaggle 入门数据集）。明确问题定义与评估 metric。
  - 开始：建立项目 notebook skeleton（sections: Summary, Data, EDA, Preprocessing, Modeling, Eval, Conclusion）。
- 周二（2.5h）
  - 数据清洗与特征工程（执行 week1/2 学到的技术：缺失处理、类别编码、engineered features）。记录实验对比。
- 周三（2.5h）
  - 模型训练：用 sklearn 训练 LogisticRegression、RandomForestBaseline、以及简单的 LightGBM（如可用）。做 5-fold CV，记录 metric（accuracy/AUC 或你选的 metric）。
- 周四（2.5h）
  - 模型优化：尝试 basic hyperparameter tuning（GridSearch 或 RandomizedSearch, 或调整学习率/树深度），并记录改进幅度。加入 model persistence（joblib.dump）。
- 周五（2.5h）
  - 结果解释与可视化：绘制 feature importance（树模型）或 SHAP（如果时间允许），写结论段落（为什么某些 feature 有用）。
  - 准备 demo 脚本（2–3 分钟要点：问题、你的方法、主要结果、可改进点）。
- 周六（2.5h）
  - 打包 deliverable：检查 notebook 可直接运行（Colab），写 README（如何运行、依赖、关键结论），录制 2–3 分钟演示视频（用屏幕录制），把所有文件 push。
- 周日（1h）月复盘（重点）
  - 做 3–5 分钟的“demo 回顾”：把 demo 存到仓库与一个短的反思笔记（本月学到的最重要三点、下月改进目标）。
  - 评估是否达到第1月验收标准（可运行 notebook、README、demo）。

第1个月交付物（必须完成）

- GitHub 仓库初始化并包含：week1_py_numpy.ipynb、week2_eda_feature.ipynb、week3_lr.ipynb、week4_project.ipynb、README（运行说明）。
- 至少一个端到端项目（week4）可在 Colab 上运行，含 model artifact（joblib/pkl）与 2–3 分钟 demo 视频。
- 周/错题本（简单 markdown 文件，记录关键错误与收获）。

Notebook 模板（每个 notebook 都按此结构）

- 标题 + 1 段 Summary（目标、数据来源、关键结论）
- 运行环境（Python 版本、依赖包） & 如何运行（Colab/GDrive 授权说明）
- 数据加载与快速预览（head/info/describe）
- EDA（关键图表与结论）
- 预处理（每一步为什么这么做）
- 模型训练（含 cross-validation）
- 评估（metric、混淆矩阵、ROC/PR）
- 结论与下一步（潜在改进点）
- 附录（重要函数/benchmark 脚本/引用资源）

几点实践建议（提高效率）

- 每次做实验都设固定 random_state 并记录 seeds；用小数据子集做调试。
- 把长期 task（如学习资料、课）用书签或 Notion 管理，避免重复搜索。
- 每日结束前花 5–10 分钟写当天笔记（what I did, blocked, next）。
- 遇到卡住的问题先记录并花不超过 45 分钟深挖，超时转到别的任务，避免陷入细节。

```
（Trees 基础与 baseline）

周一（决策树理论 + 阅读笔记）
打开一个新笔记 week5_day1_notes.md，写上今日目标：理解信息增益/基尼。
阅读 sklearn 决策树文档对应章节（链接：scikit-learn DecisionTreeClassifier），并把“分裂准则”摘录成 3 条要点写进笔记。
用 20–30 分钟查一篇短博客（“信息增益 vs gini”），把 1 个示例（小表格）手算一次，并把计算过程写入笔记。
产出：week5_day1_notes.md（含 3 个要点 + 一个手算例子）。
周二（在数据上实践 DecisionTree）
打开 week5_trees_baseline.ipynb，新建 cell：加载数据（示例代码： df = pd.read_csv(‘data.csv’)）。
用 sklearn 建模：from sklearn.tree import DecisionTreeClassifier; model = DecisionTreeClassifier(max_depth=3, random_state=42); model.fit(X_train, y_train)。
用 sklearn.tree.plot_tree(model, max_depth=2) 绘制并保存图片（或 export_text 输出树结构）。
在 notebook 写 3 条观察（例如：哪个特征最先分裂？分裂阈值大致是多少？）。
产出：notebook 中的树图和 3 条观察结论。
周三（RandomForest 原理 + 实践）
写短笔记：bagging 的作用、为何能降方差（2–3 句）。
在 notebook 中训练随机森林：from sklearn.ensemble import RandomForestClassifier; rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1); cross_val_score(rf, X, y, cv=5).mean()。记录 5-fold 的平均分。
保存模型： joblib.dump(rf, ‘models/rf_baseline.pkl’)。
产出：CV 得分记录（CSV）和模型文件。
周四（GBDT 理论入门）
在 notes 写“GBDT 基本思想”一段（残差拟合、逐步加模型、学习率作用）。
安装 LightGBM（若本地）：pip install lightgbm（或在 Colab 直接 pip 安装）。
读 LightGBM 快速开始，并把默认训练命令写入 notebook： import lightgbm as lgb; lgb.train(params, lgb.Dataset(X_train,y_train), num_boost_round=100)（示例代码）。
产出：week5_day4_notes.md + LightGBM 参考代码片段。
周五（LightGBM baseline 训练）
在 notebook 中用 lightgbm 的 sklearn 接口训练： from lightgbm import LGBMClassifier; model = LGBMClassifier(random_state=42); cross_val_score(model, X, y, cv=5).
比较 RF 与 LGBM 的 CV score，写一段短结论（哪一个更好、可能原因）。
把训练日志（每折分数）保存到 experiments/baseline_scores.csv（包含模型名、fold、score）。
产出：baseline_scores.csv + notebook 更新。
周六（整理 + 提交）
把本周所有代码整理成 week5_trees_baseline.ipynb 的单一目录结构（Data/Models/Results/Notes）。
在 repo 写一个小 README（week5/README.md）：如何复现（Colab 链接或本地命令），列出依赖（pip freeze > requirements.txt）。
Git 提交并 push（git add、commit、push），在 issue tracker 写本周卡点（如果有）。
产出：commit 到 GitHub，README 与 requirements.txt。
周6（Pipeline 与预处理）

周一（学习 Pipeline 基本概念）
在 notes 中列出 ColumnTransformer、Pipeline 的作用与常见错误（例如：漏 fit-transform 的顺序）。
看 sklearn Pipeline 的官方示例，并复制到 week6_pipeline_tune.ipynb（保留注释）。
产出：week6_day1_notes.md + 复制的示例 notebook cell。
周二（实现数值/类别处理 pipeline）
在 notebook 里写预处理 pipeline：
数值管道：SimpleImputer(strategy=‘median’) + StandardScaler()
类别管道：SimpleImputer(strategy=‘constant’, fill_value=‘NA’) + OneHotEncoder(handle_unknown=‘ignore’)
ColumnTransformer 将两者组合。
运行 ColumnTransformer.fit_transform(X_train) 并检查输出 shape（确认 one-hot 扩展列数）。
产出：preprocessor 对象保存在 models/preprocessor.pkl。
周三（把 pipeline 嵌入到整体流程）
构建完整 Pipeline： from sklearn.pipeline import Pipeline; pipe = Pipeline([(‘pre’, preprocessor), (‘clf’, LGBMClassifier())])
用 cross_val_score(pipe, X, y, cv=5) 做 5-fold CV 并记录时间（用 time.time() 记录运行时间）。
将 CV 得分写入 experiments/pipeline_cv.csv。
产出：pipeline_cv.csv。
周四（类别编码对比实验）
在 notebook 中实现两套 pipeline：一套用 OneHotEncoder，另一套用目标编码（可用 category_encoders.TargetEncoder 或自己做 K-fold target encoding）。
为避免泄露，若使用 target encoding 实现 K-fold mean-encoding：写函数 kfold_target_encode(X, y, col, n_splits=5) 并输出编码后的列。
运行两套 pipeline 的 5-fold CV，比较结果，记录到 experiments/encoding_comparison.csv。
产出：encoding_comparison.csv + target encode 函数实现。
周五（GridSearch/RandomizedSearch 基本调参）
在 notebook 中设置简单的 param_grid（例如：n_estimators: [50,100], max_depth: [6,8,10]）并运行 RandomizedSearchCV（n_iter=10，cv=3，n_jobs=-1）。
保存 best_params_ 到 experiments/best_params.json。
产出：best_params.json。
周六（整理与提交）
将 pipeline 代码抽成 train_pipeline.py（可接收 config.json）；确保可以从命令行运行： python train_pipeline.py --config configs/pipeline_config.json
更新 week6_pipeline_tune.ipynb，写如何用 Colab 运行 train_pipeline.py 的说明。
Git 提交并 push，写一个短 issue：下周需尝试的超参/特征。
产出：train_pipeline.py + configs 示例。
周7（特征工程与解释）

周一（高基数类别处理）
在 notebook 中统计每个类别列的唯一值数量（df[col].nunique()），挑出高基数 (>50) 列列表。
对这些列实现频率编码（freq = df[col].value_counts(normalize=True)），并把频率列加入到 X。保存编码映射到 encoders/high_cardinality.json。
产出：encoders/high_cardinality.json。
周二（实现 K-fold target encoding 安全版本）
复用上周 target encode 概念，封装成函数 safe_target_encode(X, y, col, n_splits=5, seed=42) 返回 transformed_series。
在 notebook 上对一个高基数类别做对比实验：one-hot vs freq vs target-encode 的 CV 结果。记录表格。
产出：encoding_experiment_result.csv。
周三（groupby 聚合特征）
挑选 2 个组合（例如 user_id 与 device_type）计算如下聚合：count、mean(label)、std、last_value（若有时间），并把这些新列加入训练集。
将 groupby 代码写成函数 make_agg_features(df, group_cols, agg_defs) 并放到 features/feature_utils.py。
产出：feature_utils.py + 新的训练 CSV（data_with_agg.csv）。
周四（SHAP 基本用法）
安装 shap（pip install shap），在 notebook 中对训练好的 LGBM 模型运行 shap.TreeExplainer(model).shap_values(X_sample)。
生成并保存两个图：summary_plot 与 dependence_plot（png）。在 notebook 中解释 top-5 特征为何重要（写 3–4 行结论）。
产出：shap_summary.png、shap_dependence_*.png。
周五（特征选择与降维快速尝试）
使用 sklearn.feature_selection.SelectKBest 或基于 feature importance 的阈值筛选出 top 30 特征，记录被移除的特征列表。
对比使用全部特征与只用 top30 的 CV 分数，写一行结论（是否下降/提升）。
产出：feature_selection_result.csv + removed_features.txt。
周六（整理周成果）
把本周新增的脚本（feature_utils.py、encoders 等）放入 repo 的 features/ 目录，写一个 small README：如何复现特征工程（steps）。
Git 提交并 push，写周报（week7_report.md）列出下一步要做的 stacking 策略草案。
产出：repo 提交 + 周报。
周8（Ensemble 与项目 v0 整理）

周一（学习 stacking 思路与简单代码）
阅读一篇 short blog “stacking for beginners”，并在 notes 提炼成步骤：第一层模型训练并输出 oof preds，第二层用 LR 拟合 oof preds。
在 notebook 写 stacking 函数 skeleton：def get_oof_preds(models, X, y, n_folds=5): …。
产出：stacking_skeleton.py（含 docstring）。
周二（实现 2 层 stacking）
选择三种基模型（LR、RF、LGBM），实现 get_oof_preds 并生成第一层的 oof 特征（保存到 data/oof_features.csv）。
训练二层模型（LogisticRegression）并记录 CV 得分。
产出：data/oof_features.csv + stacking_result.csv。
周三（对比 ensemble 对不同 seed/split 的鲁棒性）
用 3 个不同 seed 重复 stacking 流程，记录每次的 CV 得分，计算 mean/std（写入 experiments/stacking_seeds.csv）。
分析：若 std 较大，写出可能原因（数据泄露、过拟合第二层、基模型多样性不足）。
产出：stacking_seeds.csv。
周四（写项目 v0 的 README）
README.md 包含：问题描述、数据来源、baseline 模型与 CV 分数、如何复现（Colab 链接或本地 commands）、已知问题与下一步计划（3 条）。
在 repo 根目录建一个 demo/ 目录，放入 60 秒的项目简介脚本 demo/script.txt（说点要点，方便录音/录屏）。
产出：README.md + demo/script.txt。
周五（录制 3 分钟 demo）
使用屏幕录制工具（Loom/OBS），录 3 分钟：1) 问题 2) 你的 pipeline 3) 关键结果 4) 下一步。保存为 demo/week8_demo.mp4，上传到 repo 或 GDrive 并把链接写入 README。
产出：demo 视频链接写入 README。
周六（清理 + tag release）
确认所有 notebook/script 都能在 Colab 运行（至少跑一次），fix 小 bug。
在 GitHub 打 tag：git tag -a v0.1 -m “project v0 baseline”；git push --tags。
周末复盘并写下下月（第3月）重点：CV 策略、稳定性、Optuna 调参。
产出：release tag + 周报。
```

总体约定（再次确认）

每天约 2–2.5 小时（周一–周六），周日 1 小时复盘并更新日志。
每周必须在 notebooks/ 下提交或更新一个对应 notebook（例如 week9_cv_reproducibility.ipynb）。
所有实验记录放 experiments/（CSV/JSON），模型 artifacts 放 models/，重要脚本放 src/ 或 scripts/。
每次实验都记录随机种子（random_state）并写到 experiments/records.csv 中。
—— 周9（主题：CV 策略与可复现训练） ——
目标：掌握不同 CV 策略并把训练流程脚本化（可传 seed/config）以保证复现。

周一
新建 notebooks/week9_cv_reproducibility.ipynb，写今日目标与要比较的 CV 策略列表（kfold, stratifiedKFold, GroupKFold, TimeSeriesSplit）。
加载数据（示例代码 df = pd.read_csv(‘data/train.csv’)），并写 helper 函数 get_Xy(df)。
产出：notebook 初始框架 + get_Xy 函数。
周二
实现并运行 StratifiedKFold（sklearn.model_selection.StratifiedKFold，n_splits=5），写代码打印每个 fold 的样本分布（value_counts）。
运行简单模型（LGBMClassifier）在每个 fold 上并保存 fold score 到 experiments/cv_stratified.csv（列：fold, seed, score）。
产出：experiments/cv_stratified.csv。
周三
实现 GroupKFold（若数据有 group 列，如 customer_id），写 demo：如何指定 groups 参数并保存分组统计（每 fold 的 group count）。
运行并保存结果 experiments/cv_group.csv。
产出：experiments/cv_group.csv。
周四
如果数据为时间序列，示范 TimeSeriesSplit 的使用（保留顺序、不洗牌），并记录每个 split 的时间范围（min/max dates）。
在 notebook 写一段说明：何时用时间 split（5–6 行）。
产出：notebook 中的时间 split 小结。
周五
把训练流程脚本化：创建 src/train.py，支持命令行参数 --cv_type, --seed, --model, --config。用 argparse 解析并在 experiments/records.csv 追加一行（cv_type,seed,model,score,time）。
本地运行示例： python src/train.py --cv_type stratified --seed 42 --model lgbm --config configs/train_config.json
产出：src/train.py + configs/train_config.json。
周六
在 notebooks/week9_cv_reproducibility.ipynb 中总结不同 CV 策略的比较（表格/一段结论），并 push 到 GitHub。
写周报 week9_report.md（包含下周要尝试的 5 个 hyperparameters）。
产出：Git commit & push。
—— 周10（主题：Optuna 自动化调参） ——
目标：用 Optuna 做中等规模调参并保存最佳配置与试验日志。

周一
安装并导入 optuna（pip install optuna），阅读快速开始文档并把 minimal example 复制到 notebooks/week10_optuna_tuning.ipynb。
产出：notebook 初始 cell。
周二
在 notebook 中定义目标函数 objective(trial)：读取 configs/train_config.json，随机采样 params（num_leaves、learning_rate、n_estimators、max_depth 等），用 cross_val_score 返回平均 CV 得分。
产出：objective 函数代码。
周三
运行 optuna.study = optuna.create_study(direction=‘maximize’)，study.optimize(objective, n_trials=50, n_jobs=1)。把 study 保存为 experiments/optuna_study.pkl 与 experiments/optuna_trials.csv（包括每次 trial 的 params、score、time）。
产出：experiments/optuna_trials.csv。
周四
用 optuna.visualization.plot_param_importances(study) 和 plot_optimization_history(study) 生成图并保存（png），写 3 行分析（哪些参数重要，可能进一步搜索空间调整）。
产出：plots/optuna_importance.png。
周五
用 best_params = study.best_params 配置训练最终模型，训练整个训练集并保存模型 models/lgbm_optuna_best.pkl，同时记录验证 score。
产出：models/lgbm_optuna_best.pkl。
周六
在 notebooks/week10_optuna_tuning.ipynb 写一段“如何复现”并 push，写周报（包括 n_trials 与最优得分）。
产出：Git 提交与周报 week10_report.md。
—— 周11（主题：错误分析与分群改进） ——
目标：定位模型在哪些子群上表现差并做针对性特征改进。

周一
在 notebooks/week11_error_analysis.ipynb 写出错误分析的步骤 checklist（按分布/按 feature bucket/按时间）。
产出：notebook 初始框架。
周二
运行模型预测于验证集，生成 predictions 与 probabilities，生成 confusion_matrix，并将错分样本导出 experiments/errors_fold0.csv（包含真实、预测、prob、关键特征列）。
产出：experiments/errors_fold0.csv。
周三
对错分样本按某些候选特征（如 region、user_type、time_bin）做 groupby 统计：error_rate = wrong / total，保存到 experiments/error_by_group.csv。
找出 error_rate 高于整体的 top3 子群。
产出：experiments/error_by_group.csv + top3 子群列表。
周四
针对其中一个 top 子群设计特征：例如对 time 相关子群增加 rolling_count/last_event_interval 或对 location 子群增加 geo-encoding（lat/lon -> cluster id）。实现特征函数 add_subgroup_features(df). 放入 features/custom_features.py。
产出：features/custom_features.py。
周五
重新训练模型（pipeline 包含新特征），记录 per-group metric（整体与 top 子群），保存到 experiments/error_improvement.csv（比较 before/after）。
产出：experiments/error_improvement.csv。
周六
在 notebook 中写一小段结论（量化改进）并更新简历草稿的项目成果（如 “对 A 子群提升了 X%”）。Push & 写周报。
产出：更新的项目 README 与周报。
—— 周12（主题：稳定性与训练/推理一致性） ——
目标：检测并修复训练时与推理时的 feature 偏差与数据漂移问题。

周一
写 notebooks/week12_stability.ipynb 的目标：检测训练/推理分布差异（feature drift）。添加函数 compare_distribution(train_col, prod_col)（KS-test / wasserstein）。
产出：notebook 框架 + compare_distribution 函数。
周二
模拟推理分布（可用 holdout/近期数据），对主要数值特征做 KS test，并将 p-values 保存到 experiments/feature_drift.csv（列：feature, pvalue, drift_flag）。
产出：experiments/feature_drift.csv。
周三
若发现 drift，尝试解决方案之一：对训练数据做时间加权训练（older samples 权重较低），实现 sample_weight 传入训练函数并比较效果，记录 experiments/time_weighting.csv。
产出：experiments/time_weighting.csv。
周四
实现训练与推理 pipeline 的一致性检查小脚本 src/check_pipeline_consistency.py：加载 preprocessor.pkl，给定 sample 输入，比较训练 pipeline 输出（保存的 train stats）与当前推理输出（shape/dtype/NaN presence）。
产出：src/check_pipeline_consistency.py。
周五
写建议实践清单到 notebooks（如 feature contracts、schemas、统计摘要每日报表），并把示例 contract 存到 configs/feature_contract.json。
产出：configs/feature_contract.json。
周六
总结本周修复的要点，更新项目 README 的“生产注意事项”一节，并 push。
产出：周报与 README 更新。
—— 周13（主题：算法实现目标选择与初版实现） ——
目标：选择并开始实现一个算法模块（例如决策树分裂、简化 GBDT 增益计算或 k-means++），先实现正确性版本。

周一
在 notebooks/week13_alg_impl.ipynb 写需求文档（why, scope, input/output spec, complexity目标）。选择具体目标并写 pseudo-code。
产出：notebook + pseudo-code。
周二
新建 src/alg/impl.py，搭建 class Skeleton（例如 class SimpleGBDT: fit(), predict()），实现数据结构与接口。
产出：src/alg/impl.py 初版。
周三
实现核心算法步骤（例如单轮残差计算与一次树的拟合或 k-means 的 one-iteration），用小合成数据验证输出与 sklearn 相近。
产出：tests/test_alg_basic.py（简单 unit test）。
周四
写复杂度注释（每步的时间/空间复杂度估计）并把简单示例输出写入 notebooks（与 sklearn 对比误差/输出）。
产出：notebook 对比表格。
周五
修复 bug，增加边界条件（空类别、单样本、全相同值），并把实现提交 models/impl_v0.pkl（若有需要可保存中间结构）。
产出：models/impl_v0.pkl + commit。
周六
运行一次针对小数据集的完整 benchmark（时间/结果正确性），把结果写到 experiments/alg_impl_v0.csv。写周报。
产出：experiments/alg_impl_v0.csv。
—— 周14（主题：性能剖析与向量化/Numba） ——
目标：定位热点并用向量化/Numba 做加速实验。

周一
在 notebooks/week14_profile_optimize.ipynb 加入 profiler 使用示例（import cProfile; cProfile.run(‘func()’)），并对 src/alg/impl.py 的 fit() 做 profile。保存 raw profile。
产出：profiles/impl_v0.prof。
周二
分析 profile 输出，找出占用最多时间的函数/循环（记录 top3），并在 notebook 中标注。
产出：notebook 中的热点列表。
周三
针对一个热点函数尝试 NumPy 向量化改写（避免 Python for-loop），实现后对比时间（timeit 或 time.time），记录 speedup 到 experiments/optimize_vectorize.csv。
产出：experiments/optimize_vectorize.csv。
周四
对另一个热点尝试 Numba（@njit），在 notebook 中演示如何装饰函数并测基准（注意 warmup），保存结果。
产出：experiments/optimize_numba.csv。
周五
比较三种实现（原始 / 向量化 / Numba），写一段结论（memory tradeoffs，也给出何时选用）并 push。
产出：plots/optimize_comparison.png + 周报。
周六
若 speedup 明显，把优化代码合并回 src/alg/impl.py（带注释），并写 CHANGELOG（docs/CHANGELOG.md）。
产出：更新代码与 CHANGELOG。
—— 周15（主题：C++ / pybind（可选）或继续深度优化） ——
目标：若需要更高性能，尝试把关键函数写成 C++ 并通过 pybind11 调用；否则继续优化 Python 版并完善测试。

周一
决策是否做 C++：在 notebook 中记录当前 Python 实现的瓶颈（是否需要更高性能）。若决定做 C++，写一个简单 plan（文件：docs/cpp_plan.md）。
产出：docs/cpp_plan.md（或决策记录）。
周二
若做 C++：搭建 pybind11 小 demo（参考 pybind11 docs），写一个 C++ 函数 add(int, int) 并通过 python 调用，确保环境可编译（CMakeLists 或 setup.py）。
产出：cpp_bindings/demo 示例运行成功。
周三
把一个热点函数（如累加/聚合计算）用 C++ 实现并在 pybind11 中包装，编译并在 Python 中做基准对比。记录 experiments/cpp_benchmark.csv。
产出：experiments/cpp_benchmark.csv。
周四
若不做 C++，则把重点放在内存优化（使用 dtype 压缩、inplace 操作、避免临时数组），并记录 memory_usage（psutil or memory_profiler）。
产出：experiments/memory_optimization.csv。
周五
完善单元测试（tests/test_alg_*），覆盖 C++ 绑定或 NumPy 版本的边界条件与数值稳定性。
产出：tests/ 通过 CI 的测试脚本（本地运行）。
周六
将优化结果写入 docs/performance_report.md（包含环境、数据规模、每种实现时间/内存），并 push。
产出：docs/performance_report.md。
—— 周16（主题：可复现 benchmark 与 release） ——
目标：把实现打包为可复现的 benchmark（脚本一键运行），并发布 release。

周一
写 scripts/run_benchmarks.sh：接受参数 --size --impl（python/numba/cpp），自动记录结果到 experiments/benchmarks.csv。
产出：scripts/run_benchmarks.sh。
周二
在 CI 或本地 Docker 环境下运行多规模（n=1e3,1e5,1e6）的 benchmark，确保结果可复现并把 CSV 上传。
产出：experiments/benchmarks.csv。
周三
更新 README，写明如何一键运行 benchmark（示例命令），并打 tag v1.0：git tag -a v1.0 -m “alg implementation v1.0”。
产出：tag v1.0。
周四
准备 3–5 分钟的 demo（演示实现 + 加速比 + 场景意义），把 demo 脚本放在 demo/。录制并上传视频。
产出：demo/alg_impl_demo.mp4。
周五
发布 GitHub release（包含二进制或 wheel if available），并在 repo 首页写“benchmark summary”。
产出：GitHub release。
周六
月中或月末复盘（若此为该月末），总结可在简历/项目页使用的量化数据并 push。
产出：周报。
—— 周17（主题：FastAPI 服务基础） ——
目标：为表格项目 v1 构建基本预测 API（输入验证、model load、feature pipeline 调用）。

周1（周一）
在 src/api/ 新建 app.py，初始化 FastAPI： from fastapi import FastAPI; app=FastAPI(); 写根路由 GET /ping 返回 {“ok”:True}。
产出：src/api/app.py 基础框架。
周二
定义输入模型（pydantic）： class PredictRequest(BaseModel): feature1: float; feature2: str; …；写 POST /predict 接口 stub（返回 dummy）。
产出：src/api/app.py 有 POST /predict stub。
周三
在 /predict 中实现加载 pipeline（joblib.load(‘models/preprocessor.pkl’)）与模型（models/lgbm_optuna_best.pkl），把请求 JSON 转为 DataFrame，调用 pipeline.transform -> model.predict_proba 返回概率。
产出：api 可返回真实预测（本地 uvicorn src.api.app:app --reload 测试）。
周四
添加输入校验与异常处理（400 on bad input），添加 logging（记录 request_id、latency）。
产出：logs/ 下产生的简单 request log（本地运行一段时间后）。
周五
写简单自动化测试 scripts/test_api.sh：用 curl POST JSON 验证返回格式与时间（记录到 experiments/api_latency.csv）。
产出：scripts/test_api.sh + experiments/api_latency.csv。
周六
在 notebooks/week17_fastapi_demo.ipynb 写如何在 Colab 或本地运行 API 的步骤并 push。写周报。
产出：notebook + push。
—— 周18（主题：Docker 化服务） ——
目标：把 API 容器化并测试本地 Docker 运行。

周一
在 repo 根写 Dockerfile（基础 python:3.10-slim，复制 requirements.txt，COPY src/ /app，CMD [“uvicorn”, “src.api.app:app”, “–host”,“0.0.0.0”,“–port”,“8080”]）。
产出：Dockerfile。
周二
在本地 build 镜像： docker build -t myproject-api:latest .；运行容器并做 smoke test（curl /ping /predict），保存运行日志。
产出：build 成功截图或日志。
周三
写 docker-compose.yml（如果需要 redis/celery 做异步），并用 docker-compose up 模拟。
产出：docker-compose.yml。
周四
将镜像 push 到 GitHub Container Registry 或 Docker Hub（需要登录），写推送脚本 scripts/push_image.sh。
产出：scripts/push_image.sh。
周五
在 notebooks/week18_docker_deploy.ipynb 写如何在云主机（或本地 VM）运行该镜像的步骤（port mapping、env vars、volume mounts for models）。
产出：notebook 更新。
周六
写部署 README（如何回滚、如何查看 logs、如何限制内存/CPU），并 push。写周报。
产出：README 部署节与周报。
—— 周19（主题：ONNX 导出与推理加速） ——
目标：尝试把模型导出为 ONNX（若兼容）并使用 ONNX Runtime 做延迟对比。

周一
在 notebooks/week19_onnx_benchmark.ipynb 写目标：目标模型（sklearn/XGBoost/LightGBM 是否可导出）。尝试用 skl2onnx 或 onnxmltools 导出简单 sklearn pipeline。
产出：notebook 框架。
周二
若模型可导出：执行导出代码（示例： from skl2onnx import convert_sklearn; onx = convert_sklearn(pipe, initial_types=…); with open(‘models/model.onnx’,‘wb’) as f: f.write(onx.SerializeToString())）。
验证 ONNX 输出与 sklearn 输出一致（对相同输入断言近似相等）。
产出：models/model.onnx + 验证代码。
周三
使用 onnxruntime.InferenceSession 加载模型并做 predict，测单次延迟（time.perf_counter），记录到 experiments/onnx_latency.csv。
产出：experiments/onnx_latency.csv。
周四
对比不同运行配置：onnxruntime CPU 单线程 vs 多线程 vs sklearn pipeline（Python），并记录 throughput/latency。
产出：plots/onnx_vs_sklearn_latency.png。
周五
如果 LightGBM 无法直接导出，可考虑用 treelite 或把模型转为 XGBoost 再导出，写尝试记录（docs/onnx_attempts.md）。
产出：docs/onnx_attempts.md。
周六
总结优化建议（batching、preprocessing 缓存、use onnxruntime with op-level optimizations），写周报并更新部署文档。
产出：周报与 docs 更新。
—— 周20（主题：压力测试与监控基础） ——
目标：对服务做并发压力测试并加入基本 telemetry/logging。

周一
在 notebooks/week20_deployment_prod_ready.ipynb 记录压力测试计划（并发数、QPS、payload size）。准备测试工具（ab/hey/locust）。
产出：notebook 计划。
周二
使用 hey 或 ab 做并发测试（示例： hey -n 1000 -c 50 http://localhost:8080/predict），收集 latency 分位数并保存到 experiments/load_test_before.csv。
产出：experiments/load_test_before.csv。
周三
在 API 中加入简单 telemetry：prometheus_client（pip install prometheus_client），添加 /metrics endpoint 并记录 request_duration_seconds、request_count。
产出：src/api 增加 metrics 行为。
周四
重新运行压力测试并采集 metrics（Prometheus + Grafana 如需可用 docker-compose 简单示例），保存 experiments/load_test_after.csv。
产出：experiments/load_test_after.csv & metrics snapshot。
周五
实施单项优化（例如预加载模型进内存、减少 JSON parsing 或启用 batching），再次测试并记录改进（write experiments/load_test_optimized.csv）。
产出：experiments/load_test_optimized.csv。
周六
把压力测试结论写成短报告（what changed, improvement %, next ops steps），更新部署 README 并 push。
产出：deployment_performance_report.md。
—— 周21（主题：算法题刷题与 ML Q&A） ——
目标：开始高强度刷题并整理 ML 算法问答笔记，准备面试材料。

周一
在 leetcode_solutions/ 新增 week21/ 目录，做 2 道数组/字符串中等题（按模板写解题步骤、代码、复杂度分析），每题 commit。
产出：leecode_solutions/week21/{prob1.py,prob2.py} + README。
周二
做另外 2 道题（双指针/哈希），并在 notebooks/week21_coding_mlprep.ipynb 记录遇到的坑与优化。
产出：新增题目文件 + notebook 更新。
周三
在 docs/ml_q_and_a.md 写 10 个常见 ML 面试问答（GBDT 的梯度为什么是残差？early stopping 如何设置？bias-variance tradeoff 简述）。每条 4–6 行回答并 commit。
产出：docs/ml_q_and_a.md。
周四
练习一轮算法题 “模拟面试”：计时 45 分钟解决中等题并写出白板风格的思路（把过程录屏或写下重点）。
产出：面试记录（recording 或 markdown）。
周五
做一次 ML 算法题目答题练习（例如 “解释 GBDT 的训练流程并写出伪代码”），把答案写入 docs/gbdt_explanation.md。
产出：docs/gbdt_explanation.md。
周六
整理本周的刷题成果，更新错题本（错题原因与正确方法），并 push。写周报。
产出：错题本更新 + 周报。
—— 周22（主题：树/图/系统题 + 行为题准备） ——
目标：加强树/图/堆等题型训练并准备行为题 STAR 答案。

周一
做 2 道树/二叉树题（中等），用递归/迭代分别实现，并写复杂度分析。保存到 leetcode_solutions/week22/tree_*.py。
产出：题目代码与注释。
周二
做 2 道堆/滑动窗口题（中等），并写一个小 note: “什么时候用 heap vs balanced BST vs sliding window”。
产出：代码 + note。
周三
系统题写作：在 docs/system_designs.md 写“如何设计在线特征服务”一页（架构图、component responsibilities、latency budget、consistency strategy），用 ascii 图或简单图片。
产出：docs/system_designs.md + 图片。
周四
行为题准备：写 6 条 STAR 式回答（团队冲突、成就、失败、时间管理、技术挑战、影响力），每条 5–6 行。存 docs/behavioral_answers.md。
产出：docs/behavioral_answers.md。
周五
找人做一次 mock 行为面（或自我录像），回看录像并写 3 个可改进点（tone, structure, 时间控制）。
产出：mock 反馈与改进列表。
周六
将本周题目与要点 push 到 repo，写周报并更新求职公司名单（加入联系人/内推信息）。
产出：周报与求职名单更新。
—— 周23（主题：DP/图/项目演讲练习 + 综合 Mock） ——
目标：强化 DP/图题并反复练习项目 2–3 分钟电梯演讲，做综合 mock。

周一
做 2 道图/并查集题并写解法（union-find 实现）。保存 to leetcode_solutions/week23/。
产出：代码与注释。
周两（二）
DP 训练：做 2 个中等 DP 题（状态定义、转移写清楚），并把状态压缩/优化一遍。保存到 repo。
产出：DP 代码。
周三
写并练习 2–3 分钟项目演讲稿（docs/elevator_pitch.md），逐句计时并录三次，选最佳版本放到项目页。
产出：elevator_pitch.txt + demo recording。
周四
做一次综合 mock（coding 1题 + ML 算法问答 + 行为题），请记录评分（0-5）与 5 条改进项。若没人模拟，可自定题目并计时。
产出：mock_feedback.md。
周五
根据 mock 反馈逐条改进（复写项目演讲、补强逻辑漏洞、优化代码注释），并 push 改动。
产出：修改后的文件与 commit。
周六
整理本周所有 mock 记录，写周报并把准备好的 demo & 项目 pitch 链接放在简历顶部说明。
产出：周报更新与简历草稿改进。
—— 周24（主题：最终冲刺：简历润色、投递、复盘） ——
目标：整理最终交付物、发送首批投递并做最终复盘与下一步计划。

周一
回顾错题本与项目，总结 top 10 学习点写成 docs/learning_summary.md（可直接用作 LinkedIn 帖子）。
产出：docs/learning_summary.md。
周二
简历最终润色：项目描述量化（增加 metric、时间、具体改进数据），将 demo 链接和 GitHub 项目放在简历顶部 Projects 一栏。
产出：简历 PDF（resume_final.pdf）。
周三
准备投递邮件模板或 LinkedIn 私信模板，包含一句项目亮点（用数字支撑），保存到 docs/apply_templates.md。
产出：docs/apply_templates.md。
周四
发送首批 10 份投递（网申/内推/邮件），在 spreadsheet (applications.csv) 记录公司、岗位、联系人、投递时间、跟进时间、状态。
产出：applications.csv（首批记录）。
周五
做一次放松但有节奏的模拟：1 coding 小题 + 1 项目讲述（计时 30–40 分钟），检查状态并记录感觉。
产出：模拟记录。
周六
写 600–800 字学习心得（docs/learning_reflection.md），放到 repo 并可直接贴到 LinkedIn。做 final repo cleanup（README、requirements、license）。
产出：learning_reflection.md + repo cleanup commit。
周日（最终复盘 1 小时）
列出已完成目标、未完成项与未来 3 个月计划（例如实习后技能路线），并给自己设 3 个短期目标（每天 1 小时深造方向）。保存为 docs/next_steps.md。
—— 持续工具与交付物清单（便于管理）

notebooks/week{n}_*.ipynb（每周至少 1 个）
src/ （训练、API、alg 实现脚本）
models/ （保存的 model.pkl / onnx）
experiments/ （所有实验的 CSV/JSON 记录）
tests/ （单元测试）
demo/ （每月 demo 视频）
docs/ （说明、Q&A、系统设计、行为答案等）
leetcode_solutions/（算法题代码）

```
好，那我把你原本的6个月（24周）逐周计划，根据我们刚讨论的方向，完整融合 4 个深度学习专项周（PyTorch入门、CNN图像分类、BERT文本分类、ML+DL项目整合），并且写成统一的逐日执行清单（Mon–Sun，每天有输出目标和交付物），给你一版最终版执行版。

6个月机器学习+深度学习算法工程师实习备战计划（融合版）
第1月：数据分析与ML基础（周1–4）
让你快速熟悉Python数据分析、可视化、基础ML API。

周1–4（保持原版任务，不做改动）

第2月：经典ML与实战（周5–8）
完成表格项目v0的全过程。

周5：决策树/随机森林/GBDT
Mon 阅读决策树原理（信息增益、基尼等）—写笔记
Tue sklearn.DecisionTreeClassifier训练+画树
Wed 随机森林原理，训练RF baseline+CV
Thu 阅读GBDT思想，看LightGBM/XGBoost入门
Fri LightGBM默认参数5折CV baseline
Sat 整理week5_trees_baseline.ipynb，保存CV结果
Sun(复盘) 三点模型差异+改进方向

周6：Pipeline与调参
Mon 学ColumnTransformer+Pipeline
Tue 封装preprocessing pipeline
Wed 编码对比（one-hot / target）+效果记录
Thu pipeline+CV输出fold metrics
Fri 学GridSearchCV、RandomizedSearchCV、早停
Sat 整理week6_pipeline_tune.ipynb
Sun 列出3个下月特征/优化思路

周7：特征工程
Mon 深入类别特征编码风险与止损
Tue 实现K折target encoding
Wed 组合特征&groupby聚合
Thu 模型对比实验表
Fri 学feature importance/partial dependence/SHAP
Sat 整理week7_feature_engineering.ipynb
Sun 复盘错题本

周8：Ensemble与v0项目提交
Mon 学ensemble（bagging/stacking/blending）
Tue 简单stacking实现
Wed 跑stacking对比提升
Thu 写项目v0 README
Fri 录3分钟demo介绍baseline与计划
Sat 整理repo（requirements+model+notebook+视频）
Sun(月末复盘) 上传demo+月总结

第3月：特征工程进阶与模型稳定性（周9–12 + PyTorch入门）
周9：CV策略与复现性
Mon 学不同CV策略
Tue 用合适CV重跑baseline
Wed 多seed实验统计均值/方差
Thu 脚本化训练（train.py）接受seed+config
Fri 保存实验记录CSV/JSON
Sat 整理week9_cv_reproducibility.ipynb
Sun 列5个特征/模型变体想法

周10：Optuna调参
Mon 学Optuna基本用法
Tue 对LightGBM做n_trials=50搜索
Wed Hyperparameter importance分析
Thu 调整stacking元模型参数
Fri 实践early stopping对结果影响
Sat 整理week10_optuna_tuning.ipynb
Sun 更新README最佳实践

周11：错误分析
Mon 按群体分析错误/混淆矩阵
Tue 错误多子群定制特征
Wed 再训练，记录per-group metric
Thu 更深入SHAP解释
Fri 写output_errors.py导出错分样本详情
Sat week11_error_analysis.ipynb整理
Sun 记录3个简历可用改进成果

周11.5（新增）PyTorch入门周
目标：熟悉PyTorch基础
Mon 安装跑完60min入门前半，tensor操作
Tue Dataset+DataLoader读取MNIST，打印shape
Wed 两层全连接网络跑3epoch
Thu 学Optimizer/Loss（SGD/Adam对比）
Fri 模型保存/加载，预测单图
Sat 写PyTorch五大核心模块总结
Sun 复盘+列API盲点

周12：模型稳定性
Mon 学训练/推理特征一致性注意点
Tue 模拟训练/推理差异(drift)案例
Wed 模型压缩/裁剪入门
Thu 升级表格项目为v1（稳健pipeline+调参）
Fri 录3-5分钟v1稳定性改进demo
Sat 整理repo并打release tag
Sun(月末复盘) 下月目标定为算法实现+性能优化

第4月：算法实现与性能优化（周13–16 + CNN周）
周13：算法实现目标确定
（原版保留）

周13.5（新增）CNN图像分类周
目标：PyTorch实现CNN在CIFAR-10
Mon 下载CIFAR-10+基础transform
Tue 定义Conv2d+ReLU+MaxPool网络
Wed 训练10epoch，记录精度
Thu 尝试Dropout/BN
Fri 绘制训练曲线
Sat 封装train_cnn.py支持参数
Sun 复盘CNN优势与卷积核设置影响

周14–16（原版性能优化、可能C++加速）
（保留原版任务）

第5月：部署与在线推理（周17–20 + BERT周 + 项目整合周）
周17：FastAPI服务
（原版保留）

周17.5（新增）BERT文本分类周
目标：HuggingFace微调BERT做IMDB情感分类
Mon 安装transformers，熟悉tokenizer编码
Tue datasets加载IMDB并批量编码
Wed 定义BertForSequenceClassification
Thu Trainer API微调BERT
Fri 保存模型predict_text()
Sat 写预测demo notebook
Sun 总结Transformer核心（自注意力+微调）

周18–20（Docker部署、ONNX导出、压测监控）
（保留原版任务）

周20.5（新增）ML+DL整合周
目标：整合表格ML项目+一个DL项目为portfolio
Mon 整理ML项目结构README
Tue 整理DL项目结构README
Wed 给两个项目录demo视频
Thu README加demo链接和关键指标
Fri 简历项目经历补DL条目
Sat LinkedIn/知乎发项目路线总结
Sun 检查各展示渠道一致性/完整性

第6月：面试冲刺（周21–24）
保留原版（重点刷题+模拟面试+投递）

总结
这样改完后，你会有：

传统ML全流程项目（v0/v1）：数据处理、特征工程、调参、工程化部署
手写算法与性能优化项目：展示底层实现能力
深度学习PyTorch技能+项目：
周11.5：PyTorch基础API
周13.5：CNN图像分类项目
周17.5：BERT文本分类项目
工程化能力：FastAPI、Docker、ONNX优化、部署监控
Portfolio整合：周20.5做成完整作品集（ML+DL）
```

