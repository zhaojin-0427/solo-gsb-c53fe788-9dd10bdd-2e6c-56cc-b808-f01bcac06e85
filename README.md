# 色谱定量 HTTP 服务 (Chromatography Quantification API)

基于 **Python / FastAPI / NumPy / SciPy (HiGHS)** 的多组分色谱定量服务。

给定至多 **6** 种纯品（标准品）的时间–强度曲线及其已知正浓度、参考通道中**带编号**的内标出峰时刻、样本通道中**不带编号**的候选内标时刻（可含漏检与杂峰）、样本定量通道曲线、检测器饱和值、容差与允许的漂移斜率范围，服务：

1. **锚点匹配（内标校时）**——枚举所有满足以下条件的内标配对：
   - 至少 2 对锚点；
   - 参考、候选两侧都**严格保序且不复用**；
   - 每段（参考时刻差 / 样本时刻差）斜率落在给定 `slope_range` 内；
   - 允许参考内标**漏检**、候选中存在**杂峰**（未被配对的时刻即忽略）。
2. **共用分段线性校时**——以匹配的锚点链建立样本时间→参考时间的分段线性映射（首/末锚点之外用最边段斜率外推），把每个纯品模板插值到锚点之间的样本点上。
3. **线性叠加定量**——样本预测值为各模板的非负线性组合 `P·α`，组分浓度 = `α × 纯品浓度`。
4. **误差判定（线性规划，结果可证）**：
   - 未饱和点：`|预测 − 实测| ≤ tolerance`；
   - 饱和点（`强度 ≥ saturation`）：只需 `预测 ≥ saturation − tolerance`（单侧，允许更高）。
5. **先最大化可行锚点配对数**，再在**所有并列的最优解释**上给出每个组分浓度的**下界 / 上界**与锚点证据。重叠峰**不强行唯一归属**：若两种归属都可行，二者都会出现在证据与浓度区间中；上界无界时记为 `null`。

所有浓度界都来自线性规划在完整（非抽样）约束下的极值，是对所述模型的**可证（certified）**结果。

---

## 一、一键启动（Docker Compose）

需要 Docker 与 Docker Compose v2。

```bash
docker compose up --build
```

服务监听 **http://localhost:8000** 。健康检查：

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

停止：

```bash
docker compose down
```

> 无需 Docker 时也可本地运行：
>
> ```bash
> pip install -r requirements.txt
> uvicorn app.main:app --host 0.0.0.0 --port 8000
> ```

---

## 二、接口

### `POST /quantify`

`Content-Type: application/json`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `standards` | 数组(1..6) | 纯品列表 |
| `standards[].times` | number[] | 纯品时间轴，严格递增 |
| `standards[].intensities` | number[] | 纯品强度曲线，长度与时间轴相同，非负 |
| `standards[].concentration` | number > 0 | 该纯品的已知浓度 |
| `reference_times` | number[] | **带编号**参考内标时刻，严格递增 |
| `candidate_times` | number[] | **无编号**样本候选内标时刻，非递减（可含杂峰、漏检） |
| `sample_times` | number[] | 样本时间轴，严格递增 |
| `sample_intensities` | number[] | 样本定量通道强度，非负；`≥ saturation` 视为饱和点 |
| `saturation` | number > 0 | 检测器饱和值 |
| `tolerance` | number > 0 | 单点允许误差 |
| `slope_range` | `[low, high]` | 允许的段斜率（参考/样本），需 `0 < low ≤ high` |

### 响应

`status` 取值：

| status | 含义 |
| --- | --- |
| `ok` | 找到可行解释，返回组分浓度界与锚点证据 |
| `invalid_input` | 非法数值（NaN/无穷/负值）、长度不一致、时间轴非递增、斜率区间非法等（HTTP 400/422） |
| `insufficient_anchors` | 满足保序、不复用、斜率范围的锚点不足 2 对 |
| `infeasible` | 锚点结构存在，但没有任何匹配能在容差内解释样本（无可行解释） |

`ok` 时：

- `max_anchors`：可行的最大锚点配对数；
- `components[i].concentration_lower / _upper`：在**全部并列最优解释**上该组分浓度的可证上下界，上界无界为 `null`；
- `evidence.common_anchors`：所有解释共有的锚点（唯一确定的配对）；
- `evidence.possible_anchors`：至少一个解释中出现的锚点；
- `evidence.explanations`：并列解释（每条是一组 `{reference_index, candidate_index}`，索引分别对应 `reference_times` 与 `candidate_times` 的下标）。解释过多时列表截断（`explanations_truncated=true`），但**浓度极值与锚点并/交集仍在全部解释上精确计算**。

---

## 三、请求 / 响应示例

下列示例含 2 个组分、3 个带编号参考内标、4 个候选时刻（其中 `29.9` 为杂峰，被正确排除）。完整请求见 [`examples/request.json`](examples/request.json)，完整响应见 [`examples/response.json`](examples/response.json)。

```bash
curl -X POST http://localhost:8000/quantify \
  -H 'Content-Type: application/json' \
  --data @examples/request.json
```

响应（节选；数组长度与实际文件一致）：

```json
{
  "status": "ok",
  "max_anchors": 3,
  "structural_max_anchors": 3,
  "components": [
    { "index": 0,
      "concentration_lower": 78.0,
      "concentration_upper": 82.0 },
    { "index": 1,
      "concentration_lower": 96.0,
      "concentration_upper": 104.0 }
  ],
  "evidence": {
    "common_anchors": [
      { "reference_index": 0, "candidate_index": 0 },
      { "reference_index": 1, "candidate_index": 2 },
      { "reference_index": 2, "candidate_index": 3 }
    ],
    "possible_anchors": [
      { "reference_index": 0, "candidate_index": 0 },
      { "reference_index": 1, "candidate_index": 2 },
      { "reference_index": 2, "candidate_index": 3 }
    ],
    "explanations": [
      [
        { "reference_index": 0, "candidate_index": 0 },
        { "reference_index": 1, "candidate_index": 2 },
        { "reference_index": 2, "candidate_index": 3 }
      ]
    ],
    "explanations_truncated": false,
    "num_explanations": 1
  }
}
```

真实配比为组分 0 浓度 80、组分 1 浓度 100；可证区间分别为 `[78, 82]`、`[96, 104]`（随容差收紧），杂峰 `29.9`（候选索引 1）未被使用。

### 重叠峰不唯一归属

当两个组分在同一位置出峰、两种归属都满足容差时，服务返回多条并列 `explanations`，`common_anchors` 只包含真正唯一确定的锚点，浓度区间跨两种归属，例如两个同形共流出组分各为 `下界≈0 … 上界≈总含量`，**不会**把重叠峰强行判给某一组分。

### 无界上界

若某组分模板在所有（被采样/被饱和约束触及的）时刻上均无贡献，其浓度无法被上界约束，`concentration_upper` 返回 `null`。

---

## 四、在线进样性能

针对要求的在线规模——**12 个参考内标、24 个候选时刻、8192 个样本点、6 个组分**（含真实分段漂移、12 个杂峰、检测器饱和与两个零含量组分）——在普通多核主机上端到端约 **3 秒**返回已证结果（预算 10 秒）。可复现：

```bash
python benchmark.py     # 断言 ≤ 10s，并校验真实含量落在可证区间内
```

实现要点（保证延迟）：

- 内标匹配建模为有向无环图上的最长保序链，两阶段搜索（松弛 LP 求最长 → 定长枚举全部并列解释）；
- 用“均匀子样 + 所有结合性极值点（峰尖/平台沿）”的**松弛**分段 LP 提前剪除不可能的边与前缀，只对幸存链构建覆盖全部 8192 点的完整 LP；
- LP 通过 **HiGHS 原生 API** 一次建模型、热复用求解 2K 个浓度极值（一次可行性判定近乎免费），并对约束行做保可行域的精确缩放。

---

## 五、测试

```bash
python smoke_test.py          # 基本定量、饱和、非法输入、锚点不足、无可行解释
python correctness_test.py    # 重叠峰歧义、null 上界、漏检、杂峰、容差/饱和/斜率语义
python benchmark.py           # 12/24/8192 在线规模性能与界正确性
```

## 目录结构

```
app/
  main.py      FastAPI 路由、请求校验、错误状态
  solver.py    输入校验、锚点链枚举、分段线性校时、LP 定量与可证区间
examples/      与 README 一致的请求/响应示例
Dockerfile, docker-compose.yml, requirements.txt
benchmark.py, smoke_test.py, correctness_test.py
```
