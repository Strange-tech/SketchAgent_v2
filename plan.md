# SketchAgent 人类化绘画过程 — 修改大纲

## 1. 目标

把当前"按边缘追踪顺序 / 长度排序"的绘制，改成符合人类绘画习惯的顺序：

1. **先树干、后树叶**（语义分割 + 部件级顺序）
2. **自下而上、自左而右**（模拟树木生长方向 + 右手绘画习惯）
3. **树叶先画轮廓、再画细节**

---

## 2. 现状与问题

| 现有产物 | 作用 | 问题 |
|---|---|---|
| `img2sketch_cv.py` | CV 提取笔画（Canny / contour / both） | 几何忠实，但笔画顺序是"追踪序 / 长度序"，非人类序 |
| `naturalize.py` | LLM 自然化（重排顺序 + 平滑 t 值） | 无语义约束、无空间约束，且可能破坏几何 |
| `sketch_animation.html` | 逐笔动画 | 顺序不人类化 |
| `utils.py` | 网格 / SVG / 解析 / `extract_text_from_response` | 可直接复用 |

核心结论（前几轮已验证）：**几何形状应由 CV 从 mask 提取（忠实），"人类化"的实质是确定性的排序规则，不该交给 LLM**（LLM 对精确顺序/几何不可靠）。agent 只做可选的"节奏/风格"润色。

---

## 3. 总体架构（五阶段）

```
输入图像
  → [1] SAM3 语义分割 → trunk_mask + foliage_mask
  → [2] 按部件提取笔画（CV，忠实几何）
        树干：mask 骨架（主干 + 各级枝）
        树叶：mask 外轮廓（outline）+ mask 内部骨架/Canny（detail）
  → [3] 人类化排序（确定性规则，核心，不交给 LLM）
        部件序 → 层次序 → 空间序（下→上、左→右）→ 每笔方向
  → [4] agent 自然化（可选，LLM 只润节奏/风格，不碰几何与顺序）
  → [5] 渲染 + 逐笔动画（复用 utils.format_svg + sketch_animation.html 的 rAF reveal）
```

---

## 4. 模块细节

### 4.1 SAM3 语义分割（新增 `sam3_segment.py`）

- 输入：原图。
- 输出：`trunk_mask`、`foliage_mask`（二值），并注册到 50×50 网格 / 612×612 画布（与现有 sketch 同坐标系）。
- 方式：SAM3 + 文本提示（"trunk" / "foliage"）；若 SAM3 不支持文本提示，则用点/框提示，或接一个轻量语义分割头。
- 待定：树干与树叶重叠/遮挡区域的归属规则（建议重叠区归树干，保证树干连续）。

### 4.2 按部件提取笔画（复用 `img2sketch_cv.py` 的思路，按 mask 分别跑）

- **树干**：`trunk_mask` 骨架化 → 追踪 → 主干 + 各级枝的骨架线（中心线）。
- **树叶**：
  - `outline` = `foliage_mask` 的外边界轮廓。
  - `detail` = `foliage_mask` 内部骨架，或 原图 Canny ∩ foliage_mask 内部（细节太多时降采样/合并）。
- 每部分都输出 `<points>` + `<t_values>`，沿用现有 `strokes.xml` 格式，带 `part`（trunk/foliage）和 `layer`（outline/detail）标签。

### 4.3 人类化排序（新增 `order_strokes.py`，纯 numpy，无 GPU）

排序键（升序）：`(part, layer, anchor_y, anchor_x)`

- `part`：树干 = 0，树叶 = 1 —— **先树干后树叶**。
- `layer`：outline = 0，detail = 1 —— **树叶先轮廓后细节**（树干单层或只 outline）。
- `anchor_y`：该笔画**最下点**的 y —— **自下而上**（网格 y=1 是底部、y=50 是顶部，故按 y 升序）。
- `anchor_x`：最下点对应的 x —— **自左而右**（x 升序，作为次级键，用于同高度多分支/多笔触）。

每笔方向修正：确保**每一笔也自下而上**：
- 若 `points[0].y > points[-1].y`（首点在上、末点在下），则反转 points，并重新做弧长参数化得到 t_values。

### 4.4 agent 自然化（改造 `naturalize.py`，可选）

- 只在 [3] 之后执行，prompt 严格约束：**不改 points 几何、不改笔画顺序**，只微调 t_values 节奏 / 轻微平滑。
- 复用 DeepSeek（`deepseek-v4-flash`）+ `extract_text_from_response`。
- 若仍引入抖动/破坏几何，则跳过此步——确定性排序本身已足够"人类化"。

### 4.5 渲染与动画

- 复用 `utils.format_svg` 出 SVG/PNG；`sketch_animation.html` 的 rAF 逐笔 reveal（顺序 = [3] 输出顺序）。

---

## 5. 排序算法伪代码

```python
def humanize(strokes):  # 每个 stroke 带 part、layer
    for s in strokes:
        pts = s.points
        i = min(range(len(pts)), key=lambda k: (pts[k].y, pts[k].x))  # 最下、最左点
        s.anchor = (s.part, s.layer, pts[i].y, pts[i].x)
        # 每笔自下而上
        if pts[0].y > pts[-1].y:
            s.points = pts[::-1]
            s.t_values = arc_length_t_values(s.points)
    return sorted(strokes, key=lambda s: s.anchor)
```

注意：坐标约定是 **grid `xNyM`，y=1 底部、y=50 顶部**。做空间排序时务必确认 mask 像素坐标 → 网格坐标的映射方向（现有 `utils.pixel_to_cell` 已有该映射，可复用）。

---

## 6. 迁移到服务器的注意事项

- **服务器需要**：GPU + CUDA + SAM3 模型权重（SAM3 依赖较重，本地 Mac 跑不动）。
- **迁移文件**：
  - 复用：`utils.py`、`prompts.py`、`naturalize.py`、`img2sketch_cv.py`（CV 提取部分）、`sketch_animation.html` 模板。
  - 新增：`sam3_segment.py`（分割）、`order_strokes.py`（人类化排序，纯 CPU/numpy）。
- **环境变量**：`.env` 沿用 DeepSeek（`ANTHROPIC_API_KEY` + `ANTHROPIC_BASE_URL`）供 agent 自然化；注意别被宿主机 `ANTHROPIC_BASE_URL` 等覆盖干扰。

---

## 7. 分阶段实施（建议顺序）

1. **阶段 A**：先做 `order_strokes.py`（不依赖 SAM3，拿现有 `strokes.xml` 假数据即可验证排序逻辑 + 动画顺序）。
2. **阶段 B**：接 SAM3 出 trunk/foliage mask，验证分割质量。
3. **阶段 C**：按 mask 分别提取树干骨架 / 树叶 outline+detail。
4. **阶段 D**：三件套串起来（分割 → 提取 → 人类化排序），出动画。
5. **阶段 E**：（可选）接 agent 自然化润色节奏。

---

## 8. 验证方式

- 动画慢放（0.5×）肉眼检查：树干先出现、自下而上；树叶先轮廓后细节、自下而上、自左而右。
- 同一张树图做**旧顺序 vs 新顺序**的并排动画对比。

---

## 9. 待定问题

1. SAM3 是否支持文本提示（决定 text prompt 还是 point/box + 语义头）。
2. 树干/树叶重叠区域的归属规则。
3. 树叶 `detail` 用 mask 内部骨架还是原图 Canny；太碎时如何降采样/合并。
4. agent 自然化是否保留（若破坏几何则跳过，纯确定性排序）。
